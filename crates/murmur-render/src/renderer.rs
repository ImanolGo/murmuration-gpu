//! The first real frame (PLAN.md phase 5): a dusk sky, every bird deposited as
//! optical-depth coverage into an HDR target with multiplicative darkness, and
//! a tone-mapped output.
//!
//! The pipeline mirrors the terminal reference's `render/` module:
//!
//! 1. a ground [`camera`](crate::camera) frames the flock;
//! 2. the [`sky`](crate::sky) gradient is written to an `Rgba16Float` target,
//!    sampled by viewing elevation;
//! 3. every bird is drawn as a small footprint whose coverage darkens the HDR
//!    target with `dst *= 1 - coverage` (the plan's multiplicative blend) and,
//!    in the same pass, accumulates its *raw* optical depth into a second
//!    target so the exposure can adapt;
//! 4. the optical-depth target is histogrammed and the exposure is eased so the
//!    80th percentile of the flock's depth maps to about 70 % opacity, exactly
//!    as the reference's `Renderer::adapt_exposure` does;
//! 5. a tone-map pass writes the linear HDR target to the surface.
//!
//! Because the per-bird coverage is `1 - exp(-exposure * raw)`, the product over
//! birds is `exp(-exposure * tau)`: the multiplicative blend and the reference's
//! additive `1 - exp(-tau)` compositor give the same pixel. The exposure is read
//! from the *previous* frame's optical depth, so a frame needs no ordering
//! between the two.
//!
//! Phase 5 draws every bird with the sub-pixel bilinear / resolved Gaussian
//! footprint of the reference's `render/splat.rs`. Phase 6 puts three levels of
//! bird in front of that: a [`mesh`](crate::mesh) for the nearest, the phase-5
//! splat for the smallest, and a flapping impostor quad between them. A compute
//! pass (`shaders/lod.wgsl`) classifies every bird by its projected wingspan and
//! writes the instance lists that feed the three *indirect* draws, so the CPU
//! never touches a bird. All three levels deposit the same optical mass `a·ρ²`,
//! and the thresholds are cross-fades, so a bird that changes level keeps its
//! darkness.
//!
//! Phase 6 also carries a short HDR history (TAA): every frame is sampled at a
//! different sub-pixel offset and blended into the running image, so sub-pixel
//! birds, whose centres cross pixel boundaries every frame, stop shimmering.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use murmur_gpu::BIRD_WGSL;
use murmur_gpu::device::Gpu;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor, Extent3d, FragmentState,
    LoadOp, Operations, Origin3d, PipelineLayoutDescriptor, PrimitiveState, QuerySet,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::camera::Camera;
use crate::mesh;

/// The HDR target format: the sky and the darkness it is multiplied by.
pub const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The optical-depth target format (raw `tau` and its accumulation).
pub const OPTICS_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Exposure before the flock has been seen: the reference's old fixed `k_dark`.
const DEFAULT_EXPOSURE: f32 = 25.0;

/// Optical depth that maps to 70 % opacity: `-ln(1 - 0.7)`.
const TARGET_TAU: f32 = 1.204;

/// Time constant of the exposure adaptation \[s\].
const EXPOSURE_TAU: f32 = 4.0;

/// Bounds on the adaptive exposure.
pub const EXPOSURE_MIN: f32 = 1.0;
const EXPOSURE_MAX: f32 = 400.0;

/// Histogram bins used to locate the 80th-percentile optical depth.
const BINS: usize = 256;

/// Optical depth a full histogram bin range maps to.
const TAU_MAX: f32 = 8.0;

/// Time constant of the temporal accumulation \[s\]: the history keeps about
/// `TAA_TAU` seconds of frames, short enough that the flock's own motion does not
/// smear, long enough to quiet the wingbeat's 10 Hz flicker at 30 fps.
const TAA_TAU: f32 = 0.035;

/// Frames in the jitter sequence. Eight Halton offsets cover a pixel's area
/// evenly enough that a still camera's picture converges in a few frames.
const TAA_SAMPLES: u32 = 8;

/// Frame-time probes: frame start, LOD classify, the bird pass, the exposure
/// histogram, the temporal resolve, the tone map. Read back by
/// [`Renderer::frame_times_ms`].
pub const TIMESTAMP_SLOTS: usize = 6;

/// The per-frame camera block, matching `Scene` in `shaders/scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Scene {
    pub eye: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub _pad3: f32,
    pub viewport: [f32; 2],
    pub fov: f32,
    pub evening: f32,
    pub exposure: f32,
    pub flap_amplitude: f32,
    pub time: f32,
    pub _pad4: f32,
    /// Sub-pixel offset of this frame's sample \[px\], for temporal AA.
    pub jitter: [f32; 2],
    /// Whether a falcon is hunting (1/0).
    pub falcon: f32,
    /// How visible the foreground reed bed is (0…1).
    pub reed: f32,
    /// The falcon's position when `falcon` is 1 \[m\].
    pub falcon_pos: [f32; 3],
    pub _pad6: f32,
}

impl Scene {
    /// Builds the block from a camera and the frame's settings.
    pub fn from_camera(
        camera: &Camera,
        width: u32,
        height: u32,
        evening: f32,
        exposure: f32,
        flap_amplitude: f32,
        time: f32,
    ) -> Self {
        Self {
            eye: camera.eye.to_array(),
            _pad0: 0.0,
            forward: camera.forward.to_array(),
            _pad1: 0.0,
            right: camera.right.to_array(),
            _pad2: 0.0,
            up: camera.up.to_array(),
            _pad3: 0.0,
            viewport: [width as f32, height as f32],
            fov: camera.fov,
            evening,
            exposure,
            flap_amplitude,
            time,
            _pad4: 0.0,
            jitter: [0.0, 0.0],
            falcon: 0.0,
            reed: 0.0,
            falcon_pos: [0.0, 0.0, 0.0],
            _pad6: 0.0,
        }
    }

    /// Sets how visible the reed bed is (see [`crate::sky::reed_visibility`]).
    pub fn with_reeds(mut self, visibility: f32) -> Self {
        self.reed = visibility;
        self
    }

    /// Sets the falcon the coverage pass draws, or clears it.
    pub fn with_falcon(mut self, position: Option<glam::Vec3>) -> Self {
        match position {
            Some(p) => {
                self.falcon = 1.0;
                self.falcon_pos = p.to_array();
            }
            None => {
                self.falcon = 0.0;
                self.falcon_pos = [0.0, 0.0, 0.0];
            }
        }
        self
    }
}

/// A GPU error when building the render pipelines, so the app can report it
/// instead of panicking inside wgpu.
#[derive(Debug)]
pub struct RenderError(pub String);

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RenderError {}

/// The renderer: pipelines, the two HDR targets and the adaptive exposure.
pub struct Renderer {
    scene: Scene,
    scene_buffer: Buffer,
    size: (u32, u32),
    output_format: TextureFormat,

    sky: RenderPipeline,
    mesh: RenderPipeline,
    impostor: RenderPipeline,
    coverage: RenderPipeline,
    /// The falcon's footprint, drawn from the scene alone (PLAN.md phase 7).
    falcon: RenderPipeline,
    /// The foreground reed bed (PLAN.md phase 8).
    reeds: RenderPipeline,
    tonemap: RenderPipeline,
    taa: RenderPipeline,
    exposure_pipeline: ComputePipeline,
    lod_pipeline: ComputePipeline,

    bird_layout: BindGroupLayout,
    lod_layout: BindGroupLayout,
    tonemap_layout: BindGroupLayout,
    taa_layout: BindGroupLayout,
    exposure_layout: BindGroupLayout,

    /// The starling mesh, upload once: the same ten triangles every instance.
    mesh_vertices: Buffer,
    /// The तीन per-level instance lists: `(bird index, weight)` pairs.
    lod_lists: Option<Buffer>,
    /// The three `DrawIndirect` argument blocks the LOD pass counts into.
    lod_args: Buffer,
    /// The bird count the LOD buffers were sized for.
    lod_birds: u32,
    /// Cached bind groups per ping-pong bird buffer `(buffer_addr, count)`.
    bird_bind_cache: Vec<(usize, u32, BindGroup, BindGroup)>,

    sky_bind: BindGroup,
    falcon_bind: BindGroup,
    bird_bind: Option<BindGroup>,
    lod_bind: Option<BindGroup>,
    tonemap_bind: [BindGroup; 2],
    taa_bind: [BindGroup; 2],
    exposure_bind: BindGroup,

    /// The running HDR image the tone map reads: `accum[parity]` is history and
    /// `accum[1 - parity]` is this frame's.
    accum: [Texture; 2],
    accum_view: [TextureView; 2],
    /// The resolve's `blend.x` is this frame's history weight.
    blend: Buffer,
    /// GPU timestamps around the frame's passes, when the adapter can write
    /// them. `None` means the frame time cannot be measured, and callers must
    /// say so rather than reporting a wall clock.
    timestamps: Option<QuerySet>,
    timestamp_buffer: Buffer,
    timestamp_staging: Buffer,
    /// Frames rendered so far, for the jitter sequence and the ping-pong.
    frame: u32,
    /// Cleared when the history must be thrown away (a new or resized target).
    accum_ready: bool,
    /// Overrides the resolve's history weight for the next frame.
    history_weight: Option<f32>,
    /// Overrides the next frame's sample offset.
    fixed_jitter: Option<[f32; 2]>,

    hdr: Texture,
    hdr_view: TextureView,
    optics: Texture,
    optics_view: TextureView,

    histogram: Buffer,
    histogram_staging: Buffer,

    exposure: f32,
    exposure_ready: bool,
    async_exposure: bool,
    exposure_pending: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    exposure_elapsed: f32,
}

/// A prepared full-screen render target the tone-map pass writes into.
pub struct Target {
    /// The texture (sampled or copied from).
    pub texture: Texture,
    /// Its default view, for the tone-map pass.
    pub view: TextureView,
}

impl Renderer {
    /// Builds the renderer for a target of `output_format` and `width × height`.
    pub fn new(
        gpu: &Gpu,
        width: u32,
        height: u32,
        output_format: TextureFormat,
    ) -> Result<Self, RenderError> {
        let device = &gpu.device;
        let width = width.max(1);
        let height = height.max(1);
        // Capture every pipeline/validation error below and report it as a
        // value instead of letting wgpu panic inside `new`.
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let scene_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("render: scene"),
            size: std::mem::size_of::<Scene>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // sky.wgsl and scene.wgsl are prepended so each pass shares one Scene
        // struct, the projection helper and the sky palette.
        let scene_wgsl = include_str!("../shaders/scene.wgsl");
        let sky_wgsl = include_str!("../shaders/sky.wgsl");
        let sky_source = format!(
            "{scene_wgsl}\n{sky_wgsl}\n{}\n{}",
            include_str!("../shaders/atmosphere.wgsl"),
            include_str!("../shaders/sky_pass.wgsl")
        );
        let sky_module = shader(device, "sky", &sky_source);

        // The three bird passes and the LOD classifier all read a `Bird` and the
        // scene, so they share one prelude.
        let bird_pass_wgsl = include_str!("../shaders/bird_pass.wgsl");
        let bird_prelude = format!("{scene_wgsl}\n{BIRD_WGSL}\n{bird_pass_wgsl}");
        let coverage_module = shader(
            device,
            "coverage",
            &format!(
                "{bird_prelude}\n{}",
                include_str!("../shaders/coverage.wgsl")
            ),
        );
        let mesh_module = shader(
            device,
            "mesh",
            &format!(
                "{bird_prelude}\n{}",
                include_str!("../shaders/bird_mesh.wgsl")
            ),
        );
        let impostor_module = shader(
            device,
            "impostor",
            &format!(
                "{bird_prelude}\n{}",
                include_str!("../shaders/impostor.wgsl")
            ),
        );
        let lod_module = shader(
            device,
            "lod",
            &format!("{bird_prelude}\n{}", include_str!("../shaders/lod.wgsl")),
        );
        let falcon_module = shader(
            device,
            "falcon",
            &format!("{bird_prelude}\n{}", include_str!("../shaders/falcon.wgsl")),
        );

        let tonemap_source = include_str!("../shaders/tonemap.wgsl").to_string();
        let tonemap_module = shader(device, "tonemap", &tonemap_source);
        let taa_module = shader(device, "taa", include_str!("../shaders/taa.wgsl"));

        let exposure_module = shader(device, "exposure", include_str!("../shaders/exposure.wgsl"));

        let uniform_entry = |binding: u32, visibility: ShaderStages| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_read = |binding: u32, visibility: ShaderStages| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_write = |binding: u32, visibility: ShaderStages| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding: u32, visibility: ShaderStages| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let sky_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: sky layout"),
            entries: &[uniform_entry(0, ShaderStages::VERTEX_FRAGMENT)],
        });
        let bird_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: bird layout"),
            entries: &[
                uniform_entry(0, ShaderStages::VERTEX_FRAGMENT),
                storage_read(1, ShaderStages::VERTEX),
                storage_read(2, ShaderStages::VERTEX),
            ],
        });
        let lod_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: lod layout"),
            entries: &[
                uniform_entry(0, ShaderStages::COMPUTE),
                storage_read(1, ShaderStages::COMPUTE),
                storage_write(2, ShaderStages::COMPUTE),
                storage_write(3, ShaderStages::COMPUTE),
            ],
        });
        // The falcon reads the scene and nothing else.
        let falcon_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: falcon layout"),
            entries: &[uniform_entry(0, ShaderStages::VERTEX_FRAGMENT)],
        });
        let tonemap_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: tonemap layout"),
            entries: &[texture_entry(0, ShaderStages::FRAGMENT)],
        });
        let taa_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: taa layout"),
            entries: &[
                texture_entry(0, ShaderStages::FRAGMENT),
                texture_entry(1, ShaderStages::FRAGMENT),
                uniform_entry(2, ShaderStages::FRAGMENT),
            ],
        });
        let exposure_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: exposure layout"),
            entries: &[
                texture_entry(0, ShaderStages::COMPUTE),
                storage_write(1, ShaderStages::COMPUTE),
            ],
        });

        let pipeline_layout = |label: &'static str, layouts: &[&BindGroupLayout]| {
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &layouts.iter().map(|l| Some(*l)).collect::<Vec<_>>(),
                immediate_size: 0,
            })
        };

        let sky = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render: sky"),
            layout: Some(&pipeline_layout("render: sky layout", &[&sky_layout])),
            vertex: VertexState {
                module: &sky_module,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &sky_module,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // The three bird levels: the mesh (vertex data), the impostor quad and
        // the sub-pixel splat (both quad-less). All three darken the HDR target
        // and accumulate raw optical depth with the same blends.
        let mesh = bird_pipeline(
            device,
            "render: mesh",
            &bird_layout,
            &mesh_module,
            "vs_mesh",
            "fs_mesh",
            &[Some(VertexBufferLayout {
                array_stride: std::mem::size_of::<mesh::Vertex>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            })],
        );
        let impostor = bird_pipeline(
            device,
            "render: impostor",
            &bird_layout,
            &impostor_module,
            "vs_impostor",
            "fs_impostor",
            &[],
        );
        let coverage = bird_pipeline(
            device,
            "render: coverage",
            &bird_layout,
            &coverage_module,
            "vs_bird",
            "fs_bird",
            &[],
        );
        let falcon = bird_pipeline(
            device,
            "render: falcon",
            &falcon_layout,
            &falcon_module,
            "vs_falcon",
            "fs_falcon",
            &[],
        );
        let reeds_module = shader(
            device,
            "reeds",
            &format!("{scene_wgsl}\n{}", include_str!("../shaders/reeds.wgsl")),
        );
        let reeds = bird_pipeline(
            device,
            "render: reeds",
            &falcon_layout,
            &reeds_module,
            "vs_reeds",
            "fs_reeds",
            &[],
        );

        let tonemap = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render: tonemap"),
            layout: Some(&pipeline_layout(
                "render: tonemap layout",
                &[&tonemap_layout],
            )),
            vertex: VertexState {
                module: &tonemap_module,
                entry_point: Some("vs_tonemap"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &tonemap_module,
                entry_point: Some("fs_tonemap"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let taa = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render: taa"),
            layout: Some(&pipeline_layout("render: taa layout", &[&taa_layout])),
            vertex: VertexState {
                module: &taa_module,
                entry_point: Some("vs_taa"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &taa_module,
                entry_point: Some("fs_taa"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let exposure_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("render: exposure"),
            layout: Some(&pipeline_layout(
                "render: exposure layout",
                &[&exposure_layout],
            )),
            module: &exposure_module,
            entry_point: Some("histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        let lod_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("render: lod"),
            layout: Some(&pipeline_layout("render: lod layout", &[&lod_layout])),
            module: &lod_module,
            entry_point: Some("classify"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mesh_vertices = device.create_buffer(&BufferDescriptor {
            label: Some("render: starling mesh"),
            size: std::mem::size_of_val(mesh::VERTICES) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&mesh_vertices, 0, bytemuck::cast_slice(mesh::VERTICES));

        // The indirect draw arguments: three blocks whose instance counts start
        // at zero and are counted up by the LOD classifier.
        let lod_args = device.create_buffer(&BufferDescriptor {
            label: Some("render: lod draw args"),
            size: std::mem::size_of_val(&lod_args_template()) as u64,
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&lod_args, 0, bytemuck::cast_slice(&lod_args_template()));

        let histogram = device.create_buffer(&BufferDescriptor {
            label: Some("render: exposure histogram"),
            size: (BINS * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let histogram_staging = device.create_buffer(&BufferDescriptor {
            label: Some("render: exposure readback"),
            size: (BINS * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (hdr, hdr_view) = target_texture(device, width, height, HDR_FORMAT, "render: hdr");
        let (optics, optics_view) =
            target_texture(device, width, height, OPTICS_FORMAT, "render: optics");
        let (accum, accum_view) = history_targets(device, width, height);
        let blend = device.create_buffer(&BufferDescriptor {
            label: Some("render: taa blend"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&blend, 0, bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0]));

        // Frame-time probes, only where the adapter can write timestamps: a
        // frame time is reported from the GPU's own clock or not at all.
        let timestamps = gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
            .then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("render: frame timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: TIMESTAMP_SLOTS as u32,
                })
            });
        let timestamp_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("render: timestamps"),
            size: (TIMESTAMP_SLOTS * 8) as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let timestamp_staging = device.create_buffer(&BufferDescriptor {
            label: Some("render: timestamps readback"),
            size: (TIMESTAMP_SLOTS * 8) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sky_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: sky bind"),
            layout: &sky_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });
        let falcon_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: falcon bind"),
            layout: &falcon_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });
        let (tonemap_bind, taa_bind) = history_binds(
            device,
            &tonemap_layout,
            &taa_layout,
            &hdr_view,
            &accum_view,
            &blend,
        );
        let exposure_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: exposure bind"),
            layout: &exposure_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&optics_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: histogram.as_entire_binding(),
                },
            ],
        });

        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(RenderError(error.to_string()));
        }
        Ok(Self {
            scene: Scene::from_camera(
                &Camera::new(),
                width,
                height,
                0.5,
                DEFAULT_EXPOSURE,
                0.0,
                0.0,
            ),
            scene_buffer,
            size: (width, height),
            output_format,
            sky,
            mesh,
            impostor,
            coverage,
            falcon,
            reeds,
            tonemap,
            taa,
            exposure_pipeline,
            lod_pipeline,
            bird_layout,
            lod_layout,
            tonemap_layout,
            taa_layout,
            exposure_layout,
            mesh_vertices,
            lod_lists: None,
            lod_args,
            lod_birds: 0,
            bird_bind_cache: Vec::new(),
            sky_bind,
            falcon_bind,
            bird_bind: None,
            lod_bind: None,
            tonemap_bind,
            taa_bind,
            exposure_bind,
            accum,
            accum_view,
            blend,
            timestamps,
            timestamp_buffer,
            timestamp_staging,
            frame: 0,
            accum_ready: false,
            history_weight: None,
            fixed_jitter: None,
            hdr,
            hdr_view,
            optics,
            optics_view,
            histogram,
            histogram_staging,
            exposure: DEFAULT_EXPOSURE,
            exposure_ready: false,
            async_exposure: false,
            exposure_pending: None,
            exposure_elapsed: 0.0,
        })
    }

    /// The current adaptive exposure.
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// The window uses delayed, non-blocking exposure telemetry. Captures keep
    /// the synchronous path so their converged stills remain reproducible.
    pub fn set_async_exposure(&mut self, enabled: bool) {
        self.async_exposure = enabled;
    }

    /// Fixes the exposure and stops it adapting. A frame whose subject is one
    /// bird cannot give the histogram a sensible 80th percentile, so the
    /// measurement modes pin the exposure to what a flock settles on instead of
    /// letting a single bird's pixels choose it.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure.clamp(EXPOSURE_MIN, EXPOSURE_MAX);
        self.exposure_ready = true;
    }

    /// The GPU time of each probe in [`TIMESTAMP_SLOTS`], in milliseconds,
    /// relative to the frame's start (slot 0), as of the last submitted frame.
    /// `None` when the adapter cannot write timestamps — callers must then say
    /// the frame time is unmeasured rather than time it on the CPU.
    pub fn frame_times_ms(&self, gpu: &Gpu) -> Option<[f32; TIMESTAMP_SLOTS]> {
        self.timestamps.as_ref()?;
        let (tx, rx) = mpsc::channel();
        self.timestamp_staging
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the frame timestamps");
        rx.recv()
            .expect("the poll ran the mapping callback before returning")
            .expect("mapping the frame timestamps");
        let range = self
            .timestamp_staging
            .get_mapped_range(..)
            .expect("reading the frame timestamps");
        let ticks: &[u64] = bytemuck::cast_slice(&range);
        let start = ticks[0];
        let period = f64::from(gpu.queue.get_timestamp_period());
        let times =
            std::array::from_fn(|i| ((ticks[i].wrapping_sub(start)) as f64 * period / 1e6) as f32);
        drop(range);
        self.timestamp_staging.unmap();
        Some(times)
    }

    /// Fixes the frame's sub-pixel sample offset (`None` returns to the Halton
    /// sequence). A measurement whose frames must differ only by the thing being
    /// measured sets this to zero, so the jitter does not move the birds between
    /// frames.
    pub fn set_jitter(&mut self, jitter: Option<[f32; 2]>) {
        self.fixed_jitter = jitter;
    }

    /// Overrides the temporal-AA history weight for the frames that follow
    /// (`None` returns to the frame-time rule). A still capture ramps it down as
    /// `1 / (frame + 1)` so the frozen scene converges like a box filter — the
    /// sub-pixel jitter then acts as a real supersample — instead of keeping an
    /// exponential window of the last few frames.
    pub fn set_history_weight(&mut self, weight: Option<f32>) {
        self.history_weight = weight;
    }

    /// Discard temporal samples after a camera cut; history from another lens
    /// must not contaminate a frozen-view shimmer measurement.
    pub fn reset_history(&mut self) {
        self.accum_ready = false;
    }

    /// How many birds the last frame drew at each LOD level: mesh, impostor,
    /// splat. Read back from the indirect draw arguments, for the record.
    pub fn lod_counts(&self, gpu: &Gpu) -> [u32; LOD_LEVELS] {
        let size = std::mem::size_of_val(&lod_args_template()) as u64;
        let readback = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("render: lod counts readback"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render: lod counts"),
            });
        encoder.copy_buffer_to_buffer(&self.lod_args, 0, &readback, 0, size);
        gpu.queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the LOD count readback");
        rx.recv()
            .expect("the poll ran the mapping callback before returning")
            .expect("mapping the LOD counts");
        let range = readback
            .get_mapped_range(..)
            .expect("reading the LOD counts");
        let args: &[u32] = bytemuck::cast_slice(&range);
        let counts = std::array::from_fn(|level| args[level * 4 + 1]);
        drop(range);
        readback.unmap();
        counts
    }

    /// The current scene block.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Replaces the scene block for the next frame.
    pub fn set_scene(&mut self, gpu: &Gpu, scene: Scene) {
        self.scene = scene;
        gpu.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&scene));
    }

    /// Points the bird passes at a (new) bird storage buffer, sizing the LOD
    /// lists for `count` instances. `count` may exceed the birds the simulation
    /// actually draws; the extra lists stay empty.
    pub fn set_birds(&mut self, gpu: &Gpu, birds: &Buffer, count: u32) {
        let device = &gpu.device;
        let count = count.max(1);
        // The list buffer only depends on the bird count, and the window
        // re-points the bird buffer every frame (the simulation ping-pongs it),
        // so keep the allocation while the count holds and reuse cached bind
        // groups for the two ping-pong buffers.
        if self.lod_lists.is_none() || self.lod_birds != count {
            self.lod_lists = Some(device.create_buffer(&BufferDescriptor {
                label: Some("render: lod lists"),
                size: u64::from(count) * LOD_LEVELS as u64 * LOD_ENTRY_BYTES,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.lod_birds = count;
            self.bird_bind_cache.clear();
        }
        let key = std::ptr::from_ref(birds) as usize;
        if let Some((_, _, bird_bind, lod_bind)) = self
            .bird_bind_cache
            .iter()
            .find(|(k, c, _, _)| *k == key && *c == count)
        {
            self.bird_bind = Some(bird_bind.clone());
            self.lod_bind = Some(lod_bind.clone());
            return;
        }
        let lists = self.lod_lists.clone().expect("just created");
        let bird_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: bird bind"),
            layout: &self.bird_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.scene_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: birds.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: lists.as_entire_binding(),
                },
            ],
        });
        let lod_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: lod bind"),
            layout: &self.lod_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.scene_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: birds.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: lists.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.lod_args.as_entire_binding(),
                },
            ],
        });
        if self.bird_bind_cache.len() >= 4 {
            self.bird_bind_cache.clear();
        }
        self.bird_bind_cache
            .push((key, count, bird_bind.clone(), lod_bind.clone()));
        self.bird_bind = Some(bird_bind);
        self.lod_bind = Some(lod_bind);
    }

    /// Recreates the size-dependent targets after a resize.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.size == (width, height) {
            return;
        }
        self.size = (width, height);
        let device = &gpu.device;
        let (hdr, hdr_view) = target_texture(device, width, height, HDR_FORMAT, "render: hdr");
        let (optics, optics_view) =
            target_texture(device, width, height, OPTICS_FORMAT, "render: optics");
        let (accum, accum_view) = history_targets(device, width, height);
        let (tonemap_bind, taa_bind) = history_binds(
            device,
            &self.tonemap_layout,
            &self.taa_layout,
            &hdr_view,
            &accum_view,
            &self.blend,
        );
        self.hdr = hdr;
        self.hdr_view = hdr_view;
        self.optics = optics;
        self.optics_view = optics_view;
        self.accum = accum;
        self.accum_view = accum_view;
        self.tonemap_bind = tonemap_bind;
        self.taa_bind = taa_bind;
        // The resized history holds another size's frame: start it again.
        self.accum_ready = false;
        self.exposure_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: exposure bind"),
            layout: &self.exposure_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.optics_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.histogram.as_entire_binding(),
                },
            ],
        });
        self.scene.viewport = [width as f32, height as f32];
    }

    /// The output format this renderer's tone-map pipeline was built for.
    pub fn output_format(&self) -> TextureFormat {
        self.output_format
    }

    /// Renders one frame into `view` and updates adaptive exposure. Captures
    /// block on the small histogram; set_async_exposure enables delayed,
    /// non-blocking telemetry for the window.
    pub fn render(&mut self, gpu: &Gpu, view: &TextureView, birds: u32, dt: f32) {
        self.render_impl(gpu, view, birds, dt);
    }

    fn render_impl(&mut self, gpu: &Gpu, view: &TextureView, birds: u32, dt: f32) {
        self.exposure_elapsed += dt.max(0.0);
        self.adapt_exposure(gpu);
        if self.async_exposure {
            self.scene.exposure = self.exposure;
        }
        let read_exposure = self.exposure_pending.is_none();
        // The optical-depth histogram is an atomic counter; zero it first. The
        // indirect draws start at zero instances, with their per-level vertex
        // counts kept.
        gpu.queue.write_buffer(&self.histogram, 0, &[0u8; BINS * 4]);
        gpu.queue.write_buffer(
            &self.lod_args,
            0,
            bytemuck::cast_slice(&lod_args_template()),
        );

        // This frame's temporal-AA sample offset, and the weight the history
        // keeps. The first frame after a target is built starts the history
        // rather than blending into an empty one.
        self.scene.jitter = self.fixed_jitter.unwrap_or_else(|| taa_jitter(self.frame));
        gpu.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&self.scene));
        let alpha = match self.history_weight {
            Some(weight) => weight,
            None if self.accum_ready => 1.0 - (-dt.max(0.0) / TAA_TAU).exp(),
            None => 1.0,
        };
        self.accum_ready = true;
        gpu.queue.write_buffer(
            &self.blend,
            0,
            bytemuck::cast_slice(&[alpha, 0.0, 0.0, 0.0]),
        );
        let parity = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);

        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render: frame"),
            });
        if let Some(set) = &self.timestamps {
            encoder.write_timestamp(set, 0);
        }

        // 1. Classify every bird by its projected wingspan and append it to its
        //    level's instance list. The atomic counters the classifier takes its
        //    slots from are the same words the three draws read their instance
        //    counts from, so there is no second pass to collect them.
        if let Some(lod_bind) = &self.lod_bind {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render: lod"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lod_pipeline);
            pass.set_bind_group(0, lod_bind, &[]);
            pass.dispatch_workgroups(birds.min(self.lod_birds).div_ceil(LOD_WORKGROUP), 1, 1);
        }
        if let Some(set) = &self.timestamps {
            encoder.write_timestamp(set, 1);
        }

        // 2. Sky into the HDR target.
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render: sky"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.sky);
            pass.set_bind_group(0, &self.sky_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. The three bird levels, one indirect draw each. The optics target is
        // cleared here even with no birds, so the exposure histogram never reads
        // a stale frame; the HDR target is loaded so the sky stays.
        if let Some(bird_bind) = &self.bird_bind {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render: birds"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &self.hdr_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    }),
                    Some(RenderPassColorAttachment {
                        view: &self.optics_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, bird_bind, &[]);
            pass.set_vertex_buffer(0, self.mesh_vertices.slice(..));
            for (level, pipeline) in [&self.mesh, &self.impostor, &self.coverage]
                .into_iter()
                .enumerate()
            {
                pass.set_pipeline(pipeline);
                pass.draw_indirect(&self.lod_args, (level * ARGS_STRIDE) as u64);
            }
        }

        // The foreground: the reed bed (always in view when the camera looks
        // low enough) and the falcon, drawn on top of the flock from the scene
        // alone, into both targets so the falcon counts toward the exposure.
        if self.scene.falcon > 0.5 || self.scene.reed > 0.002 {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render: falcon"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &self.hdr_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    }),
                    Some(RenderPassColorAttachment {
                        view: &self.optics_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.scene.reed > 0.002 {
                pass.set_pipeline(&self.reeds);
                pass.set_bind_group(0, &self.falcon_bind, &[]);
                pass.draw(0..3, 0..1);
            }
            if self.scene.falcon > 0.5 {
                pass.set_pipeline(&self.falcon);
                pass.set_bind_group(0, &self.falcon_bind, &[]);
                pass.draw(0..6, 0..1);
            }
        }
        if let Some(set) = &self.timestamps {
            encoder.write_timestamp(set, 2);
        }

        // 3. Histogram the optical depth.
        {
            let (w, h) = self.size;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render: exposure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exposure_pipeline);
            pass.set_bind_group(0, &self.exposure_bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        if read_exposure {
            encoder.copy_buffer_to_buffer(
                &self.histogram,
                0,
                &self.histogram_staging,
                0,
                (BINS * 4) as u64,
            );
        }
        if let Some(set) = &self.timestamps {
            encoder.write_timestamp(set, 3);
        }

        // 4. Blend the frame into the history. This is what the tone map reads,
        //    so what is displayed is the accumulated picture, not one frame.
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render: taa"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.accum_view[1 - parity],
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.taa);
            pass.set_bind_group(0, &self.taa_bind[parity], &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(set) = &self.timestamps {
            encoder.write_timestamp(set, 4);
        }

        // 5. Tone map the accumulated HDR image to the output.
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render: tonemap"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tonemap);
            pass.set_bind_group(0, &self.tonemap_bind[1 - parity], &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(set) = &self.timestamps {
            encoder.write_timestamp(set, 5);
            encoder.resolve_query_set(set, 0..TIMESTAMP_SLOTS as u32, &self.timestamp_buffer, 0);
            encoder.copy_buffer_to_buffer(
                &self.timestamp_buffer,
                0,
                &self.timestamp_staging,
                0,
                (TIMESTAMP_SLOTS * 8) as u64,
            );
        }

        gpu.queue.submit(Some(encoder.finish()));
        if read_exposure {
            let (tx, rx) = mpsc::channel();
            self.histogram_staging
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = tx.send(result);
                });
            self.exposure_pending = Some(rx);
        }
        if !self.async_exposure {
            self.adapt_exposure(gpu);
        }
    }

    /// Reads back the histogram and eases the exposure toward the 80th
    /// percentile of the flock's optical depth, exactly as the reference does.
    fn adapt_exposure(&mut self, gpu: &Gpu) {
        let Some(rx) = &self.exposure_pending else {
            return;
        };
        if self.async_exposure {
            gpu.device
                .poll(wgpu::PollType::Poll)
                .expect("polling exposure");
            match rx.try_recv() {
                Err(mpsc::TryRecvError::Empty) => return,
                Err(error) => panic!("exposure callback disconnected: {error}"),
                Ok(result) => result.expect("mapping the exposure histogram"),
            }
        } else {
            gpu.wait_bounded(murmur_gpu::device::WAIT_BUDGET)
                .expect("waiting for exposure");
            rx.recv()
                .expect("exposure callback")
                .expect("mapping exposure");
        }
        let range = self
            .histogram_staging
            .get_mapped_range(..)
            .expect("reading the mapped histogram");
        let bins: &[u32] = bytemuck::cast_slice(&range);
        let total: u64 = bins.iter().map(|&b| u64::from(b)).sum();
        let tau80 = if total == 0 {
            None
        } else {
            let target = (total as f64 * 0.8) as u64;
            let mut seen = 0u64;
            let mut found = 0usize;
            for (i, &b) in bins.iter().enumerate() {
                seen += u64::from(b);
                if seen >= target {
                    found = i;
                    break;
                }
            }
            Some((found as f32 + 0.5) / BINS as f32 * TAU_MAX)
        };
        drop(range);
        self.histogram_staging.unmap();
        self.exposure_pending = None;
        let dt = std::mem::take(&mut self.exposure_elapsed);

        let Some(tau80) = tau80 else {
            return;
        };
        if tau80 <= 1e-6 {
            return;
        }
        let target = (TARGET_TAU / tau80).clamp(EXPOSURE_MIN, EXPOSURE_MAX);
        if !self.exposure_ready {
            self.exposure = target;
            self.exposure_ready = true;
        } else {
            self.exposure += (target - self.exposure) * (1.0 - (-dt / EXPOSURE_TAU).exp());
        }
        self.exposure = self.exposure.clamp(EXPOSURE_MIN, EXPOSURE_MAX);
    }

    /// Creates an off-screen target the tone-map pass can render into and a
    /// following readback can copy from.
    pub fn create_target(&self, gpu: &Gpu, width: u32, height: u32) -> Target {
        let (texture, view) = {
            let device = &gpu.device;
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("render: still target"),
                size: Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: self.output_format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            (texture, view)
        };
        Target { texture, view }
    }

    /// Copies a rendered target back to the host as 4-byte RGBA rows (no row
    /// padding), ready for the PNG encoder.
    pub fn read_target(&self, gpu: &Gpu, target: &Target, width: u32, height: u32) -> Vec<u8> {
        let (w, h) = (width.max(1), height.max(1));
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("render: still readback"),
            size: (padded * h) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render: still copy"),
            });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the still readback");
        rx.recv()
            .expect("the poll ran the mapping callback before returning")
            .expect("mapping the still readback");
        let range = buffer.get_mapped_range(..).expect("reading the still");
        let mut out = Vec::with_capacity((unpadded * h) as usize);
        for row in 0..h as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&range[start..start + unpadded as usize]);
        }
        drop(range);
        buffer.unmap();
        out
    }
}

/// The unbounded full-screen target usage used for the two internal buffers.
fn target_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: TextureFormat,
    label: &str,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// The two HDR images the temporal resolve ping-pongs between.
fn history_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> ([Texture; 2], [TextureView; 2]) {
    let (a, a_view) = target_texture(device, width, height, HDR_FORMAT, "render: history 0");
    let (b, b_view) = target_texture(device, width, height, HDR_FORMAT, "render: history 1");
    ([a, b], [a_view, b_view])
}

/// The two pairs of bind groups the temporal resolve uses: `taa[k]` reads
/// `accum[k]` as history and writes `accum[1 - k]`, which `tonemap[1 - k]` reads.
fn history_binds(
    device: &wgpu::Device,
    tonemap_layout: &BindGroupLayout,
    taa_layout: &BindGroupLayout,
    hdr_view: &TextureView,
    accum_view: &[TextureView; 2],
    blend: &Buffer,
) -> ([BindGroup; 2], [BindGroup; 2]) {
    let tonemap = std::array::from_fn(|k| {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: tonemap bind"),
            layout: tonemap_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&accum_view[k]),
            }],
        })
    });
    let taa = std::array::from_fn(|k| {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: taa bind"),
            layout: taa_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&accum_view[k]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: blend.as_entire_binding(),
                },
            ],
        })
    });
    (tonemap, taa)
}

/// This frame's sub-pixel sample offset, from a Halton (2, 3) sequence over
/// `TAA_SAMPLES` frames, centred so no bias is introduced.
fn taa_jitter(frame: u32) -> [f32; 2] {
    let i = frame % TAA_SAMPLES + 1;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

/// The `base`-ary radical inverse of `index`, the Halton sequence's term.
fn halton(index: u32, base: u32) -> f32 {
    let mut index = index;
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// The additive blend used for the optical-depth accumulation.
fn additive() -> BlendState {
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    }
}

/// The multiplicative darkness every bird level deposits: `dst *= 1 - coverage`,
/// with the alpha channel left as the sky wrote it.
fn darken() -> BlendState {
    BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::OneMinusSrc,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    }
}

/// Builds one of the three bird-level pipelines. They share a layout (scene,
/// birds, instance lists), the two colour targets and the blends, and differ
/// only in their shader and, for the mesh, in having a vertex buffer.
fn bird_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &BindGroupLayout,
    module: &wgpu::ShaderModule,
    vertex_entry: &str,
    fragment_entry: &str,
    buffers: &[Option<VertexBufferLayout<'_>>],
) -> RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module,
            entry_point: Some(vertex_entry),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[
                Some(ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(darken()),
                    write_mask: ColorWrites::ALL,
                }),
                Some(ColorTargetState {
                    format: OPTICS_FORMAT,
                    blend: Some(additive()),
                    write_mask: ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// How many LOD levels the classifier writes and the renderer draws. Must match
/// the `LEVEL` of the three pass shaders; `tests/lod_constants.rs` parses them
/// out of the WGSL.
pub const LOD_LEVELS: usize = 3;

/// Bytes of one instance-list entry: `(bird index, weight bits)`, a `vec2<u32>`.
const LOD_ENTRY_BYTES: u64 = 8;

/// Bytes of one `DrawIndirect` block: four `u32`.
const ARGS_STRIDE: usize = 16;

/// What `shaders/lod.wgsl` declares as its workgroup size, checked by
/// `tests/lod_constants.rs`.
pub const LOD_WORKGROUP: u32 = 64;

/// Projected wingspan \[px\] at which the mesh takes over, and the half-width of
/// the mesh↔impostor cross-fade: the same numbers as `shaders/lod.wgsl`, which
/// `tests/lod_constants.rs` checks. The app's `--lod-sweep` measures the
/// transition these describe.
pub const RHO_MESH: f32 = 6.0;
pub const LOD_BAND: f32 = 2.0;

/// Below this a bird is smaller than a pixel and the analytic splat draws it.
pub const RHO_SPLAT: f32 = 1.5;

/// Vertices the level's pipeline draws per instance: the mesh's triangle list,
/// then a quad for the impostor and one for the splat.
const LOD_VERTICES: [u32; LOD_LEVELS] = [mesh::VERTEX_COUNT, 6, 6];

/// The indirect draw arguments with zero instances: the template the renderer
/// writes back before the classifier counts the birds of a frame.
fn lod_args_template() -> [u32; LOD_LEVELS * 4] {
    let mut args = [0u32; LOD_LEVELS * 4];
    let mut level = 0;
    while level < LOD_LEVELS {
        args[level * 4] = LOD_VERTICES[level];
        level += 1;
    }
    args
}

/// Compiles a WGSL module, surfacing the shader source in the error.
fn shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    })
}
