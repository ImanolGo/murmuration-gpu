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
//! footprint of the reference's `render/splat.rs`; the mesh and LOD impostors
//! are phase 6.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use murmur_gpu::BIRD_WGSL;
use murmur_gpu::device::Gpu;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor, Extent3d, FragmentState,
    LoadOp, Operations, Origin3d, PipelineLayoutDescriptor, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, VertexState,
};

use crate::camera::Camera;

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
const EXPOSURE_MIN: f32 = 1.0;
const EXPOSURE_MAX: f32 = 400.0;

/// Histogram bins used to locate the 80th-percentile optical depth.
const BINS: usize = 256;

/// Optical depth a full histogram bin range maps to.
const TAU_MAX: f32 = 8.0;

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
        }
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
    coverage: RenderPipeline,
    tonemap: RenderPipeline,
    exposure_pipeline: ComputePipeline,

    coverage_layout: BindGroupLayout,
    tonemap_layout: BindGroupLayout,
    exposure_layout: BindGroupLayout,

    sky_bind: BindGroup,
    coverage_bind: Option<BindGroup>,
    tonemap_bind: BindGroup,
    exposure_bind: BindGroup,

    hdr: Texture,
    hdr_view: TextureView,
    optics: Texture,
    optics_view: TextureView,

    histogram: Buffer,
    histogram_staging: Buffer,

    exposure: f32,
    exposure_ready: bool,
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
            "{scene_wgsl}\n{sky_wgsl}\n{}",
            include_str!("../shaders/sky_pass.wgsl")
        );
        let sky_module = shader(device, "sky", &sky_source);

        let coverage_source = format!(
            "{scene_wgsl}\n{BIRD_WGSL}\n{}",
            include_str!("../shaders/coverage.wgsl")
        );
        let coverage_module = shader(device, "coverage", &coverage_source);

        let tonemap_source = include_str!("../shaders/tonemap.wgsl").to_string();
        let tonemap_module = shader(device, "tonemap", &tonemap_source);

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
        let coverage_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: coverage layout"),
            entries: &[
                uniform_entry(0, ShaderStages::VERTEX_FRAGMENT),
                storage_read(1, ShaderStages::VERTEX),
            ],
        });
        let tonemap_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: tonemap layout"),
            entries: &[texture_entry(0, ShaderStages::FRAGMENT)],
        });
        let exposure_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render: exposure layout"),
            entries: &[
                texture_entry(0, ShaderStages::COMPUTE),
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        let coverage = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render: coverage"),
            layout: Some(&pipeline_layout(
                "render: coverage layout",
                &[&coverage_layout],
            )),
            vertex: VertexState {
                module: &coverage_module,
                entry_point: Some("vs_bird"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &coverage_module,
                entry_point: Some("fs_bird"),
                compilation_options: Default::default(),
                targets: &[
                    // The HDR target darkens: dst *= (1 - coverage). The alpha
                    // channel is left as the sky wrote it.
                    Some(ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(BlendState {
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
                        }),
                        write_mask: ColorWrites::ALL,
                    }),
                    // The optical-depth target accumulates: dst += raw.
                    Some(ColorTargetState {
                        format: OPTICS_FORMAT,
                        blend: Some(additive()),
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

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

        let sky_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: sky bind"),
            layout: &sky_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });
        let tonemap_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: tonemap bind"),
            layout: &tonemap_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            }],
        });
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
            coverage,
            tonemap,
            exposure_pipeline,
            coverage_layout,
            tonemap_layout,
            exposure_layout,
            sky_bind,
            coverage_bind: None,
            tonemap_bind,
            exposure_bind,
            hdr,
            hdr_view,
            optics,
            optics_view,
            histogram,
            histogram_staging,
            exposure: DEFAULT_EXPOSURE,
            exposure_ready: false,
        })
    }

    /// The current adaptive exposure.
    pub fn exposure(&self) -> f32 {
        self.exposure
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

    /// Points the coverage pass at a (new) bird storage buffer.
    pub fn set_birds(&mut self, gpu: &Gpu, birds: &Buffer) {
        self.coverage_bind = Some(gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: coverage bind"),
            layout: &self.coverage_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.scene_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: birds.as_entire_binding(),
                },
            ],
        }));
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
        self.hdr = hdr;
        self.hdr_view = hdr_view;
        self.optics = optics;
        self.optics_view = optics_view;
        self.tonemap_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render: tonemap bind"),
            layout: &self.tonemap_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.hdr_view),
            }],
        });
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

    /// Renders one frame into `view` and updates the adaptive exposure for the
    /// next frame, blocking on the small exposure readback.
    pub fn render(&mut self, gpu: &Gpu, view: &TextureView, birds: u32, dt: f32) {
        self.render_impl(gpu, view, birds, dt);
    }

    fn render_impl(&mut self, gpu: &Gpu, view: &TextureView, birds: u32, dt: f32) {
        // The optical-depth histogram is an atomic counter; zero it first.
        gpu.queue.write_buffer(&self.histogram, 0, &[0u8; BINS * 4]);

        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render: frame"),
            });

        // 1. Sky into the HDR target.
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

        // 2. Bird coverage into the HDR and optical-depth targets. The optics
        // target is cleared here even with no birds, so the exposure histogram
        // never reads a stale frame; the HDR target is loaded so the sky stays.
        if let Some(coverage_bind) = &self.coverage_bind {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("render: coverage"),
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
            pass.set_pipeline(&self.coverage);
            pass.set_bind_group(0, coverage_bind, &[]);
            pass.draw(0..6, 0..birds);
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
        encoder.copy_buffer_to_buffer(
            &self.histogram,
            0,
            &self.histogram_staging,
            0,
            (BINS * 4) as u64,
        );

        // 4. Tone map the HDR target to the output.
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
            pass.set_bind_group(0, &self.tonemap_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
        self.adapt_exposure(gpu, dt);
    }

    /// Reads back the histogram and eases the exposure toward the 80th
    /// percentile of the flock's optical depth, exactly as the reference does.
    fn adapt_exposure(&mut self, gpu: &Gpu, dt: f32) {
        let (tx, rx) = mpsc::channel();
        self.histogram_staging
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("waiting for the exposure readback");
        rx.recv()
            .expect("the poll ran the mapping callback before returning")
            .expect("mapping the exposure histogram");
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

/// Compiles a WGSL module, surfacing the shader source in the error.
fn shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    })
}
