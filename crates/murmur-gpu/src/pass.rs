//! Recording one compute dispatch, optionally stamping GPU timestamps.
//!
//! `--bench` (PLAN.md phase 3) needs per-pass GPU times, so the one place that
//! starts a compute pass also owns the timestamp bookkeeping. The numbers are
//! GPU timestamps from `queue.get_timestamp_period()`, per AGENTS.md ("report
//! performance only from GPU timestamp queries or a profiler").

use wgpu::{
    BindGroup, CommandEncoder, ComputePassDescriptor, ComputePassTimestampWrites, ComputePipeline,
    QuerySet,
};

/// The passes of one simulation step, in the order they are recorded
/// (`shaders/sim.wgsl` and `shaders/grid.wgsl`).
pub const STEP_PASSES: [&str; 12] = [
    "tick",
    "grid_clear",
    "grid_histogram",
    "grid_scan_blocks",
    "grid_scan_totals",
    "grid_scan_add",
    "grid_scatter",
    "grid_sort_buckets",
    "prepare_decisions",
    "decide",
    "decide_wide",
    "integrate",
];

/// Timestamp query slots needed for one step: a begin and an end per pass.
pub const TIMESTAMPS_PER_STEP: u32 = 2 * STEP_PASSES.len() as u32;

/// Hands out the timestamp query slots of one step, two per pass.
pub struct Timeline<'a> {
    set: &'a QuerySet,
    next: u32,
}

impl<'a> Timeline<'a> {
    /// Starts a timeline over the query slots of `set`.
    pub fn new(set: &'a QuerySet) -> Self {
        Self { set, next: 0 }
    }

    /// The `(query_set, begin, end)` indices for the next pass.
    fn slot(&mut self) -> (&'a QuerySet, u32, u32) {
        let begin = self.next;
        self.next += 2;
        (self.set, begin, begin + 1)
    }
}

/// Records one compute dispatch of `pipeline`, stamping it when `timeline` is
/// given.
pub fn dispatch(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    workgroups: u32,
    timeline: Option<&mut Timeline<'_>>,
) {
    let timestamp_writes = timeline.map(|timeline| {
        let (query_set, begin, end) = timeline.slot();
        ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }
    });
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: None,
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}
