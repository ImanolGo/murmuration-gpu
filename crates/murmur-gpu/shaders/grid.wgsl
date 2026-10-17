// The spatial grid: the passes PLAN.md phase 3 adds to replace the brute-force
// neighbour search in `sim.wgsl::decide` with an acceleration structure.
//
// Mirrors `murmur-ref/src/neighbours.rs`, the CPU oracle's hash grid:
//
//   cell index  <- Grid::coord        (via `cell_of` in common.wgsl)
//   histogram   <- Grid::rebuild      (bucket counts, with atomics)
//   scan        <- Grid::rebuild      (the bucket start offsets)
//   scatter     <- Grid::rebuild      (which bird sits in each sorted slot)
//   query       <- Grid::for_each_within (in sim.wgsl; this file only builds)
//
// The oracle keeps a linked list per bucket. A GPU walks a contiguous range far
// better, so here the birds are sorted by bucket instead: the histogram counts
// them, an exclusive prefix scan turns the counts into start offsets, and a
// scatter writes each bird's index into its bucket's range. The cells, the hash
// and the exact distance check are the oracle's, so the neighbour sets agree.
//
// The world is unbounded (birds roam far outside the roost), so cells hash into
// a fixed-size table instead of a dense grid over a bounding box. That needs no
// per-step recentring pass and no bound on where the flock flies; the table is
// sized from the bird count so collisions stay rare (see `Grid::buckets_for`).
//
// The scan is two-level ("workgroup scan plus block sums"): one workgroup scans
// a 2048-bucket block and writes its total, a single workgroup scans the block
// totals, and a third pass adds the block total back. Buckets are a power of
// two at most 2^22, so at most 2048 blocks and one workgroup covers the totals.
//
// Two buffers play two roles each, so the pipeline needs only eight storage
// buffers (the WebGPU limit). Neither reuse races: `counts` is the histogram
// until the level-1 scan has read it, then `grid_scan_add` re-seeds it as the
// scatter cursor; `block_sums` holds the block totals until the single
// `grid_scan_totals` workgroup has read them all into shared memory, then holds
// the block start offsets.

const WG: u32 = 256u;
const ELEMS: u32 = 8u;
const BLOCK: u32 = WG * ELEMS; // buckets per scan workgroup

@group(0) @binding(1) var<storage, read> src: array<Bird>;
@group(0) @binding(4) var<storage, read_write> bucket: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(8) var<storage, read_write> block_sums: array<u32>;

// --- Grid::rebuild: cell index and histogram -------------------------------

// Clears the bucket counts. A separate pass, because every histogram write must
// see a zeroed table (a compute pass boundary is a memory barrier).
@compute @workgroup_size(WG)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= P.buckets {
        return;
    }
    atomicStore(&counts[b], 0u);
}

// Records each bird's bucket and counts one per bucket.
@compute @workgroup_size(WG)
fn grid_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    // A bird the evening has not yet spawned, or one that has landed, is not in
    // the flock: it must not enter the grid. It would otherwise sit at the
    // origin with every other empty slot and make every query that walks that
    // bucket or falls back to the brute-force walk scan the whole capacity.
    if src[i].live == 0u {
        bucket[i] = u32(P.buckets);
        return;
    }
    let b = bucket_of(cell_of(src[i].pos));
    bucket[i] = b;
    atomicAdd(&counts[b], 1u);
}

// --- exclusive prefix scan -------------------------------------------------

var<workgroup> scan_vals: array<u32, BLOCK>;
var<workgroup> scan_sums: array<u32, WG>;

// An exclusive scan of the in-block totals, in place (Hillis and Steele).
fn scan_sums_exclusive(t: u32) {
    var x = scan_sums[t];
    for (var off = 1u; off < WG; off = off * 2u) {
        var y = 0u;
        if t >= off {
            y = scan_sums[t - off];
        }
        workgroupBarrier();
        x = x + y;
        workgroupBarrier();
        scan_sums[t] = x;
        workgroupBarrier();
    }
}

// Level 1: each workgroup scans one 2048-bucket block and publishes the block's
// exclusive offsets plus its total.
@compute @workgroup_size(WG)
fn grid_scan_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    let base = wid.x * BLOCK;

    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let k = base + t * ELEMS + e;
        var v = 0u;
        if k < P.buckets {
            v = atomicLoad(&counts[k]);
        }
        scan_vals[t * ELEMS + e] = v;
    }
    workgroupBarrier();

    // Each thread scans its own run; `sum` is the run total.
    var sum = 0u;
    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let idx = t * ELEMS + e;
        let v = scan_vals[idx];
        scan_vals[idx] = sum;
        sum = sum + v;
    }
    scan_sums[t] = sum;
    workgroupBarrier();

    scan_sums_exclusive(t);

    let thread_offset = scan_sums[t] - sum;
    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let k = base + t * ELEMS + e;
        if k < P.buckets {
            offsets[k] = thread_offset + scan_vals[t * ELEMS + e];
        }
    }
    if t == 0u {
        block_sums[wid.x] = scan_sums[WG - 1u];
    }
}

var<workgroup> total_vals: array<u32, BLOCK>;
var<workgroup> total_sums: array<u32, WG>;

// Level 2: one workgroup scans the block totals into block start offsets, in
// place (all reads into shared memory happen before any write).
@compute @workgroup_size(WG)
fn grid_scan_totals(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let blocks = (P.buckets + BLOCK - 1u) / BLOCK;

    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let k = t * ELEMS + e;
        var v = 0u;
        if k < blocks {
            v = block_sums[k];
        }
        total_vals[t * ELEMS + e] = v;
    }
    workgroupBarrier();

    var sum = 0u;
    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let idx = t * ELEMS + e;
        let v = total_vals[idx];
        total_vals[idx] = sum;
        sum = sum + v;
    }
    total_sums[t] = sum;
    workgroupBarrier();

    var x = total_sums[t];
    for (var off = 1u; off < WG; off = off * 2u) {
        var y = 0u;
        if t >= off {
            y = total_sums[t - off];
        }
        workgroupBarrier();
        x = x + y;
        workgroupBarrier();
        total_sums[t] = x;
        workgroupBarrier();
    }

    let thread_offset = total_sums[t] - sum;
    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let k = t * ELEMS + e;
        if k < blocks {
            block_sums[k] = thread_offset + total_vals[t * ELEMS + e];
        }
    }
}

// Level 3: add each block's start offset back and re-seed `counts` as the
// scatter cursor. `offsets` keeps the bucket start so the query in `sim.wgsl`
// always sees a stable range.
@compute @workgroup_size(WG)
fn grid_scan_add(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    let base = wid.x * BLOCK;
    let add = block_sums[wid.x];
    for (var e = 0u; e < ELEMS; e = e + 1u) {
        let k = base + t * ELEMS + e;
        if k < P.buckets {
            let v = offsets[k] + add;
            offsets[k] = v;
            atomicStore(&counts[k], v);
        }
    }
}

// --- scatter ---------------------------------------------------------------

// Places each bird's index into its bucket's range. Which slot a bird gets is
// whatever the atomics decide, so this pass alone is not reproducible; the next
// one puts each range in a fixed order.
@compute @workgroup_size(WG)
fn grid_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.n {
        return;
    }
    // `bucket[i] == P.buckets` marks an inactive bird (see `grid_histogram`).
    if bucket[i] >= P.buckets {
        return;
    }
    let slot = atomicAdd(&counts[bucket[i]], 1u);
    sorted[slot] = i;
}

// Sorts each bucket's range by bird index, one workgroup thread per bucket.
//
// A parallel scatter's order depends on which thread got there first, so without
// this the sum inside a bird's decision would differ from run to run, and the
// same command on the same machine would not twice give the same flock. The
// model is deterministic (AGENTS.md), so the GPU is too: this restores a fixed
// order, which is also what makes the CPU/GPU comparison reproducible.
//
// The table is sized at about two birds per bucket ([`Grid::buckets_for`]), so
// nearly every bucket holds one bird and the inner loop does nothing; a bucket
// holds a handful. Insertion sort is O(k^2) per bucket, and is only slow if a
// state clumps hundreds of birds into one 4 m cell.
@compute @workgroup_size(WG)
fn grid_sort_buckets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= P.buckets {
        return;
    }
    let start = offsets[b];
    let end = offsets[b + 1u];
    for (var i = start + 1u; i < end; i = i + 1u) {
        let key = sorted[i];
        var j = i;
        while (j > start && sorted[j - 1u] > key) {
            sorted[j] = sorted[j - 1u];
            j = j - 1u;
        }
        sorted[j] = key;
    }
}
