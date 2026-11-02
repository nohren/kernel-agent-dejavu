# Evaluating an agent-optimized Triton kernel

I created a Claude-driven agent loop that optimized Double-P’s Triton k-means assignment kernel, achieving a **1.38× median speedup** on an A100 across 16 real Qwen2.5-1.5B layer–KV-head key sets. The sixth candidate, **iter06**, passed the specified correctness tolerance at 2,048 centroids.

This comparison covers the original and iter06, with ablations to explain the improvement. Measurements and reproduction steps are in [`kernel_explained.ipynb`](notebooks/kernel_explained.ipynb) and [`results/`](results/); compiled TTGIR, PTX, and SASS are in [`ir/`](ir/).

## Workload and measurement

For each key, the kernel selects the centroid with the largest dot product, adds the key to that centroid’s sum, and increments its count. Each launch processes eight independent key sets, each containing 32,768 keys of dimension 128 and 2,048 unit-length centroids. Inputs are BF16; sums use FP32.

The evaluation uses 16 layer–KV-head key sets from two text prompts. For individual measurements, each key set is replicated across all eight batch slots, preserving the launch shape. These are kernel timings, not end-to-end model speedups.

The harness checks candidates against an FP32 reference, benchmarks original and candidate in interleaved rounds within the same process, and repeats correctness checks after timing. The agent receives benchmark results and, when enabled, Nsight Compute reports to guide subsequent revisions.

## Results

| Input | Original | iter06 | Speedup |
|---|---:|---:|---:|
| Evaluation A: eight distinct key sets | 0.899 ms | 0.654 ms | 1.374× |
| Evaluation B: eight distinct key sets | 0.905 ms | 0.657 ms | 1.381× |
| Individual key sets, each replicated eight times | 0.889–0.921 ms | 0.655–0.668 ms | 1.36–1.40× |

The median of the 16 individual-case speedups is **1.38×**. For every key, the chosen score must be within $2^{-12}\lVert k\rVert_2$ of the reference maximum; sums and counts must agree with the kernel’s labels. At 2,048 centroids, iter06 passes both evaluation sets, with a worst measured cosine-score gap of **$1.9\times10^{-4}$**, below the $2.44\times10^{-4}$ tolerance.

## Two kernel changes

**Packed argmax with a deferred reduction.** The original reduces each score tile to a maximum and index inside the centroid loop. iter06 replaces low score bits with the centroid index, maintains elementwise maxima across iterations, and performs one reduction after the loop. This reduces repeated reduction work at the cost of score precision.

```python
# Inside the centroid loop:
bits = ip.to(tl.int32, bitcast=True)
packed = ((bits & MASK_HI) | idx[None, :]).to(tl.float32, bitcast=True)
best = tl.maximum(best, packed)

# After the loop:
row_best = tl.max(best, axis=1)
max_idx = row_best.to(tl.int32, bitcast=True) & MASK_LO
```

With $C$ centroids, packing uses $b=\lceil\log_2 C\rceil$ index bits, leaving $23-b$ FP32 fraction bits. Scores in the same truncated bin can exchange order; the bin width bounds the packing-only loss.

For a nonzero key and unit-length centroids, the idealized packing-only regret bound is:

$$
\frac{s_{\mathrm{best}}-s_{\mathrm{chosen}}}{\lVert k\rVert_2}
\le 2^{-(23-b)},
\qquad b=\lceil\log_2 C\rceil.
$$

This bound excludes dot-product rounding error.

**Flat reload epilogue.** After storing assignments, iter06 reloads keys in flat chunks before atomic accumulation. The kernel uses `@triton.jit(do_not_specialize=["stride_one"])`, and the caller passes `stride_one=1`. Preventing specialization to 1 changes the compiled layout: neighboring lanes update consecutive columns instead of every fourth column.

```python
ii = tl.arange(0, EPI_ROWS * D)
r_loc = ii // D
cc = (ii % D) * stride_one
k_base = K + b * N * D
s_base = S + b * C * D

for r0 in range(0, BLOCK_N, EPI_ROWS):
    rr = start_n + r0 + r_loc
    rmask = rr < N
    lab = tl.load(m_base + rr, mask=rmask, other=0)
    kv = tl.load(k_base + rr * D + cc, mask=rmask, other=0.0)
    tl.atomic_add(
        s_base + lab * D + cc, kv.to(tl.float32),
        mask=rmask, sem="relaxed",
    )
```

## Ablation and profiling

The ablation uses one key set. Variants retain the original tile settings unless noted. All pass the specified correctness checks at 2,048 centroids.

**Matching tile settings closes the gap.** iter06 uses 64 centroids per tile instead of 32, and reloads 16 keys per epilogue chunk instead of 32. Applying both settings to the combined variant gives:

| Variant | Time | Speedup vs original |
|---|---:|---:|
| Packed argmax + flat reload, runtime stride | 0.714 ms | 1.265× |
| Same, with iter06’s tile settings | 0.660 ms | 1.370× |
| Full iter06 | 0.653 ms | 1.379× |

With those settings, the ablation kernel produces identical SASS after normalizing kernel names. The individual contributions of the two settings are not isolated.

Nsight Compute reports the following. These locked-clock profiling durations are separate from the benchmark timings above.

| Variant | Atomic sectors/request | Executed warp instructions | Duration (µs) |
|---|---:|---:|---:|
| Original | 16.03 | 159.8M | 1,085 |
| Rewritten baseline | 16.03 | 163.8M | 1,108 |
| Packed argmax only | 16.03 | 110.2M | 1,026 |
| Flat reload only, runtime stride | 4.12 | 173.5M | 908 |
| Both changes, constant stride | 16.03 | 117.2M | 988 |
| Both changes, runtime stride | 4.12 | 120.4M | 833 |
| Same, with iter06’s tile settings | 4.12 | 115.8M | 768 |
| Full iter06 | 4.12 | 115.8M | 767 |

**Packing cuts instructions; the runtime-stride flat reload cuts sectors.** Packing removes 31–33% of executed instructions in the matched comparisons, while leaving sectors per request unchanged. Every variant issues 1,056,768 atomic requests. Overall, iter06 executes **27.6% fewer instructions** and touches **about 3.9× fewer atomic sectors** than the original.

### How the threads are assigned

TTGIR records the mapping from tensor elements to threads. Each sum atomic compiles to one scalar FP32 reduction per element a thread holds (`RED.E.ADD.F32.FTZ.RN.STRONG.GPU` in SASS), so every variant issues 512 sum-atomic requests and four count-atomic requests per block. The layout changes which addresses neighboring lanes access within each request:

| Sum-atomic layout | Columns accessed by one warp instruction | 32-byte sectors |
|---|---|---:|
| Original; flat reload with constant stride | 0, 4, …, 124 | 16 |
| Flat reload with runtime stride | 0, 1, …, 31 | 4 |

Each sector holds eight FP32 values. Consecutive columns place eight lanes’ accesses in each sector; every-fourth-column accesses place only two.

**The constant stride also adds layout conversions.** With the stride visible, Triton loads 8 values per thread (16-byte loads) but gives the atomic 4 per thread, so it converts the keys and their addresses between the two layouts. With the runtime stride, one layout serves both the loads and the atomic:

```mlir
// Constant stride (ir/v4/v4.ttgir), epilogue loop
#blocked  = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
%kv_71 = tt.load %kv_70, %rmask_64, %cst_4 : tensor<4096x!tt.ptr<bf16>, #blocked>
%12 = ttg.convert_layout %kv_71 : tensor<4096xbf16, #blocked> -> tensor<4096xbf16, #blocked4>
%14 = ttg.convert_layout %11 : tensor<4096x!tt.ptr<f32>, #blocked> -> tensor<4096x!tt.ptr<f32>, #blocked4>
%15 = tt.atomic_rmw fadd, relaxed, gpu, %14, %13, %rmask_65 : (tensor<4096x!tt.ptr<f32>, #blocked4>, ...)

// Runtime stride (ir/v5/v5.ttgir), epilogue loop
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
%kv_66 = tt.load %kv_65, %rmask_60, %cst_4 : tensor<4096x!tt.ptr<bf16>, #blocked>
%13 = tt.atomic_rmw fadd, relaxed, gpu, %11, %12, %rmask_60 : (tensor<4096x!tt.ptr<f32>, #blocked>, ...)
```

The conversions pass through shared memory. In PTX, the constant-stride epilogue loop body contains 20 shared-memory stores, 20 shared-memory loads, and 20 barriers; the runtime-stride loop body contains none. These are static instruction counts.

```
// ir/v4/v4.ptx, epilogue loop (excerpt)
@%p14 ld.global.v4.b32 { %r504, %r505, %r506, %r507 }, [ %rd79 + 0 ];
bar.sync 0;
st.shared.v4.b32 [%r44], {%r504, %r505, %r509, %r510};
st.shared.v4.b32 [%r45+2048], {%r506, %r507, %r511, %r512};
bar.sync 0;
ld.shared.v4.b32 {%r595, %r596, %r597, %r598}, [%r46];
...
@%p18 atom.global.gpu.relaxed.add.f32 %r521, [ %rd83 + 0 ], %r522;
```

The original's tile atomic also converts the key tile, from 8 to 4 consecutive columns per thread, before the atomic. The timing comparisons therefore include both improved atomic access locality and removed layout-conversion work. Per-variant counts and the full excerpts are in [`results/remainder_and_sectors.txt`](results/remainder_and_sectors.txt).

## Limits

Results cover one A100, one benchmark shape, and one model’s keys. Packing trades score precision for centroid-index bits: its regret bound doubles whenever the centroid count doubles.

Stress tests appending random unit-length centroids observed errors exceeding the fixed $2^{-12}$ tolerance at 4,096 and 8,192 centroids, consistent with that tradeoff. These added centroids are not fresh k-means solutions. The effect of these errors on downstream model quality has not been evaluated.

Source: [Double-P](https://arxiv.org/abs/2602.05191). Terminology: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).


