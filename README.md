# Evaluating an agent-optimized Triton kernel

I created a Claude driven agent loop which optimized Double-P’s Triton k-means assignment kernel achieving a **1.38× median speedup** on an A100 across 16 real Qwen2.5-1.5B layer KV head key sets. This kernel was the sixth candidate, **iter06**, and passed the specified correctness tolerance at 2,048 centroids.

This comparison covers the original and iter06. Measurements and reproduction steps are in [`kernel_explained.ipynb`](kernel_explained.ipynb) and [`results/`](results/).

## Workload and measurement

For each key, the kernel selects the centroid with the largest dot product, adds the key to that centroid’s sum, and increments its count. Each launch processes eight independent key sets, each containing 32,768 keys of dimension 128 and 2,048 unit-length centroids. Inputs are BF16; sums use FP32.

The evaluation uses 16 layer KV head key sets from two text prompts. For individual measurements, each key set is replicated across all eight batch slots, preserving the launch shape. These are kernel timings, not end to end model speedups.

The harness checks candidates against an FP32 reference, benchmarks original and candidate in interleaved rounds within the same process, and repeats correctness checks after timing. The agent receives benchmark results and, when enabled, Nsight Compute reports to guide subsequent revisions.

## Results

| Input | Original | iter06 | Speedup |
|---|---:|---:|---:|
| Evaluation A: eight distinct key sets | 0.899 ms | 0.654 ms | 1.374× |
| Evaluation B: eight distinct key sets | 0.905 ms | 0.657 ms | 1.381× |
| Individual key sets, each replicated eight times | 0.889–0.921 ms | 0.655–0.668 ms | 1.36–1.40× |

The median individual-case speedup is **1.38×**. For every key, the chosen score must be within $2^{-12}\lVert k\rVert_2$ of the reference maximum; sums and counts must agree with the kernel’s labels. At 2,048 centroids, iter06 passes both evaluation sets, with a worst measured cosine-score gap of **$1.9\times10^{-4}$**, below the $2.44\times10^{-4}$ tolerance.

## Two kernel changes

**Packed argmax with a deferred reduction.** The original reduces each score tile to a maximum and index inside the centroid loop. iter06 replaces the low score bits with the centroid index, maintains elementwise maxima across iterations, and performs one reduction after the loop. This reduces repeated reduction work and sacrifices score precision.

With $C$ centroids, the index occupies $b=\lceil\log_2 C\rceil$ bits, leaving $p=23-b$ FP32 fraction bits. Packing can change the winner within a truncated-score bin. The discarded bits sum to less than one unit of the last retained place; the binary exponent scales that bin’s width.

For a nonzero key and unit-length centroids, the idealized packing-only regret bound is:

$$
\frac{s_{\mathrm{best}}-s_{\mathrm{chosen}}}{\lVert k\rVert_2}
\le 2^{-(23-b)}.
$$
The bound doubles with each doubling of $C$.
```python
   for start in range(0, C, BLOCK_K):
        x = tl.load(x_ptrs)
        ip = tl.dot(k, x)
        bits = ip.to(tl.int32, bitcast=True)
        idx = start + offs_k
        packed = ((bits & MASK_HI) | idx[None, :]).to(tl.float32, bitcast=True) # bit packing max idx into score
        best = tl.maximum(best, packed) # max index conditional updates completely removed
        x_ptrs += BLOCK_K * D

    row_best = tl.max(best, axis=1) # reduction op outside the loop
    max_idx = row_best.to(tl.int32, bitcast=True) & MASK_LO
   ```

**Flat reload epilogue.** After storing assignments, iter06 reloads keys in flat chunks before atomic accumulation. Profiling shows more compact atomic accesses. A stride argument supplied as 1 uses Triton’s `do_not_specialize`; removing it reduces performance in the ablation. Its exact compiler-level effect remains unresolved.

 ```python
   @triton.jit(do_not_specialize=["stride_one"])
   def _assign_packed_kernel(
    K, X, S, CNT, M,
    N, C, stride_one,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
    MASK_HI: tl.constexpr, MASK_LO: tl.constexpr, REDUCE_EACH: tl.constexpr,
    EPI_ROWS: tl.constexpr,
    ):
   ```
   ```python
      #epilogue, flat reload and accumulate EPI_ROWS at a time
      # same number of global atomic requests, sectors per atomic request drops by 4x
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
          tl.atomic_add(s_base + lab * D + cc, kv.to(tl.float32), mask=rmask, sem="relaxed")
   ```

## Ablation and profiling

The ablation holds the original launch configuration fixed on one key set:

| Variant | Speedup vs original |
|---|---:|
| Rewritten baseline | 0.980× |
| Packed argmax only | 1.060× |
| Flat reload only (runtime stride)| 1.153× |
| Packed argmax + flat reload (constant stride) | 1.082× |
| Packed argmax + flat reload (runtime stride) | 1.252× |
| Full iter06 | 1.368× |

All ablation variants pass the checks at 2,048 centroids. The rewritten baseline is 2% slower than the original; the additional improvement from 1.252× to 1.368× is not isolated.

Nsight Compute reports:

| Metric | Original | iter06 |
|---|---:|---:|
| Global atomic requests | 1,056,768 | 1,056,768 |
| Sectors per atomic request | 16.0 | 4.12 |
| Achieved occupancy | 18.01% | 18.05% |

The same request count touches **about 3.9× fewer 32-byte sectors**, supporting improved atomic-access locality. Occupancy is essentially unchanged. These full-kernel counters do not assign a precise share of the speedup to either change.

## Limits

Results cover one A100, one benchmark shape, and one model’s keys. Packing trades score precision for centroid-index bits: its regret bound doubles whenever the centroid count doubles. Tests at 4,096 and 8,192 centroids observed errors exceeding the fixed $2^{-12}$ tolerance, consistent with that tradeoff. Whether the larger $C$ meaningfully affect downstream model quality has not been evaluated.

Source: [Double-P](https://arxiv.org/abs/2602.05191). Terminology: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

