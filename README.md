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

This bound excludes dot-product rounding error. At 2,048 centroids, `MASK_HI = -2048` clears 11 low bits and `MASK_LO = 2047` extracts the index. The general technique is to defer cross-element reductions when an elementwise running state can preserve the required result; packing additionally trades precision for simpler index tracking.

**Flat reload epilogue.** After storing assignments, iter06 reloads keys in flat chunks before atomic accumulation. The kernel uses `@triton.jit(do_not_specialize=["stride_one"])`, and the caller passes `stride_one=1`. In this flat epilogue, preventing specialization to 1 changes the compiled layout: neighboring lanes update consecutive columns instead of every fourth column. The excerpt below runs after labels are stored and the block synchronizes; base pointers select the current batch item.

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

## What the ablations establish

The following Nsight Compute measurements use one key set and locked clocks; they are separate from the benchmark timings above. All variants pass the specified checks at 2,048 centroids. **Constant** and **runtime** strides both equal 1; only whether the compiler knows that value differs. “Packed” includes deferred reduction; “tile” accumulates from the existing key tile.

| Variant | Argmax | Accumulation | Column stride | Time (µs) | Atomic sectors/request |
|---|---|---|---|---:|---:|
| Original | Plain | Tile | Constant | 1,105 | 16.03 |
| v1: rewritten baseline | Plain | Tile | Constant | 1,107 | 16.03 |
| v2 | Packed | Tile | Constant | 1,026 | 16.03 |
| v3 | Plain | Flat reload | Runtime | 908 | 4.12 |
| v4 | Packed | Flat reload | Constant | 998 | 16.03 |
| v5 | Packed | Flat reload | Runtime | 838 | 4.12 |
| v8 | Plain | Tile | Runtime | 1,417 | 31.90 |
| v9 | Packed | Tile | Runtime | 1,548 | 31.90 |
| v6: matched tile settings | Packed | Flat reload | Runtime | 768 | 4.12 |
| iter06 | Packed | Flat reload | Runtime | 776 | 4.12 |

**Packing reduces instructions.** Matched comparisons show 31–33% fewer executed warp instructions, with unchanged sectors/request. Overall, iter06 executes 115.8M versus the original’s 159.8M instructions: **27.6% fewer**.

**Flat reload and runtime stride work together here.** Flattening alone leaves sectors/request at 16.03. Runtime stride on the tile epilogue increases them to 31.90. Their combination reaches 4.12. Every variant issues 1,056,768 atomic requests, so the improvement is in how those requests access memory.

**Tile settings close the remaining gap.** v6 changes v5’s `BLOCK_K` from 32 to 64 and `EPI_ROWS` from 32 to 16, reproducing iter06’s SASS after normalizing kernel names. 
## Reading the thread mapping beneath Triton

The GPU partitions each thread block into warps of 32 consecutive threads.
Each thread occupies a warp lane numbered 0–31. A warp scheduler issues
an instruction to the warp’s active threads.

Triton’s layout determines which tensor elements each thread owns.
Changing the layout changes the elements assigned to threads, but it does
not change how the GPU partitions threads into warps. The TTGIR below exposes
this element to thread mapping. 

For these kernels, each block contains 128 threads: four warps of 32.
To understand atomic access efficiency, examine the destination address
used by each active thread when its warp executes one atomic instruction.

Each cell shows one thread’s update during the same warp level atomic
instruction. `s[b, c, d]` stores component `d` of centroid `c`’s accumulated
sum in batch `b`. Atomic updates are written as += for readability.

| Implementation | Thread at lane 0 | Thread at lane 1 | Thread at lane 2 |
|---|---|---|---|
| Original tile; flat reload with constant stride | `s[b, c, 0] += key[0]` | `s[b, c, 4] += key[4]` | `s[b, c, 8] += key[8]` |
| Flat reload with runtime stride | `s[b, c, 0] += key[0]` | `s[b, c, 1] += key[1]` | `s[b, c, 2] += key[2]` |
| Tile with runtime stride | `s[b, c0, 0] += key0[0]` | `s[b, c1, 0] += key1[0]` | `s[b, c0, 0] += key2[0]` |

In the first two rows, threads process components of the same key,
assigned to centroid `c`. In the third row, threads process different
keys, whose chosen centroids `c0`, `c1`, and `c100` etc. may coincide.

A sector is an aligned 32-byte chunk of memory in a cache line or device memory. For instance an L1 or L2 cache line is four sectors i.e 128 bytes. [see here](https://docs.nvidia.com/nsight-compute/ProfilingGuide). This means that each aligned sector can contain eight FP32 values. With 32 active threads, aligned memory access touches four sectors.  Accessing columns `0, 4, …, 124` touches 16 sectors, two threads access values in each sector.

**For scalar atomic instructions.** Threads participate together in each warp instruction, then process their other values through additional instructions.

| Atomic instruction | Lane 0 updates | Lane 1 updates | Lane 2 updates | … Lane 31 updates |
|---|---:|---:|---:|---:|
| First | 0 | 4 | 8 | 124 |
| Second | 1 | 5 | 9 | 125 |
| Third | 2 | 6 | 10 | 126 |
| Fourth | 3 | 7 | 11 | 127 |

Read **across a row** for the work performed by threads together. Read **down a column** for one thread’s work across instructions.


This explains the inefficiency: **each row touches 16 sectors**, even though each thread’s four values are adjacent. Coalescing concerns addresses accessed **across threads in one instruction**. **`sizePerThread = [1]` means one consecutive element per thread before the layout moves to the next thread**. 

For a simple **one-warp example** covering 128 values:

| Instruction | Lane 0 | Lane 1 | Lane 2 | … Lane 31 |
|---|---:|---:|---:|---:|
| First | 0 | 1 | 2 | 31 |
| Second | 32 | 33 | 34 | 63 |
| Third | 64 | 65 | 66 | 95 |
| Fourth | 96 | 97 | 98 | 127 |

Each thread still handles four values, but they are distributed differently:

- **Old lane 0:** `0, 1, 2, 3`
- **New lane 0:** `0, 32, 64, 96`

Both perform **four warp instructions and 128 additions**. The old mapping touches 16 sectors per instruction; the new mapping touches four.



 TTGIR evidence is the flat epilogue’s load and atomic layouts. Below, aliases are renamed and operations abbreviated for readability:

```
// v4: layouts used by the flat epilogue.
// Key loads use #blocked; sum atomics use #blocked4.
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// v5: key loads and sum atomics use the same layout.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```
For these 1D layouts, `sizePerThread = [4]` assigns groups of four
consecutive elements to each thread. In the first scalar atomic
instruction, the threads at warp lanes 0, 1, and 2 therefore update
elements 0, 4, and 8. With `sizePerThread = [1]`, those same threads
update elements 0, 1, and 2.

## Limits
Results cover one A100, one benchmark shape, and one model’s keys. Packing trades score precision for centroid index bits: its regret bound doubles whenever the centroid count doubles. Tests at 4,096 and 8,192 centroids observed errors exceeding the fixed $2⁻¹²$ tolerance, consistent with that tradeoff. Whether the larger $C$ meaningfully affect downstream model quality has not been evaluated.

Source: [Double-P](https://arxiv.org/abs/2602.05191). Terminology: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).


