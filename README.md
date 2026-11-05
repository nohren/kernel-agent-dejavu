# Investigating an agent optimized Triton kernel

**1.38× median kernel speedup on A100 across 16 real Qwen2.5-1.5B layer/head key sets, using Triton 3.6.0.**

I built a Claude-driven optimization loop for [Double-P’s](https://arxiv.org/abs/2602.05191) k-means assignment kernel. **The agent generated and benchmarked revisions. I investigated why the winning candidate, `iter06`, worked**, using controlled ablations, Nsight Compute, and TTGIR/PTX/SASS inspection.

The kernel assigns keys to centroids and accumulates their sums and counts. Each launch processes eight sets of 32,768 keys, with dimension 128 and 2,048 centroids.

## Main findings

- **Defer the reduction.** Moving argmax reduction after the centroid loop reduced repeated work. With the same epilogue and launch settings, deferral without packing reduced profiled runtime from **918 to 865 µs**. Packing indices into score bits reduced it further to **839 µs**, a 3% improvement.  This may not be worth the precision loss.
- **Match the layout to the instruction.** Under Triton 3.6, the agent’s flat epilogue with a runtime stride of 1 reduced atomic sectors/request from **16 to 4.12**. Compiler IR showed neighboring lanes accessing consecutive columns instead of every fourth column.
- **Better coalescing does not guarantee a faster kernel.** Triton 3.8 fixed the measured coalescing issue, but the original kernel’s runtime did not improve in the profiling comparison below.

## Current question

**Why does the original kernel remain slow after Triton fixes atomic coalescing?**

| Original kernel, profiled | Triton 3.6 | Triton 3.8 |
|---|---:|---:|
| Atomic sectors/request | 16.00 | 4.12 |
| Registers/thread | 166 | 254 |
| Achieved occupancy | 18.0% | 12.2% |
| Runtime | 1,099 µs | 1,125 µs |

Register usage rises and occupancy falls, but these counters do not establish what causes the remaining performance gap. Even with equal sector efficiency in 3.8, the flat runtime-stride variant remains faster than its constant-stride counterpart. **That is where the investigation currently stops.**

## Correctness and scope

At 2,048 centroids ($2^{11}$), `iter06` passed the specified normalized score gap tolerance: **1.9 × 10⁻⁴ worst measured gap**, below **2⁻¹² ≈ 2.44 × 10⁻⁴**. Tests at 4,096 and 8,192 centroids exceeded that fixed tolerance as expected. Results are kernel level measurements on A100, not end to end model speedups or model quality evaluations.  Packing indices only provides a 3% speedup overall so may not be worth it.

[Read the analysis](/ANALYSIS.md) for full tables, methodology, the Triton 3.8 comparison, and limits. [Reproduction notebook](notebooks/kernel_explained.ipynb) · [Results](results/) · [Compiler IR](ir/) · [Agent loop](agent/agent_loop.py) · [Winning kernel](iter06.py)
