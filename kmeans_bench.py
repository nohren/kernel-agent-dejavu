"""Correctness, benchmark and profiling harness for Double-P's Triton k-means assign kernel.

The kernel is copied from sparse_attention/index/cluster/kmeans.py (_triton_assign_kernel). Shapes mirror how
segment_k_means calls it for LLaMA-3.1-8B (8 kv heads, head_dim 128, ~16 keys per cluster, 1 segment per 8K tokens).

    python kmeans_bench.py check                  # correctness + determinism vs fp32 reference, per dtype
    python kmeans_bench.py bench                  # time Triton vs unfused PyTorch, achieved TFLOP/s per shape
    python kmeans_bench.py iter                   # one full Lloyd iteration: zeroing vs assign (E) vs update (M)
    python kmeans_bench.py jit                    # first-call (compile) latency across new sequence lengths
    python kmeans_bench.py search SHAPE           # config search gated on correctness; reports best vs original
    python kmeans_bench.py ncu VARIANT SHAPE [k=v ...]   # one warmup + one measured launch, for Nsight Compute
"""

import os
import random
import statistics
import sys
import tempfile
import time

# fresh compile cache so first-call timings include real JIT compilation
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="triton_cache_"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DEV = "cuda"

# dense (no sparsity) tensor-core peak TFLOP/s per input format and DRAM GB/s; first name match wins.
# "fp32" is TF32 on sm_80+ (Triton's default for fp32 tl.dot) and plain FMA on Turing.
GPU_TABLE = [
    ("H100 PCIe", {"bf16": 756, "fp16": 756, "fp8": 1513, "fp32": 378}, 2000),
    ("H100", {"bf16": 989, "fp16": 989, "fp8": 1979, "fp32": 494}, 3350),
    ("A100", {"bf16": 312, "fp16": 312, "fp32": 156}, 1555),  # 2039 GB/s on the 80 GB part, set below
    ("L4", {"bf16": 121, "fp16": 121, "fp8": 242, "fp32": 60}, 300),
    ("T4", {"fp16": 65, "fp32": 8.1}, 320),
]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp8": torch.float8_e4m3fn}


def gpu_info():
    p = torch.cuda.get_device_properties(0)
    peaks, dram = {}, None
    for key, pk, bw in GPU_TABLE:
        if key in p.name:
            peaks, dram = pk, bw
            break
    if "A100" in p.name and "80GB" in p.name:
        dram = 2039
    return {
        "name": p.name, "cc": (p.major, p.minor), "sms": p.multi_processor_count,
        "mem_gb": round(p.total_memory / 1e9, 1), "l2_mb": round(getattr(p, "L2_cache_size", 0) / 2**20, 1),
        "peak_tflops": peaks, "dram_gbs": dram,
    }


def supported_formats(cc):
    fmts = ["fp16", "fp32"]
    if cc >= (8, 0):
        fmts.append("bf16")  # bf16 tensor cores arrived with Ampere (A100)
    if cc >= (8, 9):
        fmts.append("fp8")  # fp8 e4m3 tensor cores: Ada (L4) and Hopper (H100)
    return fmts


GPU = gpu_info() if torch.cuda.is_available() else None
# input format: bf16 where the hardware has it (what Double-P stores keys in), otherwise fp16.
# Override with KMEANS_DTYPE=fp16|bf16|fp32|fp8.
DTYPE_NAME = os.environ.get("KMEANS_DTYPE") or ("bf16" if GPU and GPU["cc"] >= (8, 0) else "fp16")
DTYPE = DTYPES[DTYPE_NAME]


def peak_tflops(fmt=None):
    return ((GPU or {}).get("peak_tflops") or {}).get(fmt or DTYPE_NAME)


def describe_gpu():
    g = GPU
    peak = peak_tflops()
    return (f"{g['name']} (sm_{g['cc'][0]}{g['cc'][1]}, {g['sms']} SMs, {g['mem_gb']} GB, L2 {g['l2_mb']} MB, "
            f"DRAM ~{g['dram_gbs'] or '?'} GB/s, {DTYPE_NAME} tensor-core peak ~{peak or '?'} TFLOP/s dense)")

# name: (batch rows, tokens per row, centroids per row) for head_dim 128
SHAPES = {
    "seg-32K": (8 * 4, 8192, 512),  # iterations 1-9 at 32K: 8 kv heads x 4 segments
    "final-32K": (8, 32768, 2048),  # final global pass at 32K
    "seg-128K": (8 * 16, 8192, 512),  # iterations 1-9 at 128K: 16 segments
    "final-128K": (8, 131072, 8192),  # final global pass at 128K
}
HEAD_DIM = 128


# ---------------------------------------------------------------------------------------------------------------------
# original kernel (verbatim logic from kmeans.py): every size and stride is tl.constexpr
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def assign_orig(
    K, X, S, C, M,
    stride_kz: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    stride_xz: tl.constexpr, stride_xk: tl.constexpr, stride_xd: tl.constexpr,
    stride_sz: tl.constexpr, stride_sk: tl.constexpr, stride_sd: tl.constexpr,
    stride_cz: tl.constexpr, stride_ck: tl.constexpr,
    stride_mz: tl.constexpr, stride_mn: tl.constexpr,
    num_tokens: tl.constexpr, num_centroids: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)
    if start_n >= num_tokens:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens
    k_ptrs = K + batch_idx * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    x_ptrs = X + batch_idx * stride_xz + offs_k[None, :] * stride_xk + offs_d[:, None] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn
    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
    max_val = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)
    for start_k in tl.range(0, num_centroids, BLOCK_K):
        x = tl.load(x_ptrs)
        ip = tl.dot(k, x).to(tl.float32)
        tmp_max_val, tmp_max_idx = tl.max(ip, axis=1, return_indices=True)
        tmp_max_idx += start_k
        max_idx = tl.where(tmp_max_val > max_val, tmp_max_idx, max_idx)
        max_val = tl.maximum(tmp_max_val, max_val)
        x_ptrs += BLOCK_K * stride_xk
    tl.store(m_ptrs, max_idx, mask=n_mask)
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, k.to(tl.float32), mask=n_mask[:, None], sem="relaxed")
    tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem="relaxed")


# ---------------------------------------------------------------------------------------------------------------------
# variant: identical body, but sizes and strides are runtime arguments (no recompile per sequence length)
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def assign_dyn(
    K, X, S, C, M,
    stride_kz, stride_kn, stride_kd,
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    stride_mz, stride_mn,
    num_tokens, num_centroids,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)
    if start_n >= num_tokens:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens
    k_ptrs = K + batch_idx * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    x_ptrs = X + batch_idx * stride_xz + offs_k[None, :] * stride_xk + offs_d[:, None] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn
    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
    max_val = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)
    for start_k in tl.range(0, num_centroids, BLOCK_K):
        x = tl.load(x_ptrs)
        ip = tl.dot(k, x).to(tl.float32)
        tmp_max_val, tmp_max_idx = tl.max(ip, axis=1, return_indices=True)
        tmp_max_idx += start_k
        max_idx = tl.where(tmp_max_val > max_val, tmp_max_idx, max_idx)
        max_val = tl.maximum(tmp_max_val, max_val)
        x_ptrs += BLOCK_K * stride_xk
    tl.store(m_ptrs, max_idx, mask=n_mask)
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, k.to(tl.float32), mask=n_mask[:, None], sem="relaxed")
    tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem="relaxed")


# ---------------------------------------------------------------------------------------------------------------------
# M-step kernel (verbatim logic from kmeans.py _triton_update_kernel): centroid = sum / count, optional L2 normalize
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def update_orig(
    X, S, C,
    stride_xz: tl.constexpr, stride_xk: tl.constexpr, stride_xd: tl.constexpr,
    stride_sz: tl.constexpr, stride_sk: tl.constexpr, stride_sd: tl.constexpr,
    stride_cz: tl.constexpr, stride_ck: tl.constexpr,
    num_centroids: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, NORMORLIZE: tl.constexpr,
):
    start_k = tl.program_id(0) * BLOCK_K
    batch_idx = tl.program_id(1)
    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < num_centroids
    x_ptrs = X + batch_idx * stride_xz + offs_k[:, None] * stride_xk + offs_d[None, :] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_k[:, None] * stride_sk + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz + offs_k[:, None] * stride_ck
    s = tl.load(s_ptrs, mask=k_mask[:, None], other=0.0)
    c = tl.load(c_ptrs, mask=k_mask[:, None], other=0)
    x_mask = c > 0
    x = s / c
    if NORMORLIZE:
        x /= tl.sqrt(tl.sum(x * x, axis=-1, keep_dims=True))
    tl.store(x_ptrs, x.to(X.type.element_ty), mask=x_mask)


def launch_update(cent, s, cnt, normalize=True, block_k=128):
    b, c, d = cent.shape
    update_orig[(triton.cdiv(c, block_k), b, 1)](
        cent, s, cnt, *cent.stride(), *s.stride(), *cnt.stride(), c,
        BLOCK_K=block_k, BLOCK_D=d, NORMORLIZE=normalize, num_warps=4, num_stages=1,
    )


KERNELS = {"orig": assign_orig, "dyn": assign_dyn}
DEFAULT_CFG = {"block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 2}  # what kmeans.py uses


def make_inputs(b, n, c, dtype=None, seed=0):
    dtype = dtype or DTYPE
    g = torch.Generator(device=DEV).manual_seed(seed)
    data = torch.randn(b, n, HEAD_DIM, device=DEV, generator=g).to(dtype)
    cent = torch.randn(b, c, HEAD_DIM, device=DEV, generator=g)
    cent = (cent / cent.norm(dim=-1, keepdim=True)).to(dtype)  # spherical k-means keeps unit centroids
    return data, cent


def alloc_outputs(data, cent):
    b, n, d = data.shape
    c = cent.shape[1]
    s = torch.zeros(b, c, d, device=DEV, dtype=torch.float32)
    cnt = torch.zeros(b, c, device=DEV, dtype=torch.int32)
    m = torch.empty(b, n, device=DEV, dtype=torch.int32)
    return s, cnt, m


def launch(variant, data, cent, s, cnt, m, block_n=128, block_k=32, num_warps=4, num_stages=2):
    b, n, d = data.shape
    c = cent.shape[1]
    assert c % block_k == 0, "centroid loads are unmasked: num_centroids must be a multiple of BLOCK_K"
    KERNELS[variant][(triton.cdiv(n, block_n), b, 1)](
        data, cent, s, cnt, m,
        *data.stride(), *cent.stride(), *s.stride(), *cnt.stride(), *m.stride(),
        n, c,
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_D=d, num_warps=num_warps, num_stages=num_stages,
    )


def run(variant, data, cent, **cfg):
    s, cnt, m = alloc_outputs(data, cent)
    launch(variant, data, cent, s, cnt, m, **cfg)
    return s, cnt, m


def torch_assign(data, cent):
    # unfused reference: GEMM -> argmax -> scatter-add, materializing the [B, N, C] score matrix
    ip = torch.bmm(data, cent.transpose(1, 2))
    m = ip.argmax(-1)
    b, c, d = cent.shape
    s = torch.zeros(b, c, d, device=DEV, dtype=torch.float32).scatter_add_(
        1, m[..., None].expand(-1, -1, d), data.float())
    cnt = torch.zeros(b, c, device=DEV, dtype=torch.float32).scatter_add_(1, m, torch.ones_like(m, dtype=torch.float32))
    return s, cnt.int(), m


def verify(data, cent, s, cnt, m, max_tokens=4096):
    """Compare a Triton result against exact fp32 math. Returns a dict of metrics."""
    n_chk = min(data.shape[1], max_tokens)
    d32, c32 = data[:, :n_chk].float(), cent.float()
    ip32 = torch.bmm(d32, c32.transpose(1, 2))
    ref = ip32.argmax(-1)
    mm = m[:, :n_chk].long()
    best = ip32.gather(-1, ref[..., None]).squeeze(-1)
    got = ip32.gather(-1, mm[..., None]).squeeze(-1)
    knorm = d32.norm(dim=-1).clamp_min(1e-30)
    # sums / counts must be consistent with the kernel's own assignment
    b, c, d = cent.shape
    s_ref = torch.zeros(b, c, d, device=DEV, dtype=torch.float64).scatter_add_(
        1, m.long()[..., None].expand(-1, -1, d), data.float().double())  # via fp32: fp8 has no direct cast
    cnt_ref = torch.zeros(b, c, device=DEV, dtype=torch.float64).scatter_add_(
        1, m.long(), torch.ones_like(m, dtype=torch.float64))
    return {
        "argmax_agree": (mm == ref).double().mean().item(),
        # how much worse the chosen centroid's score is than the true best (0 = exact or tie)
        "max_regret": (best - got).max().item(),
        # the same gap divided by |key|: a cosine-similarity gap, since centroids are unit norm
        "max_rel_regret": ((best - got) / knorm).max().item(),
        "sum_rel_err": ((s.double() - s_ref).abs().max() / s_ref.abs().max()).item(),
        "counts_exact": bool(torch.equal(cnt.double(), cnt_ref)),
    }


# chosen centroid's score within REL_TOL * |key| of the best: 16x finer than bf16's 2^-8 relative precision. Scale-free,
# unlike an absolute tolerance (real keys range from |k| ~20 to ~800), and loose enough for near-tie flips and for
# approximations whose error is provably below it.
REL_TOL = 2.0 ** -12


def passes(v, rel_tol=REL_TOL):
    return v["counts_exact"] and v["sum_rel_err"] < 1e-5 and v["max_rel_regret"] <= rel_tol


def flops_and_bytes(b, n, c, elem):
    flops = 2.0 * b * n * c * HEAD_DIM
    min_bytes = b * n * HEAD_DIM * elem + b * c * HEAD_DIM * elem + b * n * 4  # keys + centroids once + labels
    atomic_bytes = b * n * HEAD_DIM * 4  # one fp32 atomic add per key element
    return flops, min_bytes, atomic_bytes


def bench_fn(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def interleaved(fns, zero, rounds=7):
    """Time each fn in rotating order for `rounds` rounds, after holding the GPU busy so clocks settle.

    Each fn zeroes the accumulators and then launches, as a real k-means iteration does (this also leaves s in L2 the
    way the kernel finds it in practice). `zero` alone is timed in the same rounds and subtracted, so the returned
    {name: [ms per round]} is kernel time only. Also returns the per-round zeroing times.
    """
    fns = {**fns, "_zero": zero}
    names = list(fns)
    for f in fns.values():  # compile first
        f()
    triton.testing.do_bench(fns[names[0]], warmup=500, rep=500)
    raw = {k: [] for k in names}
    for r in range(rounds):
        for k in names[r % len(names):] + names[:r % len(names)]:
            raw[k].append(bench_fn(fns[k]))
    zeros = raw.pop("_zero")
    return {k: [t - z for t, z in zip(ts, zeros)] for k, ts in raw.items()}, zeros


def timed_launch(variant, data, cent, s, cnt, m, **cfg):
    """Zero the accumulators, then launch; the timed unit for interleaved()."""
    def f():
        s.zero_()
        cnt.zero_()
        launch(variant, data, cent, s, cnt, m, **cfg)
    return f


def zeroer(s, cnt):
    def f():
        s.zero_()
        cnt.zero_()
    return f


def bench_kernel(variant, data, cent, s, cnt, m, rounds=3, **cfg):
    """Median kernel-only ms (zeroing subtracted) and median zeroing ms."""
    t, zeros = interleaved({"k": timed_launch(variant, data, cent, s, cnt, m, **cfg)}, zeroer(s, cnt), rounds)
    return statistics.median(t["k"]), statistics.median(zeros)


# ---------------------------------------------------------------------------------------------------------------------
def cmd_check():
    b, n, c = 8, 8192, 512
    print(f"GPU: {describe_gpu()}")
    print(f"correctness at B={b}, N={n}, C={c}, D={HEAD_DIM} vs exact fp32 math (first 4096 tokens per row)")
    fmts = supported_formats(GPU["cc"])
    if GPU["cc"] < (8, 0):
        print("  bf16 skipped: no bf16 tensor cores before sm_80; fp32 runs on CUDA cores (no TF32 before sm_80)")
    for fmt in fmts:
        label = fmt + (" (TF32 dot)" if fmt == "fp32" and GPU["cc"] >= (8, 0) else "")
        data, cent = make_inputs(b, n, c, DTYPES[fmt])
        try:
            s, cnt, m = run("orig", data, cent, **DEFAULT_CFG)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  {label:16s} FAILED to compile/run: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
            continue
        v = verify(data, cent, s, cnt, m)
        s2, cnt2, m2 = run("orig", data, cent, **DEFAULT_CFG)
        ms, _ = bench_kernel("orig", data, cent, *alloc_outputs(data, cent), **DEFAULT_CFG)
        tflops = 2.0 * b * n * c * HEAD_DIM / ms / 1e9
        print(f"  {label:16s} agree={v['argmax_agree']:.5f} regret={v['max_regret']:.2e} "
              f"sum_err={v['sum_rel_err']:.1e} counts_exact={v['counts_exact']}  "
              f"labels_identical_rerun={torch.equal(m, m2)} "
              f"sum_bits_differ_rerun={(s != s2).double().mean().item():.2%}  "
              f"time={ms:.3f} ms ({tflops:.1f} TFLOP/s, peak {peak_tflops(fmt) or '?'})")


def cmd_bench():
    print(f"GPU: {describe_gpu()}   input format: {DTYPE_NAME}")
    print(f"{'shape':11s} {'B':>4s} {'N':>7s} {'C':>5s} | {'triton ms':>9s} {'TFLOP/s':>7s} {'%peak':>6s} "
          f"{'zero us':>7s} | {'torch ms':>9s} {'speedup':>7s}")
    elem = torch.empty(0, dtype=DTYPE).element_size()
    peak = peak_tflops()
    for name, (b, n, c) in SHAPES.items():
        data, cent = make_inputs(b, n, c)
        ms, zero_ms = bench_kernel("orig", data, cent, *alloc_outputs(data, cent), **DEFAULT_CFG)
        flops, _, _ = flops_and_bytes(b, n, c, elem)
        tflops = flops / ms / 1e9
        pct = f"{tflops / peak:6.1%}" if peak else f"{'?':>6s}"
        # unfused PyTorch materializes the [B, N, C] score matrix; skip when it would not fit comfortably
        score_bytes = b * n * c * 2
        if DTYPE_NAME != "fp8" and score_bytes < 0.25 * GPU["mem_gb"] * 1e9:
            tms = bench_fn(lambda: torch_assign(data, cent))
            tcell = f"{tms:9.3f} {tms / (ms + zero_ms):6.2f}x"  # torch_assign allocates its own zeroed outputs
        else:
            tcell = f"{'skip':>9s} {'-':>7s}"
        print(f"{name:11s} {b:4d} {n:7d} {c:5d} | {ms:9.3f} {tflops:7.2f} {pct} {zero_ms * 1e3:7.1f} | {tcell}")


def cmd_iter():
    # one Lloyd iteration exactly as _triton_k_means_train runs it: allocate zeroed accumulators, assign, update
    print(f"GPU: {describe_gpu()}   input format: {DTYPE_NAME}")
    print(f"{'shape':11s} | {'zeros us':>8s} {'assign ms':>9s} {'update us':>9s} | {'iteration ms':>12s} "
          f"{'assign share':>12s} {'other':>8s}")
    for name, (b, n, c) in SHAPES.items():
        data, cent = make_inputs(b, n, c)
        work = cent.clone()  # the update writes centroids in place; keep the inputs untouched
        s, cnt, m = alloc_outputs(data, cent)
        d = data.shape[-1]

        def zeros():
            torch.zeros(b, c, d, device=DEV, dtype=torch.float32)
            torch.zeros(b, c, device=DEV, dtype=torch.int32)

        def full_iteration():
            s_ = torch.zeros(b, c, d, device=DEV, dtype=torch.float32)
            cnt_ = torch.zeros(b, c, device=DEV, dtype=torch.int32)
            launch("orig", data, work, s_, cnt_, m, **DEFAULT_CFG)
            launch_update(work, s_, cnt_)

        launch("orig", data, cent, s, cnt, m, **DEFAULT_CFG)  # realistic counts for the update timing
        t_zero = bench_fn(zeros)
        t_assign = bench_fn(lambda: launch("orig", data, cent, s, cnt, m, **DEFAULT_CFG))
        t_update = bench_fn(lambda: launch_update(work, s, cnt))
        t_iter = bench_fn(full_iteration)
        other = t_iter - t_assign
        print(f"{name:11s} | {t_zero * 1e3:8.1f} {t_assign:9.3f} {t_update * 1e3:9.1f} | {t_iter:12.3f} "
              f"{t_assign / t_iter:12.1%} {other * 1e3:6.1f}us")


def cmd_jit():
    random.seed(0)
    lengths = [random.randint(6000, 12000) for _ in range(5)]
    c = 512
    for variant in ["orig", "dyn"]:
        firsts, steadies = [], []
        for n in lengths:
            data, cent = make_inputs(8, n, c)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(variant, data, cent, **DEFAULT_CFG)
            torch.cuda.synchronize()
            firsts.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            run(variant, data, cent, **DEFAULT_CFG)
            torch.cuda.synchronize()
            steadies.append((time.perf_counter() - t0) * 1e3)
        print(f"{variant:5s} new lengths {lengths}")
        print(f"      first call ms: {[round(x, 1) for x in firsts]}")
        print(f"      steady call ms: {[round(x, 2) for x in steadies]}")


def cmd_search(shape):
    b, n, c = SHAPES[shape]
    data, cent = make_inputs(b, n, c)
    s, cnt, m = alloc_outputs(data, cent)
    rows = []
    for block_n in [64, 128, 256]:
        for block_k in [32, 64, 128]:
            if c % block_k:
                continue
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3, 4]:
                    cfg = dict(block_n=block_n, block_k=block_k, num_warps=num_warps, num_stages=num_stages)
                    try:
                        vs, vcnt, vm = run("dyn", data, cent, **cfg)
                        torch.cuda.synchronize()
                    except Exception as e:  # out of shared memory / registers
                        rows.append((cfg, None, f"launch failed: {type(e).__name__}"))
                        continue
                    v = verify(data, cent, vs, vcnt, vm)
                    if not passes(v):
                        rows.append((cfg, None, f"REJECTED: {v}"))
                        continue
                    ms = bench_fn(timed_launch("dyn", data, cent, s, cnt, m, **cfg))  # coarse: zeroing included
                    rows.append((cfg, ms, f"agree={v['argmax_agree']:.4f}"))
    ok = sorted([r for r in rows if r[1] is not None], key=lambda r: r[1])
    bad = [r for r in rows if r[1] is None]
    print(f"{shape}: {len(ok)} configs passed the correctness gate, {len(bad)} failed or were rejected")
    for cfg, _, note in bad[:5]:
        print(f"  x {cfg}: {note[:120]}")

    # the coarse pass times configs minutes apart while clocks drift, so re-time the top 5 against the original in
    # interleaved rounds; dyn at the default config and orig at the best config separate the variant from the config
    top = ok[:5]
    fns = {"orig default": timed_launch("orig", data, cent, s, cnt, m, **DEFAULT_CFG),
           "dyn default": timed_launch("dyn", data, cent, s, cnt, m, **DEFAULT_CFG),
           "orig best cfg": timed_launch("orig", data, cent, s, cnt, m, **top[0][0])}
    for cfg, _, _ in top:
        fns[f"dyn {cfg}"] = timed_launch("dyn", data, cent, s, cnt, m, **cfg)
    t, zeros = interleaved(fns, zeroer(s, cnt))
    print(f"re-timed in interleaved rounds (kernel time, zeroing {statistics.median(zeros) * 1e3:.1f} us subtracted), "
          f"vs the original {DEFAULT_CFG}:")
    for name, ts in t.items():
        ratios = sorted(o / x for o, x in zip(t["orig default"], ts))
        print(f"  {statistics.median(ts):8.3f} ms  {statistics.median(ratios):6.3f}x "
              f"(rounds {ratios[0]:.3f}-{ratios[-1]:.3f})  {name}")


def cmd_ncu(variant, shape, overrides):
    b, n, c = SHAPES[shape]
    cfg = dict(DEFAULT_CFG)
    for kv in overrides:
        k, v = kv.split("=")
        cfg[k] = int(v)
    data, cent = make_inputs(b, n, c)
    run(variant, data, cent, **cfg)  # launch 1: compile + warmup (skip it in ncu)
    torch.cuda.synchronize()
    run(variant, data, cent, **cfg)  # launch 2: the one to profile
    torch.cuda.synchronize()
    print(f"profiled {variant} on {shape} with {cfg}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        cmd_check()
    elif mode == "bench":
        cmd_bench()
    elif mode == "iter":
        cmd_iter()
    elif mode == "jit":
        cmd_jit()
    elif mode == "search":
        cmd_search(sys.argv[2] if len(sys.argv) > 2 else "final-32K")
    elif mode == "ncu":
        cmd_ncu(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)
