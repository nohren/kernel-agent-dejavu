import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["stride_one"])
def _assign_packed_kernel(
    K, X, S, CNT, M,
    N, C, stride_one,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
    MASK_HI: tl.constexpr, MASK_LO: tl.constexpr, REDUCE_EACH: tl.constexpr,
    EPI_ROWS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    k_ptrs = K + b * N * D + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

    x_ptrs = X + b * C * D + offs_k[None, :] * D + offs_d[:, None]

    if REDUCE_EACH:
        best = tl.full([BLOCK_N], float("-inf"), tl.float32)
    else:
        best = tl.full([BLOCK_N, BLOCK_K], float("-inf"), tl.float32)
    for start in range(0, C, BLOCK_K):
        x = tl.load(x_ptrs)
        ip = tl.dot(k, x)
        bits = ip.to(tl.int32, bitcast=True)
        idx = start + offs_k
        packed = ((bits & MASK_HI) | idx[None, :]).to(tl.float32, bitcast=True)
        if REDUCE_EACH:
            best = tl.maximum(best, tl.max(packed, axis=1))
        else:
            best = tl.maximum(best, packed)
        x_ptrs += BLOCK_K * D

    if REDUCE_EACH:
        row_best = best
    else:
        row_best = tl.max(best, axis=1)
    max_idx = row_best.to(tl.int32, bitcast=True) & MASK_LO

    m_base = M + b * N
    tl.store(m_base + offs_n, max_idx, mask=n_mask)
    tl.atomic_add(CNT + b * C + max_idx, tl.zeros_like(max_idx) + 1, mask=n_mask, sem="relaxed")
    tl.debug_barrier()

    # coalesced scatter-add epilogue: 1-D chunks, one element per thread, lanes on consecutive columns
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


# (BLOCK_N, BLOCK_K, num_warps, num_stages, REDUCE_EACH)
_CONFIGS = [
    (128, 32, 4, 2, False),
    (128, 32, 4, 2, True),
    (128, 32, 4, 3, True),
    (128, 64, 4, 2, False),
    (128, 64, 4, 3, False),
    (128, 128, 8, 2, False),
    (256, 64, 8, 2, False),
    (256, 64, 4, 2, True),
    (256, 32, 8, 2, False),
]

_CACHE = {}


def _run(cfg, data, cent, s, cnt, m):
    bn, bk, nw, ns, red = cfg
    B, N, D = data.shape
    C = cent.shape[1]
    idx_bits = max(1, (C - 1).bit_length())
    grid = (triton.cdiv(N, bn), B, 1)
    _assign_packed_kernel[grid](
        data, cent, s, cnt, m,
        N, C, 1,
        BLOCK_N=bn, BLOCK_K=bk, D=D,
        MASK_HI=-(1 << idx_bits), MASK_LO=(1 << idx_bits) - 1, REDUCE_EACH=red,
        EPI_ROWS=16,
        num_warps=nw, num_stages=ns,
    )


def _tune(data, cent, s, cnt, m):
    C = cent.shape[1]
    s2 = torch.zeros_like(s)
    cnt2 = torch.zeros_like(cnt)
    m2 = torch.empty_like(m)

    def make(cfg):
        def f():
            s2.zero_()
            cnt2.zero_()
            _run(cfg, data, cent, s2, cnt2, m2)
        return f

    best = None
    for cfg in _CONFIGS:
        if C % cfg[1] != 0:
            continue
        try:
            _run(cfg, data, cent, s2, cnt2, m2)
            torch.cuda.synchronize()
            t = triton.testing.do_bench(make(cfg), warmup=10, rep=60)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    return best[1] if best is not None else (128, 32, 4, 2, False)


def launch(data, cent, s, cnt, m):
    data = data.contiguous()
    cent = cent.contiguous()
    B, N, D = data.shape
    C = cent.shape[1]
    key = (B, N, C, D)
    cfg = _CACHE.get(key)
    if cfg is None:
        cfg = _tune(data, cent, s, cnt, m)
        _CACHE[key] = cfg
    _run(cfg, data, cent, s, cnt, m)