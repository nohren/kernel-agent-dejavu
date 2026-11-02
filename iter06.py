import statistics

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------------------------------------
# bf16 path (fallback)
# ------------------------------------------------------------------------------------------------
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

    ii = tl.arange(0, EPI_ROWS * D)
    r_loc = ii // D
    cc = (ii % D) * stride_one
    k_base = K + b * N * D
    s_base = S + b * C * D
    for r0 in range(0, BLOCK_N, EPI_ROWS):
        rr = start_n + r0 + r_loc
        rmask = rr < N
        kv = tl.load(k_base + rr * D + cc, mask=rmask, other=0.0)
        lab = tl.load(m_base + rr, mask=rmask, other=0)
        tl.atomic_add(s_base + lab * D + cc, kv.to(tl.float32), mask=rmask, sem="relaxed")


# ------------------------------------------------------------------------------------------------
# int8 path: centroid quantization prepass + int8 GEMM (per-slot max) + exact fp32 rescoring
# ------------------------------------------------------------------------------------------------
@triton.jit
def _cmax_kernel(X, P, C, NP, R: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    offs_r = pid * R + tl.arange(0, R)
    offs_d = tl.arange(0, D)
    x = tl.load(X + b * C * D + offs_r[:, None] * D + offs_d[None, :],
                mask=offs_r[:, None] < C, other=0.0).to(tl.float32)
    mx = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    tl.store(P + b * NP + pid, mx)


@triton.jit
def _cquant_kernel(X, P, XQ, C, NP, R: tl.constexpr, D: tl.constexpr, NP2: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    offs_p = tl.arange(0, NP2)
    pm = tl.load(P + b * NP + offs_p, mask=offs_p < NP, other=0.0)
    g = tl.max(pm, axis=0)
    scale = 127.0 / tl.maximum(g, 1e-30)
    offs_r = pid * R + tl.arange(0, R)
    offs_d = tl.arange(0, D)
    rmask = offs_r[:, None] < C
    ptr_off = b * C * D + offs_r[:, None] * D + offs_d[None, :]
    x = tl.load(X + ptr_off, mask=rmask, other=0.0).to(tl.float32)
    v = x * scale
    q = tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127)
    tl.store(XQ + ptr_off, q.to(tl.int8), mask=rmask)


@triton.jit(do_not_specialize=["stride_one"])
def _assign_i8_kernel(
    K, XQ, X, S, CNT, M,
    N, C, stride_one,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
    IDX_SCALE: tl.constexpr, MASK_LO: tl.constexpr, EPI_ROWS: tl.constexpr,
    NCAND: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    NEG: tl.constexpr = -2147483647

    k_ptrs = K + b * N * D + offs_n[:, None] * D + offs_d[None, :]
    kf0 = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(kf0), axis=1)
    inv = 127.0 / tl.maximum(amax, 1e-30)
    v = kf0 * inv[:, None]
    kq = tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int32)
    kq = tl.minimum(tl.maximum(kq, -127), 127).to(tl.int8)

    xq_ptrs = XQ + b * C * D + offs_k[None, :] * D + offs_d[:, None]
    m1 = tl.full([BLOCK_N, BLOCK_K], NEG, tl.int32)
    for start in range(0, C, BLOCK_K):
        xq = tl.load(xq_ptrs)
        ip = tl.dot(kq, xq, out_dtype=tl.int32)
        p = ip * IDX_SCALE + (start + offs_k)[None, :]
        m1 = tl.maximum(m1, p)
        xq_ptrs += BLOCK_K * D

    # exact fp32 rescoring of the NCAND best slot maxima
    x_base = X + b * C * D
    kf = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    t = m1
    bi = tl.zeros([BLOCK_N], dtype=tl.int32)
    bs = tl.full([BLOCK_N], float("-inf"), tl.float32)
    for j in tl.static_range(NCAND):
        f = tl.max(t, axis=1)
        if j + 1 < NCAND:
            t = tl.where(t == f[:, None], NEG, t)
        ij = f & MASK_LO
        cj = tl.load(x_base + ij[:, None] * D + offs_d[None, :]).to(tl.float32)
        sj = tl.sum(kf * cj, axis=1)
        u = sj > bs
        bi = tl.where(u, ij, bi)
        bs = tl.where(u, sj, bs)
    max_idx = bi

    m_base = M + b * N
    tl.store(m_base + offs_n, max_idx, mask=n_mask)
    tl.atomic_add(CNT + b * C + max_idx, tl.zeros_like(max_idx) + 1, mask=n_mask, sem="relaxed")
    tl.debug_barrier()

    ii = tl.arange(0, EPI_ROWS * D)
    r_loc = ii // D
    cc = (ii % D) * stride_one
    k_base = K + b * N * D
    s_base = S + b * C * D
    for r0 in range(0, BLOCK_N, EPI_ROWS):
        rr = start_n + r0 + r_loc
        rmask = rr < N
        kv = tl.load(k_base + rr * D + cc, mask=rmask, other=0.0)
        lab = tl.load(m_base + rr, mask=rmask, other=0)
        tl.atomic_add(s_base + lab * D + cc, kv.to(tl.float32), mask=rmask, sem="relaxed")


# ------------------------------------------------------------------------------------------------
# configs:
#   ("bf16", BLOCK_N, BLOCK_K, warps, stages, REDUCE_EACH, EPI)
#   ("i8",   BLOCK_N, BLOCK_K, warps, stages, EPI, NCAND)
# ------------------------------------------------------------------------------------------------
_SHAPES_I8 = [
    (64, 128, 8, 3),
    (64, 128, 8, 4),
    (64, 128, 4, 4),
    (128, 128, 8, 3),
    (128, 128, 16, 3),
]
_CONFIGS = [("i8", bn, bk, nw, ns, 16, nc) for (bn, bk, nw, ns) in _SHAPES_I8 for nc in (2, 3, 4)]
_CONFIGS.append(("bf16", 128, 32, 4, 2, True, 16))

_CACHE = {}
_SCRATCH = {}
_CR = 64


def _scratch(cent):
    B, C, D = cent.shape
    key = (B, C, D, cent.device)
    sc = _SCRATCH.get(key)
    if sc is None:
        np_ = triton.cdiv(C, _CR)
        sc = (torch.empty(B, np_, device=cent.device, dtype=torch.float32),
              torch.empty(B, C, D, device=cent.device, dtype=torch.int8), np_)
        _SCRATCH[key] = sc
    return sc


def _run(cfg, data, cent, s, cnt, m):
    B, N, D = data.shape
    C = cent.shape[1]
    idx_bits = max(1, (C - 1).bit_length())
    if cfg[0] == "bf16":
        _, bn, bk, nw, ns, red, epi = cfg
        _assign_packed_kernel[(triton.cdiv(N, bn), B, 1)](
            data, cent, s, cnt, m, N, C, 1,
            BLOCK_N=bn, BLOCK_K=bk, D=D,
            MASK_HI=-(1 << idx_bits), MASK_LO=(1 << idx_bits) - 1, REDUCE_EACH=red,
            EPI_ROWS=min(epi, bn), num_warps=nw, num_stages=ns,
        )
    else:
        _, bn, bk, nw, ns, epi, nc = cfg
        part, cq, np_ = _scratch(cent)
        _cmax_kernel[(np_, B, 1)](cent, part, C, np_, R=_CR, D=D, num_warps=4)
        _cquant_kernel[(np_, B, 1)](cent, part, cq, C, np_, R=_CR, D=D,
                                    NP2=max(2, triton.next_power_of_2(np_)), num_warps=4)
        _assign_i8_kernel[(triton.cdiv(N, bn), B, 1)](
            data, cq, cent, s, cnt, m, N, C, 1,
            BLOCK_N=bn, BLOCK_K=bk, D=D,
            IDX_SCALE=(1 << idx_bits), MASK_LO=(1 << idx_bits) - 1,
            EPI_ROWS=min(epi, bn), NCAND=nc, num_warps=nw, num_stages=ns,
        )


def _tune(data, cent, s, cnt, m):
    C = cent.shape[1]
    idx_bits = max(1, (C - 1).bit_length())
    s2 = torch.zeros_like(s)
    cnt2 = torch.zeros_like(cnt)
    m2 = torch.empty_like(m)

    def make(cfg):
        def f():
            s2.zero_()
            cnt2.zero_()
            _run(cfg, data, cent, s2, cnt2, m2)
        return f

    valid = []
    for cfg in _CONFIGS:
        if C % cfg[2] != 0:
            continue
        if cfg[0] == "i8":
            if idx_bits > 11 or C // cfg[2] < 2 or cfg[2] < 64:
                continue
        try:
            _run(cfg, data, cent, s2, cnt2, m2)
            torch.cuda.synchronize()
        except Exception:
            continue
        valid.append(cfg)
    if not valid:
        return ("bf16", 128, 32, 4, 2, False, 16)
    if len(valid) == 1:
        return valid[0]

    fns = [make(c) for c in valid]
    try:
        triton.testing.do_bench(fns[0], warmup=200, rep=300)
    except Exception:
        pass
    times = [[] for _ in valid]
    rounds = 3
    for r in range(rounds):
        order = list(range(len(valid)))
        order = order[r % len(order):] + order[:r % len(order)]
        for i in order:
            try:
                times[i].append(triton.testing.do_bench(fns[i], warmup=5, rep=30))
            except Exception:
                times[i].append(float("inf"))
    meds = [statistics.median(t) for t in times]
    best_i = min(range(len(valid)), key=lambda i: meds[i])
    return valid[best_i]


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