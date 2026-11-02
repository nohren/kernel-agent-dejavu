"""Evaluate one candidate implementation of the k-means assign kernel and print one JSON line.

A candidate is a Python module defining `launch(data, cent, s, cnt, m)` that fills the outputs using Triton kernels.
Runs in its own process so a crashing or hanging kernel cannot take down the agent loop.

    python eval_candidate.py CANDIDATE.py SHAPE                  # correctness gate + benchmark vs the original kernel
    python eval_candidate.py CANDIDATE.py SHAPE --real GATE.pt   # also gate on real keys at SHAPE, and time on them
    python eval_candidate.py CANDIDATE.py SHAPE --profile        # one warmup + one launch in an NVTX range, for ncu
    python eval_candidate.py CANDIDATE.py --heldout HELDOUT.pt   # correctness on held-out real keys only, no timing
"""

import argparse
import importlib.util
import json
import re
import statistics
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kmeans_bench as kb  # noqa: E402

# (batch rows, tokens, centroids): ragged token counts exercise the masking paths
CHECK_SHAPES = [(3, 1000, 128), (2, 4173, 512)]
ROUNDS = 7  # interleaved candidate/original timing rounds
# torch-side compute that would bypass Triton (tl.argmax and friends are allowed)
FORBIDDEN = [r"torch\.matmul", r"torch\.bmm", r"torch\.einsum", r"torch\.mm\(", r"(?<!tl)\.argmax\(", r"scatter_add",
             r"torch_assign", r"kmeans_bench", r"F\.linear", r"(?<![\w.])@(?![\w.])", r"tensordot", r"addb?mm",
             r"torch\.inner", r"torch\.dot", r"torch\.mv\("]


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.launch


def load_real(path):
    """Real keys file from make_real_keys.py: keys [B, N, D], unit-norm centroids [B, C, D], one (layer, head) per row."""
    d = torch.load(path, map_location="cpu")
    return d["keys"].to(kb.DEV, kb.DTYPE), d["cent"].to(kb.DEV, kb.DTYPE), [tuple(r) for r in d["rows"]]


def outputs_ok(data, cent, s, cnt, m, c, max_tokens=4096):
    b, n, _ = data.shape
    if int(m.min()) < 0 or int(m.max()) >= c:
        return {"shape": [b, n, c], "error": f"labels out of range [{int(m.min())}, {int(m.max())}]"}, False
    v = kb.verify(data, cent, s, cnt, m, max_tokens=max_tokens)
    return {"shape": [b, n, c], **v}, kb.passes(v)


def check(launch, b, n, c, seed):
    data, cent = kb.make_inputs(b, n, c, seed=seed)
    s, cnt, m = kb.alloc_outputs(data, cent)
    launch(data, cent, s, cnt, m)
    torch.cuda.synchronize()
    return outputs_ok(data, cent, s, cnt, m, c)


def rows_ok(keys, cents, s, cnt, m, rows):
    """Per (layer, head) row, every token checked."""
    results, ok = [], True
    for i, (layer, head) in enumerate(rows):
        sl = slice(i, i + 1)
        res, passed = outputs_ok(keys[sl], cents[sl], s[sl], cnt[sl], m[sl], cents.shape[1], max_tokens=1 << 30)
        res.pop("shape")
        results.append({"layer": layer, "head": head, **res})
        ok &= passed
    return results, ok


def real_rows(launch, keys, cents, rows):
    s, cnt, m = kb.alloc_outputs(keys, cents)
    launch(keys, cents, s, cnt, m)
    torch.cuda.synchronize()
    return rows_ok(keys, cents, s, cnt, m, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("shape", nargs="?")
    ap.add_argument("--real", help="real keys at SHAPE: added to the gate and used for timing and profiling")
    ap.add_argument("--heldout", help="only check correctness on these real keys")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    out = {"candidate": args.candidate, "shape": args.shape, "passed": False}
    try:
        src = Path(args.candidate).read_text()
        if not Path(args.candidate).name.endswith("_baseline.py"):  # only harness-written baselines may call kmeans_bench
            bad = [f for f in FORBIDDEN if re.search(f, src)]
            if bad:
                out["error"] = f"forbidden constructs (compute must be in Triton kernels): {bad}"
                print(json.dumps(out))
                return
        launch = load(args.candidate)

        if args.heldout:
            res, ok = real_rows(launch, *load_real(args.heldout))
            out.update(passed=ok, heldout=res)
            print(json.dumps(out))
            return

        b, n, c = kb.SHAPES[args.shape]
        rows = None
        if args.real:
            data, cent, rows = load_real(args.real)
            assert (data.shape[0], data.shape[1], cent.shape[1]) == (b, n, c), "real keys must be at the target shape"
        else:
            data, cent = kb.make_inputs(b, n, c)
        s, cnt, m = kb.alloc_outputs(data, cent)

        if args.profile:
            launch(data, cent, s, cnt, m)  # compile + warmup
            torch.cuda.synchronize()
            s.zero_()
            cnt.zero_()
            torch.cuda.nvtx.range_push("measure")
            launch(data, cent, s, cnt, m)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return

        checks, ok = [], True
        for i, (cb, cn, cc) in enumerate(CHECK_SHAPES + [(b, n, c)]):
            res, passed = check(launch, cb, cn, cc, seed=100 + i)
            checks.append(res)
            ok &= passed
        out["checks"] = checks
        if rows:
            out["real_checks"], real_ok = real_rows(launch, data, cent, rows)
            ok &= real_ok
        if not ok:
            out["error"] = "correctness gate failed (see checks" + (" and real_checks)" if rows else ")")
            print(json.dumps(out))
            return

        def cand():
            s.zero_()
            cnt.zero_()
            launch(data, cent, s, cnt, m)

        base = kb.timed_launch("orig", data, cent, s, cnt, m, **kb.DEFAULT_CFG)

        # GPU clocks drift between processes by more than most candidate differences, so time the candidate and the
        # original in rotating order in this process and score the median per-round ratio of kernel times
        t, zeros = kb.interleaved({"cand": cand, "base": base}, kb.zeroer(s, cnt), rounds=ROUNDS)

        # the timed calls are never checked, so check one more call that looks exactly like them: a candidate that
        # skips work on repeated calls would otherwise pass the gate and time near zero
        m.fill_(-1)
        cand()
        torch.cuda.synchronize()
        res, ok = rows_ok(data, cent, s, cnt, m, rows) if rows else outputs_ok(data, cent, s, cnt, m, c)
        if not ok:
            out.update(error="outputs wrong on a repeated call after timing (every call must do the full work)",
                       repeat_check=res)
            print(json.dumps(out))
            return

        ratios = sorted(bt / ct for bt, ct in zip(t["base"], t["cand"]))
        ms, base_ms = statistics.median(t["cand"]), statistics.median(t["base"])
        zero_ms = statistics.median(zeros)
        flops, _, _ = kb.flops_and_bytes(b, n, c, data.element_size())
        peak = kb.peak_tflops()
        tflops = flops / ms / 1e9
        out.update(passed=True, ms=ms, baseline_ms=base_ms, zero_ms=zero_ms, speedup=statistics.median(ratios),
                   speedup_range=[ratios[0], ratios[-1]],
                   speedup_incl_zeroing=(base_ms + zero_ms) / (ms + zero_ms), tflops=tflops,
                   pct_of_peak=(tflops / peak) if peak else None, dtype=kb.DTYPE_NAME,
                   timed_on="real keys" if rows else "synthetic")
    except Exception as e:  # compile errors, launch errors, illegal memory access
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
        out["traceback"] = traceback.format_exc()[-3000:]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
