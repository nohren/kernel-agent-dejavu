"""Agent loop for kernel optimization: Claude proposes Triton kernel variants, a harness gates each one on correctness
against an fp32 reference, benchmarks it against the original kernel, optionally profiles it with Nsight Compute, and
feeds the results back. The fastest correct kernel is kept. No human in the loop between iterations.

Colab (A100 / L4 / H100 GPU runtime; T4 also works with fp16), with kmeans_bench.py and eval_candidate.py in the
same folder:
    !pip install -q -U anthropic
    import os; from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")   # add the key under Colab "Secrets"
    !python agent_loop.py --shape final-32K --iters 8 --ncu

With real keys from make_real_keys.py (gate on and time on one set, report on a held-out set after the run):
    !python agent_loop.py --shape final-32K --iters 15 --patience 6 --ncu \
        --real-gate real_keys_gate.pt --heldout real_keys_heldout.pt [--seed previous_run/iter06.py]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-5-5"
# USD per million tokens for claude-opus-5-5 (input, output, cache read, cache write ~1.25x input)
PRICE = {"in": 4.0, "out": 20.0, "cache_read": 0.20, "cache_write": 5.0}

BASELINE_SRC = '''# BASELINE: the original kernel from sparse_attention/index/cluster/kmeans.py, via kmeans_bench.py
import kmeans_bench as kb


def launch(data, cent, s, cnt, m):
    kb.launch("orig", data, cent, s, cnt, m, **kb.DEFAULT_CFG)
'''

TUNED_SRC = '''# TUNED BASELINE: the best launch config from the brute-force search, on the kernel variant the search used
import kmeans_bench as kb


def launch(data, cent, s, cnt, m):
    kb.launch("dyn", data, cent, s, cnt, m, **{cfg})
'''

SYSTEM_TEMPLATE = """You are a GPU kernel engineer optimizing one Triton kernel. Each turn you receive measurements \
from an automated harness and return one new candidate implementation. There is no human in the loop.

## Hardware
{hardware}

## Operation (k-means assignment step with fused accumulation)
Inputs: data [B, N, D] {dtype} contiguous (keys), cent [B, C, D] {dtype} contiguous (unit-norm centroids), D = 128.
Outputs (zeroed / allocated by the harness before every call):
- m [B, N] int32: m[b, n] = argmax_c dot(data[b, n], cent[b, c])
- s [B, C, D] fp32 (zeroed): s[b, c] += data[b, n] for every n assigned to c
- cnt [B, C] int32 (zeroed): cnt[b, c] += 1 for every n assigned to c
Target shape: {shape} = (B, N, C) = {bnc}. C is always a multiple of 128; N may be any positive integer.

## Contract
Return a complete, self-contained Python module that imports only torch, triton and triton.language and defines
    def launch(data, cent, s, cnt, m): ...
which fills s, cnt and m using Triton kernels. All compute must happen in Triton kernels (no torch matmul, \
argmax or scatter ops; allocating scratch buffers with torch is fine). You may change anything else: tiling, \
num_warps, num_stages, loop structure, number of kernels, persistent scheduling, how atomics are issued, etc.

## Correctness gate (a candidate that fails is discarded)
Checked against exact fp32 math on synthetic inputs, (3, 1000, 128), (2, 4173, 512) and the target shape (randn keys, \
unit-norm randn centroids){real_gate}:
- labels in range, and for every checked key the chosen centroid's score is within {rel_tol:.3g} * |key| of the best \
score (a cosine-similarity gap, since centroids are unit norm; 16x finer than bf16's relative precision). Near-tie \
flips pass; approximate scoring passes only if its error on these keys stays below that bound.
- s matches the sum of data rows for the kernel's own labels (relative error < 1e-5)
- cnt matches exactly
- after timing, one more call on the timing inputs is checked the same way, so every call must do the full work

## Objective
Maximize speedup on the target shape{timed_on}: the median over interleaved rounds, in one process, of (original \
kernel time / candidate kernel time). Each timed call zeroes s and cnt and then launches (triton.testing.do_bench); \
the zeroing alone is timed in the same rounds and subtracted, so ms, speedup and tflops are kernel-only. Absolute \
times drift between processes with GPU clocks, so compare candidates by speedup, and treat differences inside the \
reported speedup_range as noise.

## Reference implementation and harness
The harness (kmeans_bench.py) is below. `assign_orig` with DEFAULT_CFG is the baseline you must beat.

```python
{harness}
```

## Response format
1. At most 6 sentences: the bottleneck you infer from the latest measurements and the single main change you \
are making to address it.
2. Exactly one ```python fenced block containing the complete candidate module.
"""


REAL_GATE_NOTE = """, and on real post-RoPE keys from Qwen2.5-1.5B at exactly the target shape: 8 rows, each one \
(layer, head) pair with 32768 keys and 2048 unit-length centroids, every key checked. The keys are prepared the way \
Double-P clusters them (first 4 and last 64 prompt tokens dropped, the head's mean key subtracted) and the centroids \
come from Double-P's segment k-means. Real keys have larger outlier channels than Gaussian ones, and neighbouring \
tokens often share a centroid. Per-row results come back as real_checks. A held-out set of real \
keys from other layers, heads and prompts is checked after the run and never shown to you, so do not special-case \
the gate data"""


def evaluate(path, shape, real=None, timeout=900):
    cmd = [sys.executable, str(HERE / "eval_candidate.py"), str(path), shape] + (["--real", real] if real else [])
    return run_json(cmd, timeout)


def heldout(path, data_path, timeout=900):
    return run_json([sys.executable, str(HERE / "eval_candidate.py"), str(path), "--heldout", data_path], timeout)


def run_json(cmd, timeout):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"timed out after {timeout}s (hang or very slow compile)"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"passed": False, "error": f"evaluator crashed (exit {proc.returncode}): {proc.stderr[-2000:]}"}
    return json.loads(lines[-1])


def profile(path, shape, real=None, timeout=900):
    cmd = ["ncu", "--nvtx", "--nvtx-include", "measure/", "--section", "SpeedOfLight", "--section", "Occupancy",
           "--section", "LaunchStats", "--section", "WarpStateStats",
           sys.executable, str(HERE / "eval_candidate.py"), str(path), shape, "--profile"]
    cmd += ["--real", real] if real else []
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"(profiler unavailable: {type(e).__name__})"
    text = proc.stdout
    start = text.find("==PROF==")
    text = text[start:] if start >= 0 else text
    if "ERR_NVGPUCTRPERM" in text or "No kernels were profiled" in text:
        return "(profiler could not collect counters on this machine)"
    return text[-7000:]


def extract_code(text):
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def summarize(result):
    keep = {k: result[k] for k in ["passed", "ms", "baseline_ms", "zero_ms", "speedup", "speedup_range",
                                   "speedup_incl_zeroing", "tflops", "pct_of_peak", "dtype", "timed_on", "error",
                                   "checks", "real_checks", "repeat_check"]
            if k in result}
    if "traceback" in result and not result.get("passed"):
        keep["traceback_tail"] = result["traceback"][-1500:]
    return json.dumps(keep, indent=1, default=str)


def call_claude(client, system, messages, effort):
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=32000,
        system=system,
        messages=messages,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": effort},
        cache_control={"type": "ephemeral"},  # caches the growing conversation prefix turn over turn
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on a fallback model if a safety classifier declines
    ) as stream:
        return stream.get_final_message()


def cost(usage):
    return (usage.input_tokens * PRICE["in"] + usage.output_tokens * PRICE["out"]
            + (usage.cache_read_input_tokens or 0) * PRICE["cache_read"]
            + (usage.cache_creation_input_tokens or 0) * PRICE["cache_write"]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="final-32K")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many iterations without improvement")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--ncu", action="store_true", help="feed Nsight Compute sections back to the model")
    ap.add_argument("--tuned", nargs="*", default=[], metavar="KEY=VAL",
                    help="best launch config a config search found for the original kernel (e.g. num_stages=4); "
                         "the agent is told about it and has to beat it")
    ap.add_argument("--real-gate", help="real keys at the target shape (make_real_keys.py): gated on and timed on")
    ap.add_argument("--heldout", help="held-out real keys, checked for every passing candidate after the run only")
    ap.add_argument("--seed", help="a kernel from an earlier run to start from; its code and results go to the agent")
    ap.add_argument("--out", default=str(HERE / "runs"), help="parent folder for run folders (e.g. on Google Drive)")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import kmeans_bench as kb

    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log = open(run_dir / "log.jsonl", "w")
    system = SYSTEM_TEMPLATE.format(shape=args.shape, bnc=kb.SHAPES[args.shape], hardware=kb.describe_gpu(),
                                    dtype=kb.DTYPE_NAME, harness=(HERE / "kmeans_bench.py").read_text(),
                                    rel_tol=kb.REL_TOL, real_gate=REAL_GATE_NOTE if args.real_gate else "",
                                    timed_on=", timed on the real gate keys" if args.real_gate else "")
    print(f"GPU: {kb.describe_gpu()}   input format: {kb.DTYPE_NAME}", flush=True)

    base_path = run_dir / "iter00_baseline.py"
    base_path.write_text(BASELINE_SRC)
    base = evaluate(base_path, args.shape, args.real_gate)
    if not base.get("passed"):
        sys.exit(f"baseline failed: {summarize(base)}")
    base_prof = profile(base_path, args.shape, args.real_gate) if args.ncu else "(profiling disabled)"
    lo, hi = base["speedup_range"]
    print(f"baseline: {base['ms']:.3f} ms ({base['tflops']:.2f} TFLOP/s); noise floor, original vs itself: "
          f"{base['speedup']:.3f}x (rounds {lo:.3f}-{hi:.3f})", flush=True)
    best = {"iter": 0, "speedup": 1.0, "path": base_path, "name": "the baseline"}
    passing = [(best["name"], base_path, 1.0)]  # every gate-passing kernel, for the held-out check at the end

    tuned, tuned_note = None, ""
    if args.tuned:
        cfg = {**kb.DEFAULT_CFG, **{k: int(v) for k, v in (kv.split("=") for kv in args.tuned)}}
        tuned_path = run_dir / "iter00_tuned_baseline.py"  # the _baseline.py suffix exempts it from the import check
        tuned_path.write_text(TUNED_SRC.format(cfg=cfg))
        tuned = evaluate(tuned_path, args.shape, args.real_gate)
        if not tuned.get("passed"):
            sys.exit(f"tuned baseline failed: {summarize(tuned)}")
        print(f"tuned baseline {cfg}: {tuned['ms']:.3f} ms, {tuned['speedup']:.3f}x vs original", flush=True)
        best = {"iter": 0, "speedup": tuned["speedup"], "path": tuned_path, "name": "the tuned baseline"}
        passing.append((best["name"], tuned_path, tuned["speedup"]))
        tuned_note = (f"A brute-force search over launch configs found {cfg} on the runtime-size variant (assign_dyn), "
                      f"measured at {tuned['speedup']:.3f}x vs the original. Config tuning alone reaches that, so a "
                      "candidate only counts if its speedup is higher.\n\n")

    seed_note = ""
    if args.seed:
        seed_src = Path(args.seed).read_text()
        seed_path = run_dir / "iter00_seed.py"
        seed_path.write_text(seed_src)
        seed = evaluate(seed_path, args.shape, args.real_gate)
        if seed.get("passed"):
            print(f"seed kernel {args.seed}: {seed['ms']:.3f} ms, {seed['speedup']:.3f}x vs original", flush=True)
            passing.append(("the seed kernel", seed_path, seed["speedup"]))
            if seed["speedup"] > best["speedup"]:
                best = {"iter": 0, "speedup": seed["speedup"], "path": seed_path, "name": "the seed kernel"}
        else:
            print(f"seed kernel {args.seed} FAILED the gate: {seed.get('error', '')[:120]}", flush=True)
        seed_note = ("Starting point: a kernel from an earlier run, measured under this gate:\n"
                     f"{summarize(seed)}\n\n```python\n{seed_src}```\n\n")

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": (
        f"Baseline measurements on {args.shape}:\n{summarize(base)}\n\nNsight Compute (baseline):\n{base_prof}\n\n"
        f"{tuned_note}{seed_note}Propose candidate 1.")}]
    total_cost, stale = 0.0, 0
    for it in range(1, args.iters + 1):
        try:
            resp = call_claude(client, system, messages, args.effort)
        except anthropic.RateLimitError as e:
            print(f"rate limited, stopping: {e.message}")
            break
        except anthropic.APIStatusError as e:
            print(f"API error {e.status_code}, stopping: {e.message}")
            break
        except anthropic.APIConnectionError:
            print("network error, stopping")
            break
        messages.append({"role": "assistant", "content": resp.content})  # full content, append-only history
        total_cost += cost(resp.usage)
        if resp.stop_reason == "refusal":
            print(f"iteration {it}: request declined, stopping")
            break

        text = "".join(b.text for b in resp.content if b.type == "text")
        reasoning = "".join(b.thinking for b in resp.content if b.type == "thinking")
        code = extract_code(text)
        cand_path = run_dir / f"iter{it:02d}.py"
        if code is None:
            result = {"passed": False, "error": "no ```python block found in the response"}
        else:
            cand_path.write_text(code)
            result = evaluate(cand_path, args.shape, args.real_gate)
        prof = profile(cand_path, args.shape, args.real_gate) if (args.ncu and result.get("passed")) else ""

        improved = result.get("passed") and result["speedup"] > best["speedup"]
        if improved:
            best = {"iter": it, "speedup": result["speedup"], "path": cand_path, "name": f"candidate {it}"}
            stale = 0
        else:
            stale += 1
        if result.get("passed"):
            passing.append((f"candidate {it}", cand_path, result["speedup"]))
            lo, hi = result["speedup_range"]
            status = f"{result['ms']:.3f} ms, {result['speedup']:.3f}x vs original (rounds {lo:.3f}-{hi:.3f})"
            if tuned:
                status += f", {result['speedup'] / tuned['speedup']:.3f}x vs tuned"
        else:
            status = f"FAILED: {result.get('error', '')[:120]}"
        print(f"iter {it:2d}: {status}{'  <- best' if improved else ''}   (API cost so far ${total_cost:.2f})",
              flush=True)
        log.write(json.dumps({"iter": it, "result": result, "hypothesis": text.split("```")[0].strip(),
                              "reasoning_summary": reasoning, "usage": resp.usage.to_dict()}, default=str) + "\n")
        log.flush()
        if stale >= args.patience:
            print(f"no improvement for {args.patience} iterations, stopping")
            break

        messages.append({"role": "user", "content": (
            f"Candidate {it} results:\n{summarize(result)}\n\n"
            + (f"Nsight Compute (candidate {it}):\n{prof}\n\n" if prof else "")
            + f"Best so far: {best['name']} at {best['speedup']:.3f}x vs the original"
            + (f" (tuned baseline {tuned['speedup']:.3f}x)" if tuned else "") + ".\n"
            f"Propose candidate {it + 1}.")})

    (run_dir / "best_kernel.py").write_text(Path(best["path"]).read_text())
    print(f"\nbest: {best['name']} at {best['speedup']:.3f}x vs the original"
          + (f", {best['speedup'] / tuned['speedup']:.3f}x vs the tuned baseline" if tuned else "")
          + f". Files in {run_dir}. API cost ~${total_cost:.2f}")

    if args.heldout:
        # the test set: never shown to the agent, so it shows whether passing the gate generalizes
        print(f"\nheld-out real keys ({args.heldout}), every gate-passing kernel:")
        with open(run_dir / "heldout.jsonl", "w") as f:
            for name, path, speedup in passing:
                r = heldout(path, args.heldout)
                rows = r.get("heldout", [])
                worst = max((row.get("max_rel_regret", float("inf")) for row in rows), default=float("inf"))
                failed = [f"L{row['layer']}h{row['head']}" for row in rows
                          if not (row.get("counts_exact") and row.get("max_rel_regret", 1) <= kb.REL_TOL
                                  and row.get("sum_rel_err", 1) < 1e-5)]
                verdict = "pass" if r.get("passed") else f"FAIL {' '.join(failed) or r.get('error', '')[:80]}"
                mark = "  <- best" if Path(path) == Path(best["path"]) else ""
                print(f"  {name:18s} {speedup:6.3f}x  worst cos gap {worst:.1e}  {verdict}{mark}", flush=True)
                f.write(json.dumps({"name": name, "path": str(path), "speedup": speedup, **r}, default=str) + "\n")


if __name__ == "__main__":
    main()
