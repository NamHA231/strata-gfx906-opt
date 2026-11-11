#!/usr/bin/env python3
"""Sweep the decode levers on one gfx906 card and print one row per arm.

The arms that need an engine restart (STRATA_EXP_MODE, STRATA_HC_REQ8) and the arms that do not
(--spec, prompt length) are separated on purpose: restarting a server costs a model load, and on a
32 GB MI50 the load is ~4 minutes, so the cheap arms are swept first against a running server.

What it measures, all from GET /metrics (serve/server.py:3738, serve/prometheus.py:88-94):
  tokens/window   output_tokens / (requests + a per-request window count we cannot see)  -> reported
                  as output_tokens and decode_ms, plus drafts/window from the draft counters.
  accept          drafts_accepted / drafts_offered  (what --spec actually bought)
  ms/token        decode_ms / output_tokens
  hit             expert-cache and prompt-cache reuse (reused / prompt_tokens)

Run with -I if the current directory holds untrusted Python:
  python3 -I tools/sweep_gfx906_decode.py --base http://127.0.0.1:8080

Nothing here changes the server.  For the env arms, start the server yourself with the env var set
and run this script again; the label is only what you pass with --label.
"""
import argparse
import json
import sys
import time
import urllib.request

PROMPTS = [
    ("short", "Giải thích ngắn gọn thuật toán Dijkstra.", 160),
    ("code", "Viết hàm Python parse RFC3339 timestamp về datetime có timezone, kèm test.", 400),
    ("long", "Viết một hàm Python đọc file CSV lớn theo từng chunk 8192 dòng, tính trung bình "
             "trượt 5 cửa sổ cho mọi cột số, xử lý giá trị thiếu, không dùng pandas, kèm docstring, "
             "kiểm thử biên và giải thích lựa chọn bộ nhớ. " * 12, 700),
]


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post_chat(base, prompt, max_tokens, model, timeout=600):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--label", default="arm", help="free-text name of this arm (EXP_MODE=7, spec=4, ...)")
    ap.add_argument("--reps", type=int, default=2, help="repetitions per prompt (first rep is usually cold)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json-out", default=None, help="append one JSON line per measurement to this file")
    args = ap.parse_args()

    models = get_json(args.base + "/v1/models")["data"]
    model = args.model or models[0]["id"]
    eng = get_json(args.base + "/metrics")["engine"]
    print(f"# {args.label}\n# model={model} spec={eng.get('spec')} kv={eng.get('kv')} "
          f"expert_slots={eng.get('expert_slots')} expert_cache_mib={eng.get('expert_cache_mib')} "
          f"vram_free_mib={eng.get('vram_free_mib')}")
    hdr = f"{'prompt':<7} {'rep':>3} {'ptok':>6} {'otok':>5} {'wall_s':>7} {'ms/tok':>7} " \
          f"{'acc%':>6} {'reuse%':>7}"
    print(hdr)

    for name, prompt, cap in PROMPTS:
        for rep in range(1, args.reps + 1):
            before = get_json(args.base + "/metrics")["totals"]
            wall, resp = post_chat(args.base, prompt, cap, model)
            after = get_json(args.base + "/metrics")["totals"]
            u = resp.get("usage", {}) or {}
            otok = u.get("completion_tokens") or 0
            ptok = u.get("prompt_tokens") or 0
            d_off = (after.get("drafts_offered", 0) or 0) - (before.get("drafts_offered", 0) or 0)
            d_acc = (after.get("drafts_accepted", 0) or 0) - (before.get("drafts_accepted", 0) or 0)
            d_ms = (after.get("decode_ms", 0.0) or 0.0) - (before.get("decode_ms", 0.0) or 0.0)
            pt = (after.get("prompt_tokens", 0) or 0) - (before.get("prompt_tokens", 0) or 0)
            reused = (after.get("reused", 0) or 0) - (before.get("reused", 0) or 0)
            ms_tok = d_ms / otok if otok else float("nan")
            acc = 100.0 * d_acc / d_off if d_off else float("nan")
            rej = 100.0 * reused / pt if pt else float("nan")
            print(f"{name:<7} {rep:>3} {ptok:>6} {otok:>5} {wall:>7.2f} {ms_tok:>7.2f} "
                  f"{acc:>6.1f} {rej:>7.1f}")
            if args.json_out:
                rec = {"label": args.label, "prompt": name, "rep": rep, "prompt_tokens": ptok,
                       "output_tokens": otok, "wall_s": round(wall, 3), "decode_ms": round(d_ms, 1),
                       "ms_per_token": round(ms_tok, 3), "drafts_offered": d_off,
                       "drafts_accepted": d_acc, "accept": round(acc, 3) if d_off else None,
                       "prompt_reuse": round(rej, 3) if pt else None,
                       "engine": {k: eng.get(k) for k in ("spec", "kv", "expert_slots",
                                                          "expert_cache_mib", "vram_free_mib")}}
                with open(args.json_out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
