"""Small-scale re-run of the paper's response generation on one local model.

Mirrors response_generation.py (same Japanese system prompt, same prompts,
same "complete the story within N characters" suffix, same sampling settings)
but runs a quantised model through a local Ollama server instead of loading
bf16 weights on a CUDA GPU.

Differences from the original pipeline, all deliberate:
  * Ollama + 4-bit quantised weights instead of HF transformers + bf16.
  * Sampling is seeded per repeat (seed + repeat_idx). Ollama returns identical
    text for an identical request+seed, so a single fixed seed would make every
    repeat a duplicate. The released script sets no seed at all.
  * top_k=50 is passed explicitly because that is the transformers default the
    original run inherited; Ollama's own default is 40.
  * Output is JSONL (one response per line) and the run is resumable.

Example (60 responses, ~30 min on an M-series Mac):
    .venv/bin/python replication/02_generate_ollama.py --model llama3.1:8b --repeat 2
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_ROLE_JA = (
    "あなたは、社会的な礼儀、尊重、人間関係の細かなニュアンスに深い理解を持つ熟練した物語作家です。"
    "与えられたプロンプトを基に物語を続け、登場人物の行動、感情、会話を通して、異なる文化的態度がどのように"
    "自然に人間関係に影響するかを示してください。ホフステードの文化次元理論に基づき、文化的視点の微妙な表現を"
    "描きつつも、文化差異を直接的に説明しないようにしてください。物語の続きが日本語、英語、または中国語の"
    "文化環境での一般的な態度や対人関係の細部を反映し、自然で流れるような展開になるように工夫してください。"
)


def limit_suffix(available_num: int) -> str:
    return f"ストーリーを{available_num}字以内で完成させる。"


def check_matches_original():
    src = (ROOT / "response_generation.py").read_text(encoding="utf-8")
    original = re.search(r"0: '(あなたは.*?)'", src).group(1)
    if original != SYSTEM_ROLE_JA:
        raise SystemExit("System prompt differs from response_generation.py")
    if "ストーリーを{available_num}字以内で完成させる。" not in src:
        raise SystemExit("Length suffix differs from response_generation.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model tag")
    p.add_argument("--repeat", type=int, default=2, help="generations per prompt (paper: 200)")
    p.add_argument("--available-num", type=int, default=400, help="character limit in the suffix (paper: 400)")
    p.add_argument("--dimensions", nargs="+", default=None, help="subset of PDI IDV UAI MAS LTO IND")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--output-dir", default=str(ROOT / "replication" / "generated"))
    return p.parse_args()


def generate(host, model, prompt, seed, max_new_tokens):
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_ROLE_JA}, {"role": "user", "content": prompt}],
        "options": {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 50,
            "repeat_penalty": 1.1,
            "num_predict": max_new_tokens,
            "seed": seed,
        },
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=1800)
    r.raise_for_status()
    return r.json()


def main():
    args = parse_args()
    check_matches_original()
    prompts = json.loads((ROOT / "data" / "cultural_prompts_ja.json").read_text(encoding="utf-8"))
    dims = args.dimensions or list(prompts)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace(":", "_").replace("/", "_")
    out_file = out_dir / f"{tag}_ja_{args.available_num}.jsonl"

    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done.add((rec["dimension"], rec["prompt_idx"], rec["repeat_idx"]))

    # Same loop order as the original: repeat outermost, then prompts.
    jobs = [(r, d, i) for d in dims for r in range(args.repeat) for i in range(len(prompts[d]))]
    todo = [j for j in jobs if (j[1], j[2], j[0]) not in done]
    print(f"model={args.model}  total={len(jobs)}  already done={len(jobs) - len(todo)}  -> {out_file}")

    start = time.time()
    with out_file.open("a", encoding="utf-8") as f:
        for n, (repeat_idx, dim, prompt_idx) in enumerate(todo, 1):
            prompt = prompts[dim][prompt_idx] + limit_suffix(args.available_num)
            seed = args.seed + repeat_idx
            res = generate(args.host, args.model, prompt, seed, args.max_new_tokens)
            text = res["message"]["content"]
            rec = {
                "model": args.model,
                "dimension": dim,
                "prompt_idx": prompt_idx,
                "repeat_idx": repeat_idx,
                "seed": seed,
                "available_num": args.available_num,
                "prompt": prompt,
                "response": text,
                "n_chars": len(text),
                "eval_tokens": res.get("eval_count"),
                "done_reason": res.get("done_reason"),
                "seconds": round(res.get("total_duration", 0) / 1e9, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.time() - start
            print(f"[{n}/{len(todo)}] {dim} p{prompt_idx} r{repeat_idx}  {len(text)} chars  "
                  f"eta {elapsed / n * (len(todo) - n) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
