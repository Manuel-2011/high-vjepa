#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Blinded two-alternative forced-choice app for comparing world-model latents.

Reads the `pairs_2afc.json` that `evals/generate_flow_decoder_panels.py` writes
alongside panel 6, and asks a human, one trial at a time: given the current frame
and the true next frame, which of these two reconstructions is closer to the
truth?

Three properties make the answers worth collecting, and all three are enforced
here rather than left to the operator:

  * Blinding. Model names never reach the browser. Each trial carries an opaque
    trial id; the mapping from side to model lives only in this process and in
    the results file.
  * Side randomization. Which model is shown left is drawn per trial from a seed
    derived from the trial id, so it is stable across a reload (a rater who
    refreshes does not see the pair flip) but uncorrelated with model identity.
  * Trial order randomization, seeded once per session, so a rater's fatigue
    does not load onto one model.

Responses append to `responses.jsonl` in the panel directory. A tally with a
two-sided sign test is shown after every answer - a forced-choice preference is
only interpretable with a count attached, and the app makes it hard to eyeball
"model A looked better" off three trials.

This is the whole of the specified web UI. Nothing beyond it is in scope.

Run
---
    pip install gradio
    python -m app.flow_decoder.app_2afc \
        --pairs preliminary_experiments/evals/vitl/flow_decoder_panels/pairs_2afc.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple


def load_pairs(path: Path) -> Tuple[dict, List[dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pairs = payload.get("pairs", [])
    if not pairs:
        raise SystemExit(
            f"{path} holds no pairs. Panel 6 only produces pairs when two or more decoders are compared - "
            "re-run generate_flow_decoder_panels.py with at least two --decoder entries."
        )
    return payload, pairs


def trial_id(pair: dict) -> str:
    """Stable opaque id for a pair, so side assignment survives a reload."""
    key = f"{pair['index']}|{pair['left']['model']}|{pair['right']['model']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def build_trials(pairs: List[dict], root: Path, seed: int) -> List[dict]:
    """Randomize side assignment per trial and the order across trials."""
    trials = []
    for pair in pairs:
        tid = trial_id(pair)
        # Seeded from the trial id, not the session, so a refresh is stable.
        flip = random.Random(tid).random() < 0.5
        a, b = (pair["right"], pair["left"]) if flip else (pair["left"], pair["right"])
        trials.append(
            {
                "trial_id": tid,
                "index": pair["index"],
                "input": str(root / pair["input"]),
                "truth": str(root / pair["truth"]),
                "A": {"model": a["model"], "image": str(root / a["image"])},
                "B": {"model": b["model"], "image": str(root / b["image"])},
            }
        )
    random.Random(seed).shuffle(trials)
    return trials


def sign_test_p(wins: int, total: int) -> float:
    """Two-sided exact binomial p-value against p = 0.5.

    Written out rather than pulled from scipy so the reported number is auditable
    and the app has no hard dependency beyond gradio.
    """
    if total == 0:
        return 1.0
    from math import comb

    extreme = min(wins, total - wins)
    tail = sum(comb(total, k) for k in range(0, extreme + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def tally(responses_path: Path, models: List[str]) -> str:
    """Markdown tally of every head-to-head, with counts and a sign test."""
    if not responses_path.exists():
        return "_No responses yet._"
    records = []
    with open(responses_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return "_No responses yet._"

    head_to_head: Dict[Tuple[str, str], Counter] = {}
    ties = 0
    for record in records:
        if record.get("choice") == "tie":
            ties += 1
            continue
        winner, loser = record["chosen_model"], record["other_model"]
        key = tuple(sorted((winner, loser)))
        head_to_head.setdefault(key, Counter())[winner] += 1

    lines = [f"**{len(records)} response(s)** ({ties} tie(s))", "", "| pair | wins | p (sign test) |", "|---|---|---|"]
    for a, b in combinations(sorted(models), 2):
        counter = head_to_head.get(tuple(sorted((a, b))), Counter())
        wins_a, wins_b = counter.get(a, 0), counter.get(b, 0)
        total = wins_a + wins_b
        if not total:
            lines.append(f"| {a} vs {b} | no data | - |")
            continue
        p = sign_test_p(wins_a, total)
        lines.append(f"| {a} vs {b} | {wins_a} - {wins_b} | {p:.3f} |")
    lines += [
        "",
        "_A preference is only readable with its count. With a handful of trials the sign test will not "
        "separate anything, which is the honest answer rather than a missing one._",
    ]
    return "\n".join(lines)


def build_interface(pairs_path: Path, seed: int, responses_path: Path):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "the 2AFC app needs gradio: pip install gradio\n"
            "Everything else in the flow-decoder pipeline runs without it."
        ) from exc

    payload, pairs = load_pairs(pairs_path)
    root = pairs_path.parent
    trials = build_trials(pairs, root, seed)
    models = payload.get("models", [])
    sampler = payload.get("sampler", {})

    def render(position: int):
        if position >= len(trials):
            return None, None, None, None, "**Done - every trial answered.**", tally(responses_path, models)
        trial = trials[position]
        header = f"Trial {position + 1} / {len(trials)} - clip sample {trial['index']}"
        return (
            trial["input"],
            trial["truth"],
            trial["A"]["image"],
            trial["B"]["image"],
            header,
            tally(responses_path, models),
        )

    def record(position: int, choice: str):
        if position >= len(trials):
            return (position, *render(position))
        trial = trials[position]
        if choice == "tie":
            chosen, other = None, None
        else:
            chosen = trial[choice]["model"]
            other = trial["B" if choice == "A" else "A"]["model"]
        with open(responses_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trial_id": trial["trial_id"],
                        "index": trial["index"],
                        "choice": choice,
                        "chosen_model": chosen,
                        "other_model": other,
                        "left_model": trial["A"]["model"],
                        "right_model": trial["B"]["model"],
                        "sampler": sampler,
                    }
                )
                + "\n"
            )
        position += 1
        return (position, *render(position))

    with gr.Blocks(title="Flow-decoder 2AFC") as demo:
        gr.Markdown(
            "# Which reconstruction is closer to the truth?\n"
            "Both images below were produced by the **same decoder architecture, the same fixed training "
            "budget and the same noise seed**, differing only in which frozen world model supplied the "
            "latent. Model identities are hidden and the left/right assignment is randomized.\n\n"
            f"Sampler: `{sampler}`"
        )
        header = gr.Markdown()
        with gr.Row():
            input_image = gr.Image(label="x_t (current frame)", interactive=False, height=260)
            truth_image = gr.Image(label="x_{t+1} (ground truth)", interactive=False, height=260)
        with gr.Row():
            image_a = gr.Image(label="A", interactive=False, height=300)
            image_b = gr.Image(label="B", interactive=False, height=300)
        with gr.Row():
            choose_a = gr.Button("A is closer", variant="primary")
            choose_tie = gr.Button("Can't tell")
            choose_b = gr.Button("B is closer", variant="primary")
        results = gr.Markdown()
        position = gr.State(0)

        outputs = [position, input_image, truth_image, image_a, image_b, header, results]
        choose_a.click(lambda p: record(p, "A"), inputs=position, outputs=outputs)
        choose_b.click(lambda p: record(p, "B"), inputs=position, outputs=outputs)
        choose_tie.click(lambda p: record(p, "tie"), inputs=position, outputs=outputs)
        demo.load(lambda: (0, *render(0)), outputs=outputs)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blinded 2AFC app over flow-decoder reconstructions.")
    parser.add_argument(
        "--pairs",
        default="preliminary_experiments/evals/vitl/flow_decoder_panels/pairs_2afc.json",
        help="pairs_2afc.json written by evals/generate_flow_decoder_panels.py.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Trial-order seed for this session.")
    parser.add_argument(
        "--responses",
        default=None,
        help="Where to append responses. Defaults to responses.jsonl next to the pairs file.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Expose a public gradio link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"{pairs_path} not found; run evals/generate_flow_decoder_panels.py first.")
    responses_path = Path(args.responses) if args.responses else pairs_path.parent / "responses.jsonl"
    demo = build_interface(pairs_path, args.seed, responses_path)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
