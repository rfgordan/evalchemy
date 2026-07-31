"""Generate the static gsm-robust dataset (data/gsm_robust.jsonl).

The checked-in JSONL is the exact realization the task runs, so reviewers can
read every perturbed item and diffs stay legible. Regenerate (only when the
design changes) with:

    python -m eval.chat_benchmarks.zeroeval.data_prep.generate_gsm_robust

Each line is one instance:
    id                    "<zero-eval id>::<condition>"
    condition             one of robustness.CONDITIONS
    robust_seed           per-instance transform seed
    original_question     verbatim zero-eval gsm question
    question              perturbed text (== original for history conditions)
    answer                gold answer, verbatim from zero-eval
    changed               question != original_question
    number_words_affected spoken-number homophone swap occurred
    history_train_indices openai/gsm8k train indices whose Q/A pairs are
                          prepended as prior chat turns (history conditions
                          only, else null); content is materialized at load
                          so the file stays reviewable in size

Requires the zeroeval extra plus `cmudict` (homophone condition).
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset

from eval.chat_benchmarks.zeroeval.src.robustness import (
    HISTORY_CONDITIONS,
    assign_conditions,
    history_indices,
    perturb_question,
)

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "gsm_robust.jsonl"
DEFAULT_SEED = 20260731


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--perturbations-per-item", type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
    n_train = len(load_dataset("openai/gsm8k", "main", split="train"))
    assigned = assign_conditions(len(dataset), args.seed, args.perturbations_per_item)

    OUT_PATH.parent.mkdir(exist_ok=True)
    n_lines = 0
    with open(OUT_PATH, "w") as f:
        for ind, (item, conditions) in enumerate(zip(dataset, assigned)):
            for condition in conditions:
                if condition in HISTORY_CONDITIONS:
                    record = {
                        "id": f"{item['id']}::{condition}",
                        "condition": condition,
                        "robust_seed": args.seed,
                        "original_question": item["question"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "changed": True,
                        "number_words_affected": False,
                        "history_train_indices": history_indices(
                            n_train, ind, HISTORY_CONDITIONS[condition], args.seed
                        ),
                    }
                else:
                    meta = perturb_question(item["question"], condition, args.seed + ind)
                    record = {
                        "id": f"{item['id']}::{condition}",
                        "condition": condition,
                        "robust_seed": meta["robust_seed"],
                        "original_question": item["question"],
                        "question": meta["text"],
                        "answer": item["answer"],
                        "changed": meta["changed"],
                        "number_words_affected": meta["number_words_affected"],
                        "history_train_indices": None,
                    }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_lines += 1
    print(f"wrote {OUT_PATH}: {n_lines} instances")


if __name__ == "__main__":
    main()
