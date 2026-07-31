"""gsm_robust benchmark: chat-mode GSM8K robustness to prompt noise and
irrelevant chat history (marin-community/marin#7776, wish-list
marin-community/marin#7090 item 16).

Runs two zeroeval tasks in one invocation:
  gsm-clean   -- the static unperturbed companion (clean reference)
  gsm-robust  -- the checked-in perturbed realization
                 (zeroeval/data/gsm_robust.jsonl): six seeded
                 meaning-preserving noise conditions plus prior-turn
                 irrelevant-history conditions, ~165 items each

Reported metrics: clean accuracy, per-condition accuracy, and changed-only /
spoken-number slices. See eval/chat_benchmarks/zeroeval/src/robustness.py for
the transform definitions and data_prep/generate_gsm_robust.py for dataset
regeneration.
"""

import logging
from typing import List, Optional

from eval.chat_benchmarks.zeroeval.eval_instruct import ZeroEvalBenchmark, ZeroEvalConfig


class GSMRobustBenchmark(ZeroEvalBenchmark):
    def __init__(
        self,
        tasks: List[str] = ["gsm-clean", "gsm-robust"],
        config: Optional[ZeroEvalConfig] = None,
        max_tokens: Optional[int] = 4096,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(
            tasks=tasks,
            config=config,
            max_tokens=max_tokens,
            debug=debug,
            logger=logger,
            system_instruction=system_instruction,
        )
