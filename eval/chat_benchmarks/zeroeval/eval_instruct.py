from argparse import Namespace
from dataclasses import dataclass
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.chat_benchmarks.zeroeval.src.task_configs import prompt_generation
from eval.task import BaseBenchmark

from .src.unified_utils import mapping_task_names, save_outputs
from .src.evaluation.zebra_grid_eval import eval_model as zebra_grid_eval_model, load_private_solutions
from .src.evaluation.crux_eval import eval_model as crux_eval_model
from .src.evaluation.math_eval import eval_model as math_eval_model


@dataclass
class ZeroEvalConfig:
    # Dataset configuration
    start_index: int = 0
    end_index: int = -1


class ZeroEvalBenchmark(BaseBenchmark):
    """
    ZeroEval benchmark for a number of tasks and benchmarks.
    """

    def __init__(
        self,
        tasks: List[str] = ["zebra-grid", "numersense-v2", "crux", "math-l5"],
        config: Optional[ZeroEvalConfig] = None,
        max_tokens: Optional[int] = 4096,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger, system_instruction=system_instruction)
        self.tasks = tasks
        self.config = config or ZeroEvalConfig()
        self.max_new_tokens = int(max_tokens) if max_tokens is not None else 4096
        self.debug = debug

    def load_dataset(self, data_name: str) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load dataset for ZeroEval tasks.
        """
        if data_name == "gsm-robust":
            return self._load_gsm_robust()
        if data_name == "gsm-clean":
            return self._load_gsm_clean()
        try:
            chat_history = []
            id_strs = []
            extracted_chats = []
            metadata = {}
            dataset, id_name = mapping_task_names(data_name)

            if self.debug:
                dataset = dataset.select(range(min(10, len(dataset))))
                self.logger.info(f"Debug mode: using {len(dataset)} examples")

            # Process each item
            prompt_generation_args = Namespace(run_name="")
            for ind, item in enumerate(dataset):
                id_strs.append(item.get(id_name, f"{data_name}#{ind}"))
                prompt = prompt_generation(data_name, item, prompt_generation_args)
                chat_history.append([prompt])
                extracted_chats.append([{"content": prompt, "role": "user"}])
                for key in item:
                    if key not in metadata:
                        metadata[key] = []
                    metadata[key].append(item[key])

            self.logger.info(f"Finished processing {data_name} dataset.")

            # Apply index limits
            if self.config.end_index < 0:
                self.config.end_index = len(id_strs)

            slice_range = slice(self.config.start_index, self.config.end_index)
            return (
                id_strs[slice_range],
                chat_history[slice_range],
                extracted_chats[slice_range],
                {k: v[slice_range] for k, v in metadata.items()},
            )

        except Exception as e:
            self.logger.error(f"Error loading dataset for task {data_name}: {e}")
            raise e

    @staticmethod
    def _gsm_robust_breakdown(parsed_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Per-condition accuracy over all items and over actually-changed
        items (no-ops otherwise dilute the delta), plus the recoverable
        spoken-number slice for the homophone condition."""
        by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for item in parsed_results:
            by_condition.setdefault(item["condition"], []).append(item)

        def acc(items: List[Dict[str, Any]]) -> float:
            return 100.0 * sum(1 for i in items if i["matched"] is True) / len(items)

        breakdown = {}
        for condition, items in sorted(by_condition.items()):
            breakdown[f"gsm-robust:{condition}"] = acc(items)
            no_answer = sum(1 for i in items if i["matched"] == "No answer extracted")
            breakdown[f"gsm-robust:{condition}_no_answer"] = 100.0 * no_answer / len(items)
            changed = [i for i in items if i.get("changed")]
            if changed and len(changed) != len(items):
                breakdown[f"gsm-robust:{condition}_changed_only"] = acc(changed)
            numword = [i for i in items if i.get("number_words_affected")]
            if numword:
                breakdown[f"gsm-robust:{condition}_number_words"] = acc(numword)
        return breakdown

    def _load_gsm_clean(self) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """Static unperturbed companion of gsm-robust (data/gsm_clean.jsonl)."""
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm_clean.jsonl")
        with open(data_file) as f:
            records = [json.loads(line) for line in f]
        if self.debug:
            records = records[:10]
            self.logger.info(f"Debug mode: using {len(records)} instances")

        prompt_generation_args = Namespace(run_name="")
        chat_history, id_strs, extracted_chats, metadata = [], [], [], {}
        for record in records:
            id_strs.append(record["id"])
            prompt = prompt_generation("gsm", record, prompt_generation_args)
            extracted_chats.append([{"content": prompt, "role": "user"}])
            chat_history.append([prompt])
            for key, value in record.items():
                metadata.setdefault(key, []).append(value)

        if self.config.end_index < 0:
            self.config.end_index = len(id_strs)
        slice_range = slice(self.config.start_index, self.config.end_index)
        return (
            id_strs[slice_range],
            chat_history[slice_range],
            extracted_chats[slice_range],
            {k: v[slice_range] for k, v in metadata.items()},
        )

    def _load_gsm_robust(self) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """gsm with seeded meaning-preserving perturbations (marin-community/marin#7776).

        Instances come verbatim from the checked-in static realization
        (data/gsm_robust.jsonl, produced by data_prep/generate_gsm_robust.py):
        noise conditions carry a perturbed question; history conditions keep
        the question verbatim and prepend the embedded unrelated train-split
        exchanges as prior chat turns. The static gsm-clean task is the
        paired clean reference; no network access is needed.
        """
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm_robust.jsonl")
        with open(data_file) as f:
            records = [json.loads(line) for line in f]
        if self.debug:
            records = records[:10]
            self.logger.info(f"Debug mode: using {len(records)} instances")

        prompt_generation_args = Namespace(run_name="")
        chat_history, id_strs, extracted_chats, metadata = [], [], [], {}
        for record in records:
            id_strs.append(record["id"])
            prompt = prompt_generation("gsm", record, prompt_generation_args)
            history = []
            for exchange in record.get("history_exchanges") or []:
                history.append({"role": "user", "content": exchange["question"]})
                history.append({"role": "assistant", "content": exchange["answer"]})
            extracted_chats.append(history + [{"content": prompt, "role": "user"}])
            chat_history.append([prompt])
            for key, value in record.items():
                if key == "history_exchanges":
                    continue
                if key == "history_train_indices":
                    key, value = "history_messages", 2 * len(value or [])
                metadata.setdefault(key, []).append(value)

        self.logger.info(f"Finished loading gsm-robust: {len(id_strs)} instances.")
        if self.config.end_index < 0:
            self.config.end_index = len(id_strs)
        slice_range = slice(self.config.start_index, self.config.end_index)
        return (
            id_strs[slice_range],
            chat_history[slice_range],
            extracted_chats[slice_range],
            {k: v[slice_range] for k, v in metadata.items()},
        )

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for ZeroEval tasks.

        Args:
            model (LM): Language model instance

        Returns:
            Dict[str, Any]: Dictionary containing file paths and temporary directory
        """
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        results = {}

        for task in self.tasks:
            self.logger.info(f"Generating responses for task: {task}")

            # Load data
            try:
                id_strs, chat_history, extracted_chats, metadata = self.load_dataset(task)
            except Exception as e:
                self.logger.error(f"Error loading data for task {task}: {e}")
                continue

            self.logger.info(f"Finished loading data for task {task}.")

            # Apply template
            model_inputs = [model.apply_chat_template(chat) for chat in extracted_chats]

            output_path = os.path.join(temp_dir, f"{task}.json")
            results[task] = output_path

            # Generate responses
            self.logger.info("Generating responses for Zero Eval...")
            all_instances = [
                Instance(
                    "generate_until",
                    None,
                    (
                        inputs,
                        {
                            "temperature": 0.0,
                            "max_new_tokens": self.max_new_tokens,
                            "do_sample": False,
                        },
                    ),
                    idx,
                )
                for idx, inputs in enumerate(model_inputs)
            ]

            outputs = self.compute(model, all_instances)

            if model.rank != 0:
                continue

            outputs = [[output] for output in outputs]

            # Save outputs
            save_args = Namespace(
                data_name=task,
                model_name="",
                engine="",
                repetition_penalty=0.0,
                temperature=0.0,
                top_p=0.0,
                max_tokens=self.max_new_tokens,
            )
            save_outputs(save_args, id_strs, outputs, chat_history, metadata, model_inputs, output_path)

        results["temp_dir_obj"] = temp_dir_obj
        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate responses for ZeroEval tasks.
        """
        temp_dir_obj = results["temp_dir_obj"]
        temp_dir = temp_dir_obj.name
        del results["temp_dir_obj"]

        eval_results = {}

        for task, filepath in results.items():
            try:
                # Special handling for zebra-grid task
                if task == "zebra-grid":
                    load_private_solutions()
                    result, _ = zebra_grid_eval_model("%", filepath)
                    eval_results[task] = float(result["Puzzle Acc"])
                    eval_results[f"{task}_cell_acc"] = float(result["Cell Acc"])
                else:
                    # Handle other tasks (numersense-v2, crux, math-l5)
                    math_tasks = ["numersense-v2", "math-l5", "gsm", "gsm-clean", "gsm-robust"]
                    eval_func = math_eval_model if task in math_tasks else crux_eval_model
                    result, parsed_results = eval_func("%", filepath)
                    eval_results[task] = float(result["Acc"])
                    if task == "gsm-robust":
                        eval_results.update(self._gsm_robust_breakdown(parsed_results))

                # Common metrics for all tasks
                eval_results[f"{task}_no_answer"] = float(result["No answer"])
                eval_results[f"{task}_reason_lens"] = float(result["Reason Lens"])

            except Exception as e:
                self.logger.error(f"Error evaluating responses for task {task}: {e}")
                raise e

        temp_dir_obj.cleanup()
        return eval_results
