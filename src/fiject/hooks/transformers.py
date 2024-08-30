"""
Fiject integration for HuggingFace transformers.

Example usage:
```
    trainer = Trainer(
        model=...,
        ...
        callbacks=[
            EvaluateBeforeTrainingCallback(),
            FijectCallback("loss",  evals_between_commits=10),
            FijectCallback("other", evals_between_commits=10, metric_names_with_formatting={"pr": "Pr", "re": "Re", "f1": "$F_1$", "acc": "Accuracy"])
        ],
        ...
    )
    trainer.train()
```
"""
from transformers.trainer_callback import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
from fiject import LineGraph, CacheMode
import logging as logger
from typing import List, Dict


class FijectCallback(TrainerCallback):
    """
    Callback object passed to a HuggingFace Trainer that will add an evaluation point to a Fiject graph, and will commit
    the result every N evaluations.

    You can either choose the evaluation metrics yourself, or let it default to the metric used to rank models.
    """

    def __init__(self, plot_name: str, metric_names_with_formatting: Dict[str,str]=None, evals_between_commits: int=-1):  # Metrics default to the eval loss.
        self.graph = LineGraph(plot_name, CacheMode.WRITE_ONLY, overwriting=True)
        self.evals_per_commit = evals_between_commits
        self.evals_so_far = 0
        self.dont_commit_again_if_you_are_still_at = -1

        self.metrics_tracked = None if not metric_names_with_formatting \
                          else {handle[handle.startswith("eval_")*len("eval_"):]: formatted for handle,formatted in metric_names_with_formatting.items()}

    def _automatic_metric_name(self, args: TrainingArguments):
        """
        Based on the early-stopping callback, which also accesses the evaluation loss
        https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_callback.py#L586
        """
        ranking_metric = args.metric_for_best_model
        ranking_metric = ranking_metric[ranking_metric.startswith("eval_")*len("eval_"):]
        self.metrics_tracked = {ranking_metric: ranking_metric.replace("_", "-")}

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        if not self.metrics_tracked:
            self._automatic_metric_name(args)

        for result_name, result_name_formatted in self.metrics_tracked.items():
            key = "eval_" + result_name
            if key not in metrics:
                logger.warning(f"Huh? You asked to track metric '{key}' but you're not computing it!")
                continue

            self.graph.add(result_name_formatted, state.global_step, metrics.get(key))

        self.evals_so_far += 1  # By incrementing first, you never commit after just one iteration.
        if self.evals_per_commit > 0 and self.evals_so_far % self.evals_per_commit == 0:
            self._commit(state.global_step)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):  # Every time a checkpoint is made, also save a graph.
        self._commit(state.global_step)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._commit(state.global_step)

    def _commit(self, current_time: int):
        if self.dont_commit_again_if_you_are_still_at == current_time:
            return

        self.dont_commit_again_if_you_are_still_at = current_time
        self.graph.commit(legend_position="upper right", x_label="Training batches", y_label="Validation set performance",
                          do_points=False, grid_linewidth=0.1)
