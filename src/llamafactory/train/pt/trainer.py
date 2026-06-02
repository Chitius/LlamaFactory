# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        loss = super().compute_loss(model, inputs, *args, **kwargs)
        self._store_mtp_metrics(model, num_items_in_batch=kwargs.get("num_items_in_batch"))
        return loss

    def _get_metric_model(self, model):
        if hasattr(self, "accelerator"):
            try:
                model = self.accelerator.unwrap_model(model)
            except Exception:
                pass

        while hasattr(model, "module"):
            model = model.module

        return model

    def _get_loss_log_scale(self, num_items_in_batch: Optional["torch.Tensor"] = None) -> float:
        if not self.model_accepts_loss_kwargs or num_items_in_batch is None:
            return 1.0

        scale = float(getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps))
        if self.args.average_tokens_across_devices:
            scale *= float(self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu)

        return scale

    def _store_mtp_metrics(self, model, num_items_in_batch: Optional["torch.Tensor"] = None) -> None:
        metric_model = self._get_metric_model(model)
        if getattr(metric_model, "last_main_loss", None) is None:
            return

        split = "train" if metric_model.training else "eval"
        loss_log_scale = self._get_loss_log_scale(num_items_in_batch)
        self._stored_metrics[split]["main_loss"].append(metric_model.last_main_loss.float().item() * loss_log_scale)
        self._stored_metrics[split]["mtp_loss"].append(metric_model.last_mtp_loss.float().item() * loss_log_scale)

        mtp_correct = getattr(metric_model, "last_mtp_correct", None)
        mtp_total = getattr(metric_model, "last_mtp_total", None)
        if mtp_correct is not None and mtp_total is not None:
            self._stored_metrics[split]["mtp_correct"].append(mtp_correct.float().item())
            self._stored_metrics[split]["mtp_total"].append(mtp_total.float().item())

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Add MTP metrics collected during compute_loss to Trainer logs."""
        split = "train" if "loss" in logs else "eval"
        prefix = "" if split == "train" else "eval_"
        stored_metrics = self._stored_metrics[split]

        metric_values = []
        metric_keys = []
        for key in ("main_loss", "mtp_loss"):
            values = stored_metrics.get(key, [])
            if values:
                metric_keys.append(f"{prefix}{key}")
                metric_values.append(torch.tensor(values, dtype=torch.float, device=self.accelerator.device).mean())

        if metric_values:
            reduced_values = self.accelerator.reduce(torch.stack(metric_values), "mean").tolist()
            for key, value in zip(metric_keys, reduced_values):
                logs[key] = value

        mtp_correct = stored_metrics.get("mtp_correct", [])
        mtp_total = stored_metrics.get("mtp_total", [])
        if mtp_correct and mtp_total:
            counts = torch.tensor(
                [sum(mtp_correct), sum(mtp_total)], dtype=torch.float, device=self.accelerator.device
            )
            mtp_correct_sum, mtp_total_sum = self.accelerator.reduce(counts, "sum").tolist()
            if mtp_total_sum > 0:
                logs[f"{prefix}mtp_acc"] = mtp_correct_sum / mtp_total_sum

        if split in self._stored_metrics:
            del self._stored_metrics[split]

        return super().log(logs, *args, **kwargs)
