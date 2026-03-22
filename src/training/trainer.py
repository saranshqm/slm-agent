"""
Phi-3.5 agent fine-tuning (QLoRA) with correct labels, trainable new-token
embeddings, and step-by-step diagnostics.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TOOL_SPECIAL_TOKENS = [
    "<tool_use>",
    "</tool_use>",
    "<tool_name>",
    "</tool_name>",
    "<parameters>",
    "</parameters>",
]


def _format_example(instruction: str, input_text: str | None, output: str) -> tuple[str, str]:
    """Returns (prompt_prefix, full_text). Loss is applied only on tokens after prompt_prefix."""
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt, prompt + output


class AgentDataCollator:
    """Pad input_ids / attention_mask; pad labels with -100 (ignore index)."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.tokenizer.pad(
            pad_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            lab = f["labels"]
            L = min(len(lab), max_len)
            labels[i, :L] = torch.tensor(lab[:L], dtype=torch.long)
        batch["labels"] = labels
        return batch


class MetricsLoggingCallback(TrainerCallback):
    """Log loss, LR, epoch, and perplexity on each reported step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = [f"step={state.global_step}"]
        if "loss" in logs and logs["loss"] is not None:
            parts.append(f"train_loss={logs['loss']:.4f}")
            try:
                ppl = math.exp(min(20.0, logs["loss"]))
                parts.append(f"train_ppl={ppl:.2f}")
            except OverflowError:
                parts.append("train_ppl=inf")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.4f}")
        log.info(" | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        el = float(metrics["eval_loss"])
        try:
            ppl = math.exp(min(20.0, el))
        except OverflowError:
            ppl = float("inf")
        log.info(
            "eval @ step=%s | eval_loss=%.4f | eval_ppl=%.2f",
            state.global_step,
            el,
            ppl,
        )


class AgentTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def _load_config(self, path: str) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup_model_and_tokenizer(self) -> None:
        model_id = self.config["model"]["name"]
        log.info("Loading tokenizer (%s)...", model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="right",
        )

        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": TOOL_SPECIAL_TOKENS}
        )
        log.info("Added %d tool special tokens (expected %d).", added, len(TOOL_SPECIAL_TOKENS))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        qcfg = self.config.get("quantization", {})
        use_4bit = bool(qcfg.get("load_in_4bit", True))
        quant_config = None
        if use_4bit:
            compute_dtype = torch.float16
            bnb_dtype = qcfg.get("bnb_4bit_compute_dtype", "float16")
            if isinstance(bnb_dtype, str) and "bfloat16" in bnb_dtype.lower():
                compute_dtype = torch.bfloat16
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", False)),
            )
            log.info("Using 4-bit quantization (QLoRA).")

        log.info("Loading model...")
        torch_dtype = torch.bfloat16 if self.config["training"].get("bf16") else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch_dtype if not use_4bit else None,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lcfg = self.config["lora"]
        modules_to_save = lcfg.get("modules_to_save") or ["embed_tokens", "lm_head"]
        lora_config = LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["lora_alpha"],
            target_modules=lcfg["target_modules"],
            lora_dropout=lcfg["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        tcfg = self.config["training"]
        if tcfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _tokenize_batch(self, examples: dict[str, list]) -> dict[str, Any]:
        max_len = int(self.config["data"]["max_seq_length"])
        prompts: list[str] = []
        full_texts: list[str] = []

        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        ):
            inp = (input_text or "").strip()
            prompt, full = _format_example(instruction, inp if inp else None, output)
            prompts.append(prompt)
            full_texts.append(full)

        enc = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        all_input_ids: list[list[int]] = []
        all_attention: list[list[int]] = []
        all_labels: list[list[int]] = []

        for i, prompt in enumerate(prompts):
            cut = len(prompt)
            ids = enc["input_ids"][i]
            offsets = enc["offset_mapping"][i]
            labels: list[int] = []
            for tid, (start, _end) in zip(ids, offsets):
                if start >= cut:
                    labels.append(int(tid))
                else:
                    labels.append(-100)
            all_input_ids.append(ids)
            all_attention.append(enc["attention_mask"][i])
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "attention_mask": all_attention, "labels": all_labels}

    def prepare_datasets(self) -> None:
        data = self.config["data"]
        train_path = data["train_dataset"]
        eval_path = data["eval_dataset"]
        log.info("Loading JSON datasets...")

        train_ds = load_dataset("json", data_files=train_path)["train"]
        eval_ds = load_dataset("json", data_files=eval_path)["train"]

        self.train_dataset = train_ds.map(
            self._tokenize_batch,
            batched=True,
            batch_size=32,
            remove_columns=train_ds.column_names,
            desc="Tokenizing train",
        )
        self.eval_dataset = eval_ds.map(
            self._tokenize_batch,
            batched=True,
            batch_size=32,
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval",
        )
        log.info("Train rows: %d | Eval rows: %d", len(self.train_dataset), len(self.eval_dataset))

    def _response_label_stats(self, split: str, n: int = 512) -> None:
        ds = self.train_dataset if split == "train" else self.eval_dataset
        total_lab = 0
        masked = 0
        end = min(n, len(ds))
        for i in range(end):
            for x in ds[i]["labels"]:
                total_lab += 1
                if x == -100:
                    masked += 1
        log.info(
            "[%s] label positions (first %d rows): %.1f%% masked (instruction/padding)",
            split,
            end,
            100.0 * masked / max(1, total_lab),
        )

    def verify_setup(self) -> None:
        """Sanity checks: special token ids, one forward pass, decoded sample."""
        assert self.model is not None and self.tokenizer is not None
        for t in TOOL_SPECIAL_TOKENS:
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if tid is None or tid < 0:
                log.warning("Token %s missing from vocab.", t)
            else:
                log.info("Token %s -> id %s", t, tid)

        self._response_label_stats("train")
        self._response_label_stats("eval")

        collator = AgentDataCollator(self.tokenizer)
        batch = collator([self.train_dataset[0], self.train_dataset[1]])
        self.model.eval()
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
        loss = float(out.loss.item())
        log.info("Sanity forward on 2 examples: loss=%.4f (finite=%s)", loss, math.isfinite(loss))

        ids = self.train_dataset[0]["input_ids"]
        lab = self.train_dataset[0]["labels"]
        vis_ids = [tid for tid, lb in zip(ids, lab) if lb != -100][:80]
        snippet = self.tokenizer.decode(vis_ids, skip_special_tokens=False)
        log.info("First example response-prefix decode (truncated): %s", snippet[:400].replace("\n", "\\n"))

    def setup_training_args(self) -> TrainingArguments:
        t = self.config["training"]
        ev = self.config.get("evaluation", {})
        fp16 = bool(t.get("fp16", False))
        bf16 = bool(t.get("bf16", True))

        report = t.get("report_to", "none")
        if report in (None, "", "none"):
            report_list: list[str] = []
        elif isinstance(report, str):
            report_list = [report]
        else:
            report_list = list(report)

        return TrainingArguments(
            output_dir=t["output_dir"],
            num_train_epochs=float(t["num_train_epochs"]),
            max_steps=int(t.get("max_steps", -1)),
            per_device_train_batch_size=int(t["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
            learning_rate=float(t["learning_rate"]),
            weight_decay=float(t.get("weight_decay", 0.0)),
            max_grad_norm=float(t.get("max_grad_norm", 1.0)),
            lr_scheduler_type=str(t.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(t.get("warmup_ratio", 0.03)),
            logging_steps=int(t.get("logging_steps", 10)),
            save_steps=int(t.get("save_steps", 500)),
            eval_strategy=str(ev.get("evaluation_strategy", "steps")),
            eval_steps=int(ev.get("eval_steps", t.get("save_steps", 500))),
            save_strategy=str(ev.get("save_strategy", "steps")),
            load_best_model_at_end=bool(ev.get("load_best_model_at_end", True)),
            metric_for_best_model=str(ev.get("metric_for_best_model", "eval_loss")),
            greater_is_better=bool(ev.get("greater_is_better", False)),
            fp16=fp16 and not bf16,
            bf16=bf16,
            gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
            optim=str(t.get("optim", "adamw_torch")),
            group_by_length=bool(t.get("group_by_length", False)),
            remove_unused_columns=False,
            report_to=report_list,
            save_total_limit=int(t.get("save_total_limit", 3)),
            logging_first_step=True,
            prediction_loss_only=True,
        )

    def train(self) -> Trainer:
        self.setup_model_and_tokenizer()
        self.prepare_datasets()
        self.verify_setup()

        args = self.setup_training_args()
        collator = AgentDataCollator(
            self.tokenizer,
            pad_to_multiple_of=int(self.config["training"].get("pad_to_multiple_of", 8)),
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            callbacks=[MetricsLoggingCallback()],
        )

        log.info("Starting training...")
        trainer.train()

        out_dir = Path(self.config["training"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        log.info("Saved adapter + tokenizer to %s", out_dir.resolve())

        self._maybe_run_rl_phase()

        return trainer

    def _maybe_run_rl_phase(self) -> None:
        rl_cfg = self.config.get("rl") or {}
        if not rl_cfg.get("enabled", False):
            return
        import sys

        tdir = str(Path(__file__).resolve().parent)
        if tdir not in sys.path:
            sys.path.insert(0, tdir)
        import rl_phase

        rl_phase.run_rl_phase(self.config, self.model, self.tokenizer)


if __name__ == "__main__":
    cfg = Path(__file__).resolve().parents[2] / "config" / "training_config.yaml"
    AgentTrainer(str(cfg)).train()
