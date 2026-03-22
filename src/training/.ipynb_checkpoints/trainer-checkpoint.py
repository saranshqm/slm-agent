"""
Corrected Trainer for PHI-3.5 Agent Fine-tuning (LoRA)
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from datasets import load_dataset
from transformers import BitsAndBytesConfig

# =========================
# LOGGING FIX
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class AgentTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None

    def _load_config(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # =========================
    # MODEL + TOKENIZER
    # =========================
    def setup_model_and_tokenizer(self):
        log.info("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=True,
        )

        # 🔥 ADD SPECIAL TOKENS (CRITICAL)
        special_tokens = {
            "additional_special_tokens": [
                "<tool_use>", "</tool_use>",
                "<tool_name>", "</tool_name>",
                "<parameters>", "</parameters>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Tokenizer size: {len(self.tokenizer)}")

        # =========================
        # LOAD MODEL (FIXED)
        # =========================
        log.info("Loading model...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.config["quantization"]["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
        )

        # 🔥 CRITICAL: resize embeddings
        model.resize_token_embeddings(len(self.tokenizer))

        model = prepare_model_for_kbit_training(model)

        # =========================
        # LoRA
        # =========================
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(model, lora_config)

        log.info("Model ready.")

    # =========================
    # DATASET
    # =========================
    def prepare_datasets(self):
        log.info("Loading datasets...")

        train_dataset = load_dataset(
            "json",
            data_files=self.config["data"]["train_dataset"]
        )["train"]

        eval_dataset = load_dataset(
            "json",
            data_files=self.config["data"]["eval_dataset"]
        )["train"]

        def tokenize(example):
            text = f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""

            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",  # 🔥 FIX
                max_length=self.config["data"]["max_seq_length"],
            )

            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        self.train_dataset = train_dataset.map(tokenize)
        self.eval_dataset = eval_dataset.map(tokenize)

        log.info(f"Train size: {len(self.train_dataset)}")
        log.info(f"Eval size: {len(self.eval_dataset)}")

    # =========================
    # TRAINING ARGS
    # =========================
    def setup_training_args(self):
        return TrainingArguments(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            logging_steps=50,
            save_steps=500,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=True,
            remove_unused_columns=False,
            report_to="none",  # 🔥 remove wandb
        )

    # =========================
    # TRAIN
    # =========================
    def train(self):
        self.setup_model_and_tokenizer()
        self.prepare_datasets()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=self.setup_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        log.info("Starting training...")
        trainer.train()

        log.info("Saving model...")

        output_dir = self.config["training"]["output_dir"]

        # 🔥 CRITICAL FIX (ROOT CAUSE)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        log.info(f"Model saved to {output_dir}")

        return trainer


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    trainer = AgentTrainer("config/training_config.yaml")
    trainer.train()