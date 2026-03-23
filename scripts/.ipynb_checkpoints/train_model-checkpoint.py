# """
# Production-ready PHI-3.5 LoRA Trainer (OOM-safe, no wandb, fixed padding).
# """

# import torch
# import yaml
# import uuid
# import logging
# from typing import Any
# from pathlib import Path

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback,
#     BitsAndBytesConfig,
# )
# from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
# from datasets import load_dataset

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)


# class AgentTrainer:
#     def __init__(self, config_path: str):
#         self.config = self._load_config(config_path)
#         self.tokenizer = None
#         self.model = None
#         self.trainer = None

#     def _load_config(self, path: str) -> dict[str, Any]:
#         with open(path, "r") as f:
#             return yaml.safe_load(f)

#     # =========================
#     # MODEL + TOKENIZER
#     # =========================
#     def setup_model_and_tokenizer(self):
#         log.info("Loading tokenizer...")

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config["model"]["name"],
#             trust_remote_code=True,
#             padding_side="right",
#         )

#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         # Add tool tokens
#         self.tokenizer.add_special_tokens({
#             "additional_special_tokens": [
#                 "<tool_use>", "</tool_use>",
#                 "<tool_name>", "</tool_name>",
#                 "<parameters>", "</parameters>",
#             ]
#         })

#         # Quantization config (QLoRA)
#         quant_config = None
#         if self.config["quantization"]["load_in_4bit"]:
#             log.info("Using 4-bit QLoRA...")
#             quant_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4",
#             )

#         log.info("Loading model...")

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config["model"]["name"],
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             quantization_config=quant_config,
#         )

#         self.model.resize_token_embeddings(len(self.tokenizer))

#         if quant_config:
#             self.model = prepare_model_for_kbit_training(self.model)

#         # Enable gradient checkpointing (OOM saver)
#         self.model.gradient_checkpointing_enable()

#         # LoRA
#         lora_config = LoraConfig(
#             r=self.config["lora"]["r"],
#             lora_alpha=self.config["lora"]["lora_alpha"],
#             target_modules=self.config["lora"]["target_modules"],
#             lora_dropout=self.config["lora"]["lora_dropout"],
#             bias="none",
#             task_type=TaskType.CAUSAL_LM,
#         )

#         self.model = get_peft_model(self.model, lora_config)
#         self.model.print_trainable_parameters()

#     # =========================
#     # DATA
#     # =========================
#     def prepare_datasets(self):
#         log.info("Loading datasets...")

#         train_dataset = load_dataset("json", data_files=self.config["data"]["train_dataset"])["train"]
#         eval_dataset = load_dataset("json", data_files=self.config["data"]["eval_dataset"])["train"]

#         def tokenize(examples):
#             texts = []

#             for i in range(len(examples["instruction"])):
#                 system = examples.get("system", [""] * len(examples["instruction"]))[i]
#                 instruction = examples["instruction"][i]
#                 input_text = examples["input"][i] or ""
#                 output = examples["output"][i]

#                 if input_text:
#                     user_msg = f"{instruction}\n\nInput:\n{input_text}"
#                 else:
#                     user_msg = instruction
                
#                 messages = []
#                 if system:
#                     messages.append({"role": "system", "content": system})
#                 messages.append({"role": "user", "content": user_msg})
                
#                 prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#                 text = prompt + output + self.tokenizer.eos_token

#                 texts.append(text)

#             return self.tokenizer(
#                 texts,
#                 truncation=True,
#                 padding=False,
#                 max_length=self.config["data"]["max_seq_length"],
#             )

#         self.train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
#         self.eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

#         log.info(f"Train size: {len(self.train_dataset)}")
#         log.info(f"Eval size: {len(self.eval_dataset)}")

#     # =========================
#     # COLLATOR (FIXED)
#     # =========================
#     def data_collator(self, features):
#         batch = self.tokenizer.pad(
#             features,
#             padding=True,
#             return_tensors="pt",
#             pad_to_multiple_of=8,
#         )

#         labels = batch["input_ids"].clone()
#         labels[batch["attention_mask"] == 0] = -100
#         batch["labels"] = labels

#         return batch

#     # =========================
#     # TRAINING ARGS
#     # =========================
#     def setup_training_arguments(self):
#         cfg = self.config["training"]

#         return TrainingArguments(
#             output_dir=cfg["output_dir"],
#             num_train_epochs=cfg["num_train_epochs"],
#             per_device_train_batch_size=cfg["per_device_train_batch_size"],
#             per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
#             gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

#             learning_rate=cfg["learning_rate"],
#             weight_decay=cfg["weight_decay"],
#             logging_steps=cfg["logging_steps"],
#             save_steps=cfg["save_steps"],

#             # OOM SAFETY
#             fp16=True,
#             gradient_checkpointing=True,

#             # Replace warmup_ratio
#             warmup_steps=int(cfg["max_steps"] * cfg.get("warmup_ratio", 0.03)),

#             lr_scheduler_type=cfg["lr_scheduler_type"],
#             max_steps=cfg["max_steps"],

#             report_to=[],
#             eval_strategy="steps",
#             eval_steps=self.config["evaluation"]["eval_steps"],
#             save_strategy="steps",

#             load_best_model_at_end=True,
#             remove_unused_columns=False,

#             run_name=f"phi3-{uuid.uuid4().hex[:6]}",
#         )

#     # =========================
#     # TRAIN
#     # =========================
#     def train(self):
#         try:
#             self.setup_model_and_tokenizer()
#             self.prepare_datasets()

#             training_args = self.setup_training_arguments()

#             self.trainer = Trainer(
#                 model=self.model,
#                 args=training_args,
#                 train_dataset=self.train_dataset,
#                 eval_dataset=self.eval_dataset,
#                 processing_class=self.tokenizer,
#                 data_collator=self.data_collator,
#                 callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
#             )

#             self.trainer.train()

#         except RuntimeError as e:
#             if "out of memory" in str(e).lower():
#                 log.error("CUDA OOM detected! Trying recovery...")

#                 torch.cuda.empty_cache()

#                 # Reduce batch size dynamically
#                 self.config["training"]["per_device_train_batch_size"] = max(
#                     1, self.config["training"]["per_device_train_batch_size"] // 2
#                 )

#                 log.info(f"Retrying with smaller batch size: {self.config['training']['per_device_train_batch_size']}")

#                 return self.train()

#             else:
#                 raise e

#     # =========================
#     # EVAL
#     # =========================
#     def evaluate_model(self):
#         results = self.trainer.evaluate()
#         for k, v in results.items():
#             print(f"{k}: {v}")
#         return results

#     # =========================
#     # SAVE
#     # =========================
#     def save_model_for_inference(self, path: str):
#         self.model.save_pretrained(path)
#         self.tokenizer.save_pretrained(path)

#         with open(Path(path) / "config.yaml", "w") as f:
#             yaml.dump(self.config, f)

#         log.info(f"Model saved to {path}")


# # =========================
# # MAIN
# # =========================
# def main():
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     args = parser.parse_args()

#     trainer = AgentTrainer(args.config)

#     trainer.train()
#     trainer.evaluate_model()
#     trainer.save_model_for_inference("./models/phi3-agent-final")


# if __name__ == "__main__":
#     main()