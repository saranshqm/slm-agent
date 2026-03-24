# =========================
# INSTALL (if needed)
# =========================
# pip install unsloth trl datasets
# install unsloth first
# pip install unsloth
import sys
import types
from transformers import modeling_utils

# Monkey patch for shard_checkpoint
def shard_checkpoint(state_dict, max_shard_size, weights_name):
    """
    A simple shard_checkpoint implementation that doesn't actually shard.
    This is a fallback for older transformers versions.
    """
    # Just return the original state dict without sharding
    return [(state_dict, weights_name)]

# Attach the function to modeling_utils if it doesn't exist
if not hasattr(modeling_utils, 'shard_checkpoint'):
    modeling_utils.shard_checkpoint = shard_checkpoint
    print("✓ Monkey patched shard_checkpoint into modeling_utils")

# Also add to the module if needed for direct imports
if not hasattr(modeling_utils, 'shard_checkpoint'):
    setattr(modeling_utils, 'shard_checkpoint', shard_checkpoint)


    
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset

# =========================
# CONFIG
# =========================
MODEL_NAME = "unsloth/Qwen2.5-1.5B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs/agent-model"

# =========================
# LOAD MODEL
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    quantization_config=bnb_config,
)

# =========================
# 🔥 ADD SPECIAL TOKENS (CRITICAL)
# =========================
special_tokens = {
    "additional_special_tokens": [
        "<tool_use>", "</tool_use>",
        "<tool_name>", "</tool_name>",
        "<parameters>", "</parameters>",
    ]
}

tokenizer.add_special_tokens(special_tokens)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 🔥 IMPORTANT
model.resize_token_embeddings(len(tokenizer))

# =========================
# LoRA
# =========================
model = FastLanguageModel.get_peft_model(
    model,
    r=256,
     target_modules = [
        # attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.05,
)

# =========================
# LOAD YOUR DATASET
# =========================
train_dataset = load_dataset(
    "json",
    data_files="./data/processed/train_dataset.json"
)["train"]

eval_dataset = load_dataset(
    "json",
    data_files="./data/processed/eval_dataset.json"
)["train"]

# =========================
# FORMAT DATA (CRITICAL)
# =========================
def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {"text": text}

train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# =========================
# TRAIN
# =========================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",  # 🔥 IMPORTANT
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        report_to="none",  # no wandb
    ),
)

trainer.train()

# =========================
# 🔥 SAVE (CRITICAL FIX)
# =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to {OUTPUT_DIR}")