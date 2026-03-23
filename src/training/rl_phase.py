"""
Optional post-SFT RL phase (GRPO-style group relative advantages) focused on tool use + reasoning.
Plug-in: enabled via config `rl.enabled`. Does not alter SFT dataset or AgentTrainer layout beyond a hook.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from tool_rl_rewards import score_tool_completion, summarize_batch_rewards

log = logging.getLogger(__name__)


def _patch_dynamic_cache_for_phi3_hub() -> None:
    """Hub Phi-3 vs Transformers 4.48+: restore seen_tokens, get_max_length, get_usable_length on DynamicCache."""
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length(0))
        if not hasattr(DynamicCache, "get_max_length"):

            def get_max_length(self, layer_idx: int = 0) -> int:
                return self.get_max_cache_shape(layer_idx)

            DynamicCache.get_max_length = get_max_length
        if not hasattr(DynamicCache, "get_usable_length"):

            def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
                max_length = self.get_max_cache_shape(layer_idx)
                prev = self.get_seq_length(layer_idx)
                if max_length is not None and max_length > 0 and prev + new_seq_length > max_length:
                    return max_length - new_seq_length
                return prev

            DynamicCache.get_usable_length = get_usable_length
    except Exception as ex:
        log.debug("DynamicCache hub-compat patch skipped: %s", ex)


def _format_prompt_only(tokenizer: Any, system: str, instruction: str, input_text: str | None) -> str:
    user_msg = instruction
    if input_text and input_text.strip():
        user_msg += f"\n\nInput:\n{input_text.strip()}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_prompt_dataset(config: dict[str, Any], split: str, tokenizer: Any) -> list[str]:
    data = config["data"]
    path = data["train_dataset"] if split == "train" else data["eval_dataset"]
    ds = load_dataset("json", data_files=path)["train"]
    prompts: list[str] = []
    for row in ds:
        system = row.get("system") or ""
        inst = row.get("instruction") or ""
        inp = row.get("input") or ""
        prompts.append(_format_prompt_only(tokenizer, system, inst, inp if inp else None))
    return prompts


def _get_device(model: torch.nn.Module) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def completion_log_prob(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    completion: str,
    device: torch.device,
) -> torch.Tensor:
    """Sum log pi(completion | prompt) under the model (differentiable)."""
    p_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
    c_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if c_ids.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    full = torch.cat([p_ids, c_ids], dim=1)
    out = model(full)
    logits = out.logits[:, :-1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    p_len = p_ids.shape[1]
    c_len = c_ids.shape[1]
    total = torch.zeros((), device=device, dtype=log_probs.dtype)
    for j in range(c_len):
        pos = p_len - 1 + j
        tid = full[0, pos + 1]
        total = total + log_probs[0, pos, tid]
    return total


@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    if device.type == "cuda":
        g = torch.Generator(device=device)
    else:
        g = torch.Generator()
    g.manual_seed(int(seed) % (2**32))
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    model.eval()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(0.01, temperature) if temperature > 0 else 1.0,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        generator=g,
    )
    model.train()
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False).strip()


def _rl_eval_pass(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_prompts: list[str],
    cfg: dict[str, Any],
    device: torch.device,
    step: int,
    out_dir: Path,
    num_samples: int,
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    sample_text = ""
    max_new = int(cfg.get("max_new_tokens", 384))
    rng = random.Random(int(cfg.get("seed", 42)) + step)
    subset = eval_prompts[: min(len(eval_prompts), num_samples)]
    for i, prompt in enumerate(subset):
        text = generate_one(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new,
            temperature=0.0,
            top_p=1.0,
            seed=rng.randint(0, 2**31 - 1),
        )
        scores.append(score_tool_completion(text))
        if i == 0:
            sample_text = text[:1200]
    model.train()
    stats = summarize_batch_rewards(scores)
    stats["step"] = float(step)
    log.info(
        "[RL eval] step=%s mean_reward=%.4f max=%.4f std=%.4f",
        step,
        stats["mean"],
        stats["max"],
        stats["std"],
    )
    if sample_text:
        log.info("[RL eval] first sample (trunc): %s", sample_text.replace("\n", "\\n")[:500])

    metrics_path = out_dir / "rl_metrics.jsonl"
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**stats, "split": "eval_greedy"}) + "\n")

    return stats


def _save_rl_checkpoint(model: torch.nn.Module, tokenizer: Any, out_dir: Path, step: int) -> None:
    ckpt = out_dir / f"checkpoint-rl-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    log.info("[RL] saved %s", ckpt.resolve())


def run_rl_phase(config: dict[str, Any], model: torch.nn.Module, tokenizer: Any) -> None:
    """
    GRPO-style updates: per prompt, sample G completions, reward, center advantages within group, policy grad.
    """
    rl = config.get("rl") or {}
    if not rl.get("enabled", False):
        return

    _patch_dynamic_cache_for_phi3_hub()

    device = _get_device(model)
    train_prompts = _load_prompt_dataset(config, "train", tokenizer)
    eval_prompts = _load_prompt_dataset(config, "eval", tokenizer)
    if not train_prompts:
        log.warning("[RL] No training prompts; skipping RL phase.")
        return

    out_root = Path(rl.get("output_dir", "./results_rl"))
    out_root.mkdir(parents=True, exist_ok=True)

    max_steps = int(rl.get("max_steps", 100))
    save_steps = int(rl.get("save_steps", 50))
    eval_steps = int(rl.get("eval_steps", 25))
    num_generations = int(rl.get("num_generations", 4))
    max_new_tokens = int(rl.get("max_new_tokens", 384))
    batch_prompts = int(rl.get("per_device_prompts", 2))
    lr = float(rl.get("learning_rate", 1e-6))
    temperature = float(rl.get("temperature", 0.85))
    top_p = float(rl.get("top_p", 0.95))
    reward_scale = float(rl.get("reward_scale", 1.0))
    max_seq = int(config["data"]["max_seq_length"])
    eval_sample_n = int(rl.get("eval_num_prompts", 12))
    base_seed = int(rl.get("seed", config["data"].get("seed", 42)))

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        log.error("[RL] No trainable parameters; skip RL.")
        return
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    if was_gc:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            was_gc = False

    model.train()
    rng = random.Random(base_seed)
    global_step = 0

    log.info(
        "[RL] starting | max_steps=%s G=%s batch_prompts=%s lr=%s out=%s",
        max_steps,
        num_generations,
        batch_prompts,
        lr,
        out_root.resolve(),
    )

    while global_step < max_steps:
        batch = [train_prompts[rng.randrange(len(train_prompts))] for _ in range(batch_prompts)]
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        step_raw_rewards: list[float] = []

        for prompt in batch:
            if len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) > max_seq - max_new_tokens - 8:
                continue

            rewards: list[float] = []
            completions: list[str] = []
            for _g in range(num_generations):
                seed = rng.randint(0, 2**31 - 1)
                with torch.no_grad():
                    comp = generate_one(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    )
                completions.append(comp)
                rw = score_tool_completion(comp) * reward_scale
                rewards.append(rw)
                step_raw_rewards.append(rw)

            mean_r = sum(rewards) / len(rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)) + 1e-8
            advantages = [(r - mean_r) / std_r for r in rewards]

            denom = max(1, len(batch) * num_generations)
            for comp, adv in zip(completions, advantages):
                if not comp.strip():
                    continue
                lp = completion_log_prob(model, tokenizer, prompt, comp, device)
                loss = -(float(adv) * lp) / denom
                loss.backward()
                step_losses.append(float(loss.detach().item()))

        if step_losses:
            torch.nn.utils.clip_grad_norm_(trainable, float(rl.get("max_grad_norm", 1.0)))
            optimizer.step()

        global_step += 1
        if step_losses:
            rmean = sum(step_raw_rewards) / max(1, len(step_raw_rewards))
            log.info(
                "[RL] step=%s policy_loss=%.6f | sampled_reward_mean=%.4f (n=%s)",
                global_step,
                sum(step_losses),
                rmean,
                len(step_raw_rewards),
            )
        else:
            log.info("[RL] step=%s (skipped batch; prompts too long)", global_step)

        if global_step % eval_steps == 0:
            _rl_eval_pass(
                model,
                tokenizer,
                eval_prompts,
                rl,
                device,
                global_step,
                out_root,
                eval_sample_n,
            )

        if global_step % save_steps == 0:
            _save_rl_checkpoint(model, tokenizer, out_root, global_step)

    if global_step % save_steps != 0:
        _save_rl_checkpoint(model, tokenizer, out_root, global_step)
    _rl_eval_pass(
        model,
        tokenizer,
        eval_prompts,
        rl,
        device,
        global_step,
        out_root,
        eval_sample_n,
    )

    if was_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    log.info("[RL] finished. Artifacts under %s", out_root.resolve())


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
