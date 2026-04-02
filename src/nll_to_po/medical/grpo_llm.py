import os
import time
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import secrets
from typing import Optional

import tyro
from peft import LoraConfig, PeftModel
import torch

from transformers import AutoModelForCausalLM, AutoProcessor  # , BitsAndBytesConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig  # , get_peft_config, ModelConfig

from nll_to_po.training.utils import set_seed_everywhere
from nll_to_po.llm.embed_cov import (
    compute_U_star_from_covariance,
    compute_U_star_from_trace,
)
from nll_to_po.medical.utils import (
    answer_correctness_reward,
    format_compliance_reward,
    prepare_dataset,
    embedding_reward_func_constructor_pubmedqa,
)


@dataclass
class TrainConfig:
    # Model and data
    model_name: str = "Qwen/Qwen3-0.6B"
    """HuggingFace model ID or local path."""
    dataset_name: str = "bigbio/pubmed_qa"
    """HuggingFace dataset ID."""
    dataset_size: int = -1
    """Number of training samples. -1 uses the full dataset."""
    output_dir_root: str = "logs"
    """Root directory for checkpoints and TensorBoard logs."""
    version: str = "v15"
    """Version tag appended to the output directory name."""
    seed: Optional[int] = None
    """Random seed. If None, a random seed is generated."""
    adapter_path: Optional[str] = None
    """Path to a SFT adapter (local dir) to load and merge before GRPO training."""

    # Reward
    format_weight: float = 0.5
    """Weight of the format-compliance reward (relative to answer-correctness and embedding)."""
    embed: bool = True
    """Use embedding-based Mahalanobis reward."""
    reward_embedding_model: str = "NeuML/pubmedbert-base-embeddings-8M"
    """HuggingFace model ID for the reward embedding model."""
    lam: float = 0.001
    """Lambda: KL penalty coefficient (beta in GRPO) and U_star regularization."""
    u_star_type: str = "id"
    """U_star matrix type: 'id' (identity), 'cov' (from covariance.pt), 'trace' (from trace.json)."""
    pooling: str = "cls"
    """Pooling strategy for the embedding model: 'mean', 'cls', or 'last'."""
    answer_only: bool = False
    """Embed only the short yes/no answer; if False, embed the long_answer explanation."""
    sentence_transformer: bool = True
    """Use SentenceTransformer API for embedding; otherwise use AutoModel."""

    # PEFT / LoRA
    use_peft: bool = True
    """Use LoRA (PEFT) for parameter-efficient fine-tuning."""
    r_peft: int = 8
    """LoRA rank."""

    # Training schedule
    n_epochs: int = 1
    """Number of training epochs (used when n_steps <= 0)."""
    n_steps: int = 400
    """Max training steps. Set to -1 to train for n_epochs instead."""
    max_completion_length: int = 1024
    """Maximum number of tokens generated per completion during training."""
    max_prompt_length: int = 256
    """Maximum prompt length in tokens."""
    use_eval_dataset: bool = False
    """Evaluate on a held-out validation split during training."""

    # Optimizer / batching
    learning_rate: float = 5e-5
    """AdamW learning rate."""
    per_device_train_batch_size: int = 4
    """Per-device training batch size."""
    gradient_accumulation_steps: int = 2
    """Gradient accumulation steps (effective batch = batch_size * accum * n_gpus)."""
    num_generations: int = 8
    """Number of completions generated per prompt for GRPO comparison."""
    save_steps: int = 20
    """Save a checkpoint every N steps."""


def main(cfg: TrainConfig) -> None:
    seed = cfg.seed if cfg.seed is not None else secrets.randbits(32)
    set_seed_everywhere(seed)

    # ###########################################
    # ******** Load Covariance matrix ***********
    # ###########################################
    output_path = (
        Path("results/cov/")
        / f"{cfg.dataset_name.replace('/', '-')}"
        / cfg.reward_embedding_model.split("/")[-1]
    )
    if os.path.exists(output_path / "covariance.pt") and cfg.u_star_type == "cov":
        Sigma = torch.load(output_path / "covariance.pt")
        U_star = compute_U_star_from_covariance(Sigma, cfg.lam)
        label = "U-star-cov"
    elif os.path.exists(output_path / "trace.json") and cfg.u_star_type == "trace":
        with open(output_path / "trace.json") as f:
            data = json.load(f)
        trace = data["covariance_trace"]
        n = data["embedding_dim"]
        if data["model_name"] != cfg.reward_embedding_model:
            print(
                f"Warning: model name in trace file ({data['model_name']}) does not match "
                f"reward embedding model ({cfg.reward_embedding_model})."
            )
        U_star = compute_U_star_from_trace(trace, n, cfg.lam)
        label = "U-star-trace"
    else:
        U_star = None
        label = "Id"

    # ###########################################
    # ******** Load Model & Tokenizer ***********
    # ###########################################

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        attn_implementation="flash_attention_2",  # Change to Flash Attention if GPU has support
        dtype="bfloat16",  # Change to bfloat16 if GPU has support
        use_cache=True,  # Whether to cache attention outputs to speed up inference
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     # bnb_4bit_compute_dtype=torch.float16,
        #     # bnb_4bit_use_double_quant=True,
        #     # bnb_4bit_quant_type="nf4"
        # )
        device_map="auto",
    )
    tokenizer = AutoProcessor.from_pretrained(cfg.model_name, padding_side="left")
    if cfg.adapter_path is not None:
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
        model = model.merge_and_unload()

    # #################################
    # ******** Load Dataset ***********
    # #################################

    dataset = load_dataset(cfg.dataset_name)
    dataset = prepare_dataset(dataset["train"])
    if cfg.dataset_size > 0:
        dataset = dataset.select(range(cfg.dataset_size))

    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    print(f"one training example: {train_dataset[0]}")

    # #######################################
    # ******* Trainer (GRPO) config *********
    # #######################################

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    embed_tag = (
        "["
        + cfg.reward_embedding_model.split("/")[-1]
        + "]["
        + label
        + "]["
        + cfg.pooling
        + "]"
        if cfg.embed
        else ""
    )
    steps_tag = f"s:{cfg.n_steps}" if cfg.n_steps > 0 else f"e:{cfg.n_epochs}"
    peft_tag = f"[peft_{cfg.r_peft}]" if cfg.use_peft else ""
    adapter_tag = (
        f"[adapter-{cfg.adapter_path.split('/')[-1]}]" if cfg.adapter_path else ""
    )
    output_dir = (
        f"{cfg.output_dir_root}/{cfg.model_name.split('/')[-1]}/"
        f"{cfg.dataset_name.split('/')[-1]}/"
        f"{embed_tag}[{cfg.lam}]{peft_tag}[{steps_tag}][{cfg.version}]{adapter_tag}trl-grpo-{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    peft_config = LoraConfig(
        r=cfg.r_peft,
        lora_alpha=2 * cfg.r_peft,
        lora_dropout=0.05,
        # target_modules=[
        #     "q_proj", "k_proj", "v_proj", "o_proj",
        #     "gate_proj", "up_proj", "down_proj",
        # ],
        target_modules=["q_proj", "v_proj"],
    )

    reward_funcs = [
        format_compliance_reward,
        answer_correctness_reward,
        embedding_reward_func_constructor_pubmedqa(
            model=cfg.reward_embedding_model,
            U_star=U_star,
            pooling=cfg.pooling,
            max_length=cfg.max_completion_length + 256,
            answer_only=cfg.answer_only,
            sentence_transformer=cfg.sentence_transformer,
        ),
    ]
    reward_weights = [cfg.format_weight, float(not cfg.embed), float(cfg.embed)]
    reward_weights = [w / sum(reward_weights) for w in reward_weights]
    print(f"Reward weights: {reward_weights}")

    training_args = GRPOConfig(
        do_eval=cfg.use_eval_dataset,
        eval_on_start=cfg.use_eval_dataset,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.n_epochs,
        max_steps=cfg.n_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=8,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=cfg.max_prompt_length,
        num_generations=cfg.num_generations,
        beta=cfg.lam,
        gradient_checkpointing=False,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=True,
        output_dir=output_dir,
        logging_steps=1,
        save_steps=cfg.save_steps,
        report_to="tensorboard",
        push_to_hub=False,
        log_completions=True,
        reward_weights=reward_weights,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if cfg.use_eval_dataset else None,
        peft_config=peft_config,
    )

    # ##########################
    # ******* Training *********
    # ##########################

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    torch.cuda.synchronize()
    t0 = time.time()
    trainer_stats = trainer.train()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Total training time: {round(t1 - t0, 2)} seconds.")

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    trainer.save_model(output_dir)
    # trainer.push_to_hub(dataset_name=dataset_id)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
