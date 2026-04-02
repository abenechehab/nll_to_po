import os
import time
from dataclasses import dataclass
from datetime import datetime

import tyro
from peft import LoraConfig
import torch

from transformers import AutoModelForCausalLM, AutoProcessor  # , BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from nll_to_po.medical.utils import prepare_sft_dataset


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

    # PEFT / LoRA
    use_peft: bool = True
    """Use LoRA (PEFT) for parameter-efficient fine-tuning."""

    # Training schedule
    max_steps: int = 400
    """Max training steps."""
    max_length: int = 2048
    """Maximum input sequence length in tokens."""
    save_steps: int = 20
    """Save a checkpoint every N steps."""

    # Optimizer / batching
    learning_rate: float = 5e-5
    """AdamW learning rate."""
    per_device_train_batch_size: int = 8
    """Per-device training batch size."""
    gradient_accumulation_steps: int = 2
    """Gradient accumulation steps (effective batch = batch_size * accum * n_gpus)."""


def main(cfg: TrainConfig) -> None:
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

    # #################################
    # ******** Load Dataset ***********
    # #################################

    dataset = load_dataset(cfg.dataset_name)
    dataset = prepare_sft_dataset(
        dataset["train"], num_samples=cfg.dataset_size if cfg.dataset_size > 0 else None
    )

    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    print(f"one training example: {train_dataset[0]}")

    # #######################################
    # ******* Trainer (SFT) config **********
    # #######################################

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    peft_tag = "[peft]" if cfg.use_peft else ""
    output_dir = (
        f"{cfg.output_dir_root}/{cfg.model_name.split('/')[-1]}/"
        f"{cfg.dataset_name.split('/')[-1]}/"
        f"{peft_tag}[{cfg.version}]trl-sft-{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=[
        #     "q_proj", "k_proj", "v_proj", "o_proj",
        #     "gate_proj", "up_proj", "down_proj",
        # ],
        target_modules=["k_proj", "q_proj", "v_proj"],
    )

    training_args = SFTConfig(
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        report_to="tensorboard",
        save_steps=cfg.save_steps,
        output_dir=output_dir,
        max_length=cfg.max_length,
        # use_liger_kernel=True,
        # activation_offloading=True,
        gradient_checkpointing=False,
        push_to_hub=False,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config if cfg.use_peft else None,
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
