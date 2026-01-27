import os
import time
from datetime import datetime
import json
from pathlib import Path
import secrets

from peft import LoraConfig
import torch

from transformers import AutoModelForCausalLM, AutoProcessor  # , BitsAndBytesConfig
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig  # , get_peft_config, ModelConfig

from nll_to_po.training.utils import set_seed_everywhere
from nll_to_po.training.reward import (
    equation_reward_func,
    embedding_reward_func_constructor,
    format_reward_func,
)
from nll_to_po.training.data import SYSTEM_PROMPT_ORCA
from nll_to_po.llm.embed_cov import (
    compute_U_star_from_covariance,
    compute_U_star_from_trace,
)


# FP8 = False
OUTPUT_DIR_ROOT = "logs"
MODEL_NAME = "Qwen/Qwen3-1.7B"  # "openai/gpt-oss-120b"  # "Qwen/Qwen3-8B"
DATASET_NAME = "HuggingFaceTB/Countdown-Task-GOLD"  # "HuggingFaceTB/Countdown-Task-GOLD"  # "Jiayi-Pan/Countdown-Tasks-3to4"  # "openai/gsm8k"  # "microsoft/orca-math-word-problems-200k"
DATASET_SIZE = -1
VERSION = "v13"
EMBED = True
REWARD_EMBEDDING_MODEL = (
    "google/embeddinggemma-300m"  # "google/gemma-3-1b-it"  # "roberta-large"
)
LAMBDA = 0.001
U_STAR_TYPE = "trace"  # "cov" or "trace" or else
POOLING = "cls"  # "mean" or "cls" (for causalLM models, "cls" means last token)
USE_PEFT = True
R_PEFT = 8  # Rank for PEFT LoRA
SEED = secrets.randbits(32)
MAX_COMPLETION_LENGTH = 1024  # 1024
USE_EVAL_DATASET = False
N_EPOCHS = 1
N_STEPS = 400
ANSWER_ONLY = True
SENTENCE_TRANSFORMER = True  # If True, use SentenceTransformer for embedding; else use BERT-based / causal models

set_seed_everywhere(SEED)


# ###########################################
# ******** Load Covariance matrix ***********
# ###########################################
output_path = (
    Path("results/cov/")
    / f"{DATASET_NAME.replace('/', '-')}"
    / REWARD_EMBEDDING_MODEL.split("/")[-1]
)
# From covariance
if os.path.exists(output_path / "covariance.pt") and U_STAR_TYPE == "cov":
    Sigma = torch.load(output_path / "covariance.pt")
    U_star = compute_U_star_from_covariance(Sigma, LAMBDA)
    label = "U-star-cov"
elif os.path.exists(output_path / "trace.json") and U_STAR_TYPE == "trace":
    # From trace
    with open(output_path / "trace.json") as f:
        data = json.load(f)
    trace = data["covariance_trace"]
    n = data["embedding_dim"]  # Get dimension from metadata
    if data["model_name"] != REWARD_EMBEDDING_MODEL:
        print(
            f"Warning: model name in trace file ({data['model_name']}) does not match reward embedding model ({REWARD_EMBEDDING_MODEL})."
        )
    U_star = compute_U_star_from_trace(trace, n, LAMBDA)
    label = "U-star-trace"
else:
    U_star = None
    label = "Id"

# ###########################################
# ******** Load Model & Tokenizer ***********
# ###########################################

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2",  # Change to Flash Attention if GPU has support
    dtype="bfloat16",  # Change to bfloat16 if GPU has support
    use_cache=True,  # Whether to cache attention outputs to speed up inference
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
    #     # bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
    #     # bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
    #     # bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    # )
    device_map="auto",
)
tokenizer = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="left")

# #################################
# ******** Load Dataset ***********
# #################################

# Load dataset from Hugging Face Hub
if "HuggingFaceTB" in DATASET_NAME:
    dataset = load_dataset(DATASET_NAME, "verified_Qwen2.5-7B-Instruct")["train"]
elif "gsm8k" in DATASET_NAME:
    dataset = load_dataset(DATASET_NAME, "main", split="train")
else:
    dataset = load_dataset(DATASET_NAME, split="train")
# select a random subset of 50k samples
# dataset = dataset.shuffle(seed=42).select(range(DATASET_SIZE))

# gemerate r1 prompt with a prefix for the model to already start with the thinking process
if "HuggingFaceTB" in DATASET_NAME:
    from nll_to_po.training.data import generate_r1_prompt_answer

    dataset = dataset.map(lambda x: generate_r1_prompt_answer(x["messages"], tokenizer))
elif "gsm8k" in DATASET_NAME:

    def format_gsm8k(example):
        rationale, answer = example["answer"].split("####")
        return {
            "prompt": f"Question: {example['question']}\nAnswer:",
            "answer": answer.strip(),
            "rationale": rationale.strip(),
        }

    dataset = dataset.map(lambda x: format_gsm8k(x))
elif "orca" in DATASET_NAME:

    def format_orca(example):
        r1_prefix = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_ORCA,
            },
            {
                "role": "user",
                "content": example["question"],
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, add_generation_prompt=True
            ),
        }

    dataset = dataset.map(lambda x: format_orca(x))
else:
    raise NotImplementedError(f"Dataset {DATASET_NAME} not implemented.")

if DATASET_SIZE > 0:
    dataset = dataset.select(range(DATASET_SIZE))

# split the dataset into train and validation
train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# train_dataset = dataset
print(f"one training example: {train_dataset[0]}")

# #######################################
# ******* Trainer (GRPO) config *********
# #######################################

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"{OUTPUT_DIR_ROOT}/{MODEL_NAME.split('/')[-1]}/{DATASET_NAME.split('/')[-1]}/{'[' + REWARD_EMBEDDING_MODEL.split('/')[-1] + ']' + '[' + label + ']' + '[' + POOLING + ']' if EMBED else ''}[{LAMBDA}]{'[peft_' + str(R_PEFT) + ']' if USE_PEFT else ''}[{'s:' + str(N_STEPS) + ']' if N_STEPS > 0 else 'e:' + str(N_EPOCHS)}[{VERSION}]trl-grpo-{timestamp}"
os.makedirs(output_dir, exist_ok=True)

peft_config = LoraConfig(
    r=R_PEFT,
    lora_alpha=2 * R_PEFT,
    lora_dropout=0.05,
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    # ],
    target_modules=["q_proj", "v_proj"],
)

if "HuggingFaceTB" in DATASET_NAME:
    reward_funcs = [
        format_reward_func,
        equation_reward_func,
        embedding_reward_func_constructor(
            model=REWARD_EMBEDDING_MODEL,
            U_star=U_star,
            pooling=POOLING,
            dataset=DATASET_NAME,
            max_length=MAX_COMPLETION_LENGTH + 256,
            answer_only=ANSWER_ONLY,
            sentence_transformer=SENTENCE_TRANSFORMER,
        ),
    ]
    reward_weights = [0.0, float(not EMBED), float(EMBED)]
else:
    assert EMBED, "For non countdown datasets, EMBED must be True."
    reward_funcs = [
        embedding_reward_func_constructor(
            model=REWARD_EMBEDDING_MODEL,
            U_star=U_star,
            pooling=POOLING,
            dataset=DATASET_NAME,
            max_length=MAX_COMPLETION_LENGTH + 256,
            answer_only=ANSWER_ONLY,
            sentence_transformer=SENTENCE_TRANSFORMER,
        )
    ]
    reward_weights = [1.0]

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    do_eval=USE_EVAL_DATASET,
    eval_on_start=USE_EVAL_DATASET,
    learning_rate=5e-5,
    num_train_epochs=N_EPOCHS,
    max_steps=N_STEPS,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    # Parameters that control the data preprocessing
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=8,
    max_completion_length=MAX_COMPLETION_LENGTH,  # default: 256            # Max completion length produced during training
    max_prompt_length=256,  # default: 512                # Max prompt length of the input prompt used for generation during training
    # GRPO specific parameters
    num_generations=8,  # 2, # default: 8                  # Number of generations produced during training for comparison
    # num_generations_eval=4,
    beta=LAMBDA,
    gradient_checkpointing=False,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=False,
    bf16=True,
    # Parameters related to reporting and saving
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    save_steps=20,  # Save model checkpoint every N steps
    report_to="tensorboard",  # Experiment tracking tool
    # trackio_space_id = output_dir,
    # Hub integration
    push_to_hub=False,
    log_completions=False,
    reward_weights=reward_weights,  # Weights for each reward function
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset if USE_EVAL_DATASET else None,
    peft_config=peft_config,
)

# ##########################
# ******* Training *********
# ##########################

# GPU memory stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

torch.cuda.synchronize()
t0 = time.time()

trainer_stats = trainer.train()  # training

torch.cuda.synchronize()
t1 = time.time()
print(f"Total training time: {round(t1 - t0, 2)} seconds.")

# GPU memory stats after training
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

# save model
trainer.save_model(output_dir)
# trainer.push_to_hub(dataset_name=dataset_id)

# ################################
# ******* Inference **************
# ################################
