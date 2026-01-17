import os
import time
from datetime import datetime

from peft import LoraConfig
import torch

from transformers import AutoModelForCausalLM, AutoProcessor  # , BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# from nll_to_po.training.data import evaluate


# FP8 = False
OUTPUT_DIR_ROOT = "logs"
MODEL_NAME = "Qwen/Qwen3-4B"  # "openai/gpt-oss-120b"  # "Qwen/Qwen3-8B"
DATASET_NAME = "openai/gsm8k"  # "HuggingFaceTB/Countdown-Task-GOLD"  # "Jiayi-Pan/Countdown-Tasks-3to4"  # "openai/gsm8k"
VERSION = "v6"
USE_RATIONALE_GSM8K = False


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

# generate sft text or prompt/completion field
if "HuggingFaceTB" in DATASET_NAME:

    def tokenize(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }
elif "gsm8k" in DATASET_NAME:
    # gsm8k
    def tokenize(example):
        rationale, answer = example["answer"].split("####")
        if USE_RATIONALE_GSM8K:
            return {
                "prompt": f"Question: {example['question']}\nAnswer:",
                "completion": f" {rationale.strip()}\nFinal Answer: {answer.strip()}",
            }
        else:
            return {
                "prompt": f"Question: {example['question']}\nAnswer:",
                "completion": f" {answer.strip()}",
            }
else:
    raise NotImplementedError(f"Dataset {DATASET_NAME} not implemented.")

dataset = dataset.map(lambda x: tokenize(x))

# split the dataset into train and test
# train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = dataset

print(f"one training example: {train_dataset[0]}")

print("Evaluating training accuracy before SFT...")
# evaluate(dataset=dataset, model=model, tokenizer=tokenizer)

# #######################################
# ******* Trainer (SFT) config **********
# #######################################

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"{OUTPUT_DIR_ROOT}/{MODEL_NAME}/{DATASET_NAME.split('/')[-1]}/{'[cot]' if USE_RATIONALE_GSM8K else ''}[{VERSION}]trl-sft-{timestamp}"
os.makedirs(output_dir, exist_ok=True)


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
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
    target_modules=["k_proj", "q_proj", "v_proj"],
)

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    # Training schedule / optimization
    per_device_train_batch_size=2,  # Batch size per GPU
    gradient_accumulation_steps=8,  # Gradients are accumulated over multiple steps → effective batch size = 2 * 8 = 16
    warmup_steps=100,
    num_train_epochs=2,  # Number of full dataset passes. For shorter training, use `max_steps` instead (this case)
    # max_steps = 200,
    learning_rate=5e-5,  # Learning rate for the optimizer
    optim="paged_adamw_8bit",  # Optimizer
    # Logging / reporting
    logging_steps=10,  # Log training metrics every N steps
    report_to="tensorboard",  # Experiment tracking tool
    # trackio_space_id=output_dir,          # HF Space where the experiment tracking will be saved
    output_dir=output_dir,  # Where to save model checkpoints and logs
    max_length=512,  # Maximum input sequence length
    # use_liger_kernel=True,  # Enable Liger kernel optimizations for faster training
    # activation_offloading=True,  # Offload activations to CPU to reduce GPU memory usage
    gradient_checkpointing=False,  # Save memory by re-computing activations during backpropagation
    # Hub integration
    push_to_hub=False,  # Automatically push the trained model to the Hugging Face Hub
    # The model will be saved under your Hub account in the repository named `output_dir`
    # gradient_checkpointing_kwargs={
    #     "use_reentrant": False
    # },  # To prevent warning message
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
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
# ******* Inference / Eval *******
# ################################

# print("Evaluating training accuracy after SFT...")
# evaluate(dataset=dataset, model=model, tokenizer=tokenizer)
