import os
from datetime import datetime
import re
from io import BytesIO
import base64
import time

import torch

from transformers import AutoProcessor, BitsAndBytesConfig  # , FineGrainedFP8Config
from transformers import Mistral3ForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


# FP8 = False
OUTPUT_DIR_ROOT = "logs"
SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"  # "mistralai/Ministral-3-3B-Instruct-2512-BF16"  # "unsloth/Ministral-3-3B-Instruct-2512"


# #################################
# ******** Load Dataset ***********
# #################################

dataset_id = "lmms-lab/multimodal-open-r1-8k-verified"
train_dataset = load_dataset(dataset_id, split="train[:5%]")


def make_conversation(example):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    return {
        "prompt": conversation,
        "image": example["image"],
    }


train_dataset = train_dataset.map(make_conversation)

print(f"one training example: {train_dataset[0]}")

# #################################
# ******* Load Model *************
# #################################

# uncomment if we need tokenizer during training
# processor = AutoProcessor.from_pretrained(
#     MODEL_NAME,
#     padding_side="left",
#     fix_mistral_regex=True if "mistral" in MODEL_NAME.lower() else False,
# )

# if FP8:
#     model_name = "mistralai/Ministral-3-3B-Instruct-2512"
#     quantization_config = FineGrainedFP8Config(dequantize=False)
# else:
# model_name = "mistralai/Ministral-3-3B-Instruct-2512-BF16" # "unsloth/Ministral-3-3B-Instruct-2512"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load the model in 4-bit precision to save memory
    bnb_4bit_compute_dtype=torch.float16,  # Data type used for internal computations in quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization to improve accuracy
    bnb_4bit_quant_type="nf4",  # Type of quantization. "nf4" is recommended for recent LLMs
)

if "mistral" in MODEL_NAME.lower():
    model_constructor = Mistral3ForConditionalGeneration
else:
    model_constructor = Qwen3VLForConditionalGeneration

print(f"Loading model {MODEL_NAME} using {model_constructor}")

model = model_constructor.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)

# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

# #################################
# ******* Reward functions ********
# #################################


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"

    matches = []
    for item in completions:
        if isinstance(item, list):
            text = item[0]["content"]
        else:
            text = item
        match = re.match(pattern, text, re.DOTALL | re.MULTILINE)
        matches.append(match)

    return [1.0 if match else 0.0 for match in matches]


def len_reward(completions, solution, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = []
    for item in completions:
        if isinstance(item, list):
            text = item[0]["content"]
        else:
            text = item
        contents.append(text)

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparsable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


# #######################################
# ******* Trainer (GRPO) config *********
# #######################################

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"{OUTPUT_DIR_ROOT}/{MODEL_NAME}/trl-grpo-{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    learning_rate=2e-5,
    # num_train_epochs=1,
    max_steps=100,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024,  # default: 256            # Max completion length produced during training
    num_generations=2,  # 2, # default: 8                  # Number of generations produced during training for comparison
    max_prompt_length=2048,  # default: 512                # Max prompt length of the input prompt used for generation during training
    fp16=True,
    bf16=False,
    # Parameters related to reporting and saving
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    report_to="tensorboard",  # Experiment tracking tool
    # trackio_space_id = output_dir,
    # Hub integration
    push_to_hub=False,
    log_completions=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, len_reward],
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
# ******* Inference **************
# ################################

adapter_model = f"{output_dir}"

model = model_constructor.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

tokenizer = AutoProcessor.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    fix_mistral_regex=True if "mistral" in MODEL_NAME.lower() else False,
)

problem = train_dataset[0]["problem"]
image = train_dataset[0]["image"]

buffer = BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
            {"type": "text", "text": problem},
        ],
    },
]

print(f"messages: {messages}")

tokenized = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True
)
tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
tokenized["pixel_values"] = tokenized["pixel_values"].to(
    dtype=torch.bfloat16, device="cuda"
)
image_sizes = [tokenized["pixel_values"].shape[-2:]]

output = model.generate(
    **tokenized,
    image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]) :])

print(f"model response:\n{decoded_output}")
