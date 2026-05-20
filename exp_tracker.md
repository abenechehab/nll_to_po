# Catastrophic forgetting

Base: HuggingFaceTB/SmolLM2-360M-Instruct
adapter_path: None

## Phase 1: GRPO on Countdown

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853  |
| 2  | logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840  |
| 3  | logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14][adapter-[peft][v13]trl-sft-20260413-151207]trl-grpo-20260415-104358  |

## Phase 2: 

### SFT on PubMedQA (in order)

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853][v15]trl-sft-20260424-233216  |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840][v15]trl-sft-20260425-053058  |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14][adapter-[peft][v13]trl-sft-20260413-151207]trl-grpo-20260415-104358][v15]trl-sft-20260416-123225  |

### GRPO on PubMedQA (in order)

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853]trl-grpo-20260424-042637  |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840]trl-grpo-20260425-001819  |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14][adapter-[peft][v13]trl-sft-20260413-151207]trl-grpo-20260415-104358]trl-grpo-20260416-123508  |

# PubMedQA main experiments

## SmolLM2-360M

### embed Id

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260425-221637  |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260426-044301  |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260426-112225  |

### Oracle reward

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[0.001][peft_8][s:400][v15]trl-grpo-20260426-173658  |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[0.001][peft_8][s:400][v15]trl-grpo-20260426-210927  |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[0.001][peft_8][s:400][v15]trl-grpo-20260427-015915 |

### SFT

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][v15]trl-sft-20260427-062028 |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][v15]trl-sft-20260427-070552 |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][v15]trl-sft-20260427-075122 |

# Ablation

# # pubmedqa with embedding gemma

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260428-115454 (100 steps only, needs to be repeated) |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260428-135557 |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260428-203531 |

# # pubmedqa with answer only

| seed | path        |
|------|-------------|
| 1  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260429-030037 |
| 2  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260429-070313 |
| 3  | logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15]trl-grpo-20260429-112056 |