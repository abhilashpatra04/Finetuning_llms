## Fine-tuning LLaMA-2 7B

This repository contains a Jupyter Notebook (Finetuning LLAMA-2 7B) which demonstrates a minimal workflow to fine-tune Meta's Llama-2-7b-chat model using:

- Hugging Face Transformers
- Hugging Face Datasets
- BitsAndBytes 4-bit quantization (nf4)
- PEFT (Parameter-Efficient Fine-Tuning) using LoRA
- Trainer API for a short training run

This README explains the notebook, the code cells, and the key concepts used. It also includes setup instructions, troubleshooting tips, and suggestions for further improvements.

## Files

- `finetuning_LLAMA_2_7B.ipynb` — main notebook that contains the full experiment and the code shown below.

When you run the notebook it will create a folder named `finetunedModel/` containing training checkpoints (e.g., `checkpoint-20`).

## High-level workflow in the notebook

1. Environment checks and dependency installation
2. GPU check and tidy CUDA environment variables
3. Login to Hugging Face Hub (for download/upload if required)
4. Load base Llama-2 7B chat model with 4-bit quantization (BitsAndBytes)
5. Load and preprocess datasets with `datasets`
6. Tokenize concatenated input/output text
7. Prepare the model for k-bit training and attach a LoRA adapter (PEFT)
8. Train using Hugging Face `Trainer` with a short smoke-run configuration
9. Load the PEFT-fine-tuned adapter and run inference
10. Optionally save the model to Google Drive

## Notebook cell-by-cell explanation and concepts

Below are the important code snippets and the concepts they illustrate.

### 1) Dependency installation

The notebook uses inline `%pip install` magic to install required packages. Key dependencies:

- `transformers` — model & training APIs
- `datasets` — data loading and preprocessing
- `peft` — parameter-efficient fine-tuning utilities (LoRA)
- `bitsandbytes` — 4/8-bit quantized kernels and optimizers
- `accelerate` — (optional) multi-GPU / mixed-precision support
- `GPUtil` — optional GPU utility helper

Example commands used in the notebook (notebook magics):

```powershell
%pip install peft
%pip install accelerate
%pip install bitsandBytes
%pip install transformers
%pip install datasets
%pip install GPUtil
```

Notes:
- In a regular PowerShell or terminal, remove the leading `%`.
- You may prefer installing these into a virtual environment.

### 2) GPU / CUDA environment

The notebook checks for GPU availability via PyTorch and sets `CUDA_VISIBLE_DEVICES=0`:

- `torch.cuda.is_available()` — confirms CUDA device presence
- `os.environ["CUDA_VISIBLE_DEVICES"]` — restricts which GPUs are visible to the process

If you do not have a GPU, the notebook will fall back to CPU, but training a 7B model on CPU is not feasible.

### 3) Hugging Face authentication

The notebook uses `huggingface_hub.notebook_login()` to authenticate with Hugging Face. This is required if you are loading private models or pushing artifacts to the Hub.

You can also run in CLI:

```powershell
huggingface-cli login
```

### 4) Loading Llama-2 7B in 4-bit (BitsAndBytes)

Important configuration used:

- BitsAndBytesConfig with:
  - `load_in_4bit=True` — loads weights in 4-bit to reduce memory
  - `bnb_4bit_use_double_quant=True` — uses double quantization trick
  - `bnb_4bit_quant_type='nf4'` — NF4 quantization (a low-bit quant scheme)
  - `bnb_4bit_compute_dtype=torch.bfloat16` — compute in bfloat16 for better range (if supported)

Why quantize?
- 7B models require a lot of GPU memory. 4-bit quantization reduces memory footprint significantly and enables fine-tuning or inference on single high-memory GPUs (A100-80GB, A6000, RTX 4090 with caution).

Notes and caveats:
- Quantized models and training require `bitsandbytes` and the right CUDA/driver combination.
- Some combinations may require using `device_map` or `load_in_4bit` alternatives.

### 5) Data loading and preprocessing

The notebook loads three datasets from the Hub and maps them to a common format {input, output}:

- `mlabonne/guanaco-llama2-1k`
- `gretelai/synthetic_text_to_sql`
- `meowterspace45/bird-sql-train-with-reasoning`

Each dataset is mapped to the minimal schema of `{"input": ..., "output": ...}` and then concatenated via `datasets.concatenate_datasets`.

Important points:
- The notebook assumes text fields exist on the datasets; in practice you should inspect and filter examples to match the task.
- The notebook concatenates `input + output` into a single text sequence for language modeling. Depending on your objective (next-token prediction or supervised fine-tuning), you may want to add prompts, separators, or create labels to avoid teaching the model to predict the input.

### 6) Tokenization

The notebook uses `LlamaTokenizer` (with `trust_remote_code=True` for some community models) and ensures a `pad_token` exists.

Tokenization step simply concatenates `input` and `output` text and tokenizes the resulting string. For causal LM fine-tuning this is a common simple approach, but note:

- For instruction datasets you often want to mask loss on the input prompt and compute loss only on the output tokens. The notebook uses the naive approach which computes loss across the entire concatenation.

### 7) Prepare for k-bit training and LoRA (PEFT)

Key steps:

- `model.gradient_checkpointing_enable()` — enables gradient checkpointing to reduce activation memory at the cost of extra compute.
- `prepare_model_for_kbit_training(model)` — from `peft` prepares a quantized model for training (freezes some parameters, enables gradient hooks).
- `LoraConfig(...)` and `get_peft_model(model, config)` — attach LoRA adapters to the model.

LoRA explanation (short):
- LoRA (Low-Rank Adaptation) inserts small trainable rank-decomposition matrices into selected weight projections (e.g., query/key/value projections). Instead of updating all model weights, LoRA trains a small number of parameters (far fewer than the full model), substantially reducing GPU memory usage and training time.

Important LoRA hyperparameters in the notebook:

- `r=8` — rank of the LoRA update matrices (controls capacity)
- `lora_alpha=64` — scaling factor
- `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` — modules to attach LoRA to (commonly attention projections)
- `bias="none"`, `lora_dropout=0.05`
- `task_type="CAUSAL_LM"`

### 8) Trainer & training arguments

The notebook uses `transformers.Trainer` with a quick smoke-run configuration:

- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=2` — accumulates gradients to simulate larger batch size
- `num_train_epochs=3` and `max_steps=20` — `max_steps` will stop the run early (for quick testing)
- `learning_rate=1e-4`
- `optim="paged_adamw_8bit"` — uses an 8-bit optimizer provided by bitsandbytes if available
- `data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)` — for causal language modeling

Notes:
- This configuration is for a short smoke test, not production training. `max_steps=20` means the run will stop quickly.
- When using PEFT/LoRA, only the adapter weights are trained and saved; the base model remains frozen.

### 9) Loading the fine-tuned adapter and running inference

After training, the notebook demonstrates how to load the base model (with the same quantization config) and then load the adapter using:

```python
from peft import PeftModel
modelFinetuned = PeftModel.from_pretrained(base_model, "finetunedModel/checkpoint-20")
```

For inference the notebook tokenizes a prompt and calls `modelFinetuned.generate(...)` to produce output. Remember to set `model.config.use_cache=False` during training but you can enable caching for faster generation at inference time.

### 10) Saving to Google Drive (optional)

Example cells show how to mount Google Drive in Colab and save the tokenizer and model weights to a Drive path. In a local environment, you can save to any accessible path via `model.save_pretrained(path)` and `tokenizer.save_pretrained(path)`.

## Requirements / Recommended environment

- Linux or Windows with a recent NVIDIA GPU. For 4-bit training: 40GB+ GPU memory is recommended (A100-40/80GB, A6000, RTX 4090 with mitigation and careful tuning).
- CUDA-compatible drivers for `bitsandbytes`.
- Python 3.10+ is recommended.

Install minimal Python dependencies:

```powershell
pip install peft accelerate bitsandbytes transformers datasets huggingface_hub GPUtil
```

If you plan to run large training, install `accelerate` and configure it:

```powershell
pip install accelerate
accelerate config
```

## How to run (local, high-level)

1. Create and activate a Python virtual environment.
2. Install dependencies (see section above).
3. Authenticate with Hugging Face (if needed):

```powershell
huggingface-cli login
```

4. Open the notebook in Jupyter and run cells sequentially.

Or convert notebook steps to a Python script and run.

## Troubleshooting & tips

- Out of memory errors: reduce `per_device_train_batch_size`, enable gradient checkpointing (already in notebook), or use larger GPUs.
- bitsandbytes import/initialization errors: ensure the CUDA driver and toolkit are compatible with the installed `bitsandbytes` wheel. For Windows, `bitsandbytes` support can be more limited — check the library docs.
- Tokenization mismatch: ensure the tokenizer and model share the same vocab and special tokens. The notebook adds `pad_token` if missing.
- Loss only on response (recommended): for supervised fine-tuning, mask the input portion so loss is computed only on the target tokens. The notebook uses the naive concatenation approach.

## Suggestions & next steps (improvements)

- Add proper instruction-style prompting and label masks so loss is computed only on target tokens.
- Use an `Accelerate` config for multi-GPU or offloading to CPU to support bigger batch sizes.
- Save only the LoRA adapter with `peft` utilities and share the adapter separately from the base model.
- Add evaluation metrics for validation set and generate-samples logging.

## Quick concept glossary

- Quantization (4-bit / nf4): reduces precision of weights to reduce VRAM usage. `nf4` is a scheme optimized for LLMs.
- bfloat16: floating point format with large dynamic range useful for mixed precision compute.
- PEFT / LoRA: add low-rank trainable adapters to a frozen base model. Greatly reduces trainable parameters and memory use.
- Gradient checkpointing: trades compute for memory by recomputing activations on the backward pass.
- DataCollatorForLanguageModeling(mlm=False): prepares inputs for causal LM training.

## License / attribution

This notebook uses Meta's Llama-2 model weights and multiple datasets published on Hugging Face. Follow the license terms for each model/dataset you use. This repository is a demonstration and does not include model weights.

## Contact / questions

If you want the README extended with explicit command-line scripts, a reproducible training script, or masking logic (loss masking for supervised fine-tuning), tell me which format you prefer (script vs notebook) and target hardware and I will provide it.
