from transformers import AutoTokenizer, AutoModelForCausalLM

# Define model name mappings
model_choices = {
    "deepseek-r1-7b": "deepseek/deepseek-r1-7b",
    "mistral-7b": "mistralai/Mistral-7B",
    "gemma2-9b": "google/gemma-2-9b",
    "llama3.1-8b": "meta-llama/Llama-3-8B",
}

# Choose which one(s) to download — you can loop or select
models_to_download = ["deepseek-r1-7b", "mistral-7b", "gemma2-9b", "llama3.1-8b"]

for model_key in models_to_download:
    model_name = model_choices[model_key]
    print(f"Downloading {model_key}: {model_name}")

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Download model weights (default FP32 or 16-bit if specified)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # optionally force to CPU to avoid OOM if no large GPU
        device_map="cpu",
    )

    print(f"✅ Finished downloading {model_key}\n")


