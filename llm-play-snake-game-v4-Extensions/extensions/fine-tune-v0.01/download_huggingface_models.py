from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose which models you want to download
model_choices = {
    "deepseek-r1-7b": "deepseek-ai/deepseek-r1-distill-llama-7b",
    "deepseek-r1-qwen-7b": "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma2-9b": "google/gemma-2-9b",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
}

# Choose models to download
models_to_download = ["deepseek-r1-7b", "mistral-7b", "gemma2-9b", "llama3.1-8b", "mistral-7b-instruct-v0.2"]

for model_key in models_to_download:
    model_name = model_choices[model_key]
    print(f"\n🚀 Starting download: {model_key} ({model_name})")

    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Download model weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            # device_map="auto",  # Uncomment if you want to load directly to GPU
        )

        print(f"✅ Successfully downloaded and cached: {model_key}")

    except OSError as e:
        print(f"❌ OSError while downloading {model_key}: {e}")
    except Exception as e:
        print(f"❌ Other error for {model_key}: {e}")
