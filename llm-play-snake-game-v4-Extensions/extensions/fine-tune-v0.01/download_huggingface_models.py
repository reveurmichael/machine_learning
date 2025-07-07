from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Use public instruction-tuned models only
model_choices = {
    "mistral-7b-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
}

# Choose models to download
models_to_download = [
    "mistral-7b-instruct-v0.2",
]

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
