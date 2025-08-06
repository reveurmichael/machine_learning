import os

USE_HF_MIRROR_ENDPOINT = 1

# Set HF endpoint
if USE_HF_MIRROR_ENDPOINT == 1:
    os.environ["HF_ENDPOINT"] = (
        "https://hf-mirror.com"  # or, you can put on the terminal: export HF_ENDPOINT=https://hf-mirror.com
    )

else:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["DISABLE_TF"] = "1"

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

model_choices = {
    "openai-gpt-oss-20b": "openai/gpt-oss-20b",
    "deepseek-r1-llama-7b": "deepseek-ai/deepseek-r1-distill-llama-7b",
    "deepseek-r1-qwen-7b": "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma2-9b": "google/gemma-2-9b",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    # 🆕 New Gemma 3 models for Snake game fine-tuning
    "gemma-3n-e4b-it": "google/gemma-3n-e4b-it",
    # 🆕 Additional Gemma 3 models (standard versions for fine-tuning)
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    # 🆕 New Qwen3 and DeepScaleR models
    "qwen3-4b": "Qwen/Qwen3-4B",
    "deepscaler-1.5b": "agentica-org/DeepScaleR-1.5B-Preview",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models using official or mirror endpoint."
    )

    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(model_choices.keys()),
        default=list(model_choices.keys()),
        help="List of model keys to download (default: all models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for model_key in args.models:
        model_name = model_choices[model_key]
        print(f"\n🚀 Starting download: {model_key} ({model_name})")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                # device_map="auto",
            )
            print(f"✅ Successfully downloaded and cached: {model_key}")
        except OSError as e:
            print(f"❌ OSError while downloading {model_key}: {e}")
        except Exception as e:
            print(f"❌ Other error for {model_key}: {e}")


if __name__ == "__main__":
    main()
