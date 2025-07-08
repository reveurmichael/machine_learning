import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

model_choices = {
    "deepseek-r1-7b": "deepseek-ai/deepseek-r1-distill-llama-7b",
    "deepseek-r1-qwen-7b": "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma2-9b": "google/gemma-2-9b",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models using official or mirror endpoint."
    )
    parser.add_argument(
        "--endpoint",
        choices=["official", "mirror"],
        default="official",
        help="Choose 'official' to use huggingface.co or 'mirror' to use hf-mirror.com",
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

    if args.endpoint == "mirror":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"

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
