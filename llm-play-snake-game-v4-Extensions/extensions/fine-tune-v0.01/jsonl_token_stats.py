import argparse
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Supported models mapping
model_choices = {
    "deepseek-r1-llama-7b": "deepseek-ai/deepseek-r1-distill-llama-7b",
    "deepseek-r1-qwen-7b": "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma2-9b": "google/gemma-2-9b",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma-3n-e4b-it": "google/gemma-3n-e4b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "deepscaler-1.5b": "agentica-org/DeepScaleR-1.5B-Preview",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate token statistics for JSONL dataset entries using LLM tokenizer."
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to your JSONL file (each line must contain 'prompt' and 'completion')."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-4b-it",
        choices=list(model_choices.keys()),
        help="Model name to determine tokenizer (default: gemma-3-4b-it)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = model_choices[args.model]
    print(f"Loading tokenizer for model: {args.model} ({model_name})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    token_counts = []

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing entries"):
            data = json.loads(line)
            text = data["prompt"] + data["completion"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts.append(len(tokens))

    token_counts = np.array(token_counts)

    print("\n📊 Token Statistics:")
    print(f"Total entries: {len(token_counts)}")
    print(f"Min tokens: {token_counts.min()}")
    print(f"Max tokens: {token_counts.max()}")
    print(f"Mean tokens: {token_counts.mean():.2f}")
    print(f"Median tokens: {np.median(token_counts):.2f}")
    print(f"25th percentile: {np.percentile(token_counts, 25):.2f}")
    print(f"75th percentile: {np.percentile(token_counts, 75):.2f}")
    print(f"Std deviation: {token_counts.std():.2f}")

if __name__ == "__main__":
    main()
