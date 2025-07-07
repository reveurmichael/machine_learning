## Example: DON'T REMOVE THIS COMMENT
# python finetune_snake_qlora.py --model llama3.1-8b --data /home/utseus22/machine_learning/llm-play-snake-game-v4-Extensions/logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250708_010930/bfs/BFS_dataset.jsonl


import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb


def get_supported_models():
    return {
        "deepseek-r1-7b": "deepseek/deepseek-r1-7b",
        "mistral-7b": "mistralai/Mistral-7B",
        "gemma2-9b": "google/gemma-2-9b",
        "llama3.1-8b": "meta-llama/Llama-3-8B",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Snake game LLM with QLoRA (4-bit) on Snake JSONL dataset"
    )
    parser.add_argument(
        "--model",
        choices=get_supported_models().keys(),
        default="gemma2-9b",
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to snake game JSONL dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./finetuned_snake",
        help="Directory to save adapters and checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Maximum sequence length for tokens"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every X steps"
    )
    return parser.parse_args()


def prepare_local_model(model_key: str, hf_id: str) -> str:
    """
    Export the Ollama model to a Hugging Face transformers format directory.
    Returns the path to the exported folder.
    """
    models_root = os.path.expanduser("~/.ollama/models")
    export_dir = os.path.join(models_root, model_key + "_hf")
    # Clean up previous export
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    print(f"Exporting '{model_key}' via Ollama to transformers at {export_dir}...")
    subprocess.run([
        "ollama", "export", "--format", "transformers",
        model_key, export_dir
    ], check=True)
    return export_dir


def main():
    args = parse_args()
    model_map = get_supported_models()
    hf_id = model_map[args.model]

    # LOCAL_PATHS = GGUF directory, not used directly now
    LOCAL_KEYS = set(get_supported_models().keys())

    if args.model in LOCAL_KEYS:
        # Export via Ollama to HF-compatible directory
        model_src = prepare_local_model(args.model, hf_id)
        print(f"Loading '{args.model}' from exported transformers folder: {model_src}")
        model = AutoModelForCausalLM.from_pretrained(
            model_src,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True
        )
    else:
        print(f"Loading '{args.model}' from HF: {hf_id}")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )

    # Always load tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # QLoRA wrappers only when loaded from HF
    if args.model not in LOCAL_KEYS:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Load and tokenize
    ds = load_dataset("json", data_files=args.data)
    def tokenize_fn(batch):
        text = batch['prompt'] + batch['completion']
        return tokenizer(text, truncation=True, max_length=args.max_length, padding='max_length')
    tokenized = ds['train'].map(tokenize_fn, batched=True, remove_columns=['prompt','completion'])

    # Training args
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"{args.model}_{ts}")
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to='none'
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Training complete. Models saved to {out_dir}")

if __name__ == '__main__':
    main()


