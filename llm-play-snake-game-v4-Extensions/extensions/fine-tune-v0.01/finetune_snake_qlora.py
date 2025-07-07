import argparse
import json
import os
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb


# Supported models and their Hugging Face identifiers
def get_supported_models():
    return {
        "deepseek-r1-7b": "deepseek/deepseek-r1-7b",
        "mistral-7b": "mistralai/Mistral-7B",
        "gemma2-9b": "google/gemma-2-9b",
        "llama3.1-8b": "meta-llama/Llama-3-8B",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune snake game LLM with QLoRA (4-bit) on Snake JSONL dataset"
    )
    parser.add_argument(
        "--model",
        choices=get_supported_models().keys(),
        default="gemma2-9b",
        help="Model to fine-tune",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to snake game JSONL dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_snake",
        help="Directory to save adapters and checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--accumulation", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X steps"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_map = get_supported_models()
    hf_id = model_map[args.model]

    # Local Ollama GGUF paths
    LOCAL_PATHS = {
        "deepseek-r1-7b": "~/.ollama/models/deepseek-r1-7b/gguf/deepseek-r1-7b.gguf",
        "mistral-7b": "~/.ollama/models/mistral-7b/gguf/mistral-7b.gguf",
        "gemma2-9b": "~/.ollama/models/gemma2-9b/gguf/gemma2-9b.gguf",
        "llama3.1-8b": "~/.ollama/models/llama3.1-8b/gguf/llama3.1-8b.gguf",
    }

    # Decide model source: local GGUF or Hugging Face
    if args.model in LOCAL_PATHS:
        model_src = LOCAL_PATHS[args.model]
        print(f"Loading model '{args.model}' from local path: {model_src}")
        model = AutoModelForCausalLM.from_pretrained(
            model_src, trust_remote_code=True, device_map="auto"
        )
    else:
        print(f"Loading model '{args.model}' from Hugging Face: {hf_id}")
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

    # Tokenizer always from HF (public)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # Prepare for QLoRA if model loaded from HF
    if args.model not in LOCAL_PATHS:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset("json", data_files=args.data)

    def tokenize_fn(batch):
        inputs = batch["prompt"] + batch["completion"]
        return tokenizer(
            inputs, truncation=True, max_length=args.max_length, padding="max_length"
        )

    tokenized = dataset["train"].map(
        tokenize_fn, batched=True, remove_columns=["prompt", "completion"]
    )

    # Training args
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Models saved to {output_dir}")

    args = parse_args()
    model_map = get_supported_models()
    model_name = model_map[args.model]

    # tokenzier and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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

    # prepare for QLoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # load dataset
    dataset = load_dataset("json", data_files=args.data)

    def tokenize_fn(batch):
        inputs = batch["prompt"] + batch["completion"]
        return tokenizer(
            inputs, truncation=True, max_length=args.max_length, padding="max_length"
        )

    tokenized = dataset["train"].map(
        tokenize_fn, batched=True, remove_columns=["prompt", "completion"]
    )

    # training args
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Models saved to {output_dir}")


if __name__ == "__main__":
    main()
