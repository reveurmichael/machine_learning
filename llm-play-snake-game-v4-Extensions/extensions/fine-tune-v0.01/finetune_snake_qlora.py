## Example: DON'T REMOVE THIS COMMENT
# python finetune_snake_qlora.py --model llama3.1-8b --data /home/utseus22/machine_learning/llm-play-snake-game-v4-Extensions/logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250708_010930/bfs/BFS_dataset.jsonl

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
        choices=list(get_supported_models().keys()) + ["all"],
        default="gemma2-9b",
        help="Model to fine-tune (use 'all' to train all supported models)",
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
    model_name = model_map[args.model]

    try:
        # tokenzier and model
        print(f"Loading tokenizer for {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model {args.model}...")
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
        print("Preparing model for QLoRA training...")
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
        print(f"Loading dataset from {args.data}...")
        dataset = load_dataset("json", data_files=args.data)

        def tokenize_fn(batch):
            inputs = batch["prompt"] + batch["completion"]
            return tokenizer(
                inputs, truncation=True, max_length=args.max_length, padding="max_length"
            )

        print("Tokenizing dataset...")
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

        print(f"Starting training for {args.model}...")
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Training complete. Models saved to {output_dir}")
        
    except Exception as e:
        print(f"Error training model {args.model}: {str(e)}")
        print(f"Skipping {args.model} and continuing...")
        return False
    
    return True


if __name__ == "__main__":
    args = parse_args()
    
    # If a specific model is specified, train only that one
    if args.model != "all":
        success = main()
        if not success:
            print(f"Failed to train {args.model}")
            exit(1)
    else:
        # Train all supported models sequentially
        model_map = get_supported_models()
        successful_models = []
        failed_models = []
        
        for model_name in model_map.keys():
            print(f"\n{'='*50}")
            print(f"Training model: {model_name}")
            print(f"{'='*50}")
            
            # Temporarily set the model argument
            original_model = args.model
            args.model = model_name
            
            try:
                success = main()
                if success:
                    successful_models.append(model_name)
                    print(f"✓ Successfully trained {model_name}")
                else:
                    failed_models.append(model_name)
                    print(f"✗ Failed to train {model_name}")
            except Exception as e:
                failed_models.append(model_name)
                print(f"✗ Exception during training of {model_name}: {str(e)}")
            
            # Restore original model argument
            args.model = original_model
        
        # Print summary
        print(f"\n{'='*50}")
        print("TRAINING SUMMARY")
        print(f"{'='*50}")
        print(f"Successful models: {successful_models}")
        print(f"Failed models: {failed_models}")
        print(f"Total successful: {len(successful_models)}/{len(model_map)}")
        
        if failed_models:
            print(f"\nFailed models: {failed_models}")
            exit(1)
