## Example: DON'T REMOVE THIS COMMENT
# python finetune_snake_qlora.py --model gemma2-9b --data /home/utseus22/machine_learning/llm-play-snake-game-v4-Extensions/logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250708_010930/bfs/BFS_dataset.jsonl

import os

# Completely disable TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TF"] = "1"

import argparse
import json
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb


# Supported models and their Hugging Face identifiers
def get_supported_models():
    return {
        "deepseek-r1-7b": "deepseek-ai/deepseek-r1-distill-llama-7b",
        "deepseek-r1-qwen-7b": "deepseek-ai/deepseek-r1-distill-qwen-7b",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "gemma2-9b": "google/gemma-2-9b",
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune snake game LLM with LoRA (no TensorFlow) on Snake JSONL dataset"
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
        "--epochs", type=int, default=2, help="Number of training epochs (reduced for Snake task)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--accumulation", type=int, default=4, help="Gradient accumulation steps (reduced for better updates)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (reduced for stability)")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length (increased for Snake reasoning)"
    )
    parser.add_argument(
        "--save_steps", type=int, default=250, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--lora_r", type=int, default=32, help="LoRA rank (increased for better capacity)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=64, help="LoRA alpha (scaled with rank)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_map = get_supported_models()
    model_name = model_map[args.model]

    try:
        # Tokenizer and model
        print(f"Loading tokenizer for {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Loading model {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Prepare for QLoRA
        print("Preparing model for QLoRA training...")
        model = prepare_model_for_kbit_training(model)
        
        # Enhanced LoRA config for Snake game reasoning
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()

        # Load dataset
        print(f"Loading dataset from {args.data}...")
        dataset = load_dataset("json", data_files=args.data)

        def tokenize_fn(batch):
            """
            Optimized tokenization for Snake game prompt-completion format.
            Focuses on learning to generate the completion given the prompt.
            """
            # Combine prompt and completion with proper formatting
            full_text = []
            for prompt, completion in zip(batch["prompt"], batch["completion"]):
                # Format: prompt + completion, with attention only on completion for loss
                full_text.append(prompt + completion)
            
            # Tokenize the full text
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_length,
                padding=False,  # Let data collator handle padding
                return_tensors=None
            )
            
            # Create labels for causal language modeling
            # We want to predict the completion part only
            labels = []
            for i, (prompt, completion) in enumerate(zip(batch["prompt"], batch["completion"])):
                # Tokenize prompt separately to find where completion starts
                prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full_tokens = tokenized["input_ids"][i]
                
                # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for completion
                label = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
                
                # Pad/truncate to match input length
                if len(label) > len(full_tokens):
                    label = label[:len(full_tokens)]
                elif len(label) < len(full_tokens):
                    label.extend([-100] * (len(full_tokens) - len(label)))
                
                labels.append(label)
            
            tokenized["labels"] = labels
            return tokenized

        print("Tokenizing dataset...")
        tokenized = dataset["train"].map(
            tokenize_fn, 
            batched=True, 
            remove_columns=["prompt", "completion"],
            desc="Tokenizing"
        )

        # Training args optimized for Snake game
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
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            dataloader_drop_last=True,
            remove_unused_columns=False,
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=tokenized,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print(f"Starting training for {args.model}...")
        print(f"Dataset size: {len(tokenized)} examples")
        print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * args.accumulation}")
        
        trainer.train()
        
        # Save model and tokenizer
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training config for reference
        config_file = os.path.join(output_dir, "training_config.json")
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2)
        
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
