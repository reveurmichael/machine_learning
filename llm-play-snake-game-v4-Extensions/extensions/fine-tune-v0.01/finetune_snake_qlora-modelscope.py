# DON'T REMOVE THIS COMMENT
# DON'T REMOVE THIS COMMENT

# 512 tokens, 200 games
# python finetune_snake_qlora-modelscope.py --model llama3.1-8b --data /home/utseus22/machine_learning/llm-play-snake-game-v4-Extensions/logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250712_152139/bfs-512/BFS-512_dataset.jsonl --max_length 512 

# Important information that should NEVER BE REMOVED:
# 1. The script is for torch 2.3
# 2. By default, the script is for 4-bit quantization. Though 16-bit quantization is also supported with argparse.

# ANY OF THE ABOVE COMMENTS SHOULD NEVER BE REMOVED

import os

USE_HF_MIRROR_ENDPOINT = 1
if USE_HF_MIRROR_ENDPOINT == 1:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # or, you can put on the terminal: export HF_ENDPOINT=https://hf-mirror.com
else:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TF"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


import argparse
import json
import warnings
from datetime import datetime
from typing import Dict, List
import inspect

# 🚨 CRITICAL: Patch torch checkpoint BEFORE any ML libraries import it
import torch
import torch.utils.checkpoint

# Suppress specific warnings for cleaner logs
warnings.filterwarnings("ignore", message=".*use_cache=True.*incompatible.*gradient checkpointing.*")

# Aggressive patch to eliminate all checkpoint warnings
original_checkpoint = torch.utils.checkpoint.checkpoint

def patched_checkpoint(fn, *args, **kwargs):
    kwargs['use_reentrant'] = False
    return original_checkpoint(fn, *args, **kwargs)

torch.utils.checkpoint.checkpoint = patched_checkpoint

# Enable CuDNN benchmark for optimized kernel selection and potential memory improvements, especially under WSL2
torch.backends.cudnn.benchmark = True

from peft import prepare_model_for_kbit_training as original_prepare

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# =====================
# MODEL CONFIGURATION
# =====================
# 🎯 Centralized model configuration for easy maintenance and model-specific optimizations
MODEL_CONFIGS = {
    "llama3.1-8b": {
        "model_name": "/home/utseus22/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct",
        "use_attn_eager": False,
    },
    "mistral-7b": {
        "model_name": "/home/utseus22/.cache/modelscope/hub/models/rubraAI/Mistral-7B-Instruct-v0.3",
        "use_attn_eager": False,
    },
}

def get_supported_models() -> Dict[str, str]:
    return {key: value["model_name"] for key, value in MODEL_CONFIGS.items()}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune snake game LLM with LoRA (4-bit DEFAULT, no TensorFlow) on Snake JSONL dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=list(get_supported_models().keys()) + ["all"],
        default="gemma2-9b",
        help="Model to fine-tune (use 'all' to train all supported models)"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to snake game JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./finetuned_snake", help="Directory to save adapters and checkpoints")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (reduced for Snake task)")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--accumulation", type=int, default=1, help="Gradient accumulation steps (set to 1 for minimal memory footprint)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (reduced for stability)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length (lowered to reduce memory usage)")
    parser.add_argument("--save_steps", type=int, default=250, help="Save checkpoint every X steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (reduced to save GPU memory)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaled with reduced rank)")
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization and use 16-bit precision (uses more memory)"
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=8,
        help="Pad sequences to a multiple of this value (default: 8, advanced override)",
    )
    return parser.parse_args()

# =====================
# MODEL LOADING & PREP
# =====================
def load_model_and_tokenizer(model_key: str, use_4bit: bool = True):
    model_cfg = MODEL_CONFIGS[model_key]
    model_name = model_cfg["model_name"]
    use_attn_eager = model_cfg.get("use_attn_eager", False)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to EOS token: {tokenizer.pad_token}")
    print(f"Loading model {model_name}...")
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_args = {}
    if use_attn_eager:
        attn_args["attn_implementation"] = "eager"
        print("Using eager attention (recommended for Gemma2)!")
    
    if use_4bit:
        print("Loading model in 4-bit mode (QLoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            **attn_args
        )
    else:
        print("Loading model in 16-bit mode (no device_map to prevent meta tensor issues)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch.float16,
            load_in_4bit=False,
            trust_remote_code=True,
            **attn_args
        )
        print("Safely transferring model to GPU using .to_empty()...")
        model = model.to_empty(device=torch.device("cuda"))
    
    # 🧹 WSL2 GPU memory management: Clear cache and reduce fragmentation
    print("🧹 Clearing GPU cache and optimizing memory for WSL2...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Ensure all operations complete
    
    # Report memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 GPU Memory after model load: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return model, tokenizer

def patched_prepare_model_for_kbit_training(model, *args, **kwargs):
    sig = inspect.signature(original_prepare)
    if "use_reentrant" in sig.parameters and "use_reentrant" not in kwargs:
        kwargs["use_reentrant"] = False
    return original_prepare(model, *args, **kwargs)

import peft
peft.prepare_model_for_kbit_training = patched_prepare_model_for_kbit_training

def prepare_model_for_training(model, use_4bit: bool = True):
    print("Preparing model for LoRA training...")
    if use_4bit:
        print("Applying k-bit training preparation for 4-bit quantization...")
        model = patched_prepare_model_for_kbit_training(model)
    else:
        print("Enabling gradient checkpointing for 16-bit memory efficiency...")
        model.gradient_checkpointing_enable(use_reentrant=False)
    return model

def detect_lora_target_modules(model) -> List[str]:
    print("Detecting LoRA target modules automatically...")
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split(".")[-1]
            target_modules.add(module_name)
    target_modules = list(target_modules)
    if not target_modules:
        print("No Linear modules detected, using common fallback...")
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 🚨 Exclude lm_head to avoid tied embedding issues
    if "lm_head" in target_modules:
        print("Removing 'lm_head' from LoRA target modules to avoid tied embedding issues.")
        target_modules.remove("lm_head")
    
    print(f"Detected target LoRA modules: {target_modules}")
    return target_modules

def create_lora_config(args, target_modules: List[str]) -> LoraConfig:
    print(f"Creating LoRA config with rank={args.lora_r}, alpha={args.lora_alpha}")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

# =====================
# DATA PROCESSING
# =====================
def create_tokenization_function(tokenizer, max_length: int):
    def tokenize_fn(batch):
        full_text = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        labels = []
        for i, (prompt, completion) in enumerate(zip(batch["prompt"], batch["completion"])):
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_tokens = tokenized["input_ids"][i]
            label = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
            if len(label) > len(full_tokens):
                label = label[:len(full_tokens)]
            elif len(label) < len(full_tokens):
                label.extend([-100] * (len(full_tokens) - len(label)))
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized
    return tokenize_fn

def load_and_tokenize_dataset(data_path: str, tokenizer, max_length: int):
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path)
    tokenize_fn = create_tokenization_function(tokenizer, max_length)
    print("Tokenizing dataset...")
    tokenized = dataset["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=["prompt", "completion"],
        desc="Tokenizing"
    )
    print(f"Dataset tokenization complete. Size: {len(tokenized)} examples")
    return tokenized

# =====================
# TRAINING CONFIGURATION
# =====================
def create_training_arguments(args, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
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
        dataloader_num_workers=0,  # 🚨 Disable multiprocessing for WSL2 memory efficiency
    )

def create_data_collator(tokenizer, training_args, pad_to_multiple_of) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of if training_args.fp16 or training_args.fp16 is None else None,
    )

# =====================
# MAIN TRAINING FUNCTION
# =====================
def main() -> bool:
    args = parse_args()
    try:
        use_4bit = not args.no_4bit
        print(f"\n{'='*60}")
        print(f"🚀 STARTING ROBUST FINE-TUNING FOR MODEL: {args.model}")
        print(f"{'='*60}")
        print(f"Quantization mode: {'4-bit (QLoRA)' if use_4bit else '16-bit'}")
        model, tokenizer = load_model_and_tokenizer(args.model, use_4bit)
        model = prepare_model_for_training(model, use_4bit)
        target_modules = detect_lora_target_modules(model)
        lora_config = create_lora_config(args, target_modules)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        tokenized_dataset = load_and_tokenize_dataset(args.data, tokenizer, args.max_length)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
        training_args = create_training_arguments(args, output_dir)
        data_collator = create_data_collator(tokenizer, training_args, args.pad_to_multiple_of)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Dataset size: {len(tokenized_dataset)} examples")
        print(f"Training epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * args.accumulation}")
        print(f"Learning rate: {args.lr}")
        print(f"LoRA rank: {args.lora_r}")
        print(f"Quantization: {'4-bit (QLoRA)' if use_4bit else '16-bit'}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        print("Starting training...")
        
        # 🎯 Calculate training progress info
        total_steps = len(tokenized_dataset) // (args.batch_size * args.accumulation) * args.epochs
        print(f"📊 Training Progress: {total_steps} total steps expected")
        print(f"⏱️  Estimated time: ~{total_steps * 2:.0f} seconds (~{total_steps * 2 / 3600:.1f} hours)")
        print(f"💾 Checkpoints will be saved every {args.save_steps} steps")
        
        trainer.train()
        print(f"\nSaving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        config_file = os.path.join(output_dir, "training_config.json")
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2)
        adapter_dir = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(adapter_dir, safe_serialization=True)
        print(f"✅ LoRA adapter saved to {adapter_dir}")
        print(f"✅ Training complete! Model saved to {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Error training model {args.model}: {str(e)}")
        print(f"Skipping {args.model} and continuing...")
        return False

if __name__ == "__main__":
    args = parse_args()
    if args.model != "all":
        success = main()
        if not success:
            print(f"❌ Failed to train {args.model}")
            exit(1)
        else:
            print(f"✅ Successfully trained {args.model}")
    else:
        print(f"\n{'='*60}")
        print("BATCH TRAINING MODE - Training all supported models")
        print(f"{'='*60}")
        model_map = get_supported_models()
        successful_models = []
        failed_models = []
        for model_name in model_map.keys():
            print(f"\n{'='*60}")
            print(f"TRAINING MODEL: {model_name}")
            print(f"{'='*60}")
            original_model = args.model
            args.model = model_name
            try:
                success = main()
                if success:
                    successful_models.append(model_name)
                    print(f"✅ Successfully trained {model_name}")
                else:
                    failed_models.append(model_name)
                    print(f"❌ Failed to train {model_name}")
            except Exception as e:
                failed_models.append(model_name)
                print(f"❌ Exception during training of {model_name}: {str(e)}")
            args.model = original_model
        print(f"\n{'='*60}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful models: {successful_models}")
        print(f"❌ Failed models: {failed_models}")
        print(f"Success rate: {len(successful_models)}/{len(model_map)} ({len(successful_models)/len(model_map)*100:.1f}%)")
        if failed_models:
            print(f"\nFailed models: {failed_models}")
            exit(1)
        else:
            print("\nAll models trained successfully!")
