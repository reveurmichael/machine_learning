import os

USE_HF_MIRROR_ENDPOINT = 1
if USE_HF_MIRROR_ENDPOINT == 1:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # or, you can put on the terminal: export HF_ENDPOINT=https://hf-mirror.com
else:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TF"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()
torch._dynamo.disable()

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── UPDATE THIS ────────────────────────────────────────────────────────────
# Point to your fine‑tuned adapter folder:
checkpoint_dir = "/home/utseus22/machine_learning/llm-play-snake-game-v4-Extensions/extensions/fine-tune-v0.01/finetuned_snake/deepseek-r1-qwen-7b_20250714_150706/checkpoint-53450/"
# Base model ID (must match what you fine‑tuned against):
base_model_id  = "deepseek-ai/deepseek-r1-distill-qwen-7b"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)

# 2. Load base model
base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="cuda:0",
   # attn_implementation="eager"
   # offload_folder="offload",       # optional: offload to disk if you run OOM
   # offload_state_dict=True,
)

# 3. Wrap with your LoRA adapter
model = PeftModel.from_pretrained(
    base,
    checkpoint_dir,
    device_map="cuda:0",
)

model.eval()

# 4. Prepare your prompt
prompt = """You are playing Snake on a 10x10 grid. The coordinate system is (0,0) at bottom-left to (9,9) at top-right. Movement: UP=y+1, DOWN=y-1, RIGHT=x+1, LEFT=x-1.

Current game state:
- Snake head position: (3, 3)
- Apple position: (2, 3)
- Snake body positions: [[4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [8, 7], [8, 8], [8, 9], [7, 9], [6, 9], [5, 9], [4, 9], [3, 9]]
- Snake length: 19

What is the best move to make? Consider:
1. Path to the apple
2. Avoiding collisions with walls and snake body
3. Maximizing score and survival

Choose from: UP, DOWN, LEFT, RIGHT
"""

# 5. Tokenize + generate
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,    # 不要太大，先安全测试
        temperature=0.0,       
        do_sample=False,       # 先不采样，排除 multinomial 错误
    )

# 6. Decode & print
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
