import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── UPDATE THIS ────────────────────────────────────────────────────────────
# Point to your fine‑tuned adapter folder:
checkpoint_dir = "/home/utseus22/machine_learning/llm-play-snake-game-v4-Extensions/extensions/fine-tune-v0.01/finetuned_snake/gemma-3-4b-it_20250712_152346/"
# Base model ID (must match what you fine‑tuned against):
base_model_id  = "google/gemma-3-4b-it"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)

# 2. Load base model
base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
   # offload_folder="offload",       # optional: offload to disk if you run OOM
   # offload_state_dict=True,
)

# 3. Wrap with your LoRA adapter
model = PeftModel.from_pretrained(
    base,
    checkpoint_dir,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()

# 4. Prepare your prompt
prompt = """You are playing Snake on a 10x10 grid. The coordinate system is (0,0) at bottom-left to (9,9) at top-right. Movement: UP=y+1, DOWN=y-1, RIGHT=x+1, LEFT=x-1.\n\nCurrent game state:\n- Snake head position: (3, 3)\n- Apple position: (2, 3)\n- Snake body positions: [[4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [8, 7], [8, 8], [8, 9], [7, 9], [6, 9], [5, 9], [4, 9], [3, 9]]\n- Snake length: 19\n\nWhat is the best move to make? Consider:\n1. Path to the apple\n2. Avoiding collisions with walls and snake body\n3. Maximizing score and survival\n\nChoose from: UP, DOWN, LEFT, RIGHT
"""

# 5. Tokenize + generate
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )

# 6. Decode & print
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
