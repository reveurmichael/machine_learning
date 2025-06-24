from __future__ import annotations

import sys
import os
from pathlib import Path

# Fix Python path for Streamlit execution
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

import streamlit as st
from extensions.llm_finetune_v0_03 import FineTuneConfig, EvaluationSuite  # noqa: E402

st.set_page_config(page_title="LLM Fine-tune v0.03", layout="wide")
st.title("🚀 LLM Fine-tuning – v0.03")

# --------------------------------------------------
# Sidebar – configuration
# --------------------------------------------------
st.sidebar.header("Configuration")
model_name = st.sidebar.text_input("Base model", value="mistralai/Mistral-7B-v0.1")
output_dir = st.sidebar.text_input("Output directory", value="logs/extensions/models/finetune_v0.03/")

uploaded_files = st.sidebar.file_uploader("Upload JSONL dataset(s)", accept_multiple_files=True, type="jsonl")

epochs = st.sidebar.slider("Epochs", 1, 3, 1)
batch_size = st.sidebar.selectbox("Batch size", [1, 2, 4, 8], index=1)
block_size = st.sidebar.slider("Block size", 128, 2048, 512, 128)
use_lora = st.sidebar.checkbox("Use LoRA", value=True)
lora_rank = st.sidebar.slider("LoRA rank", 4, 64, 8)

if st.sidebar.button("Start fine-tuning", disabled=not uploaded_files):
    # Save uploaded files to a temp dir inside output_dir
    dataset_paths = []
    out_dir = Path(output_dir)
    (out_dir / "datasets").mkdir(parents=True, exist_ok=True)
    with st.spinner("Saving uploaded datasets…"):
        for uf in uploaded_files:
            dest = out_dir / "datasets" / uf.name
            dest.write_bytes(uf.getbuffer())
            dataset_paths.append(dest)

    cfg = FineTuneConfig(
        model_name_or_path=model_name,
        output_dir=out_dir,
        dataset_paths=dataset_paths,
        num_train_epochs=epochs,
        per_device_batch_size=batch_size,
        learning_rate=2e-5,
        block_size=block_size,
        use_lora=use_lora,
        lora_rank=lora_rank,
    )

    st.session_state["run_cfg"] = cfg  # Store for later evaluation

    st.success("Configuration saved – open terminal to run fine-tuning with CLI:")
    st.code(
        f"python -m extensions.llm_finetune_v0_03.cli train "
        f"--model {model_name} --datasets {' '.join(str(p) for p in dataset_paths)} "
        f"--output {out_dir} --epochs {epochs} --batch {batch_size}"
    )

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
st.header("Quick Perplexity Evaluation")
model_dir_eval = st.text_input("Fine-tuned model directory")
if st.button("Compute perplexity"):
    eval_paths = [Path(f) for f in uploaded_files] if uploaded_files else []
    if not model_dir_eval or not eval_paths:
        st.warning("Provide model directory and dataset(s) for evaluation.")
    else:
        with st.spinner("Evaluating…"):
            suite = EvaluationSuite(Path(model_dir_eval), eval_paths)
            report = suite.quick_report()
        st.success(f"Perplexity: {report['perplexity']:.2f}") 