from __future__ import annotations

import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

import streamlit as st
from extensions.distillation_v0_03 import DistilConfig, EvaluationSuite  # noqa: E402

st.set_page_config(page_title="LLM Distillation v0.03", layout="wide")
st.title("🔄 Knowledge Distillation – v0.03")

# Sidebar config
st.sidebar.header("Distillation Config")
teacher_model = st.sidebar.text_input("Teacher model / path")
student_model = st.sidebar.text_input("Student base model", value="google/gemma-2b")
output_dir = st.sidebar.text_input("Output directory", value="logs/extensions/models/distill_v0.03/")

uploaded = st.sidebar.file_uploader("Datasets (.jsonl)", accept_multiple_files=True, type="jsonl")

epochs = st.sidebar.slider("Epochs", 1, 5, 3)
batch = st.sidebar.selectbox("Batch size", [2, 4, 8], index=1)
lr = st.sidebar.number_input("Learning rate", 1e-6, 1e-3, 5e-5, format="%e")
alpha = st.sidebar.slider("Alpha (CE weight)", 0.0, 1.0, 0.1)
T = st.sidebar.slider("Temperature", 1.0, 5.0, 2.0)
use_lora = st.sidebar.checkbox("LoRA student", value=True)
lora_rank = st.sidebar.slider("LoRA rank", 4, 64, 8)

if st.sidebar.button("Save config", disabled=not (teacher_model and uploaded)):
    out_dir = Path(output_dir)
    (out_dir / "datasets").mkdir(parents=True, exist_ok=True)
    dataset_paths = []
    for uf in uploaded:
        dest = out_dir / "datasets" / uf.name
        dest.write_bytes(uf.getbuffer())
        dataset_paths.append(dest)

    cfg = DistilConfig(
        teacher_path=teacher_model,
        student_model=student_model,
        dataset_paths=dataset_paths,
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_batch_size=batch,
        learning_rate=lr,
        alpha=alpha,
        temperature=T,
        use_lora_student=use_lora,
        lora_rank=lora_rank,
    )
    st.session_state["distil_cfg"] = cfg
    st.success("Config saved. Run distillation via CLI:")
    st.code(
        f"python -m extensions.distillation_v0_03.cli train "
        f"--teacher {teacher_model} --student {student_model} "
        f"--datasets {' '.join(str(p) for p in dataset_paths)} "
        f"--output {out_dir} --epochs {epochs} --batch {batch} "
        f"--alpha {alpha} --temperature {T} {'--no-lora-student' if not use_lora else ''}"
    )

# Evaluation section
st.header("Quick Perplexity Eval")
model_dir_eval = st.text_input("Distilled model directory")
if st.button("Compute PPL"):
    if not model_dir_eval or not uploaded:
        st.warning("Provide model dir & dataset(s)")
    else:
        suite = EvaluationSuite(Path(model_dir_eval), [Path(p) for p in dataset_paths])
        ppl = suite.perplexity()
        st.success(f"Perplexity: {ppl:.2f}") 