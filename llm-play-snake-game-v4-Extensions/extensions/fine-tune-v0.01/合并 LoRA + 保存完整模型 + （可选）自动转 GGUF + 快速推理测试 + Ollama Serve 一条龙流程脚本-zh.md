# 🚀 完整 LoRA 合并与部署流水线

一个全面的指南，用于合并 LoRA 适配器、转换为优化格式，并使用多种服务选项部署微调的大语言模型。

## 📋 目录

1. [前置条件与设置](#前置条件与设置)
2. [LoRA 合并与卸载](#lora-合并与卸载)
3. [模型转换与优化](#模型转换与优化)
4. [部署选项](#部署选项)
5. [测试与验证](#测试与验证)
6. [生产环境部署](#生产环境部署)
7. [高级技术](#高级技术)
8. [故障排除](#故障排除)

## 🛠 前置条件与设置

### 系统要求
- **GPU**: NVIDIA A100/H100（推荐）、RTX 4090+ 或 V100
- **内存**: 32GB+（大型模型需要 64GB+）
- **存储**: 500GB+ NVMe SSD
- **CUDA**: 11.8+ 或 12.1+

### 环境设置
```bash
# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes
pip install vllm ollama huggingface_hub safetensors

# 用于量化和转换
pip install auto-gptq optimum[onnxruntime-gpu]
pip install llama-cpp-python --force-reinstall --no-cache-dir

# 克隆 llama.cpp 用于 GGUF 转换
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j $(nproc)
```

## 🔗 LoRA 合并与卸载

### 方法 1: 标准 PEFT 合并
```python
#!/usr/bin/env python3
"""
高级 LoRA 合并脚本，支持多种合并策略
"""
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_lora_adapters(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    merge_strategy: str = "linear",
    weights: list = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    max_memory: dict = None
):
    """
    使用高级策略将 LoRA 适配器与基础模型合并。
    
    参数:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA 适配器路径或路径列表
        output_path: 合并模型的输出目录
        merge_strategy: 合并多个适配器的策略
        weights: 加权合并的权重
        dtype: 内存优化的模型数据类型
        device_map: 设备映射策略
        max_memory: 每个设备的最大内存
    """
    
    # 使用优化加载基础模型
    logger.info(f"从 {base_model_path} 加载基础模型")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"  # 适用于支持的模型
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 处理多个 LoRA 适配器
    if isinstance(lora_adapter_path, list):
        # 加载第一个适配器
        model = PeftModel.from_pretrained(model, lora_adapter_path[0])
        
        # 添加额外的适配器
        for i, adapter_path in enumerate(lora_adapter_path[1:], 1):
            model.load_adapter(adapter_path, adapter_name=f"adapter_{i}")
        
        # 使用指定策略合并多个适配器
        if merge_strategy in ["linear", "ties", "dare_linear", "magnitude_prune"]:
            adapter_names = [f"adapter_{i}" for i in range(len(lora_adapter_path))]
            if weights is None:
                weights = [1.0 / len(adapter_names)] * len(adapter_names)
            
            model.add_weighted_adapter(
                adapters=adapter_names,
                weights=weights,
                adapter_name="merged_adapter",
                combination_type=merge_strategy,
                density=0.7 if "prune" in merge_strategy else None
            )
            model.set_adapters("merged_adapter")
    else:
        # 单个适配器
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    # 合并并卸载
    logger.info("将 LoRA 权重与基础模型合并...")
    merged_model = model.merge_and_unload()
    
    # 清理 GPU 内存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # 保存合并后的模型
    logger.info(f"保存合并后的模型到 {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(
        output_path,
        save_safetensors=True,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(output_path)
    
    # 保存模型卡片
    model_card = f"""
---
library_name: transformers
license: apache-2.0
base_model: {base_model_path}
tags:
- fine-tuned
- lora-merged
inference: false
---

# 合并后的模型

此模型是通过将 LoRA 适配器与基础模型合并而创建的。

## 基础模型
- {base_model_path}

## LoRA 适配器
- {lora_adapter_path}

## 合并策略
- {merge_strategy}

## 使用方法
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_path}")
tokenizer = AutoTokenizer.from_pretrained("{output_path}")
```
"""
    
    with open(Path(output_path) / "README.md", "w") as f:
        f.write(model_card)
    
    logger.info("✅ LoRA 合并成功完成！")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="基础模型路径")
    parser.add_argument("--lora_adapter", required=True, help="LoRA 适配器路径")
    parser.add_argument("--output_path", required=True, help="输出目录")
    parser.add_argument("--merge_strategy", default="linear", choices=["linear", "ties", "dare_linear", "magnitude_prune"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    merge_lora_adapters(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output_path,
        merge_strategy=args.merge_strategy,
        dtype=dtype_map[args.dtype]
    )
```

### 方法 2: 高级多适配器合并
```python
def advanced_multi_adapter_merge(adapters_config: dict, output_path: str):
    """
    使用不同策略的高级合并，适用于每种适配器类型。
    
    adapters_config 示例:
    {
        "base_model": "path/to/base",
        "adapters": [
            {"path": "adapter1", "weight": 0.6, "type": "task"},
            {"path": "adapter2", "weight": 0.4, "type": "style"}
        ],
        "strategy": "ties",
        "density": 0.7
    }
    """
    # 复杂合并场景的实现
    pass
```

## ⚡ 模型转换与优化

### 为 llama.cpp 进行 GGUF 转换
```python
#!/usr/bin/env python3
"""
高级 GGUF 转换，支持多种量化选项
"""
import subprocess
import os
from pathlib import Path
import json

def convert_to_gguf(
    model_path: str,
    output_dir: str,
    quantization: str = "Q4_K_M",
    vocab_type: str = "spm",  # spm, bpe, auto
    context_length: int = 32768,
    rope_freq_base: float = None,
    rope_freq_scale: float = None
):
    """
    将 HuggingFace 模型转换为 GGUF 格式并进行优化。
    """
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型信息
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    model_name = config.get("_name_or_path", model_path.name)
    
    # 步骤 1: 转换为 GGUF FP16
    fp16_path = output_dir / f"{model_name}-fp16.gguf"
    
    convert_cmd = [
        "python", "llama.cpp/convert-hf-to-gguf.py",
        str(model_path),
        "--outfile", str(fp16_path),
        "--outtype", "f16",
        "--vocab-type", vocab_type,
        "--pad-vocab"
    ]
    
    if context_length:
        convert_cmd.extend(["--ctx", str(context_length)])
    
    if rope_freq_base:
        convert_cmd.extend(["--rope-freq-base", str(rope_freq_base)])
    
    if rope_freq_scale:
        convert_cmd.extend(["--rope-freq-scale", str(rope_freq_scale)])
    
    print(f"🔄 转换为 FP16 GGUF: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)
    
    # 步骤 2: 量化为指定格式
    if quantization != "f16":
        quantized_path = output_dir / f"{model_name}-{quantization}.gguf"
        
        quant_cmd = [
            "llama.cpp/quantize",
            str(fp16_path),
            str(quantized_path),
            quantization
        ]
        
        print(f"🔄 量化为 {quantization}: {' '.join(quant_cmd)}")
        subprocess.run(quant_cmd, check=True)
        
        # 删除 FP16 文件以节省空间
        if fp16_path.exists():
            fp16_path.unlink()
        
        final_path = quantized_path
    else:
        final_path = fp16_path
    
    print(f"✅ GGUF 转换完成: {final_path}")
    return final_path

# 量化选项及说明
QUANTIZATION_OPTIONS = {
    "Q2_K": "极小，高质量损失",
    "Q3_K_S": "小，极高质量损失",
    "Q3_K_M": "中等，高质量损失",
    "Q3_K_L": "大，高质量损失",
    "Q4_0": "传统，小，极高质量损失",
    "Q4_1": "传统，小，实质性质量损失",
    "Q4_K_S": "小，较大质量损失",
    "Q4_K_M": "中等，平衡质量/大小（推荐）",
    "Q5_0": "传统，中等，平衡质量/大小",
    "Q5_1": "传统，中等，低质量损失",
    "Q5_K_S": "大，低质量损失",
    "Q5_K_M": "大，极低质量损失（推荐）",
    "Q6_K": "极大，极低质量损失",
    "Q8_0": "极大，极低质量损失",
    "F16": "极大，无质量损失",
    "F32": "最大大小，无质量损失"
}
```

### AWQ/GPTQ 量化
```python
def quantize_with_awq(model_path: str, output_path: str, bits: int = 4):
    """
    使用 AutoAWQ 量化模型以获得最大推理速度。
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    
    # 加载模型和分词器
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, 
        safetensors=True, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 量化
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM"
        }
    )
    
    # 保存量化模型
    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ AWQ 量化完成: {output_path}")
```

## 🚀 部署选项

### 选项 1: Ollama 部署
```bash
#!/bin/bash
# ollama_deploy.sh

MODEL_PATH="$1"
MODEL_NAME="$2"
QUANTIZATION="${3:-Q4_K_M}"

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "用法: $0 <model_path> <model_name> [quantization]"
    exit 1
fi

# 创建 Modelfile
cat > Modelfile << EOF
FROM ${MODEL_PATH}

# 温度控制创造性 (0.0-2.0)
PARAMETER temperature 0.7

# Top-p 控制多样性 (0.0-1.0)
PARAMETER top_p 0.9

# Top-k 限制 token 选择 (1-100)
PARAMETER top_k 40

# 重复惩罚减少重复 (1.0-1.5)
PARAMETER repeat_penalty 1.1

# 上下文窗口大小
PARAMETER num_ctx 32768

# 停止序列
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# 系统消息
SYSTEM """你是一个在贪吃蛇游戏数据上训练的 AI 助手。你可以为贪吃蛇游戏提供策略建议，分析游戏状态，并建议最佳移动。"""

# 示例对话
MESSAGE user "贪吃蛇的最佳策略是什么？"
MESSAGE assistant "贪吃蛇的关键策略包括：1) 沿墙壁和边缘保持安全，2) 创建并维护安全路径，3) 避免在角落中困住自己，4) 提前计划几步，5) 耐心收集食物。"
EOF

# 在 Ollama 中创建模型
echo "🔄 创建 Ollama 模型..."
ollama create $MODEL_NAME -f Modelfile

# 测试模型
echo "🧪 测试模型..."
echo "什么是贪吃蛇游戏的最佳策略？" | ollama generate $MODEL_NAME

echo "✅ Ollama 部署完成！"
echo "用法: ollama run $MODEL_NAME"
```

### 选项 2: vLLM 部署
```python
#!/usr/bin/env python3
"""
生产环境 vLLM 部署脚本
"""
import argparse
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

class vLLMDeployment:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.85,
        quantization: Optional[str] = None,
        dtype: str = "bfloat16"
    ):
        self.engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=False,
            disable_log_stats=False,
            trust_remote_code=True
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.app = FastAPI(title="vLLM 贪吃蛇游戏 API")
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.post("/v1/completions")
        async def generate_completion(request: CompletionRequest):
            try:
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_tokens=request.max_tokens,
                    stop=request.stop
                )
                
                results = await self.engine.generate(
                    request.prompt,
                    sampling_params,
                    request_id=f"req_{hash(request.prompt)}"
                )
                
                return {
                    "choices": [{
                        "text": output.text,
                        "finish_reason": output.finish_reason
                    } for output in results.outputs]
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "data": [{
                    "id": "snake-game-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "vllm"
                }]
            }
    
    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)

class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    stop: Optional[List[str]] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    
    args = parser.parse_args()

    deployment = vLLMDeployment(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    print(f"🚀 启动 vLLM 服务器在 {args.host}:{args.port}")
    deployment.serve(host=args.host, port=args.port)
```

### 选项 3: TGI (Text Generation Inference)
```bash
#!/bin/bash
# tgi_deploy.sh

MODEL_PATH="$1"
PORT="${2:-8080}"

docker run --gpus all --shm-size 1g -p $PORT:80 \
    -v $MODEL_PATH:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /data \
    --max-input-length 32768 \
    --max-total-tokens 65536 \
    --max-batch-prefill-tokens 8192 \
    --trust-remote-code
```

## 🧪 测试与验证

### 综合测试套件
```python
#!/usr/bin/env python3
"""
综合模型测试和验证套件
"""
import requests
import time
import json
from typing import List, Dict
import numpy as np

class ModelTester:
    def __init__(self, api_url: str, model_name: str = None):
        self.api_url = api_url
        self.model_name = model_name
        
    def test_basic_functionality(self):
        """测试基本模型功能"""
        test_prompts = [
            "当食物在正前方时，贪吃蛇的最佳移动是什么？",
            "如何在贪吃蛇游戏中避免碰撞？",
            "描述贪吃蛇游戏的最佳策略。",
            "当蛇变长时我应该做什么？",
            "如何在贪吃蛇中创建安全路径？"
        ]
        
        results = []
        for prompt in test_prompts:
            start_time = time.time()
            response = self._generate(prompt)
            end_time = time.time()
            
            results.append({
                "prompt": prompt,
                "response": response,
                "latency": end_time - start_time,
                "tokens": len(response.split()) if response else 0
            })
        
        return results
    
    def benchmark_performance(self, num_requests: int = 10):
        """基准测试模型性能"""
        prompt = "分析这个贪吃蛇游戏状态并建议最佳移动："
        
        latencies = []
        throughputs = []
        
        for i in range(num_requests):
            start_time = time.time()
            response = self._generate(prompt, max_tokens=100)
            end_time = time.time()
            
            latency = end_time - start_time
            tokens = len(response.split()) if response else 0
            throughput = tokens / latency if latency > 0 else 0
            
            latencies.append(latency)
            throughputs.append(throughput)
            
            print(f"请求 {i+1}/{num_requests}: {latency:.2f}s, {throughput:.2f} tokens/s")
        
        return {
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "avg_throughput": np.mean(throughputs),
            "total_requests": num_requests
        }
    
    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        """从模型生成响应"""
        try:
            # 根据 API 类型适配
            if "ollama" in self.api_url:
                return self._ollama_generate(prompt)
            else:
                return self._openai_generate(prompt, max_tokens)
        except Exception as e:
            print(f"生成错误: {e}")
            return ""
    
    def _ollama_generate(self, prompt: str) -> str:
        """使用 Ollama API 生成"""
        response = requests.post(
            f"{self.api_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "")
    
    def _openai_generate(self, prompt: str, max_tokens: int) -> str:
        """使用 OpenAI 兼容 API 生成"""
        response = requests.post(
            f"{self.api_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stop": ["<|im_end|>"]
            }
        )
        return response.json()["choices"][0]["text"]

if __name__ == "__main__":
    # 测试不同的部署
    deployments = [
        {"url": "http://localhost:11434", "name": "snake-model", "type": "ollama"},
        {"url": "http://localhost:8000", "name": "snake-model", "type": "vllm"},
    ]
    
    for deployment in deployments:
        print(f"\n🧪 测试 {deployment['type']} 部署...")
        tester = ModelTester(deployment["url"], deployment["name"])
        
        # 基本功能测试
        results = tester.test_basic_functionality()
        print(f"✅ 基本测试完成: 测试了 {len(results)} 个提示")
        
        # 性能基准测试
        benchmark = tester.benchmark_performance(10)
        print(f"📊 性能: {benchmark['avg_latency']:.2f}s 平均延迟, {benchmark['avg_throughput']:.2f} tokens/s")
```

## 🏭 生产环境部署

### Kubernetes 部署
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: snake-model-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: snake-model
  template:
    metadata:
      labels:
        app: snake-model
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model=/models/snake-model"
        - "--tensor-parallel-size=1"
        - "--gpu-memory-utilization=0.85"
        - "--max-model-len=32768"
        - "--host=0.0.0.0"
        - "--port=8000"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: snake-model-service
spec:
  selector:
    app: snake-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  snake-model-vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model /models/snake-model
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.85
      --max-model-len 32768
      --host 0.0.0.0
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx-proxy:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - snake-model-vllm
```

## 🔧 高级技术

### 大型模型分片
```python
def shard_large_model(model_path: str, output_dir: str, max_shard_size: str = "5GB"):
    """
    为分布式推理分片大型模型
    """
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )
    
    model.save_pretrained(
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=True
    )
    
    print(f"✅ 模型已分片到 {output_dir}")
```

### 自定义分词器优化
```python
def optimize_tokenizer(tokenizer_path: str, output_path: str):
    """
    为特定用例优化分词器
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 为贪吃蛇游戏添加特殊 token
    special_tokens = [
        "<|game_state|>", "<|move|>", "<|strategy|>", 
        "<|food_pos|>", "<|snake_pos|>", "<|score|>"
    ]
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # 保存优化后的分词器
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ 分词器已优化并保存到 {output_path}")
```

## 🚨 故障排除

### 常见问题与解决方案

#### CUDA 内存不足
```bash
# 减少批处理大小和内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# 对于 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --tensor-parallel-size 2
```

#### 模型加载错误
```python
# 带错误处理的安全模型加载
def safe_model_load(model_path: str):
    try:
        # 首先尝试使用 trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"使用 trust_remote_code 失败: {e}")
        # 回退到不使用 trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
    return model
```

#### 性能优化
```python
# 性能监控和优化
def optimize_inference_performance():
    # 启用 flash attention
    os.environ["FLASH_ATTENTION"] = "1"
    
    # 为 PyTorch 2.0+ 启用 torch.compile
    torch._dynamo.config.suppress_errors = True
    
    # 设置最佳 CUDA 设置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

## 📚 完整工作流脚本

```bash
#!/bin/bash
# complete_deployment_pipeline.sh

set -e

# 配置
BASE_MODEL="$1"
LORA_ADAPTER="$2"
OUTPUT_NAME="$3"
DEPLOYMENT_TYPE="${4:-vllm}"  # vllm, ollama, tgi

if [ -z "$BASE_MODEL" ] || [ -z "$LORA_ADAPTER" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "用法: $0 <base_model> <lora_adapter> <output_name> [deployment_type]"
    exit 1
fi

WORK_DIR="./deployment_pipeline"
MERGED_MODEL="$WORK_DIR/merged_models/$OUTPUT_NAME"
GGUF_MODEL="$WORK_DIR/gguf_models/$OUTPUT_NAME"

mkdir -p "$WORK_DIR"/{merged_models,gguf_models,deployments}

echo "🚀 启动完整部署流水线..."

# 步骤 1: 合并 LoRA
echo "📦 步骤 1: 合并 LoRA 适配器..."
python merge_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_adapter "$LORA_ADAPTER" \
    --output_path "$MERGED_MODEL" \
    --merge_strategy linear \
    --dtype bfloat16

# 步骤 2: 转换为 GGUF（可选）
if [ "$DEPLOYMENT_TYPE" = "ollama" ]; then
    echo "🔄 步骤 2: 转换为 GGUF..."
    python convert_to_gguf.py \
        --model_path "$MERGED_MODEL" \
        --output_dir "$GGUF_MODEL" \
        --quantization Q4_K_M
    MODEL_PATH="$GGUF_MODEL"
else
    MODEL_PATH="$MERGED_MODEL"
fi

# 步骤 3: 部署模型
echo "🚀 步骤 3: 使用 $DEPLOYMENT_TYPE 部署模型..."
case $DEPLOYMENT_TYPE in
    "vllm")
        python deploy_vllm.py \
            --model "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port 8000 &
        ;;
    "ollama")
        ./deploy_ollama.sh "$MODEL_PATH" "$OUTPUT_NAME"
        ;;
    "tgi")
        ./deploy_tgi.sh "$MODEL_PATH" 8080 &
        ;;
esac

# 步骤 4: 测试部署
echo "🧪 步骤 4: 测试部署..."
sleep 30  # 等待服务器启动
python test_model.py --api_url http://localhost:8000 --model_name "$OUTPUT_NAME"

echo "✅ 部署流水线成功完成！"
echo "📊 模型可用地址: http://localhost:8000"
echo "📖 文档地址: http://localhost:8000/docs"
```

## 🎯 最佳实践总结

1. **内存管理**: 使用适当的批处理大小和内存利用率
2. **量化**: 在质量和性能之间选择正确的平衡
3. **部署**: 根据用例和规模选择部署方法
4. **监控**: 实施综合日志记录和监控
5. **测试**: 在生产前始终验证模型性能
6. **安全**: 实施适当的身份验证和速率限制
7. **扩展**: 规划负载均衡的水平扩展

## 📞 支持与资源

- **文档**: 查看模型特定文档
- **社区**: 加入 vLLM、Ollama 和 Transformers 社区
- **问题**: 向各自的 GitHub 仓库报告错误
- **性能**: 使用分析工具进行优化

---

*本指南提供了部署微调贪吃蛇游戏模型的完整流水线。根据您的特定需求和硬件能力调整参数和配置。* 