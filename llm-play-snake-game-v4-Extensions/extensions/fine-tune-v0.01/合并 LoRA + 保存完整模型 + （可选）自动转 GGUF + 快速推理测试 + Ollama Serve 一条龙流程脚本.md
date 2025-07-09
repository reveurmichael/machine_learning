# 🚀 Complete LoRA Merge & Deployment Pipeline

A comprehensive guide for merging LoRA adapters, converting to optimized formats, and deploying fine-tuned LLMs with multiple serving options.

## 📋 Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [LoRA Merge & Unload](#lora-merge--unload)
3. [Model Conversion & Optimization](#model-conversion--optimization)
4. [Deployment Options](#deployment-options)
5. [Testing & Validation](#testing--validation)
6. [Production Deployment](#production-deployment)
7. [Advanced Techniques](#advanced-techniques)
8. [Troubleshooting](#troubleshooting)

## 🛠 Prerequisites & Setup

### System Requirements
- **GPU**: NVIDIA A100/H100 (recommended), RTX 4090+, or V100
- **RAM**: 32GB+ (64GB+ for large models)
- **Storage**: 500GB+ NVMe SSD
- **CUDA**: 11.8+ or 12.1+

### Environment Setup
```bash
# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes
pip install vllm ollama huggingface_hub safetensors

# For quantization and conversion
pip install auto-gptq optimum[onnxruntime-gpu]
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Clone llama.cpp for GGUF conversion
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j $(nproc)
```

## 🔗 LoRA Merge & Unload

### Method 1: Standard PEFT Merge
```python
#!/usr/bin/env python3
"""
Advanced LoRA Merge Script with Multiple Merging Strategies
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
    Merge LoRA adapters with the base model using advanced strategies.
    
    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapter or list of paths
        output_path: Output directory for merged model
        merge_strategy: Strategy for merging multiple adapters
        weights: Weights for weighted merging
        dtype: Model dtype for memory optimization
        device_map: Device mapping strategy
        max_memory: Maximum memory per device
    """
    
    # Load base model with optimizations
    logger.info(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"  # For supported models
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Handle multiple LoRA adapters
    if isinstance(lora_adapter_path, list):
        # Load first adapter
        model = PeftModel.from_pretrained(model, lora_adapter_path[0])
        
        # Add additional adapters
        for i, adapter_path in enumerate(lora_adapter_path[1:], 1):
            model.load_adapter(adapter_path, adapter_name=f"adapter_{i}")
        
        # Merge multiple adapters using specified strategy
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
        # Single adapter
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    # Merge and unload
    logger.info("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save merged model
    logger.info(f"Saving merged model to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(
        output_path,
        save_safetensors=True,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(output_path)
    
    # Save model card
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

# Merged Model

This model was created by merging LoRA adapters with the base model.

## Base Model
- {base_model_path}

## LoRA Adapters
- {lora_adapter_path}

## Merge Strategy
- {merge_strategy}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_path}")
tokenizer = AutoTokenizer.from_pretrained("{output_path}")
```
"""
    
    with open(Path(output_path) / "README.md", "w") as f:
        f.write(model_card)
    
    logger.info("✅ LoRA merge completed successfully!")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--lora_adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output_path", required=True, help="Output directory")
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

### Method 2: Advanced Multi-Adapter Merging
```python
def advanced_multi_adapter_merge(adapters_config: dict, output_path: str):
    """
    Advanced merging with different strategies per adapter type.
    
    adapters_config example:
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
    # Implementation for complex merging scenarios
    pass
```

## ⚡ Model Conversion & Optimization

### GGUF Conversion for llama.cpp
```python
#!/usr/bin/env python3
"""
Advanced GGUF Conversion with Multiple Quantization Options
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
    Convert HuggingFace model to GGUF format with optimizations.
    """
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model info
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    model_name = config.get("_name_or_path", model_path.name)
    
    # Step 1: Convert to GGUF FP16
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
    
    print(f"🔄 Converting to FP16 GGUF: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)
    
    # Step 2: Quantize to specified format
    if quantization != "f16":
        quantized_path = output_dir / f"{model_name}-{quantization}.gguf"
        
        quant_cmd = [
            "llama.cpp/quantize",
            str(fp16_path),
            str(quantized_path),
            quantization
        ]
        
        print(f"🔄 Quantizing to {quantization}: {' '.join(quant_cmd)}")
        subprocess.run(quant_cmd, check=True)
        
        # Remove FP16 file to save space
        if fp16_path.exists():
            fp16_path.unlink()
        
        final_path = quantized_path
    else:
        final_path = fp16_path
    
    print(f"✅ GGUF conversion completed: {final_path}")
    return final_path

# Quantization options with descriptions
QUANTIZATION_OPTIONS = {
    "Q2_K": "Very small, high quality loss",
    "Q3_K_S": "Small, very high quality loss",
    "Q3_K_M": "Medium, high quality loss",
    "Q3_K_L": "Large, high quality loss",
    "Q4_0": "Legacy, small, very high quality loss",
    "Q4_1": "Legacy, small, substantial quality loss",
    "Q4_K_S": "Small, greater quality loss",
    "Q4_K_M": "Medium, balanced quality/size (recommended)",
    "Q5_0": "Legacy, medium, balanced quality/size",
    "Q5_1": "Legacy, medium, low quality loss",
    "Q5_K_S": "Large, low quality loss",
    "Q5_K_M": "Large, very low quality loss (recommended)",
    "Q6_K": "Very large, extremely low quality loss",
    "Q8_0": "Very large, extremely low quality loss",
    "F16": "Extremely large, no quality loss",
    "F32": "Maximum size, no quality loss"
}
```

### AWQ/GPTQ Quantization
```python
def quantize_with_awq(model_path: str, output_path: str, bits: int = 4):
    """
    Quantize model using AutoAWQ for maximum inference speed.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    
    # Load model and tokenizer
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, 
        safetensors=True, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Quantize
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM"
        }
    )
    
    # Save quantized model
    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ AWQ quantization completed: {output_path}")
```

## 🚀 Deployment Options

### Option 1: Ollama Deployment
```bash
#!/bin/bash
# ollama_deploy.sh

MODEL_PATH="$1"
MODEL_NAME="$2"
QUANTIZATION="${3:-Q4_K_M}"

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_path> <model_name> [quantization]"
    exit 1
fi

# Create Modelfile
cat > Modelfile << EOF
FROM ${MODEL_PATH}

# Temperature controls creativity (0.0-2.0)
PARAMETER temperature 0.7

# Top-p controls diversity (0.0-1.0)
PARAMETER top_p 0.9

# Top-k limits token choices (1-100)
PARAMETER top_k 40

# Repeat penalty reduces repetition (1.0-1.5)
PARAMETER repeat_penalty 1.1

# Context window size
PARAMETER num_ctx 32768

# Stop sequences
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# System message
SYSTEM """You are a helpful AI assistant trained on Snake game data. You can provide strategic advice for Snake gameplay, analyze game states, and suggest optimal moves."""

# Example conversation
MESSAGE user "What's the best strategy for Snake?"
MESSAGE assistant "The key strategies for Snake include: 1) Stay along walls and edges for safety, 2) Create and maintain safe paths, 3) Avoid trapping yourself in corners, 4) Plan several moves ahead, and 5) Be patient with food collection."
EOF

# Create model in Ollama
echo "🔄 Creating Ollama model..."
ollama create $MODEL_NAME -f Modelfile

# Test the model
echo "🧪 Testing model..."
echo "What is the optimal Snake game strategy?" | ollama generate $MODEL_NAME

echo "✅ Ollama deployment completed!"
echo "Usage: ollama run $MODEL_NAME"
```

### Option 2: vLLM Deployment
```python
#!/usr/bin/env python3
"""
Production vLLM Deployment Script
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
        self.app = FastAPI(title="vLLM Snake Game API")
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
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    
    args = parser.parse_args()
    
    deployment = vLLMDeployment(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    print(f"🚀 Starting vLLM server at {args.host}:{args.port}")
    deployment.serve(host=args.host, port=args.port)
```

### Option 3: TGI (Text Generation Inference)
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

## 🧪 Testing & Validation

### Comprehensive Testing Suite
```python
#!/usr/bin/env python3
"""
Comprehensive model testing and validation suite
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
        """Test basic model functionality"""
        test_prompts = [
            "What is the best move in Snake when the food is directly ahead?",
            "How do I avoid collisions in Snake game?",
            "Describe the optimal Snake game strategy.",
            "What should I do when the snake is getting long?",
            "How do I create safe paths in Snake?"
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
        """Benchmark model performance"""
        prompt = "Analyze this Snake game state and suggest the best move:"
        
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
            
            print(f"Request {i+1}/{num_requests}: {latency:.2f}s, {throughput:.2f} tokens/s")
        
        return {
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "avg_throughput": np.mean(throughputs),
            "total_requests": num_requests
        }
    
    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response from model"""
        try:
            # Adapt based on API type
            if "ollama" in self.api_url:
                return self._ollama_generate(prompt)
            else:
                return self._openai_generate(prompt, max_tokens)
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def _ollama_generate(self, prompt: str) -> str:
        """Generate using Ollama API"""
        response = requests.post(
            f"{self.api_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "")
    
    def _openai_generate(self, prompt: str, max_tokens: int) -> str:
        """Generate using OpenAI-compatible API"""
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
    # Test different deployments
    deployments = [
        {"url": "http://localhost:11434", "name": "snake-model", "type": "ollama"},
        {"url": "http://localhost:8000", "name": "snake-model", "type": "vllm"},
    ]
    
    for deployment in deployments:
        print(f"\n🧪 Testing {deployment['type']} deployment...")
        tester = ModelTester(deployment["url"], deployment["name"])
        
        # Basic functionality test
        results = tester.test_basic_functionality()
        print(f"✅ Basic tests completed: {len(results)} prompts tested")
        
        # Performance benchmark
        benchmark = tester.benchmark_performance(10)
        print(f"📊 Performance: {benchmark['avg_latency']:.2f}s avg latency, {benchmark['avg_throughput']:.2f} tokens/s")
```

## 🏭 Production Deployment

### Kubernetes Deployment
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

## 🔧 Advanced Techniques

### Model Sharding for Large Models
```python
def shard_large_model(model_path: str, output_dir: str, max_shard_size: str = "5GB"):
    """
    Shard large models for distributed inference
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
    
    print(f"✅ Model sharded to {output_dir}")
```

### Custom Tokenizer Optimization
```python
def optimize_tokenizer(tokenizer_path: str, output_path: str):
    """
    Optimize tokenizer for specific use case
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Add special tokens for Snake game
    special_tokens = [
        "<|game_state|>", "<|move|>", "<|strategy|>", 
        "<|food_pos|>", "<|snake_pos|>", "<|score|>"
    ]
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Save optimized tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Tokenizer optimized and saved to {output_path}")
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### CUDA Out of Memory
```bash
# Reduce batch size and memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# For vLLM
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --tensor-parallel-size 2
```

#### Model Loading Errors
```python
# Safe model loading with error handling
def safe_model_load(model_path: str):
    try:
        # Try with trust_remote_code first
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Failed with trust_remote_code: {e}")
        # Fallback without trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
    return model
```

#### Performance Optimization
```python
# Performance monitoring and optimization
def optimize_inference_performance():
    # Enable flash attention
    os.environ["FLASH_ATTENTION"] = "1"
    
    # Enable torch.compile for PyTorch 2.0+
    torch._dynamo.config.suppress_errors = True
    
    # Set optimal CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

## 📚 Complete Workflow Script

```bash
#!/bin/bash
# complete_deployment_pipeline.sh

set -e

# Configuration
BASE_MODEL="$1"
LORA_ADAPTER="$2"
OUTPUT_NAME="$3"
DEPLOYMENT_TYPE="${4:-vllm}"  # vllm, ollama, tgi

if [ -z "$BASE_MODEL" ] || [ -z "$LORA_ADAPTER" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: $0 <base_model> <lora_adapter> <output_name> [deployment_type]"
    exit 1
fi

WORK_DIR="./deployment_pipeline"
MERGED_MODEL="$WORK_DIR/merged_models/$OUTPUT_NAME"
GGUF_MODEL="$WORK_DIR/gguf_models/$OUTPUT_NAME"

mkdir -p "$WORK_DIR"/{merged_models,gguf_models,deployments}

echo "🚀 Starting complete deployment pipeline..."

# Step 1: Merge LoRA
echo "📦 Step 1: Merging LoRA adapters..."
python merge_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_adapter "$LORA_ADAPTER" \
    --output_path "$MERGED_MODEL" \
    --merge_strategy linear \
    --dtype bfloat16

# Step 2: Convert to GGUF (optional)
if [ "$DEPLOYMENT_TYPE" = "ollama" ]; then
    echo "🔄 Step 2: Converting to GGUF..."
    python convert_to_gguf.py \
        --model_path "$MERGED_MODEL" \
        --output_dir "$GGUF_MODEL" \
        --quantization Q4_K_M
    MODEL_PATH="$GGUF_MODEL"
else
    MODEL_PATH="$MERGED_MODEL"
fi

# Step 3: Deploy model
echo "🚀 Step 3: Deploying model with $DEPLOYMENT_TYPE..."
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

# Step 4: Test deployment
echo "🧪 Step 4: Testing deployment..."
sleep 30  # Wait for server to start
python test_model.py --api_url http://localhost:8000 --model_name "$OUTPUT_NAME"

echo "✅ Deployment pipeline completed successfully!"
echo "📊 Model available at: http://localhost:8000"
echo "📖 Documentation: http://localhost:8000/docs"
```

## 🎯 Best Practices Summary

1. **Memory Management**: Use appropriate batch sizes and memory utilization
2. **Quantization**: Choose the right balance between quality and performance
3. **Deployment**: Select deployment method based on use case and scale
4. **Monitoring**: Implement comprehensive logging and monitoring
5. **Testing**: Always validate model performance before production
6. **Security**: Implement proper authentication and rate limiting
7. **Scaling**: Plan for horizontal scaling with load balancing

## 📞 Support & Resources

- **Documentation**: Check model-specific documentation
- **Community**: Join vLLM, Ollama, and Transformers communities
- **Issues**: Report bugs to respective GitHub repositories
- **Performance**: Use profiling tools for optimization

---

*This guide provides a complete pipeline for deploying fine-tuned Snake game models. Adapt the parameters and configurations based on your specific requirements and hardware capabilities.*
