# Building Applications with Large Language Models

---

## Overview

- What are LLMs and why should you care?
- How LLMs are transforming software development
- Types of applications you can build
- Running LLMs: Cloud vs On-device 
- Introduction to Ollama and alternatives
- Practical use cases
- Getting started

---

## What are LLMs?

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│                  Large Language Models                │
│                                                       │
│   ┌─────────┐    ┌─────────┐    ┌─────────────────┐   │
│   │         │    │         │    │                 │   │
│   │ Massive │    │ Deep    │    │ Probabilistic   │   │
│   │ Training│    │ Neural  │    │ Text            │   │
│   │ Data    │    │ Networks│    │ Generation      │   │
│   │         │    │         │    │                 │   │
│   └─────────┘    └─────────┘    └─────────────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

- Neural networks trained on vast amounts of text
- Can understand and generate human-like text
- Examples: GPT-4, Claude, Llama, DeepSeek, Mistral

---

## Why LLMs Matter

- **Natural language interface** to computing
- **Accessibility**: Non-technical users can build and interact with software
- **Automation**: Handle routine language tasks
- **Augmentation**: Enhance human capabilities
- **Adaptability**: One model, countless applications

---

## The Developer Superpower

```
             BEFORE                   |               AFTER
                                      |
  ┌────────────┐     ┌────────────┐   |   ┌────────────┐     ┌────────────┐
  │            │     │            │   |   │            │     │            │
  │  Developer ├────►│  Software  │   |   │  Developer ├────►│    LLM     │
  │            │     │            │   |   │            │     │            │
  └────────────┘     └────────────┘   |   └────────────┘     └───┬────────┘
                                      |                           │
                                      |                           ▼
                                      |                     ┌────────────┐
                                      |                     │            │
                                      |                     │  Software  │
                                      |                     │            │
                                      |                     └────────────┘
```

- Multiplies what developers can build
- Reduces implementation time from weeks to hours
- Unlocks new types of applications

---

## Types of LLM Applications

1. **Text generation**: Content creation, summaries, marketing copy
2. **Conversational**: Chatbots, virtual assistants, customer support
3. **Transformation**: Translation, paraphrasing, style conversion  
4. **Analysis**: Sentiment analysis, entity extraction, classification
5. **Creative**: Story generation, poetry, creative writing
6. **Domain-specific**: Legal, medical, scientific, educational

---

## The Application Spectrum

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│               The LLM Application Spectrum                              │
│                                                                         │
│  Simple ◄────────────────────────────────────────────────────► Complex  │
│                                                                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │         │   │         │   │         │   │         │   │         │   │
│  │  Basic  │   │ Chat    │   │ Context │   │ Tool-   │   │ Multi-  │   │
│  │  Text   │   │ Apps    │   │ Aware   │   │ using   │   │ Agent   │   │
│  │ Output  │   │         │   │ Apps    │   │ Apps    │   │ Systems │   │
│  │         │   │         │   │         │   │         │   │         │   │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Running LLMs: The Options

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                         LLM Deployment Options                         │
│                                                                        │
│  ┌─────────────────────────────┐      ┌─────────────────────────────┐  │
│  │                             │      │                             │  │
│  │         Cloud-based         │      │        On-device            │  │
│  │                             │      │                             │  │
│  │  + High performance         │      │  + Privacy                  │  │
│  │  + Latest models            │      │  + No API costs             │  │
│  │  + Scalable                 │      │  + Works offline            │  │
│  │  + No hardware requirements │      │  + Full control             │  │
│  │                             │      │                             │  │
│  │  - Costs per request        │      │  - Hardware requirements    │  │
│  │  - Privacy concerns         │      │  - Limited model size       │  │
│  │  - API limitations          │      │  - Lower performance        │  │
│  │  - Internet required        │      │                             │  │
│  │                             │      │                             │  │
│  └─────────────────────────────┘      └─────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Cloud LLM Providers

- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini
- **Cohere**: Command
- **Meta AI**: Llama 3
- **Mistral AI**: Mistral, Mixtral

---

## On-device LLM Tools

```
┌───────────────────────────────────┐
│                                   │
│            LLM Tools              │
│                                   │
│  ┌─────────┐     ┌─────────────┐  │
│  │         │     │             │  │
│  │ Ollama  │     │  llama.cpp  │  │
│  │         │     │             │  │
│  └─────────┘     └─────────────┘  │
│                                   │
│  ┌─────────┐     ┌─────────────┐  │
│  │         │     │             │  │
│  │ LM Studio│     │ MLC AI     │  │
│  │         │     │             │  │
│  └─────────┘     └─────────────┘  │
│                                   │
└───────────────────────────────────┘
```

---

## Spotlight on Ollama

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                         ┃
┃                              🦙 OLLAMA                                  ┃
┃                                                                         ┃
┃  • Run large language models locally                                    ┃
┃  • Simple CLI and API                                                   ┃
┃  • 100+ models available                                                ┃
┃  • Cross-platform (Mac, Windows, Linux)                                 ┃
┃  • GPU acceleration                                                     ┃
┃  • Customize and fine-tune models                                       ┃
┃                                                                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Popular Ollama Models

| Model | Size | Performance | Sweet Spot |
|-------|------|------------|------------|
| DeepSeek | 7B | Excellent | Great all-rounder |
| Llama 3 | 8B | Very Good | Strong reasoning |
| Phi-3 | 3.8B | Good | Small but capable |
| Mistral | 7B | Very Good | Balanced perf/size |
| DeepSeek-Coder | 6.7B | Excellent | Programming tasks |
| Neural-Chat | 7B | Good | Conversational |
| Gemma | 2B | Fair | Resource constrained |

---

## How Ollama Works

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                        Ollama Architecture                       │
│                                                                  │
│   ┌──────────┐      ┌───────────┐      ┌────────────────────┐   │
│   │          │      │           │      │                    │   │
│   │  Ollama  │      │  Model    │      │  Inference         │   │
│   │  CLI/API ├─────►│  Library  ├─────►│  Engine (llama.cpp)│   │
│   │          │      │           │      │                    │   │
│   └──────────┘      └───────────┘      └────────────────────┘   │
│         │                                        │              │
│         │                                        │              │
│         ▼                                        ▼              │
│   ┌──────────┐                           ┌────────────────────┐ │
│   │          │                           │                    │ │
│   │  Model   │                           │  CPU/GPU           │ │
│   │  Registry│                           │  Acceleration      │ │
│   │          │                           │                    │ │
│   └──────────┘                           └────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installing Ollama

- **macOS**: Download from [ollama.com](https://ollama.com)
- **Windows**: Download Windows installer from [ollama.com](https://ollama.com)
- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

---

## Basic Ollama Usage

```bash
# Start the local server
ollama serve

# Pull (download) a model
ollama pull qwen2.5:3b

# Run a model in chat mode
ollama run qwen2.5:3b

# One-shot prompt
ollama run qwen2.5:3b "What is a large language model?"

# List available models
ollama list
```

---

## Alternative: LM Studio

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                         ┃
┃                             LM STUDIO                                   ┃
┃                                                                         ┃
┃  • GUI for running LLMs locally                                         ┃
┃  • Browse and download models                                           ┃
┃  • Chat interface built-in                                              ┃
┃  • Local API server                                                     ┃
┃  • Model performance comparison                                         ┃
┃  • Chat history and settings management                                 ┃
┃                                                                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Alternative: MLC LLM

- AI framework by MLC (Machine Learning Compilation)
- Deploy LLMs on mobile devices, browsers, PCs
- Focus on efficiency and wide compatibility
- Web-based UI option
- Supports iOS/Android deployment

---

## Exciting Applications You Can Build

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                 Creative Application Ideas                                  │
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │             │   │             │   │             │   │             │     │
│  │ Interactive │   │  Personal   │   │  Content    │   │  Knowledge  │     │
│  │ Storytelling│   │  Learning   │   │  Creation   │   │  Assistant  │     │
│  │             │   │  Coach      │   │  Studio     │   │             │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │             │   │             │   │             │   │             │     │
│  │  Language   │   │  Coding     │   │  Data       │   │  Simulation │     │
│  │  Learning   │   │  Assistant  │   │  Analyzer   │   │  Generator  │     │
│  │             │   │             │   │             │   │             │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Case Study: Personal Study Assistant

```
┌─────────────────────┐         ┌─────────────────┐
│                     │         │                 │
│     User Input      │         │    Knowledge    │
│  (Study Question)   │         │     Store       │
│                     │         │                 │
└──────────┬──────────┘         └────────┬────────┘
           │                             │
           ▼                             │
┌─────────────────────┐                  │
│                     │                  │
│       LLM           │◄─────────────────┘
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│                                             │
│              Response Types                 │
│                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────┐ │
│  │            │  │            │  │        │ │
│  │ Explanation│  │  Practice  │  │  Quiz  │ │
│  │            │  │  Problems  │  │        │ │
│  └────────────┘  └────────────┘  └────────┘ │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Case Study: Story Generator

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│                  Interactive Story Generator                 │
│                                                              │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐     │
│  │            │      │            │      │            │     │
│  │ Character  │      │  Setting   │      │   Plot     │     │
│  │ Creator    │ ────►│  Builder   │ ────►│  Engine    │     │
│  │            │      │            │      │            │     │
│  └────────────┘      └────────────┘      └──────┬─────┘     │
│                                                  │           │
│                                                  ▼           │
│                                          ┌────────────────┐  │
│                      ┌──────────────────►│                │  │
│                      │                   │  Story LLM     │  │
│                      │                   │                │  │
│                      │                   └────────┬───────┘  │
│                      │                            │          │
│  ┌────────────────┐  │                    ┌──────▼───────┐   │
│  │                │  │                    │              │   │
│  │  User Choices  │──┘                    │   Story      │   │
│  │                │◄───────────────────── │   Output     │   │
│  └────────────────┘                       │              │   │
│                                           └──────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Case Study: Programming Assistant

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│                 Programming Assistant                     │
│                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │             │    │             │    │             │    │
│  │ Code        │───►│ Code        │───►│ Unit Test   │    │
│  │ Generation  │    │ Explanation │    │ Generation  │    │
│  │             │    │             │    │             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         ▲                                     │           │
│         │                                     │           │
│         │                                     ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │             │    │             │    │             │    │
│  │ User        │◄───│ Debugging   │◄───│ Testing     │    │
│  │ Request     │    │ Help        │    │ Feedback    │    │
│  │             │    │             │    │             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## Challenges and Considerations

- **Hallucinations**: LLMs can generate plausible but incorrect information
- **Context Limits**: Models have finite context windows
- **Performance**: Local models have performance/capability tradeoffs
- **Bias**: LLMs can reflect biases in their training data
- **Privacy**: Consider data sensitivity when using cloud APIs
- **Hardware Requirements**: Local models need sufficient RAM/GPU

---

## Questions?

Thank you for your attention!
