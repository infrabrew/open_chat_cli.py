# Interactive LLM Chat Client CLI

**A streaming CLI chat interface for Ollama, vLLM, and Groq with real-time token statistics**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Backend Configuration](#backend-configuration)
- [Command Reference](#command-reference)
- [Token Statistics](#token-statistics)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Architecture](#architecture)
- [Contributing](#contributing)

---

## 🎯 Overview

This interactive chat client provides a terminal-based interface for conversing with Large Language Models (LLMs) through various backends. It features real-time streaming responses, automatic token counting, and performance metrics after each interaction.

### Why This Tool?

- **Universal Interface**: Single client for multiple LLM backends
- **Real-Time Streaming**: See responses as they're generated
- **Performance Metrics**: Track token usage and generation speed
- **Conversation Management**: Automatic history management to prevent context overflow
- **Developer-Friendly**: Simple command-line interface with history support

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔄 Streaming Output** | Real-time display of AI responses as they're generated |
| **📊 Token Statistics** | Track prompt tokens, completion tokens, and total usage |
| **⚡ Speed Metrics** | Measure generation speed in tokens per second (T/s) |
| **💬 Conversation History** | Automatic management with configurable limits |
| **⌨️ Command History** | Use arrow keys to navigate previous inputs |
| **🔌 Multi-Backend** | Support for Ollama, vLLM, and Groq |
| **🧹 Chat Management** | Clear conversation with simple commands |
| **🚫 Error Handling** | Graceful handling of network issues and malformed responses |

---

## 💻 Requirements

### System Requirements

- **Python**: 3.7 or higher
- **Network**: Access to LLM backend service
- **Terminal**: Modern terminal with UTF-8 support

### Python Packages

```bash
pip install requests tiktoken prompt-toolkit
```

### Package Details

| Package | Purpose | Version |
|---------|---------|---------|
| `requests` | HTTP client for API calls | ≥2.25.0 |
| `tiktoken` | Token counting | ≥0.5.0 |
| `prompt-toolkit` | Interactive terminal features | ≥3.0.0 |

---

## 🚀 Installation

### Quick Install

```bash
# Clone or download the script
Clone
git clone https://github.com/infrabrew/open_chat_cli.py.git

# OR

wget https://raw.githubusercontent.com/infrabrew/llm-chat-client/main/open_chat_cli.py

# Install dependencies
pip install requests tiktoken prompt-toolkit

# Make executable (optional)
chmod +x open_chat_cli.py
```

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install requests tiktoken prompt-toolkit
```

---

## 📖 Usage

### Basic Usage

#### Connect to Ollama (Default)

```bash
python3 open_chat_cli.py
```

#### Connect to vLLM

```bash
python3 open_chat_cli.py --backend vllm
```

#### Connect to Groq

```bash
python3 open_chat_cli.py --backend groq --api-key YOUR_API_KEY
```

### Advanced Usage

#### Custom Model Selection

```bash
# Ollama with specific model
python3 open_chat_cli.py --backend ollama --model mistral

# vLLM with custom model
python3 open_chat_cli.py --backend vllm --model microsoft/Phi-3-mini-4k-instruct

# Groq with different model
python3 open_chat_cli.py --backend groq --model mixtral-8x7b-32768 --api-key YOUR_KEY
```

#### Custom Host Configuration

```bash
# Connect to remote Ollama instance
python3 open_chat_cli.py --host http://192.168.1.100:11434

# Connect to custom vLLM deployment
python3 open_chat_cli.py --backend vllm --host http://ml-server:8000
```

---

## 🔧 Backend Configuration

### Ollama

**Default Host**: `http://192.168.0.31:11435`  
**Default Model**: `llama3`  
**Environment Variable**: `OLLAMA_HOST`

#### Setup

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Run Ollama server
ollama serve

# Connect with chat client
python3 open_chat_cli.py --backend ollama
```

#### Available Models

- `llama3` (default)
- `llama3:70b`
- `mistral`
- `mixtral`
- `codellama`
- `phi3`

---

### vLLM

**Default Host**: `http://localhost:8000`  
**Default Model**: `meta-llama/Meta-Llama-3-8B-Instruct`  
**Environment Variable**: `VLLM_HOST`

#### Setup

```bash
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000

# Connect with chat client
python3 open_chat_cli.py --backend vllm
```

#### Performance Options

```bash
# With GPU acceleration
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --dtype float16

# With quantization
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-Chat-AWQ \
    --quantization awq
```

---

### Groq

**Default Host**: `https://api.groq.com`  
**Default Model**: `llama3-8b-8192`  
**Environment Variable**: `GROQ_API_KEY` or `OPENAI_API_KEY`

#### Setup

```bash
# Get API key from https://console.groq.com/keys

# Set environment variable
export GROQ_API_KEY="your-api-key-here"

# Or use command-line argument
python3 open_chat_cli.py --backend groq --api-key your-api-key-here
```

#### Available Models

- `llama3-8b-8192` (default, 8K context)
- `llama3-70b-8192` (70B parameters, 8K context)
- `mixtral-8x7b-32768` (32K context)
- `gemma-7b-it` (Google's Gemma)

---

## 🎮 Command Reference

### Interactive Commands

| Command | Action | Example |
|---------|--------|---------|
| Type message + Enter | Send message to AI | `What is Python?` |
| `clear chat` | Clear conversation history | `clear chat` |
| Ctrl+C or Ctrl+D | Exit the chat | - |
| Up/Down arrows | Navigate command history | - |

### Command-Line Arguments

```bash
python3 open_chat_cli.py [OPTIONS]

Options:
  --backend {ollama,vllm,groq}
                        Backend service to use (default: ollama)
  --model MODEL         Model name (backend-specific defaults provided)
  --host HOST          Override host URL for the backend service
  --api-key API_KEY    API key (required for Groq)
  -h, --help           Show help message
```

### Environment Variables

```bash
# Ollama
export OLLAMA_HOST="http://192.168.0.31:11435"

# vLLM
export VLLM_HOST="http://localhost:8000"

# Groq
export GROQ_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

---

## 📊 Token Statistics

After each AI response, the client displays token usage and performance metrics:

```
[Tokens used: 245 | Speed: 87.3 T/s]
```

### Metrics Explained

| Metric | Description | Formula |
|--------|-------------|---------|
| **Tokens used** | Total tokens (prompt + completion) | `prompt_tokens + completion_tokens` |
| **Speed** | Generation speed | `completion_tokens / elapsed_time` |

### Understanding Tokens

- **Prompt Tokens**: Tokens in your input + conversation history
- **Completion Tokens**: Tokens in the AI's response
- **1 Token**: Approximately 4 characters or 0.75 words in English

### Example Breakdown

```
You: Tell me about Python
AI: Python is a high-level, interpreted programming language...
[Tokens used: 245 | Speed: 87.3 T/s]

Breakdown:
- Prompt: ~10 tokens (your question + context)
- Completion: ~235 tokens (AI response)
- Time: ~2.7 seconds
- Speed: 235 tokens ÷ 2.7 seconds = 87.3 T/s
```

---

## 🐛 Troubleshooting

### Common Issues

#### "Connection refused" Error

**Cause**: Backend service not running or wrong host

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check if vLLM is running
curl http://localhost:8000/v1/models

# Verify host configuration
python3 open_chat_cli.py --host http://localhost:11434
```

---

#### "Groq requires --api-key" Error

**Cause**: Missing API key for Groq backend

**Solutions**:
```bash
# Option 1: Use environment variable
export GROQ_API_KEY="your-key"
python3 open_chat_cli.py --backend groq

# Option 2: Use command-line argument
python3 open_chat_cli.py --backend groq --api-key your-key
```

---

#### 413 Request Entity Too Large

**Cause**: Conversation history exceeds context window

**Solution**: The client automatically limits history to 10 messages. If this still occurs:

```python
# Edit the script to reduce MAX_HISTORY
MAX_HISTORY = 6  # Reduce from 10 to 6
```

Or use the `clear chat` command to reset:
```
You: clear chat
Chat cleared. If you need anything else, just let me know!
```

---

#### Slow Response Times

**Possible Causes**:
- Large model on limited hardware
- Network latency
- High server load

**Solutions**:
```bash
# Use smaller model
python3 open_chat_cli.py --model llama3:8b  # Instead of 70b

# Check network latency
ping 192.168.0.31

# Monitor system resources
htop  # or top
```

---

#### JSON Decode Errors

**Cause**: Malformed streaming chunks (usually temporary)

**Solution**: The client automatically skips malformed chunks. If persistent:
```bash
# Check backend logs
journalctl -u ollama -f  # For Ollama systemd service

# Restart backend service
ollama serve
```

---

## 💡 Examples

### Example Session

```bash
$ python3 open_chat_cli.py --backend ollama --model llama3
Connected to ollama model llama3

You: What is machine learning?
AI: Machine learning is a subset of artificial intelligence that involves 
training algorithms to learn patterns from data and make predictions or 
decisions without being explicitly programmed for each specific task.
[Tokens used: 89 | Speed: 124.5 T/s]

You: Give me an example
AI: Sure! A classic example is email spam filtering. The algorithm learns 
from thousands of emails labeled as "spam" or "not spam" and then can 
automatically classify new incoming emails based on patterns it learned.
[Tokens used: 156 | Speed: 118.3 T/s]

You: clear chat
Chat cleared. If you need anything else, just let me know!

You: ^C
Bye!
```

### Using with Different Backends

#### Local Ollama Setup

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Chat with Llama 3
python3 open_chat_cli.py --backend ollama --model llama3
```

#### Remote vLLM Server

```bash
# Connect to ML server
python3 open_chat_cli.py \
    --backend vllm \
    --host http://ml-server.local:8000 \
    --model meta-llama/Meta-Llama-3-70B-Instruct
```

#### Groq Cloud API

```bash
# Fast inference with Groq
export GROQ_API_KEY="gsk_..."
python3 open_chat_cli.py \
    --backend groq \
    --model llama3-70b-8192
```

---

## 🏗️ Architecture

### System Flow

```
┌─────────────────┐
│  User Terminal  │
│  (prompt_toolkit)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chat Client    │
│  - Parse input  │
│  - Manage hist. │
│  - Count tokens │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  HTTP Streaming │──────▶│   Backend    │
│  (requests)     │◀──────│ (Ollama/vLLM/│
└────────┬────────┘       │    Groq)     │
         │                └──────────────┘
         ▼
┌─────────────────┐
│  Display Output │
│  + Statistics   │
└─────────────────┘
```

### Component Breakdown

#### 1. Input Layer
- **prompt_toolkit**: Provides rich terminal input
- **InMemoryHistory**: Stores command history
- Arrow key navigation support

#### 2. Processing Layer
- **Message Management**: Maintains conversation context
- **Token Counting**: Uses tiktoken for accurate counts
- **History Trimming**: Prevents context overflow

#### 3. Network Layer
- **Streaming HTTP**: Real-time response chunks
- **Server-Sent Events**: Parses SSE format
- **Error Recovery**: Handles malformed chunks

#### 4. Output Layer
- **Real-time Display**: Immediate chunk printing
- **Statistics**: Token counts and speed calculation
- **Formatting**: Clean output with metrics

---

## 🔒 Security Considerations

### API Key Management

```bash
# ❌ Bad: Hardcoding keys
python3 open_chat_cli.py --api-key "gsk_abc123..."

# ✅ Good: Using environment variables
export GROQ_API_KEY="gsk_abc123..."
python3 open_chat_cli.py --backend groq

# ✅ Better: Using .env file (with python-dotenv)
echo "GROQ_API_KEY=gsk_abc123..." > .env
```

### Network Security

- Use HTTPS for remote connections
- Validate SSL certificates
- Consider VPN for sensitive conversations
- Don't send confidential data to public APIs

---

## 🚀 Performance Tips

### Optimize Token Usage

```bash
# Use smaller models for simple tasks
--model llama3:8b  # Instead of llama3:70b

# Clear history frequently for long sessions
# Use: clear chat command
```

### Improve Speed

```bash
# Local deployment for low latency
python3 open_chat_cli.py --host http://localhost:11434

# Use quantized models (vLLM)
--model TheBloke/Llama-2-7B-Chat-AWQ

# GPU acceleration
# Ensure backend uses GPU: nvidia-smi
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Support for additional backends (OpenAI, Anthropic)
- [ ] Configuration file support (.yaml/.toml)
- [ ] Conversation export/import
- [ ] Multi-turn conversation branching
- [ ] Voice input/output
- [ ] Web interface version
- [ ] Docker containerization

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **OpenAI**: For the tiktoken library
- **Ollama**: For the excellent local LLM platform
- **vLLM**: For high-performance inference
- **Groq**: For ultra-fast LLM APIs
- **prompt-toolkit**: For rich terminal features

---
