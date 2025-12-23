#!/usr/bin/env python3
"""
Interactive CLI chat client for Ollama / vLLM / Groq with **streaming** output.
After every turn it shows:  [Tokens used: 123 | Speed: 456.7 T/s]
"""
import argparse, json, os, sys, time, tiktoken
import requests
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")

BACKENDS = {
    "ollama": {"env": "OLLAMA_HOST", "default_host": "http://192.168.0.31:11435"},
    "vllm":   {"env": "VLLM_HOST",   "default_host": "http://localhost:8000"},
    "groq":   {"env": "GROQ_API_KEY","default_host": "https://api.groq.com"},
}

def parse_args():
    p = argparse.ArgumentParser(description="Interactive chat with streaming & token stats.")
    p.add_argument("--backend", choices=BACKENDS.keys(), default="ollama")
    p.add_argument("--model", help="Model (defaults provided per backend)")
    p.add_argument("--host", help="Override host URL")
    p.add_argument("--api-key", help="API key (needed for Groq)")
    return p.parse_args()

def get_config(args):
    cfg = BACKENDS[args.backend]
    host = args.host or os.getenv(cfg["env"], cfg["default_host"])
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") if args.backend=="groq" else None
    if args.backend=="groq" and not api_key:
        sys.exit("Groq requires --api-key or env OPENAI_API_KEY")
    if args.backend=="ollama" and args.model is None:
        args.model = "llama3"
    if args.backend=="vllm" and args.model is None:
        args.model = "meta-llama/Meta-Llama-3-8B-Instruct"
    if args.backend=="groq" and args.model is None:
        args.model = "llama3-8b-8192"
    return host, api_key, args.model

def count_tokens(text):
    return len(ENC.encode(text))

def streaming_request(host, api_key, model, messages, backend):
    """Return (assistant_text, prompt_tokens, completion_tokens) by streaming."""
    req = {"model": model, "messages": messages, "stream": True, "temperature": 0.7}
    headers = {"Content-Type": "application/json"}
    if backend == "groq":
        url = f"{host}/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        url = f"{host}/v1/chat/completions"
    r = requests.post(url, headers=headers, json=req, stream=True)
    r.raise_for_status()
    r.encoding = 'utf-8'
    assistant_chunks = []
    prompt_tokens = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.strip():
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as e:
                # Skip malformed chunks - likely due to mid-stream splits
                continue
            if "usage" in chunk and prompt_tokens is None:
                prompt_tokens = chunk["usage"]["prompt_tokens"]
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                piece = delta["content"]
                assistant_chunks.append(piece)
                print(piece, end="", flush=True)
    print()  # newline after streaming
    assistant_text = "".join(assistant_chunks)
    completion_tokens = count_tokens(assistant_text)
    if prompt_tokens is None:
        prompt_tokens = count_tokens("".join(m["content"] for m in messages))
    return assistant_text, prompt_tokens, completion_tokens

def chat_loop(host, api_key, model, backend):
    history = InMemoryHistory()
    messages = []
    MAX_HISTORY = 10  # Keep last 10 messages (5 exchanges)
    while True:
        try:
            user = prompt("You: ", history=history)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user.strip():
            continue
        # Special command to clear chat history
        if user.strip().lower() == "clear chat":
            messages = []
            print("Chat cleared. If you need anything else, just let me know!")
            continue
        messages.append({"role": "user", "content": user})
        print("AI: ", end="", flush=True)
        t0 = time.time()
        assistant, prompt_tk, comp_tk = streaming_request(host, api_key, model, messages, backend)
        messages.append({"role": "assistant", "content": assistant})
        # Trim history to prevent 413 errors
        if len(messages) > MAX_HISTORY:
            messages = messages[-MAX_HISTORY:]
        total = prompt_tk + comp_tk
        elapsed = time.time() - t0
        speed = comp_tk / elapsed
        print(f"[Tokens used: {total} | Speed: {speed:.1f} T/s]")

def main():
    args = parse_args()
    host, api_key, model = get_config(args)
    print(f"Connected to {args.backend} model {model}")
    chat_loop(host, api_key, model, args.backend)

if __name__ == "__main__":
    main()