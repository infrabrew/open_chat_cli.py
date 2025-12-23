#!/usr/bin/env python3
"""
Interactive CLI chat client for Ollama / vLLM / Groq with **streaming** output.
After every turn it shows:  [Tokens used: 123 | Speed: 456.7 T/s]

This script provides a terminal-based chat interface that connects to various
LLM backends (Ollama, vLLM, or Groq) and displays real-time streaming responses
with token usage statistics and generation speed.
"""
import argparse, json, os, sys, time, tiktoken
import requests
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

# Initialize tiktoken encoder for token counting
# Using GPT-3.5-turbo's encoding as a standard approximation
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Configuration dictionary for supported backend services
# Each backend has an environment variable and default host URL
BACKENDS = {
    "ollama": {"env": "OLLAMA_HOST", "default_host": "http://192.168.0.31:11435"},
    "vllm":   {"env": "VLLM_HOST",   "default_host": "http://localhost:8000"},
    "groq":   {"env": "GROQ_API_KEY","default_host": "https://api.groq.com"},
}

def parse_args():
    """
    Parse command-line arguments for the chat client.
    
    Returns:
        argparse.Namespace: Parsed arguments containing backend, model, host, and API key
    """
    p = argparse.ArgumentParser(description="Interactive chat with streaming & token stats.")
    p.add_argument("--backend", choices=BACKENDS.keys(), default="ollama",
                   help="Backend service to use (ollama, vllm, or groq)")
    p.add_argument("--model", help="Model name (defaults provided per backend)")
    p.add_argument("--host", help="Override host URL for the backend service")
    p.add_argument("--api-key", help="API key (required for Groq backend)")
    return p.parse_args()

def get_config(args):
    """
    Build configuration from arguments and environment variables.
    
    Determines the host URL, API key, and model name based on the selected
    backend and provided arguments. Falls back to environment variables and
    defaults when arguments are not provided.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        tuple: (host_url, api_key, model_name)
    
    Exits:
        If Groq backend is selected without an API key
    """
    cfg = BACKENDS[args.backend]
    
    # Get host URL: command-line arg > environment variable > default
    host = args.host or os.getenv(cfg["env"], cfg["default_host"])
    
    # Get API key (only needed for Groq)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") if args.backend=="groq" else None
    
    # Verify API key for Groq backend
    if args.backend=="groq" and not api_key:
        sys.exit("Groq requires --api-key or env OPENAI_API_KEY")
    
    # Set default models for each backend if not specified
    if args.backend=="ollama" and args.model is None:
        args.model = "llama3"
    if args.backend=="vllm" and args.model is None:
        args.model = "meta-llama/Meta-Llama-3-8B-Instruct"
    if args.backend=="groq" and args.model is None:
        args.model = "llama3-8b-8192"
    
    return host, api_key, args.model

def count_tokens(text):
    """
    Count the number of tokens in a text string.
    
    Uses tiktoken's GPT-3.5-turbo encoder to estimate token count.
    This provides a reasonable approximation for most models.
    
    Args:
        text (str): The text to tokenize
    
    Returns:
        int: Number of tokens in the text
    """
    return len(ENC.encode(text))

def streaming_request(host, api_key, model, messages, backend):
    """
    Send a chat completion request and stream the response in real-time.
    
    Makes an HTTP POST request to the backend's chat completions endpoint
    with streaming enabled. Prints each chunk as it arrives and tracks
    token usage for statistics.
    
    Args:
        host (str): Base URL of the backend service
        api_key (str): API key for authentication (if required)
        model (str): Model name to use for generation
        messages (list): List of message dictionaries with 'role' and 'content'
        backend (str): Backend type ('ollama', 'vllm', or 'groq')
    
    Returns:
        tuple: (assistant_text, prompt_tokens, completion_tokens)
            - assistant_text: Complete response from the AI
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
    
    Raises:
        requests.HTTPError: If the API request fails
    """
    # Build the request payload
    req = {"model": model, "messages": messages, "stream": True, "temperature": 0.7}
    
    # Set up HTTP headers
    headers = {"Content-Type": "application/json"}
    
    # Configure endpoint and authentication based on backend
    if backend == "groq":
        url = f"{host}/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Ollama and vLLM use similar endpoints
        url = f"{host}/v1/chat/completions"
    
    # Make the streaming request
    r = requests.post(url, headers=headers, json=req, stream=True)
    r.raise_for_status()  # Raise exception for HTTP errors
    r.encoding = 'utf-8'  # Ensure proper text encoding
    
    # Initialize variables for tracking the response
    assistant_chunks = []  # Store all response chunks
    prompt_tokens = None   # Will be populated from usage data if available
    
    # Process the streaming response line by line
    for line in r.iter_lines(decode_unicode=True):
        # Skip empty lines
        if not line or not line.strip():
            continue
        
        # Parse Server-Sent Events format (data: prefix)
        if line.startswith("data: "):
            data = line[6:]  # Remove "data: " prefix
            
            # Check for end-of-stream marker
            if data == "[DONE]":
                break
            
            try:
                # Parse the JSON chunk
                chunk = json.loads(data)
            except json.JSONDecodeError as e:
                # Skip malformed chunks (can happen with network splits)
                continue
            
            # Extract token usage information if available
            # Some backends provide this in the final chunk
            if "usage" in chunk and prompt_tokens is None:
                prompt_tokens = chunk["usage"]["prompt_tokens"]
            
            # Extract the content delta (new text in this chunk)
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                piece = delta["content"]
                assistant_chunks.append(piece)
                # Print immediately for streaming effect
                print(piece, end="", flush=True)
    
    # Print newline after streaming completes
    print()
    
    # Combine all chunks into the full response
    assistant_text = "".join(assistant_chunks)
    
    # Count completion tokens from the generated text
    completion_tokens = count_tokens(assistant_text)
    
    # If prompt tokens weren't provided by backend, calculate manually
    if prompt_tokens is None:
        # Concatenate all message content and count tokens
        prompt_tokens = count_tokens("".join(m["content"] for m in messages))
    
    return assistant_text, prompt_tokens, completion_tokens

def chat_loop(host, api_key, model, backend):
    """
    Main interactive chat loop.
    
    Continuously prompts the user for input, sends it to the AI backend,
    displays the streaming response, and shows token usage statistics.
    Maintains conversation history and handles special commands.
    
    Args:
        host (str): Base URL of the backend service
        api_key (str): API key for authentication (if required)
        model (str): Model name to use for generation
        backend (str): Backend type ('ollama', 'vllm', or 'groq')
    """
    # Initialize command history for arrow key navigation
    history = InMemoryHistory()
    
    # Store conversation messages (alternating user/assistant)
    messages = []
    
    # Limit history to prevent context length errors (413 Too Large)
    MAX_HISTORY = 10  # Keep last 10 messages (5 exchanges)
    
    while True:
        try:
            # Prompt user for input with history support
            user = prompt("You: ", history=history)
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D gracefully
            print("\nBye!")
            break
        
        # Skip empty inputs
        if not user.strip():
            continue
        
        # Special command to clear conversation history
        if user.strip().lower() == "clear chat":
            messages = []
            print("Chat cleared. If you need anything else, just let me know!")
            continue
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user})
        
        # Show AI response prefix
        print("AI: ", end="", flush=True)
        
        # Record start time for speed calculation
        t0 = time.time()
        
        # Send request and get streaming response
        assistant, prompt_tk, comp_tk = streaming_request(host, api_key, model, messages, backend)
        
        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": assistant})
        
        # Trim history if it exceeds maximum length
        # This prevents context window overflow and 413 errors
        if len(messages) > MAX_HISTORY:
            messages = messages[-MAX_HISTORY:]  # Keep only most recent messages
        
        # Calculate and display statistics
        total = prompt_tk + comp_tk  # Total tokens used
        elapsed = time.time() - t0    # Time taken for generation
        speed = comp_tk / elapsed     # Tokens per second
        print(f"[Tokens used: {total} | Speed: {speed:.1f} T/s]")

def main():
    """
    Main entry point for the chat client.
    
    Parses arguments, configures the backend connection, and starts
    the interactive chat loop.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Get configuration (host, API key, model)
    host, api_key, model = get_config(args)
    
    # Display connection information
    print(f"Connected to {args.backend} model {model}")
    
    # Start the interactive chat loop
    chat_loop(host, api_key, model, args.backend)

# Script entry point - only runs when executed directly
if __name__ == "__main__":
    main()
