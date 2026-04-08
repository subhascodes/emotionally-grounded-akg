"""
scripts/llm_backend.py

LLM backend using a local Ollama server for text generation.
Communicates with the Ollama HTTP API; no external cloud services.
"""

import requests

# ── Constants ────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2:7b"
TIMEOUT_SECONDS = 60


# ── Core generation ───────────────────────────────────────────────────────────

def generate_text(prompt: str) -> str:
    """
    Send a prompt to the local Ollama server and return the generated text.

    Args:
        prompt (str): The input prompt for the language model.

    Returns:
        str: Stripped generated text from the model.

    Raises:
        RuntimeError: If the server is unreachable, times out, or returns an
                      unexpected response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            raise RuntimeError(
                f"Ollama request failed: 'response' key missing in reply: {data}"
            )

        return data["response"].strip()

    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Ollama request failed: server not running — {e}")
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Ollama request failed: request timed out — {e}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: HTTP error — {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


# ── CLI test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(generate_text("Write one sentence about hope."))