"""
scripts/llm_backend.py
=======================

Backend-pluggable text generation layer for the AKG-constrained storytelling
system.

This module provides a single, backend-agnostic entry point for language model
inference.  The active backend is selected at runtime via the ``LLM_BACKEND``
environment variable, allowing the experimental setup to be reproduced on
different infrastructure without code modification.

Supported backends
------------------
* ``"groq"`` — Groq Cloud inference API via the official ``groq`` Python
  client.  Low-latency, suitable for iterative generation loops with retry.
* ``"ollama"`` — *Reserved; not yet implemented.*  Placeholder branch exists
  to document the extension point for local model serving.
* ``"openai"`` — *Reserved; not yet implemented.*  Placeholder branch exists
  for OpenAI-compatible API endpoints.

Environment variables (``.env``)
----------------------------------
::

    LLM_BACKEND=groq
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=llama3-8b-8192
    GEN_TEMPERATURE=0.7
    GEN_MAX_TOKENS=256

All variables are read once at module import time.  Missing required variables
raise ``EnvironmentError`` at call time (not import time) to keep the module
importable in test environments where generation is mocked.

Stability features
------------------
* Request timeout of 30 s is enforced on every Groq API call.
* On any network or API exception, one automatic retry is attempted after an
  exponential backoff delay of 0.5 s.  If the retry also fails, a clean
  ``RuntimeError`` is raised with context.
* A post-call delay of 0.35 s is inserted after every successful API response
  to avoid burst rate limiting on the Groq free tier.

Design notes
------------
* ``generate_text`` is a pure I/O function: given the same prompt and a fixed
  temperature it produces the same distribution of outputs.  Setting
  ``GEN_TEMPERATURE=0.0`` makes generation effectively deterministic for
  reproducible baselines.
* The module contains no retry-on-mismatch logic, no emotion detection, and
  no AKG references.  All higher-level orchestration lives in
  ``story_generator.py``.
* The ``groq`` client is instantiated lazily and cached to avoid per-call
  connection overhead in tight generation loops.

Dependencies
------------
* ``groq`` (official Groq Python client)
* ``python-dotenv``
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

_ENV_PATH: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "groq").lower().strip()
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
GEN_TEMPERATURE: float = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_MAX_TOKENS: int = int(os.environ.get("GEN_MAX_TOKENS", "256"))

# Stability constants — not user-configurable to keep the experimental
# setup reproducible without additional env variables.
_REQUEST_TIMEOUT: int = 30       # seconds; passed to Groq client as timeout
_BACKOFF_DELAY: float = 0.5      # seconds; wait before single retry attempt
_POST_CALL_DELAY: float = 0.35   # seconds; throttle after every successful call

# ---------------------------------------------------------------------------
# Backend client singletons
# ---------------------------------------------------------------------------

_groq_client = None


def _get_groq_client():
    """Return a cached Groq client, initialising it on first call.

    Raises
    ------
    EnvironmentError
        If ``GROQ_API_KEY`` is not set.
    ImportError
        If the ``groq`` package is not installed.
    """
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment before calling generate_text()."
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required for LLM_BACKEND='groq'. "
                "Install it with: pip install groq"
            ) from exc

        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _call_groq_api(prompt: str) -> str:
    """Execute a single Groq API call and return the raw response text.

    Separated from retry logic so the retry wrapper can call it cleanly.

    Parameters
    ----------
    prompt:
        Full prompt string.

    Returns
    -------
    str
        Assistant reply, stripped of leading/trailing whitespace.
    """
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=GEN_TEMPERATURE,
        max_tokens=GEN_MAX_TOKENS,
        timeout=_REQUEST_TIMEOUT,
    )
    return response.choices[0].message.content.strip()


def _generate_groq(prompt: str) -> str:
    """Generate text using the Groq Cloud API with one-shot retry on failure.

    Wraps ``_call_groq_api`` with:
    * Network/API exception catching on the first attempt.
    * Exponential backoff (``_BACKOFF_DELAY`` s) before a single retry.
    * Clean ``RuntimeError`` if the retry also fails.
    * Post-call delay (``_POST_CALL_DELAY`` s) after every successful call
      to avoid burst rate limiting.

    Parameters
    ----------
    prompt:
        Full prompt string, including any system-level instructions or
        in-context examples.

    Returns
    -------
    str
        Generated text content only, with leading/trailing whitespace stripped.

    Raises
    ------
    RuntimeError
        If both the initial attempt and the single retry raise an exception.
    """
    try:
        result = _call_groq_api(prompt)
    except Exception as first_exc:
        # Exponential backoff before single retry.
        time.sleep(_BACKOFF_DELAY)
        try:
            result = _call_groq_api(prompt)
        except Exception as second_exc:
            raise RuntimeError(
                f"Groq API call failed after one retry.\n"
                f"  First error  : {first_exc}\n"
                f"  Second error : {second_exc}"
            ) from second_exc

    # Post-call throttle to avoid burst rate limiting.
    time.sleep(_POST_CALL_DELAY)
    return result


def _generate_ollama(prompt: str) -> str:  # pragma: no cover
    """[PLACEHOLDER] Generate text via a local Ollama server.

    Not yet implemented.  Raises ``NotImplementedError`` to fail fast and
    clearly when this branch is accidentally selected.

    Parameters
    ----------
    prompt:
        Full prompt string.
    """
    raise NotImplementedError(
        "Ollama backend is not yet implemented. "
        "Set LLM_BACKEND=groq in your .env file."
    )


def _generate_openai(prompt: str) -> str:  # pragma: no cover
    """[PLACEHOLDER] Generate text via an OpenAI-compatible API endpoint.

    Not yet implemented.  Raises ``NotImplementedError`` to fail fast and
    clearly when this branch is accidentally selected.

    Parameters
    ----------
    prompt:
        Full prompt string.
    """
    raise NotImplementedError(
        "OpenAI backend is not yet implemented. "
        "Set LLM_BACKEND=groq in your .env file."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_text(prompt: str) -> str:
    """Generate text from *prompt* using the configured LLM backend.

    Routes the call to the backend selected by the ``LLM_BACKEND`` environment
    variable.  All generation parameters (model, temperature, max tokens) are
    read from the environment; they are not exposed as function arguments to
    ensure that the experimental setup is fully specified by ``.env`` and not
    by call-site defaults.

    Stability guarantees (Groq backend)
    ------------------------------------
    * Request timeout: ``30 s``.
    * On any network or API error: one automatic retry after ``0.5 s`` backoff.
    * After every successful call: ``0.35 s`` post-call delay to avoid burst
      rate limiting.
    * If both attempts fail: raises ``RuntimeError`` with both error messages.

    Parameters
    ----------
    prompt:
        Full prompt string.  The caller is responsible for constructing
        prompts that include any necessary context, instructions, or in-context
        examples.  This function performs no prompt augmentation.

    Returns
    -------
    str
        Generated text content, stripped of leading and trailing whitespace.
        The string may be empty if the model returns an empty completion.

    Raises
    ------
    EnvironmentError
        If required credentials (e.g., ``GROQ_API_KEY``) are missing.
    RuntimeError
        If the API call fails on both the initial attempt and the single retry.
    NotImplementedError
        If ``LLM_BACKEND`` is set to a backend that has not yet been
        implemented (``"ollama"``, ``"openai"``).
    ValueError
        If ``LLM_BACKEND`` is set to an unrecognised value, or *prompt* is
        empty.

    Examples
    --------
    ::

        from scripts.llm_backend import generate_text

        text = generate_text("Continue this story: She opened the letter and...")
        print(text)
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    if LLM_BACKEND == "groq":
        return _generate_groq(prompt)
    elif LLM_BACKEND == "ollama":
        return _generate_ollama(prompt)
    elif LLM_BACKEND == "openai":
        return _generate_openai(prompt)
    else:
        raise ValueError(
            f"Unrecognised LLM_BACKEND: {LLM_BACKEND!r}. "
            f"Supported values: 'groq', 'ollama', 'openai'."
        )