"""Shared Gemini client with retry and rate limiting."""

import os
import time
import re
import structlog

logger = structlog.get_logger()

_configured = False
_genai = None
_last_call_time = 0.0
MIN_CALL_INTERVAL = 4.0


def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai
        _genai = google.generativeai
    return _genai


def _ensure_configured():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai = _get_genai()
        genai.configure(api_key=api_key)
        _configured = True
        logger.info("Gemini configured")


def get_model_name():
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def _rate_limit_wait():
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()


def _extract_retry_delay(error_msg):
    match = re.search(r'retry in (\d+\.?\d*)s', error_msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'seconds:\s*(\d+)', error_msg)
    if match:
        return float(match.group(1))
    return 0.0


def create_model(system_instruction=None, temperature=0.1, max_output_tokens=1024):
    _ensure_configured()
    genai = _get_genai()
    model_name = get_model_name()

    gen_config = {"temperature": temperature, "max_output_tokens": max_output_tokens}

    try:
        gen_config = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
    except Exception:
        pass

    kwargs = {"model_name": model_name, "generation_config": gen_config}

    if system_instruction:
        try:
            kwargs["system_instruction"] = system_instruction
            model = genai.GenerativeModel(**kwargs)
            return model
        except TypeError:
            del kwargs["system_instruction"]
            model = genai.GenerativeModel(**kwargs)
            model._sys_prefix = system_instruction
            return model

    return genai.GenerativeModel(**kwargs)


def generate_with_retry(model, prompt, max_retries=5, base_delay=5.0, trace_id=None):
    _ensure_configured()

    prefix = getattr(model, "_sys_prefix", None)
    full_prompt = f"{prefix}\n\n---\n\n{prompt}" if prefix else prompt

    for attempt in range(max_retries + 1):
        try:
            _rate_limit_wait()
            logger.info("Gemini call", attempt=attempt + 1, prompt_len=len(full_prompt), trace_id=trace_id)

            response = model.generate_content(full_prompt)

            if hasattr(response, "text") and response.text:
                text = response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                text = response.candidates[0].content.parts[0].text.strip()
            else:
                raise ValueError("Empty Gemini response")

            logger.info("Gemini success", length=len(text), trace_id=trace_id)
            return text

        except Exception as e:
            error_msg = str(e)
            is_rate = any(kw in error_msg.lower() for kw in ["429", "quota", "rate", "resource", "too many"])

            if is_rate and attempt < max_retries:
                suggested = _extract_retry_delay(error_msg)
                backoff = base_delay * (2 ** attempt)
                wait = min(max(suggested + 1, backoff), 120.0)
                logger.warning("Rate limited", attempt=attempt + 1, wait=round(wait, 1), trace_id=trace_id)
                time.sleep(wait)
                continue
            else:
                logger.error("Gemini failed", error=error_msg[:200], trace_id=trace_id)
                raise

    raise Exception("All retries exhausted")