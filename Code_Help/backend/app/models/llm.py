# app/models/llm.py
from typing import Optional, Dict, Any
import json
import re
import os

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from backend.app.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

# ---------- Prompt ----------
RATE_PROMPT = PromptTemplate(
    template=(
        "You are a senior Python code reviewer. Analyze the following Python code and respond "
        "STRICTLY as compact JSON with these keys:\n"
        "  - model\n"
        "  - score (0..10)\n"
        "  - summary\n"
        "  - suggestions (exactly 5)\n"
        "  - issues\n\n"
        "CODE:\n\"\"\"\n{code}\n\"\"\"\n"
    ),
    input_variables=["code"],
)

# ---------- Model ----------
def get_chat_model() -> Optional[ChatHuggingFace]:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None

    try:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=token,
            max_new_tokens=300,
            temperature=0.2,
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        logger.error(f"LLM init failed: {e}")
        return None

# ---------- JSON extraction ----------
def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}$", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None

# ---------- LLM chunk rating ----------
def llm_rate_chunk(chat: ChatHuggingFace, code: str) -> Optional[Dict[str, Any]]:
    try:
        chain = RATE_PROMPT | chat
        resp = chain.invoke({"code": code})
        content = getattr(resp, "content", resp)

        parsed = _safe_json(str(content))
        if not parsed:
            return None

        parsed["score"] = max(0, min(10, int(parsed.get("score", 0))))
        parsed["model"] = parsed.get("model", "llm")
        parsed["suggestions"] = parsed.get("suggestions", [])[:5]
        parsed["issues"] = parsed.get("issues", [])

        return parsed
    except Exception as e:
        logger.warning(f"LLM chunk failed: {e}")
        return None
