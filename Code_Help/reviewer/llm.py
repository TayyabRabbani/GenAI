"""LLM review layer: judges whether a DSA solution is correct and reviews it.

Migrated from the old chat_model.py + llm_model.py. The chat model / chain are
built lazily and cached, so importing this module is cheap and Django management
commands (migrate, etc.) never need a working HF token.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from .topics import TOPICS, canonical_topic

load_dotenv()  # read HUGGINGFACEHUB_API_TOKEN from the .env file

# Keys here MUST stay in sync with the review() view and the front-end JS.
RATE_PROMPT = PromptTemplate(
    template=(
        "You are a senior Data Structures and Algorithms (DSA) interviewer and "
        "code reviewer. A user is solving a specific named problem. First decide "
        "whether their solution CORRECTLY and COMPLETELY solves the stated "
        "problem, then review it.\n\n"
        "Return ONLY a compact JSON object (no markdown fences, no prose) with "
        "EXACTLY these keys:\n"
        '  - "solved": boolean. Decide by asking ONE question: does this code '
        "return the correct result for every input that satisfies the problem's "
        "stated constraints? If yes, solved is true. UNREACHABLE code never "
        "makes a solution unsolved -- for example, a fallback `return []` for a "
        "'no solution' case when the problem guarantees a solution always exists "
        "is fine and must be ignored. Ignore style, naming, type hints, and "
        "micro-optimizations when deciding solved. Set false ONLY for a real bug "
        "that produces wrong output on some allowed input, wrong output format, "
        "or incomplete/missing logic.\n"
        '  - "score": a number from 0 to 10 for overall quality (correctness, '
        "efficiency, clarity). Score conservatively.\n"
        '  - "complexity": an object with "time" and "space", each a short Big-O '
        'string such as "O(n)".\n'
        '  - "summary": a concise 1-3 sentence summary of the solution and your '
        "verdict.\n"
        '  - "issues": an array of concrete problems (bugs, missed edge cases, '
        "inefficiencies). Use an empty array if there are none.\n"
        '  - "suggestions": an array of exactly 5 short, actionable improvements.\n'
        '  - "topic": the single most relevant DSA topic, chosen from EXACTLY '
        "this list: Arrays, Binary Search, Linked List, Recursion, Dynamic "
        "Programming, Stack and Queue, Sliding Window, Greedy, Trees, Graphs, "
        "Miscellaneous.\n\n"
        "A correct, working solution MUST be marked solved (true) even if it "
        "could be cleaner or faster; put any non-blocking nitpicks in "
        "suggestions, not as a reason to fail it.\n\n"
        "PROBLEM NAME: {name}\n"
        "DIFFICULTY: {difficulty}\n\n"
        "PROBLEM STATEMENT:\n{statement}\n\n"
        "USER'S SOLUTION (code):\n{solution}\n\n"
        "Return JSON only."
    ),
    input_variables=["name", "difficulty", "statement", "solution"],
)


@lru_cache(maxsize=1)
def _build_chain():
    """Build (once) the prompt | model | parser chain, or None if no token."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None
    try:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=token,
            max_new_tokens=700,
            temperature=0.2,
        )
        chat = ChatHuggingFace(llm=llm)
    except Exception:
        return None
    robust_parser = OutputFixingParser.from_llm(llm=chat, parser=JsonOutputParser())
    return RATE_PROMPT | chat | robust_parser


def chain_available() -> bool:
    return _build_chain() is not None


def _normalize(raw: dict) -> dict:
    """Coerce the LLM's JSON into the exact shape the app expects."""
    raw = raw or {}

    solved = raw.get("solved")
    if isinstance(solved, str):
        solved = solved.strip().lower() in {"true", "yes", "correct", "solved", "1"}
    solved = bool(solved)

    try:
        score = max(0.0, min(10.0, float(raw.get("score"))))
    except (TypeError, ValueError):
        score = 0.0

    comp = raw.get("complexity") or {}
    complexity = {
        "time": str(comp.get("time") or "N/A"),
        "space": str(comp.get("space") or "N/A"),
    }

    issues = [str(x) for x in (raw.get("issues") or []) if str(x).strip()]
    suggestions = [str(x) for x in (raw.get("suggestions") or []) if str(x).strip()]

    return {
        "solved": solved,
        "score": score,
        "complexity": complexity,
        "summary": str(raw.get("summary") or ""),
        "issues": issues,
        "suggestions": suggestions,
        "topic": canonical_topic(raw.get("topic")),
    }


def review_solution(name: str, statement: str, solution: str, difficulty: str) -> dict:
    """Run the LLM review and return a normalized result dict."""
    chain = _build_chain()
    if chain is None:
        raise RuntimeError("LLM unavailable: set HUGGINGFACEHUB_API_TOKEN in .env.")
    raw = chain.invoke(
        {
            "name": name,
            "difficulty": difficulty,
            "statement": statement,
            "solution": solution,
        }
    )
    return _normalize(raw)
