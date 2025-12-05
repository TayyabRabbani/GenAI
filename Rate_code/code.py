import os
import json
import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

TEXT = """
import heapq
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v, w=1, bidirectional=True):
        "Add edge from u -> v with weight w"
        self.graph[u].append((v, w))
        if bidirectional:
            self.graph[v].append((u, w))

    def bfs(self, start):
        "Breadth-First Search"
        visited = set()
        queue = deque([start])
        order = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                order.append(node)
                for neighbor, _ in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return order

    def dfs(self, start):
        "Depth-First Search"
        visited = set()
        order = []

        def dfs_rec(node):
            visited.add(node)
            order.append(node)
            for neighbor, _ in self.graph[node]:
                if neighbor not in visited:
                    dfs_rec(neighbor)

        dfs_rec(start)
        return order

    def dijkstra(self, start):
        "Dijkstra’s shortest path algorithm"
        distances = {node: float("inf") for node in self.graph}
        distances[start] = 0
        pq = [(0, start)]
        parents = {start: None}

        while pq:
            curr_dist, node = heapq.heappop(pq)
            if curr_dist > distances[node]:
                continue
            for neighbor, weight in self.graph[node]:
                distance = curr_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parents[neighbor] = node
                    heapq.heappush(pq, (distance, neighbor))
        return distances, parents

    def reconstruct_path(self, parents, start, end):
        "Reconstruct path from start to end using parents"
        path = []
        while end is not None:
            path.append(end)
            end = parents[end]
        path.reverse()
        return path if path[0] == start else []

    def display(self):
        "Display adjacency list"
        for node, edges in self.graph.items():
            print(f"{node}: {edges}")


# Example usage
if __name__ == "__main__":
    g = Graph()
    g.add_edge("A", "B", 4)
    g.add_edge("A", "C", 2)
    g.add_edge("B", "C", 1)
    g.add_edge("B", "D", 5)
    g.add_edge("C", "D", 8)
    g.add_edge("C", "E", 10)
    g.add_edge("D", "E", 2)
    g.add_edge("E", "F", 3)

    print("Adjacency List:")
    g.display()

    print("\nBFS from A:", g.bfs("A"))
    print("DFS from A:", g.dfs("A"))

    distances, parents = g.dijkstra("A")
    print("\nShortest Distances from A:", distances)
    print("Shortest Path A → F:", g.reconstruct_path(parents, "A", "F"))


"""
# =========================
# Heuristic fallback rater
# =========================
def heuristic_rate(code: str, language_id: str = "python") -> Dict[str, Any]:
    """Simple, deterministic scoring that works without LLM access."""
    lines = code.splitlines()
    n_lines = len(lines)
    long_lines = sum(1 for l in lines if len(l) > 120)
    avg_len = sum(len(l) for l in lines) / max(1, n_lines)
    # rough nesting by braces/indent
    depth = 0
    max_depth = 0
    for l in lines:
        opening = len(re.findall(r"[\{\(\[]", l))
        closing = len(re.findall(r"[\}\)\]]", l))
        depth = max(0, depth + opening - closing)
        max_depth = max(max_depth, depth)
    comment_chars = 0
    for pat in (r"#.*$", r"//.*$", r"/\*[\s\S]*?\*/"):
        for m in re.finditer(pat, code, flags=re.M):
            comment_chars += len(m.group(0))
    comment_density = comment_chars / max(1, len(code))
    # issues
    issues = []
    if '"' in lines and 'return self.name"' in code:
        issues.append("Syntax error: stray quote in return statement.")
    if long_lines > 0:
        issues.append(f"{long_lines} line(s) exceed 120 chars.")
    if max_depth >= 5:
        issues.append(f"Deep nesting detected (depth {max_depth}).")
    if comment_density < 0.02:
        issues.append("Very low comment density.")
    # score (10 = good)
    score = 10
    score -= min(4, long_lines // 3)
    score -= max(0, max_depth - 3)
    if comment_density < 0.02:
        score -= 1
    score = max(0, min(10, score))
    suggestions = [
        "Fix any syntax errors before running (e.g., stray quotes).",
        "Keep functions short and focused; reduce deep nesting.",
        "Add concise comments/docstrings for non-obvious logic.",
        "Avoid very long lines; wrap at ~100–120 characters.",
        "Write unit tests for critical paths."
    ]
    return {
        "model": "heuristic-fallback",
        "score": score,
        "summary": f"Heuristic rating based on structure, comments, and line metrics for {language_id}.",
        "suggestions": suggestions,
        "issues": issues
    }


# =========================
# LLM rater via HF Inference
# =========================
RATE_PROMPT = PromptTemplate(
    template=(
        ''' "You are a senior Data Structures and Algorithms (DSA) code reviewer. "
        "Analyze the following Python code and respond STRICTLY as compact JSON with these keys:\n"
        "  - model (string)\n"
        "  - score (number 0..10, based on algorithmic correctness, efficiency, and clarity)\n"
        "  - complexity (object with 'time' and 'space' keys, each a short string)\n"
        "  - summary (string, concise 1–3 sentences)\n"
        "  - strengths (array of 3 short strings highlighting what’s done well)\n"
        "  - suggestions (array of exactly 5 short actionable improvement points)\n"
        "  - issues (array of concrete problems such as inefficiency, edge-case errors, or code smells)\n\n"
        "Guidelines: Focus on algorithm design, data structure choice, complexity, and readability. "
        "Score conservatively. Avoid generic feedback. Keep output pure JSON.\n\n"
        "CODE (Python):\n\"\"\"\n{code}\n\"\"\"\n\n"
        "Return JSON only. No explanations or formatting outside JSON."
        '''
    ),
    input_variables=["code"]
)


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
            top_p=0.9,
            repetition_penalty=1.05,
        )
        return ChatHuggingFace(llm=llm)
    except Exception:
        return None


def robust_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the last JSON object in a string."""
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Find last {...}
    m = re.search(r"\{[\s\S]*\}\s*$", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def llm_rate_chunk(chat: ChatHuggingFace, code: str) -> Optional[Dict[str, Any]]:
    try:
        chain = RATE_PROMPT | chat
        resp = chain.invoke({"code": code})
        # ChatHuggingFace returns an AIMessage; get .content
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = "\n".join(map(str, content))
        if not isinstance(content, str):
            content = str(content)
        parsed = robust_json_from_text(content)
        if not parsed:
            return None
        # minimal validation + clamping
        parsed["model"] = str(parsed.get("model", "unknown"))
        score = parsed.get("score", 0)
        if not isinstance(score, (int, float)):
            score = 0
        parsed["score"] = max(0, min(10, int(round(float(score)))))
        parsed["summary"] = str(parsed.get("summary", "")).strip()
        suggestions = parsed.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []
        parsed["suggestions"] = [str(s).strip() for s in suggestions][:5]
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        parsed["issues"] = [str(i).strip() for i in issues][:10]
        return parsed
    except Exception:
        return None


# =========================
# Code split + aggregation
# =========================
def split_code(code: str, chunk_size: int = 300, overlap: int = 0) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(code)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        # Should not happen; return empty heuristic
        return heuristic_rate("", "python")

    # Average score
    avg_score = int(round(sum(r.get("score", 0) for r in results) / len(results)))
    # Pick first non-heuristic model if any
    model = next((r["model"] for r in results if r.get("model") and r["model"] != "heuristic-fallback"), results[0].get("model", "unknown"))
    # Merge top issues/suggestions (unique, preserve order)
    def unique_keep_order(items: List[str], limit: int) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
            if len(out) >= limit:
                break
        return out

    all_sugg = []
    all_issues = []
    for r in results:
        all_sugg.extend(r.get("suggestions", []))
        all_issues.extend(r.get("issues", []))

    merged_sugg = unique_keep_order(all_sugg, 5)
    merged_issues = unique_keep_order(all_issues, 10)

    # Short combined summary
    summaries = [r.get("summary", "") for r in results if r.get("summary")]
    combined_summary = " ".join(summaries[:3]).strip()
    if len(combined_summary) > 500:
        combined_summary = combined_summary[:500] + "..."

    return {
        "model": model,
        "score": avg_score,
        "summary": combined_summary or "Aggregated rating across chunks.",
        "suggestions": merged_sugg,
        "issues": merged_issues,
    }


# =========================
# Main runner
# =========================
def rate_code(code: str) -> Dict[str, Any]:
    # Split into chunks
    chunks = split_code(code, chunk_size=300, overlap=0)

    # Try LLM
    chat = get_chat_model()
    per_chunk: List[Dict[str, Any]] = []

    if chat is None:
        # Pure heuristic for all chunks
        for ch in chunks:
            per_chunk.append(heuristic_rate(ch, "python"))
        return aggregate_results(per_chunk)

    # LLM path with heuristics as backup per chunk
    for ch in chunks:
        rated = llm_rate_chunk(chat, ch)
        if rated is None:
            rated = heuristic_rate(ch, "python")
        per_chunk.append(rated)

    return aggregate_results(per_chunk)


def print_report(report: Dict[str, Any]) -> None:
    print("\n=== CODE RATER REPORT ===")
    print(f"Model     : {report.get('model', 'unknown')}")
    print(f"Score     : {report.get('score', 0)}/10")
    print(f"Summary   : {report.get('summary', '')}")
    print("\nSuggestions:")
    for i, s in enumerate(report.get("suggestions", []), 1):
        print(f"  {i}. {s}")
    issues = report.get("issues", [])
    if issues:
        print("\nIssues:")
        for i, it in enumerate(issues, 1):
            print(f"  - {it}")
    print("=========================\n")


if __name__ == "__main__":
    report = rate_code(TEXT)
    print_report(report)
