
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from chat_model import get_chat_model

TEXT_FOR_DEMO = """
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
        return path if path and path[0] == start else []

    def display(self):
        "Display adjacency list"
        for node, edges in self.graph.items():
            print(f"{node}: {edges}")
"""

# Build the model once here; imported by review_server
chat_model = get_chat_model()

# JSON parser + auto-fixer to coerce imperfect outputs into valid JSON
json_parser = JsonOutputParser()
robust_parser = OutputFixingParser.from_llm(llm=chat_model, parser=json_parser)

# Include format instructions to strongly bias the model to perfect JSON
format_instructions = (
    '{'
    '"score": <number 0-10>, '
    '"summary": "<1-3 sentence summary>", '
    '"suggestions": ["<exactly 5 short actionable items>"], '
    '"issues": ["<concrete issues found>"]'
    '}'
)

RATE_PROMPT = PromptTemplate(
    template=(
        "You are a senior Python code reviewer. Analyze the following Python code and respond "
        "STRICTLY as compact JSON with these keys:\n"
        "  - score (number 0..10, higher is better)\n"
        "  - summary (string, 1–3 sentences)\n"
        "  - suggestions (array of exactly 5 short strings, actionable)\n"
        "  - issues (array of short strings, concrete problems you see)\n\n"
        "Guidelines: Score conservatively. Mention syntax/style issues, readability, testability, "
        "and potential bugs. Be precise and non-generic.\n\n"
        "Return JSON only. No extra text.\n\n"
        "Expected JSON shape:\n{format_instructions}\n\n"
        "CODE (Python):\n\"\"\"\n{code}\n\"\"\"\n"
    ),
    input_variables=["code"],
    partial_variables={"format_instructions": format_instructions},
)

# Final chain exposed to the server
chain = RATE_PROMPT | chat_model | robust_parser

if __name__ == "__main__":
    print("--- Running llm_model.py self-test ---")
    resp = chain.invoke({"code": TEXT_FOR_DEMO})
    print("\n--- Test invocation result (as dict) ---")
    print(resp)
    print(f"\nAccessing the score: {resp.get('score')}")
    print(f"\nAccessing the summary: {resp.get('summary')}")
    print(f"\nAccessing the suggestions: {resp.get('suggestions')}")
