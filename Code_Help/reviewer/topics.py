"""The 11 dashboard topics and a helper to map free-form labels onto them.

Kept dependency-free (no Django, no LangChain) so it can be imported by models,
views, and the LLM layer without circular or heavy imports.
"""

# Canonical topic list — order is used for dashboard grouping.
TOPICS = [
    "Arrays",
    "Binary Search",
    "Linked List",
    "Recursion",
    "Dynamic Programming",
    "Stack and Queue",
    "Sliding Window",
    "Greedy",
    "Trees",
    "Graphs",
    "Miscellaneous",
]

# Common variants the LLM (or a user) might produce, mapped to a canonical topic.
_ALIASES = {
    "array": "Arrays",
    "arrays": "Arrays",
    "string": "Arrays",
    "strings": "Arrays",
    "hashing": "Arrays",
    "hash table": "Arrays",
    "hashmap": "Arrays",
    "two pointers": "Arrays",
    "binary search": "Binary Search",
    "linked list": "Linked List",
    "linkedlist": "Linked List",
    "recursion": "Recursion",
    "backtracking": "Recursion",
    "divide and conquer": "Recursion",
    "dynamic programming": "Dynamic Programming",
    "dp": "Dynamic Programming",
    "memoization": "Dynamic Programming",
    "stack": "Stack and Queue",
    "queue": "Stack and Queue",
    "stack and queue": "Stack and Queue",
    "monotonic stack": "Stack and Queue",
    "deque": "Stack and Queue",
    "sliding window": "Sliding Window",
    "greedy": "Greedy",
    "tree": "Trees",
    "trees": "Trees",
    "binary tree": "Trees",
    "bst": "Trees",
    "binary search tree": "Trees",
    "trie": "Trees",
    "graph": "Graphs",
    "graphs": "Graphs",
    "bfs": "Graphs",
    "dfs": "Graphs",
    "union find": "Graphs",
    "topological sort": "Graphs",
}


def canonical_topic(raw) -> str:
    """Map any label to one of TOPICS, defaulting to 'Miscellaneous'."""
    if not raw:
        return "Miscellaneous"
    s = str(raw).strip().lower()
    for t in TOPICS:               # exact (case-insensitive) match
        if s == t.lower():
            return t
    if s in _ALIASES:              # exact alias
        return _ALIASES[s]
    for key, val in _ALIASES.items():   # alias contained in the label
        if key in s:
            return val
    for t in TOPICS:               # topic name contained in the label
        if t.lower() in s:
            return t
    return "Miscellaneous"
