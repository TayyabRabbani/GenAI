from typing import List, Dict, Any

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    avg_score = round(sum(r["score"] for r in results) / len(results))

    suggestions = []
    issues = []

    for r in results:
        suggestions.extend(r.get("suggestions", []))
        issues.extend(r.get("issues", []))

    def uniq(xs, limit):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
            if len(out) == limit:
                break
        return out

    return {
        "score": avg_score,
        "summary": "Aggregated review across code sections.",
        "suggestions": uniq(suggestions, 5),
        "issues": uniq(issues, 10),
    }
