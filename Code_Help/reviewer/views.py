import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from . import llm
from .models import SolvedProblem, statement_hash
from .topics import TOPICS, canonical_topic

DIFFICULTIES = ["Easy", "Medium", "Hard"]


def _parse_json(request):
    try:
        return json.loads(request.body or "{}"), None
    except json.JSONDecodeError:
        return None, JsonResponse({"detail": "Invalid JSON body."}, status=400)


@ensure_csrf_cookie
@require_GET
def index(request):
    """Submission page (sets the csrftoken cookie for the AJAX POSTs)."""
    return render(request, "reviewer/index.html", {
        "topics": TOPICS,
        "difficulties": DIFFICULTIES,
    })


@require_POST
def review(request):
    """Judge + review a solution. Does NOT persist anything."""
    payload, err = _parse_json(request)
    if err:
        return err

    name = (payload.get("name") or "").strip()
    statement = (payload.get("statement") or "").strip()
    solution = (payload.get("solution") or "").strip()
    difficulty = (payload.get("difficulty") or "Medium").strip()

    if not name or not statement or not solution:
        return JsonResponse(
            {"detail": "Problem name, statement, and solution are all required."},
            status=400,
        )
    if difficulty not in DIFFICULTIES:
        difficulty = "Medium"

    if not llm.chain_available():
        return JsonResponse(
            {"detail": "LLM unavailable. Check HUGGINGFACEHUB_API_TOKEN in .env and restart the server."},
            status=503,
        )

    try:
        result = llm.review_solution(name, statement, solution, difficulty)
    except Exception as exc:  # network/model errors surface to the UI
        return JsonResponse({"detail": f"Review failed: {exc}"}, status=502)

    return JsonResponse(result)


@require_POST
def save(request):
    """Persist a solved problem (called by the UI only after solved == true)."""
    payload, err = _parse_json(request)
    if err:
        return err

    name = (payload.get("name") or "").strip()
    statement = (payload.get("statement") or "").strip()
    solution = (payload.get("solution") or "").strip()
    difficulty = (payload.get("difficulty") or "Medium").strip()
    topic = canonical_topic(payload.get("topic"))

    if not name or not statement or not solution:
        return JsonResponse({"detail": "Missing required fields."}, status=400)
    if difficulty not in DIFFICULTIES:
        difficulty = "Medium"

    complexity = payload.get("complexity") or {}
    score = payload.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None

    obj, created = SolvedProblem.objects.update_or_create(
        statement_sha=statement_hash(statement),
        defaults={
            "name": name,
            "statement": statement,
            "solution": solution,
            "difficulty": difficulty,
            "topic": topic,
            "score": score,
            "time_complexity": str(complexity.get("time") or "")[:60],
            "space_complexity": str(complexity.get("space") or "")[:60],
            "summary": payload.get("summary") or "",
        },
    )
    return JsonResponse({"ok": True, "id": obj.id, "created": created, "topic": topic})


@require_GET
def dashboard(request):
    """All solved problems, grouped by topic in canonical order."""
    problems = list(SolvedProblem.objects.all())

    groups = []
    topic_counts = {}
    for t in TOPICS:
        items = [p for p in problems if p.topic == t]
        topic_counts[t] = len(items)
        if items:
            groups.append({"topic": t, "items": items, "count": len(items)})

    difficulty_counts = {
        d: sum(1 for p in problems if p.difficulty == d) for d in DIFFICULTIES
    }

    return render(request, "reviewer/dashboard.html", {
        "total": len(problems),
        "groups": groups,
        "topics": TOPICS,
        "topic_counts": topic_counts,
        "difficulty_counts": difficulty_counts,
    })
