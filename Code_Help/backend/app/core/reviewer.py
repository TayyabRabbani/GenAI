from backend.app.core.chunker import split_code
from backend.app.core.aggregator import aggregate_results
from backend.app.models.llm import llm_rate_chunk
from backend.app.models.heuristic import heuristic_rate
from backend.app.models.llm import get_chat_model

def review_code(code: str) -> dict:
    chunks = split_code(code)
    chat = get_chat_model()

    results = []
    for chunk in chunks:
        if chat:
            r = llm_rate_chunk(chat, chunk)
            if r is None:
                r = heuristic_rate(chunk)
        else:
            r = heuristic_rate(chunk)
        results.append(r)

    return aggregate_results(results)
