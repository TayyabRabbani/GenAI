from fastapi import APIRouter, HTTPException
from backend.app.schemas.review import CodeReviewRequest, CodeReviewResponse
from backend.app.core.reviewer import review_code

router = APIRouter()

@router.post("/review", response_model=CodeReviewResponse)
def review(request: CodeReviewRequest):
    try:
        return review_code(request.code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
