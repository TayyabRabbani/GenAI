from pydantic import BaseModel, Field
from typing import List

class CodeReviewRequest(BaseModel):
    code: str = Field(..., min_length=1)

class CodeReviewResponse(BaseModel):
    score: int = Field(..., ge=0, le=10)
    summary: str
    suggestions: List[str]
    issues: List[str]
