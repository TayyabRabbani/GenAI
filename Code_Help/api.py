import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse  # Used to send the HTML file
from pydantic import BaseModel, Field
from typing import List
from llm_model import chat_model, chain  # Import your chain and model
import uvicorn


# --- Pydantic Models for Request and Response ---

class CodeReviewRequest(BaseModel):
    """
    The incoming request body, expecting a single 'code' field.
    """
    code: str


class CodeReviewResponse(BaseModel):
    """
    The shape of the JSON response our API will return.
    This also matches the JSON structure we ask the LLM to create.
    """
    score: int = Field(..., description="The review score from 0 to 10.")
    summary: str = Field(..., description="A 1-3 sentence summary of the review.")
    suggestions: List[str] = Field(..., description="An array of 5 actionable suggestions.")
    issues: List[str] = Field(..., description="An array of concrete issues found.")


# --- FastAPI Web Server ---
app = FastAPI(
    title="AI Code Review API",
    description="An API that uses an LLM to review Python code.",
    version="1.0.0"
)


# --- API Endpoint for the Review ---

@app.post("/review", response_model=CodeReviewResponse)
def review_code(request_body: CodeReviewRequest):
    """
    Receives Python code, sends it to an LLM for review,
    and returns a structured JSON review.
    """
    if not chat_model:
        raise HTTPException(status_code=500, detail="Chat model is not available")

    try:
        code_to_review = request_body.code

        # Run your LangChain logic
        # 'resp' is already a Python dictionary because of the OutputFixingParser
        resp = chain.invoke({"code": code_to_review})

        # Validate and return it. FastAPI will automatically check
        # if this dictionary matches the 'CodeReviewResponse' model
        # and convert it to JSON for the response.
        return resp

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- API Endpoint to serve the HTML Webpage ---

@app.get("/")
async def get_homepage():
    """
    Serves the main web_ui.html file.
    """
    # Check if the file exists in the same directory
    html_file_path = "web_ui.html"
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404,
                            detail="web_ui.html not found. Make sure it's in the same directory as api.py.")

    return FileResponse(html_file_path)


if __name__ == '__main__':
    print("Starting FastAPI server on http://localhost:5000 ...")
    # This allows you to run the server just by running `python api.py`
    uvicorn.run(app, host="0.0.0.0", port=5000)

