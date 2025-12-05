import os
from typing import Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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
        )
        print("done")
        return ChatHuggingFace(llm=llm)
    except Exception:
        print("not done")
        return None