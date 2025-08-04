import os
from operator import itemgetter
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- Load Environment Variables ---
# Make sure you have a .env file with your HUGGINGFACEHUB_API_TOKEN
load_dotenv()

# --- 1. Document Ingestion and Processing (The RAG part) ---
print("Fetching YouTube transcript...")
video_id = "Gfr50f6ZBvo"  # The ID of the YouTube video
transcript_text = ""
try:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id)

    raw_transcript = fetched.to_raw_data()
    transcript = "".join(entry["text"] for entry in raw_transcript)
    print("Transcript fetched successfully.")

except TranscriptsDisabled:
    print(f"Transcripts are disabled for video ID: {video_id}.")
except Exception as e:
    print(f"An error occurred while fetching the transcript: {e}")

if transcript:
    # --- 2. Text Splitting ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 3. embedding generation
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # --- 4. Retriever ---
    print("Creating retriever...")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("Setup complete. You can now start chatting.")
else:
    print("Could not process the video. Please check the video ID or try another one.")
    retriever = None # Set retriever to None if setup fails

# --- 5. Language Model (LLM) Setup ---
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=150,
    temperature=0.7,
)

# Wrap it as a chat model
model = ChatHuggingFace(llm=llm)
# --- 6. Prompt Engineering ---
# The prompt now includes a placeholder for chat_history
prompt_template = PromptTemplate(
    template="""
You are a helpful assistant who answers questions based on the context and conversation history.
Answer ONLY from the information given in the context.
If the context is insufficient to answer the question, politely say that you don't have enough information.

TRANSCRIPT CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

YOUR ANSWER:
""",
    input_variables=['context', 'chat_history', 'question']
)

# --- 7. RAG Chain Definition ---

def format_docs(retrieved_docs):
    """Combines the content of retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def format_chat_history(history):
    """Formats the chat history list into a readable string."""
    return "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in history])

# This chain retrieves context based on the user's question
retrieval_chain = RunnableParallel({
        "context": (lambda x: x['question']) | retriever | format_docs,
        "question": (lambda x: x['question']),
        "chat_history": (lambda x: format_chat_history(x['chat_history']))
})

# The full chain that combines retrieval, prompting, the model, and parsing
main_chain = retrieval_chain | prompt_template | model | StrOutputParser()


# --- 8. Conversational Loop ---
if retriever:
    chat_history = []
    print("\n--- YouTube Chatbot ---")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # The input for the chain is a dictionary
        chain_input = {
            "question": user_input,
            "chat_history": chat_history
        }

        # Invoke the chain to get the AI's response
        ai_response = main_chain.invoke(chain_input)
        print(f"\nAI: {ai_response}")

        # Update the chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=ai_response))
else:
    print("\nChatbot cannot start due to an issue during setup.")

