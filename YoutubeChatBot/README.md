# YouTube Transcript Chatbot

This project is a conversational AI chatbot that uses the RAG (Retrieval-Augmented Generation) pattern to answer questions based on the transcript of a specific YouTube video. The chatbot fetches the transcript, chunks it, and uses a Hugging Face model to answer user questions with information found only in the video's content.

## Features
- Fetches and processes the transcript of a given YouTube video.
- Uses a Hugging Face embedding model to create a vector store from the transcript.
- Utilizes a conversational AI model from Hugging Face to provide accurate answers.
- Maintains chat history to provide a more natural conversational experience.

## Prerequisites

- **Python 3.8+**
- A **Hugging Face API Token**. You can get one for free by creating an account and visiting your [settings page](https://huggingface.co/settings/tokens).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Python virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file:** Copy the provided `.env.example` file and rename it to `.env`.

2.  **Add your Hugging Face API Token:** Open the newly created `.env` file and paste your token where indicated.
    ```env
    HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"
    ```

## Usage

1.  **Run the chatbot script:**
    ```bash
    python chatbot.py
    ```

2.  The script will automatically fetch the transcript and initialize the chatbot. Once you see the prompt `You:`, you can start asking questions related to the video's content.

3.  **To change the video,** simply edit the `video_id` variable in the `chatbot.py` file to the ID of the video you want to use.