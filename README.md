# My LangChain Projects

This repository is a personal collection of projects, experiments, and applications built using the powerful [LangChain](https://www.langchain.com/) framework. My goal is to explore, learn, and showcase various concepts of Large Language Models (LLMs), including Retrieval-Augmented Generation (RAG), Agents, Chains, and more.

Each project is contained within its own dedicated directory and includes all the necessary files to get it up and running.

---

### Table of Contents

- [Getting Started](#getting-started)
- [Projects](#projects)
    - [1. YouTube Transcript Q&A Chatbot](#1-youtube-transcript-qa-chatbot)
    - [2. Code_help Extension](#2-code_help-extension)

---

### Getting Started

Before diving into the projects, you'll need to set up your environment.

#### Prerequisites

- **Python 3.8+**
- **Git**
- **API Keys:** Many of these projects require API keys for services like Hugging Face, OpenAI, or other LLM providers.

#### General Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **API Key Configuration:**
    Each project directory contains a `.env.example` file showing the required environment variables:
    ```bash
    cp project-name/.env.example project-name/.env
    ```
    Fill in your API keys in the newly created `.env` file.  
    **Never commit `.env` files to Git.**

---

### Projects

---

#### 1. YouTube Transcript Q&A Chatbot

A command-line chatbot that fetches the transcript of a specified YouTube video and allows you to ask questions about its content.  
This project demonstrates a simple but effective Retrieval-Augmented Generation (RAG) flow.

- **Technologies:** LangChain, Hugging Face, FAISS, `youtube-transcript-api`
- **Key Concepts:**
    - Retrieval-Augmented Generation (RAG)
    - Document Loading & Chunking
    - Embeddings & Vector Stores (FAISS)
    - Conversational Chains & Memory

---

#### 2. Code_help Extension

A lightweight LangChain-powered tool designed to analyze user code inputs and classify or assist based on predefined logic.  
Inspired by your earlier compact decision-tree-style project, this extension uses LLM reasoning to generate suggestions, classify patterns, or help debug based on user queries.

- **Technologies:** LangChain, OpenAI/Hugging Face LLMs
- **Key Concepts:**
    - Agent-based code assistance
    - LLM-driven rule reasoning
    - Minimal, modular flow for fast code help
    - Demonstrates how agents can extend traditional static logic with dynamic natural-language reasoning

---

### Contact

Feel free to connect with me to discuss these projects or other AI/ML topics!

- **LinkedIn:** https://www.linkedin.com/in/md-tayyab-rabbani-757653291
