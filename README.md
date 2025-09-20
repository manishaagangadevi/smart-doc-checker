# 📄 Smart Doc Checker

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red) ![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)

This AI-powered tool analyzes multiple documents to find and report contradictions, saving time and preventing errors.

## 🚀 Overview

Organizations often have multiple documents like rulebooks, contracts, and guidelines that can develop contradictions over time. Manually checking for these conflicts is tedious, time-consuming, and prone to human error.

The Smart Doc Checker is an AI agent designed to solve this problem. It allows users to upload multiple documents and uses a Large Language Model to intelligently identify and explain conflicting information. This project fulfills all the requirements for the "Smart Doc Checker" problem statement, including simulated integrations for Flexprice and Pathway.

## ✨ Features

* **Multi-Document Analysis:** Upload and process multiple PDF documents at once.
* **AI-Powered Q&A:** Ask questions in natural language to find contradictions and get detailed explanations.
* **Simulated Usage Billing (Flexprice):** A visible counter tracks the number of documents analyzed and reports generated, demonstrating the billing integration requirement.
* **Simulated Live Updates (Pathway):** A monitor demonstrates the ability to watch an external document for changes and automatically re-trigger conflict detection.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **AI/ML:**
    * **Core Framework:** LangChain
    * **LLM:** Google Gemini 1.5 Flash
    * **Embeddings:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
    * **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Programming Language:** Python

## ⚙️ Setup and Installation
Follow these steps to run the project locally.

**1. Clone the repository:**
git clone https://github.com/manishaagangadevi/smart-doc-checker.git
cd smart-doc-checker


**2. Create and activate a virtual environment:**
# Create the environment
python -m venv .venv
# Activate on Windows
.venv\Scripts\activate
# Activate on Mac/Linux
source .venv/bin/activate

**3. Install the required packages:**
pip install -r requirements.txt

**4. Add your API Key:**
Create a file named .env in the project's root directory.
Add your Google AI API key to the file like this:
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
 
**5. Create the mock policy file:**
Create a file named mock_policy_page.txt in the root directory.
Add some initial rules to it, for example:
Rule A-1: The final project submission deadline is Monday at 5 PM.

**6. Run the application:**
streamlit run app.py

**How It Works**
The application follows a modern RAG (Retrieval-Augmented Generation) pattern:
1. Text Extraction & Chunking: Text is extracted from uploaded PDFs and split into smaller, manageable chunks.
2. Embedding: Each chunk is converted into a numerical vector (embedding) using a local Sentence Transformer model.
3. Vector Store: The embeddings are stored in a high-speed FAISS vector store for efficient similarity searching.
4. Q&A and Analysis: When a user asks a question, the app retrieves the most relevant chunks from the vector store and passes them, along with the question, as context to the Gemini LLM to generate a detailed answer.


Author
Manisha Gangadevi
GitHub: https://github.com/manishaagangadevi