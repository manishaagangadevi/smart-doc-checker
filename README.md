# Smart Doc Checker

This project is a submission for the Flexprice x Pathway Hackathon. It's an AI-powered tool that analyzes multiple documents to find and report contradictions.

## Features

* Upload multiple PDF documents for analysis.
* AI-powered contradiction detection using a Large Language Model (Gemini 1.5 Flash).
* Simulated Flexprice integration to track usage (documents analyzed and reports generated).
* Simulated Pathway integration to monitor a "live" policy file for updates and trigger re-analysis.

## Setup and Usage

1.  **Clone the repository and navigate into the project directory.**
2.  **Create a virtual environment:** `python -m venv .venv`
3.  **Activate the environment:**
    * Windows: `.venv\Scripts\activate`
    * Mac/Linux: `source .venv/bin/activate`
4.  **Install the required packages:** `pip install -r requirements.txt`
5.  **Create a `.env` file** and add your Google AI API key: `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
6.  **Run the Streamlit app:** `streamlit run app.py`