import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CORE FUNCTIONS ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an expert document analyst. Your job is to find contradictions between the documents based on the provided context.
    Analyze the following text and answer the user's question. If you find conflicting information, clearly state the contradiction and cite which part of the text it comes from.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.session_state.reports_generated += 1
        st.write("### AI Response:")
        st.write(response["output_text"])
    else:
        st.warning("Please process the documents first.")
        
# --- PATHWAY SIMULATION FUNCTIONS ---

def read_mock_policy_file():
    try:
        with open("mock_policy_page.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "mock_policy_page.txt not found. Please create it."

def simulate_policy_update():
    new_content = """
    University Policy Document - Live (UPDATED)

    Rule A-1: The final project submission deadline is Friday at Midnight.
    Rule B-1: A minimum attendance of 65% is required to pass the course.
    Rule C-1: All library books must be returned within 14 days.
    """
    with open("mock_policy_page.txt", "w") as f:
        f.write(new_content)
    time.sleep(1) 

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Smart Doc Checker", page_icon="📄")
st.title("📄 Smart Doc Checker")
st.write("Find contradictions in your documents with AI.")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_analyzed" not in st.session_state:
    st.session_state.docs_analyzed = 0
if "reports_generated" not in st.session_state:
    st.session_state.reports_generated = 0
if "live_conflict_result" not in st.session_state:
    st.session_state.live_conflict_result = ""


# Sidebar
with st.sidebar:
    st.header("Your Documents")
    uploaded_files = st.file_uploader("Upload your PDF files here", accept_multiple_files=True, type="pdf")
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                st.session_state.docs_analyzed += len(uploaded_files)
                raw_text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Documents processed successfully!")
        else:
            st.warning("Please upload at least one PDF file.")

    st.divider()
    st.header("Usage Stats")
    st.metric(label="Documents Analyzed", value=st.session_state.docs_analyzed)
    st.metric(label="Reports Generated", value=st.session_state.reports_generated)
    
    st.divider()
    st.header("Live Policy Monitor (Simulated)")
    st.write("This shows the content of the 'live' external policy file.")
    
    policy_content = read_mock_policy_file()
    st.text_area("Live Policy Content", policy_content, height=150, key="policy_display")

    if st.button("Simulate External Update"):
        if st.session_state.vector_store:
            with st.spinner("Simulating update and re-checking for conflicts..."):
                simulate_policy_update()
                updated_text = read_mock_policy_file()
                
                # Re-run analysis with the new text, providing it in the question
                docs = st.session_state.vector_store.similarity_search("Find text relevant to: " + updated_text)
                chain = get_conversational_chain()
                
                # --- IMPROVED QUESTION FOR THE AI ---
                question = f"""
                An external policy document has just been updated. Here is the new content:
                ---
                {updated_text}
                ---
                Now, please compare this new content against the original documents. 
                Identify and explain any contradictions you find.
                """
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                
                st.session_state.live_conflict_result = response["output_text"]
                st.success("Live check complete!")
                # Force a re-run to update the text_area
                st.rerun()
        else:
            st.warning("Please process a document first to compare against.")


# Main content area
st.header("Ask About Contradictions")
user_question = st.text_input("e.g., 'Is there a conflict in the submission deadlines?'")

if user_question:
    user_input(user_question)

# Display the result of the live conflict check
if st.session_state.live_conflict_result:
    st.divider()
    st.header("🚨 Live Update Conflict Found!")
    st.warning(st.session_state.live_conflict_result)