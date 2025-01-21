import requests
import streamlit as st
from streamlit_chat import message
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Hi... I am Friday. Sudarsan's personal chatbot!")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are Friday, Sudarsan's personal chatbot.
    
    Your role is to assist a hiring manager by answering questions about Sudarsan's professional and personal background, 
    making him a standout candidate for their company.
    
    Answer all questions using the provided context only. 
    Please provide the most accurate response based on the question.
    Your response should be to the point and answer the question only.

    If a question falls outside the provided context, respond politely by stating that you are not trained on that information 
    and will check with Sudarsan for clarification.

    
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to load documents from GitHub repository
def load_documents():
    """Loads the 'about_me.pdf' file from the local directory and creates vector embeddings."""
    if not st.session_state.get("vectors_loaded", False):  # Check if already loaded
        try:
            # Local file path for the about_me.pdf file
            about_file_path = "./about_me.pdf"

            # Load the PDF file using LangChain's PyPDFLoader
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            loader = PyPDFLoader(about_file_path)
            st.session_state.docs = loader.load()

            # Create vector embeddings
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            # Single confirmation message
            st.success("The document has been successfully loaded and processed!")
            st.session_state["vectors_loaded"] = True  # Mark vectors as loaded
        except Exception as e:
            st.error(f"An error occurred during document loading: {e}")
    else:
        st.info("The document is already loaded.")

        
# Initialize session state keys
if "input" not in st.session_state:
    st.session_state["input"] = ""  # Initialize with an empty string
if "responses" not in st.session_state:
    st.session_state["responses"] = ["Hi! I am Friday. How can I assist you?"]
if "requests" not in st.session_state:
    st.session_state["requests"] = []
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "vectors_loaded" not in st.session_state:
    st.session_state["vectors_loaded"] = False

# Function to handle user input
def handle_user_input():
    """Handles user input and generates a chatbot response."""
    user_input = st.session_state["input"]  # Get the input value from session state
    if not user_input.strip():
        return  # Do nothing if the input is empty

    if "vectors" not in st.session_state or not st.session_state["vectors"]:
        st.error("Click here first.")
        return

    # Generate response
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": user_input})

    # Append to session state
    st.session_state.requests.append(user_input)
    st.session_state.responses.append(response["answer"])

    # Clear input field
    st.session_state["input"] = ""


# UI Components
response_container = st.container()
input_container = st.container()

# Chat History
with response_container:
    for i in range(len(st.session_state["responses"])):
        # Display chatbot messages
        message(st.session_state["responses"][i], is_user=False, key=f"bot_{i}")
        # Display user messages
        if i < len(st.session_state["requests"]):
            message(st.session_state["requests"][i], is_user=True, key=f"user_{i}")

# Input Section
with input_container:
    st.text_input(
        "Ask me anything about Sudarsan:",
        key="input",
        on_change=handle_user_input  # Call the function when the input changes (Enter key is pressed)
    )

# Button to load documents
if st.button("Click here first"):
    load_documents()
    #if st.session_state.get("vectors_loaded", False):  # Ensure flag is checked
        #st.success("Documents have been successfully loaded!")
