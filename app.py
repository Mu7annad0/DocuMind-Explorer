# Import necessary libraries
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# Import custom modules for RAG functionality
from rag.chain import ask_question, create_chain
from rag.config import Config
from rag.ingestor import Ingestor
from rag.retriever import create_retriever
from rag.uploader import upload_file

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMP,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )

# Cache the QA chain building function
@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    # Upload files and get their paths
    file_paths = upload_file(files)
    # Ingest documents into a vector database
    vector_db = Ingestor().ingest(doc_files=file_paths)
    # Create a retriever for the vector database
    retriever = create_retriever(llm, vector_db)
    # Create and return the QA chain
    return create_chain(llm, retriever)
 
# Asynchronous function to ask questions to the chain
async def ask_chain(question: str, chain):
    full_response = ""
    async for event in ask_question(chain, question, session_id="session-id-42"):
        if isinstance(event, str):
            # Accumulate string responses
            full_response += event
            yield full_response
        elif isinstance(event, list):
            # Yield list responses (likely source documents)
            yield event

# Function to handle document upload in the sidebar
def show_upload_documents():
    st.sidebar.title("🚀 DocuMind Explorer")
    # st.sidebar.subheader("Unleash Your Documents")
    # File uploader for PDF documents
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files to fuel the AI", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        # Display a warning if no files are uploaded
        st.sidebar.warning("Feed me PDFs, and I shall grant you wisdom! 📚✨")
        return None
    # Build and return the QA chain if files are uploaded
    with st.spinner("Decoding the secrets within your documents..."):
        return build_qa_chain(uploaded_files)

# Function to display the chat interface
def show_chat_interface(chain):
    st.title("🧠 Converse with Your Digital Library")
    
    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Greetings, knowledge seeker! What mysteries shall we unravel from your documents today?",
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Pose your query to the all-knowing AI..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Give me a moment, I'm processing............")
            full_response = ""
            async def process_response():
                nonlocal full_response
                async for response in ask_chain(prompt, chain):
                    if isinstance(response, str):
                        # Update response in real-time
                        full_response = response
                        message_placeholder.markdown(full_response + "▌")
                    elif isinstance(response, list):
                        # Display relevant source documents
                        st.write("📜 Relevant Scrolls of Knowledge:")
                        for i, doc in enumerate(response):
                            with st.expander(f"Tome {i+1}"):
                                st.write(doc.page_content)
                return full_response

            # Run the asynchronous response processing
            full_response = asyncio.run(process_response())
            message_placeholder.markdown(full_response)
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Check if conversation limit is reached
    if Config.CONV_MESSAGES > 0 and Config.CONV_MESSAGES <= len(st.session_state.messages):
        st.warning(
            "You've reached the limits of this conversation. Refresh to embark on a new quest for knowledge!"
        )

# Set up the Streamlit page configuration
st.set_page_config(page_title="DocuMind Explorer", layout="wide", page_icon="🚀")

# Customize the app's appearance
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #3C3D37;  /* Dark blue-gray color for main background */
#     }
#     [data-testid="stSidebar"] {
#         background-color: #181C14;  /* Slightly lighter blue-gray for sidebar */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Main app logic
chain = show_upload_documents()
if chain:
    show_chat_interface(chain)
else:
    st.info("Upload your sacred texts in the sidebar to begin our journey of discovery! 📚🔍")