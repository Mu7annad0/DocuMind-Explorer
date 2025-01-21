"""Streamlit App."""
import asyncio
import random
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from rag.chain import ask_question, build_qa_chain
from rag.config import Config
load_dotenv()
nest_asyncio.apply()

@st.cache_resource(show_spinner=False)
def build_chain(files):
    """return the qa Chain"""
    return build_qa_chain(files)

LOADING_MESSAGES = [
    "stand by!"
]

async def ask_chain(question: str, chain):
    """Asking"""
    full_response = ""
    assistant = st.chat_message("assistant")
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        for i, doc in enumerate(documents):
            with st.expander(f"Source #{i+1}"):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_upload_documents():
    """Show the uploaded docs"""
    holder = st.empty()
    with holder.container():
        st.header("RagBase")
        st.subheader("Get answers from your documents")
        uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        holder.empty()
        return build_chain(uploaded_files)


def show_message_history():
    """Show the message history"""
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])


def show_chat_input(chain):
    """Show the chat input for the user"""
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the async function in the event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ask_chain(prompt, chain))


# Set up the Streamlit page configuration
st.set_page_config(page_title="RagBase", page_icon="🐧")

# Custom CSS for styling
st.html(
    """
<style>
    .st-emotion-cache-p4micv {
        width: 2.75rem;
        height: 2.75rem;
    }
</style>
"""
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! What do you want to know about your documents?",
        }
    ]

# Check conversation limit
if Config.CONV_MESSAGES > 0 and Config.CONV_MESSAGES <= len(st.session_state.messages):
    st.warning(
        "You have reached the conversation limit. Refresh the page to start a new conversation."
    )
    st.stop()

# Main app logic
ch = show_upload_documents()
show_message_history()
show_chat_input(ch)
