import re
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

from rag.config import Config

# Define system prompt for the language model
SYS_PROMPT = """
You are an AI assistant with access to specific context. Your task is to answer questions based solely on the provided context. Follow these guidelines:

1. Carefully analyze the context given below.
2. When answering questions, break down complex queries into relevant sub-questions.
3. Address each sub-question systematically to build towards a comprehensive answer.
4. Provide a concise and direct final answer, focusing only on information relevant to the question.
5. Avoid including extraneous details or context not directly related to the query.
6. If the question cannot be answered based on the given context, says "I DON'T KNOW.

Context:
{context}

Remember, your responses should be informative yet concise, always staying within the scope of the provided context.
"""

# Initialize an empty dictionary to store chat histories
store = {}
# Function to get or create a chat history for a given session
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to remove URLs from text using regex
def remove_links(text: str) -> str:
    """
    Remove links from the given text.
    """
    # Define regex pattern for URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # Replace all URLs with an empty string and return the result
    return url_pattern.sub('', text)

# Function to format a list of documents into a single string
def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")
    return remove_links("\n".join(texts))

# Function to create a RAG (Retrieval-Augmented Generation) chain
def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    """
    Create a chain for RAG.
    """
    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    # Define the chain of operations
    chain = (
        RunnablePassthrough.assign(context=itemgetter("question")
        | retriever.with_config({"run_name": "context_retrieval"})
        | format_documents)
    | prompt
    | llm
    )
    # Wrap the chain with message history functionality
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    ).with_config({"run_name": "chain_answer"})

# Asynchronous function to process questions using the RAG chain
async def ask_question(chain: Runnable, question: str, session_id: str):
    # Stream events from the chain execution
    async for event in chain.astream_events(
        {"question": question},
        config={
            "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
            "configurable": {"session_id": session_id},
        },
        version="v2",
        include_names=["context_retriever", "chain_answer"],
    ):
        event_type = event["event"]
        # Yield retrieved context when retriever finishes
        if event_type == "on_retriever_end":
            yield event["data"]["output"]
        # Yield generated answer chunks as they become available
        if event_type == "on_chain_stream":
            yield event["data"]["chunk"].content