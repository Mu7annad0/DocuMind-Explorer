"""The Chain"""
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq


from rag.config import Config
from rag.uploader import upload_file
from rag.indexing import Indexing
from rag.retriever import create_retriever


SYS_PROMPT = """
You are an AI assistant with access to a specific context. 
Your primary objective is to provide accurate and relevant responses strictly based on the given context. Follow these guidelines:

1. **Thorough Context Analysis:**  
   - Carefully review and understand the provided context before responding.  
   - Identify key details and concepts to ensure comprehensive coverage.

2. **Structured Responses:**  
   - Decompose complex queries into logical sub-questions.  
   - Address each sub-question methodically to construct a well-rounded answer.  
   - Ensure responses maintain a logical flow and build towards a complete understanding.

3. **Scope Adherence:**  
   - Only respond using the information available in the provided context.  
   - If a question cannot be answered based on the given context, respond with: "I DON'T KNOW."

4. **Clarity and Conciseness:**  
   - Deliver informative yet concise answers.  
   - Avoid speculation or adding information beyond the provided context.

Context:
{context}

Stay within the boundaries of the given context and prioritize accuracy in your responses.
"""


SUBQUERIES_GENERATION_PROMPT = """
You are a helpful assistant that generates multiple search queries based on a single input query \n.
Generate multiple search queries related to: {question} \n
Output (2 queries):"""

def _init_llm() -> BaseLanguageModel:
    """Initialize the LLM"""
    return ChatGroq(
        model="llama-3.3-70b-versatile")


store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get the chat message history for a given session id"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Fuse the results of multiple retrievers using reciprocal rank fusion.
    The results will be sorted by the sum of the reciprocal ranks.
    """
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results



def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    """
    Create a chain for RAG.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    subquery_prompt = PromptTemplate(
        input_variables=["question"],
        template=SUBQUERIES_GENERATION_PROMPT
    )
    generate_queries = (
        subquery_prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

    retrieval_chain = (
        generate_queries
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # Main chain assembly
    chain = (
        RunnablePassthrough.assign(
            context = retrieval_chain,
            question = itemgetter("question")
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    ).with_config({"run_name": "chain_answer"})


async def ask_question(chain: Runnable, question: str, session_id: str):
    """Asks a question to the chain and returns a stream of events.

    Args:
        chain (Runnable): The chain to ask the question to.
        question (str): The question to ask.
        session_id (str): The session ID to use.

    Yields:
        A stream of events from the chain.
    """
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
            yield event["data"]["chunk"]


def build_qa_chain(files):
    """Builds a QA chain from a list of file paths."""
    file_paths = upload_file(files)
    vector_db = Indexing().index(doc_files=file_paths)
    llm = _init_llm()
    retriever = create_retriever(llm, vector_db)
    return create_chain(llm, retriever)
