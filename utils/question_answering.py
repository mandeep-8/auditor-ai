# utils/question_answering.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
import time
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from utils.session_logger import get_session_logger
from config import COLLECTION_NAME, SESSIONS_DIR, LOG_DIR, OUTPUT_DIR
from functools import lru_cache

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_session_persist_directory(session_id):
    """Get session-specific Chroma persist directory."""
    return os.path.join(SESSIONS_DIR, session_id, "chroma_db")

def initialize_vector_store(session_id, session_logger):
    """Initialize Chroma vector store for a specific session."""
    persist_directory = get_session_persist_directory(session_id)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        session_logger.log(
            component="Vector Store",
            message=f"Removed existing Chroma directory for session {session_id}",
            decision="Accepted",
            reason="Ensuring clean vector store",
            level="DEBUG",
            context={"persist_directory": persist_directory}
        )
    
    try:
        vector_store = Chroma(
            collection_name=f"{COLLECTION_NAME}_{session_id}",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        session_logger.log(
            component="Vector Store",
            message=f"Created Chroma collection {COLLECTION_NAME}_{session_id}",
            decision="Accepted",
            reason="Session-specific collection initialized",
            level="INFO",
            context={"persist_directory": persist_directory}
        )
        return vector_store
    except (ValueError, RuntimeError) as e:
        session_logger.log(
            component="Vector Store",
            message=f"Creating Chroma collection {COLLECTION_NAME}_{session_id}",
            decision="Rejected",
            reason=f"Initialization Error: {str(e)}",
            level="ERROR",
            context={"persist_directory": persist_directory}
        )
        raise

def index_documents(session_id, session_logger):
    """Index valid Governance Documents in Chroma for a specific session."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    docs = [
        Document(page_content=result["text"], metadata={"source": result["name"]})
        for result in st.session_state.governance_doc_validation["results"]
        if result["valid"] and result["text"]
    ]
    if not docs:
        session_logger.log(
            component="Vector Store",
            message="Indexing Governance Documents",
            decision="Rejected",
            reason="No valid Governance Documents to index",
            level="ERROR",
            context={}
        )
        raise ValueError("No valid Governance Documents to index.")

    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"{COLLECTION_NAME}_{session_id}",
            persist_directory=get_session_persist_directory(session_id)
        )
        session_logger.log(
            component="Vector Store",
            message=f"Indexed {len(chunks)} chunks in Chroma collection {COLLECTION_NAME}_{session_id}",
            decision="Accepted",
            reason="Documents indexed successfully",
            level="INFO",
            context={"files": [doc.metadata["source"] for doc in docs]}
        )
        return vector_store
    except (ValueError, RuntimeError) as e:
        session_logger.log(
            component="Vector Store",
            message=f"Chroma Indexing for session {session_id}",
            decision="Rejected",
            reason=f"Indexing Error: {str(e)}",
            level="ERROR",
            context={}
        )
        raise

def get_qa_chain():
    """Initialize RetrievalQA chain for the current session."""
    session_id = st.session_state.session_id
    session_logger = get_session_logger(LOG_DIR, session_id)
    
    vector_store = index_documents(session_id, session_logger)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.0
    )
    prompt_template = ChatPromptTemplate.from_template(
"""
    Strictly extract the most specific and relevant answer from the context. Answer in 1-5 words, comma-separated for lists. Use exact terms or phrases from the context. If the answer is not found, return "NotFound". Do not infer. Do not explain.

    - Focus on capturing precise, factual details.
    - Prefer specific nouns or named entities over generalizations.
    - Do not rephrase or add assumptions.
    - Do not return partial or vague answers.

    Question: {question}
    Context: {context}
"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    st.session_state.qa_chain = qa_chain
    session_logger.log(
        component="QA Chain",
        message=f"RetrievalQA chain initialized for session {session_id}",
        decision="Accepted",
        reason="Chain created successfully",
        level="INFO",
        context={}
    )
    return qa_chain

@lru_cache(maxsize=1000)
def cached_answer_question(question, context):
    """Cache answer for a question using the QA chain."""
    qa_chain = st.session_state.qa_chain
    session_logger = get_session_logger(LOG_DIR, st.session_state.session_id)
    start_time = time.time()
    
    try:
        response = qa_chain.invoke({"query": question, "context": context})
        answer = response["result"].strip()
        sources = [doc.metadata["source"] for doc in response["source_documents"]]
        end_time = time.time()
        
        session_logger.log_llm_call(
            component="QA Processing",
            message=f"Question: {question}",
            prompt=f"Question: {question}\nContext: {context}",
            response=answer,
            model="gpt-4o-mini",
            start_time=start_time,
            end_time=end_time,
            context={
                "question": question,
                "sources": sources,
                "answer_received": answer
            }
        )
        return answer, sources
    except (ValueError, RuntimeError) as e:
        session_logger.log(
            component="QA Processing",
            message=f"Question: {question}",
            decision="Error",
            reason=f"Answer Error: {str(e)}",
            level="ERROR",
            context={"question": question}
        )
        return "Error", []

def answer_question(qa_chain, question, session_logger):
    """Answer a question using the QA chain."""
    answer, sources = cached_answer_question(question, st.session_state.cdd_text)
    
    session_logger.log(
        component="QA Processing",
        message="Answer Question",
        decision="Answered" if answer != "Error" else "Error",
        context={
            "question": question,
            "answer": answer,
            "sources": sources,
            "llm_details": {
                "model": "gpt-4o-mini"
            }
        },
        level="DEBUG" if answer != "Error" else "ERROR"
    )
    return answer