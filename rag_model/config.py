from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)  # Add this import for Gemini LLM


vector_store = InMemoryVectorStore(
    HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )
)

prompt = hub.pull("rlm/rag-prompt")


# Initialize the Gemini LLM (you'll need to set your Google API key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
