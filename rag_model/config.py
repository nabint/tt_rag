from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import hub

from langchain_google_genai import ChatGoogleGenerativeAI

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
prompt  = hub.pull("rlm/rag-prompt")
llm     = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)