from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import hub

from langchain_google_genai import ChatGoogleGenerativeAI

print("Loaded embedding model")
embedding_model = HuggingFaceEmbeddings(
    model_name="./models/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loaded prompt from Hugging Face Hub")
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("Loaded LLM from chat gpt")
