from typing import Literal

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)  # Add this import for Gemini LLM

# Initialize the Gemini LLM (you'll need to set your Google API key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add section metadata to documents
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# Initialize embeddings Model
langchain_embeddings = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},  # Specify device if needed, 'cpu' or 'cuda'
    encode_kwargs={
        "normalize_embeddings": True,
    },  # These are passed to the .encode method
)

vector_store = InMemoryVectorStore(langchain_embeddings)
_ = vector_store.add_documents(all_splits)


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]


# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    """Analyze the user's question and determine search parameters."""
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    """Retrieve relevant documents from the vector store."""
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        k=4,  # Add k parameter to limit results
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    """Generate an answer based on the retrieved context."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

# Add edges
graph_builder.add_edge(START, "analyze_query")
graph_builder.add_edge("analyze_query", "retrieve")
graph_builder.add_edge("retrieve", "generate")

# Compile the graph
graph = graph_builder.compile()

# Example usage
def run_rag_query(question: str):
    """Run a RAG query through the graph."""
    initial_state = {"question": question}
    result = graph.invoke(initial_state)
    return result["answer"]


# Example usage:
answer = run_rag_query("What are LLM-agents?")
print(answer)
