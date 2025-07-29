from rag_model.faiss_index_utils import get_index
from rag_model.schema import State, Search
from rag_model.config import llm, prompt, embedding_model
from langchain_core.messages import HumanMessage, SystemMessage

vector_store = get_index("user_reviews_index", embedding_model)


def analyze_query(state: State):
    """Analyze the user's question and determine search parameters."""
    print("ANALYZING AND OPTIMIZING QUERY....")

    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])

    print(f"OPTIMIZED QUERY.... {query}")

    return {"query": query}


def retrieve(state: State):
    """Retrieve relevant documents from the vector store."""

    all_docs = []
    seen_doc_ids = set()

    for subquery in state["query"]["queries"]:
        sub_query_text = subquery["query"]
        print(f"Retrieving for subquery: {sub_query_text}")

        docs = vector_store.similarity_search(sub_query_text, k=4)

        for doc in docs:
            if doc.metadata.get("chunk_id") not in seen_doc_ids:
                all_docs.append(doc)
                seen_doc_ids.add(doc.metadata.get("chunk_id"))

    return {"context": all_docs}


def generate(state: State):
    """Generate an answer based on the retrieved context."""
    print("GENERATING RESULT...")

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    # messages = prompt.invoke({"question": state["question"], "context": docs_content})

    messages = [
        SystemMessage(
            content="""
                You are a helpful and polite customer care assistant.
                Always address the user warmly and professionally,
                provide clear and concise answers, and if necessary,
                reassure them or guide them to the next step.
                Also don't make the answer too long.
                If you don't know the answer, politely inform the user that you are unable to assist."""
        ),
        HumanMessage(
            content=f"Question: {state['question']}\n\nContext:\n{docs_content}"
        ),
    ]

    response = llm.invoke(messages)

    return {"answer": response.content}
