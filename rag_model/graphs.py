from rag_model.schema import State, Search


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
