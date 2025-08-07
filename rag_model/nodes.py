from rag_model.faiss_index_utils import get_index
from rag_model.schema import State, Search
from rag_model.config import llm, prompt, embedding_model
from langchain_core.messages import HumanMessage, SystemMessage

change_log_index = get_index("change_logs_index", embedding_model)
user_reviews_index = get_index("user_reviews_index", embedding_model)


def analyze_query(state: State):
    """Analyze the user's question and determine search parameters."""
    print("ANALYZING AND OPTIMIZING QUERY....", state)

    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])

    return {"query": query}


def retrieve(state: State):
    """Retrieve relevant documents from the vector store."""

    all_docs = []
    seen_doc_ids = set()

    for subquery in state["query"]["queries"]:
        sub_query_text = subquery["query"]

        if state["iteration"] == 0:
            # For the first iteration, search in Change log index
            vector_store = change_log_index
        else:
            print("Switching to user reviews index for further queries.")
            vector_store = user_reviews_index

        docs = vector_store.similarity_search(sub_query_text, k=4)

        for doc in docs:
            if doc.metadata.get("chunk_id") not in seen_doc_ids:
                all_docs.append(doc)
                seen_doc_ids.add(doc.metadata.get("chunk_id"))

    return {"context": all_docs}


def generate(state: State):
    """Generate an answer based on the retrieved context."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    if state["iteration"] == 0:
        system_msg = SystemMessage(
            content="""
            You are a helpful and polite customer care assistant.
            
            The user will provide a review, complaint, or issue they are facing with the product.
            You have access to a changelog that includes version numbers and the changes made in each version.

            Your job is to:
            1. Understand the issue described by the user.
            2. Search the changelog context to determine if the issue has already been fixed. Only get the information from the changelog to check if the issue has been addressed or not.
                a. If the issue has been fixed, inform the user politely and mention the version where it was resolved.
            3. If the issue has not been fixed, Just Say NOT FOUND and nothing else.

            Always address the user warmly and professionally.
            Keep the answer clear, concise, and not too long.
            Also don't make the answer too long.
            """
        )
    else:
        system_msg = SystemMessage(
            content="""
                You are a helpful and polite customer care assistant.
                Always address the user warmly and professionally,
                provide clear and concise answers, and if necessary,
                reassure them or guide them to the next step.
                Also don't make the answer too long.
                If you don't know the answer, politely inform the user that you are unable to assist."""
        )

    messages = [
        system_msg,
        HumanMessage(
            content=f"Question: {state['question']}\n\nContext:\n{docs_content}"
        ),
    ]

    response = llm.invoke(messages)

    if response.content.startswith("NOT FOUND") and state["iteration"] == 0:
        state["next_step"] = "retrieve"
        state["iteration"] += 1
        state["context"] = None
    else:

        state["next_step"] = "return_result"
        state["answer"] = response.content

    return state


def return_result(state: State):
    return {"answer": state["answer"]}
