from langgraph.graph import START, StateGraph

from rag_model.nodes import analyze_query, retrieve, generate, return_result
from rag_model.schema import State

print("Setting up RAG model...")
# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("return_result", return_result)

# Add edges
graph_builder.add_edge(START, "analyze_query")
graph_builder.add_edge("analyze_query", "retrieve")
graph_builder.add_edge("retrieve", "generate")

graph_builder.add_conditional_edges(
    "generate",
    lambda state: state["next_step"],
    {
        "retrieve": "retrieve",
        "return_result": "return_result",
    },
)

print("Compiling the graph...")

# Compile the graph
graph = graph_builder.compile()


# Example usage
def run_rag_query(question: str):
    """Run a RAG query through the graph."""
    initial_state = {"question": question, "iteration":0}
    result = graph.invoke(initial_state)

    return result["answer"]


# Example usage:
if __name__ == "__main__":
    print("Running RAG query...")
    while input("Do you want to run a RAG query? (yes/no): ").strip().lower() != "no":
        user_question = input("Enter your question: ")
        answer = run_rag_query(user_question)
        print(f"Answer: {answer}")
