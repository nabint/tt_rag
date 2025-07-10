from typing import List, Literal, TypedDict
from typing_extensions import Annotated
from langchain_core.documents import Document


class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str
