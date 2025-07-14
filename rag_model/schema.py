from typing import List, TypedDict
from typing_extensions import Annotated
from langchain_core.documents import Document


class SubQuery(TypedDict):
    intent: Annotated[
        str, "What this subquery is about, e.g. 'leave policy', 'evaluation process'"
    ]
    query: Annotated[str, "The sub-question related to that intent"]


class Search(TypedDict):
    queries: Annotated[
        List[SubQuery], "List of subqueries extracted from the input question"
    ]


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str
