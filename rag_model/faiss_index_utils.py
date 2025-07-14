from langchain_community.vectorstores import FAISS


def create_index(docs, index_name, embedding_model):
    faiss = FAISS.from_documents(docs, embedding_model)
    faiss.save_local(index_name)


def get_index(index_name, embedding_model):
    faiss = FAISS.load_local(
        index_name, embedding_model, allow_dangerous_deserialization=True
    )
    return faiss
