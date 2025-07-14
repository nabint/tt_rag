from data.utils import get_fixed_sized_chunks, load_pdfs_from_directory
from rag_model.faiss_index_utils import create_index
from rag_model.config import embedding_model


def main():
    docs = load_pdfs_from_directory("./data/teslatech_policy/")
    splitted_documets = get_fixed_sized_chunks(docs)

    create_index(splitted_documets, "teslatech_policy_index", embedding_model)


if __name__ == "__main__":
    main()