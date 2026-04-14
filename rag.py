from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_retriever():
    """Loads the Markdown KB, chunks by headers, embeds locally, returns retriever."""

    with open("autostream_kb.md", "r") as f:
        kb_content = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = splitter.split_text(kb_content)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    retriever = setup_retriever()
    results = retriever.invoke("How much does the Pro plan cost?")
    print(results[0].page_content)