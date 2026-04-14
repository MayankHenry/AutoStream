import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_retriever():
    """Loads the Markdown KB, chunks it, and returns a Chroma retriever."""
    
    # 1. Load the knowledge base
    file_path = "autostream_kb.md"
    
    with open(file_path, "r") as file:
        kb_content = file.read()

    # 2. Split the document based on Markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = markdown_splitter.split_text(kb_content)

    # 3. Initialize HuggingFace local embedding model (NO API KEY NEEDED)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create the local vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 5. Return it as a retriever object
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever

# Quick test to make sure it works
if __name__ == "__main__":
    retriever = setup_retriever()
    results = retriever.invoke("How much is the Pro plan?")
    print(f"Retrieved Context: {results[0].page_content}")