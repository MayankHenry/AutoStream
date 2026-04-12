import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load your Google API Key
load_dotenv()

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

    # 3. Initialize Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create the local vector store
    # This creates a local SQLite database in a folder called "chroma_db"
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 5. Return it as a retriever object for the agent to use
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever

# Quick test to make sure it works
if __name__ == "__main__":
    retriever = setup_retriever()
    results = retriever.invoke("How much is the Pro plan?")
    print(f"Retrieved Context: {results[0].page_content}")