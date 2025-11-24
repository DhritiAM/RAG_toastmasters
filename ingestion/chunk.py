# vectorize_text.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
import json

def chunk_text(path):

    try:
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        print("Loaded", len(documents), "document(s)")
    except Exception as e:
        print("Error loading documents:", e)
        return None

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk ~1000 characters
    chunk_overlap=50,    # overlap to preserve context ~100
    separators=["\n\n", "\n", ".", " "]  # use paragraph breaks when possible
    )

    try:
        chunks = splitter.split_documents(documents)
        print("Created", len(chunks), "chunks")
        return chunks
    except Exception as e:
        print("Error splitting documents:", e)
        return None


# def chunk_text(path):
#     try:
#         # Load text manually and close the file immediately
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()

#         # Convert to a LangChain Document
#         documents = [Document(page_content=text, metadata={"source": path})]

#         # Chunk using LangChain’s text splitter
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         chunks = text_splitter.split_documents(documents)

#         print(f"Loaded and chunked {len(chunks)} pieces from {path}")
#         return chunks

#     except Exception as e:
#         print(f"Error chunking {path}: {e}")
#         return None



def save_chunks_to_json(chunks, filename="chunks.json"):

    data = []

    for i, chunk in enumerate(chunks):
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
            meta = chunk.metadata
        else:
            text = str(chunk)
            meta = {}
        data.append({
            "id": i,
            "text": text,
            "metadata": meta
        })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


