from retriever import Retriever
from generate import LocalLLMGenerator
from reranker import Reranker
from sentence_transformers import SentenceTransformer

def build_context(chunks):
    return "\n\n".join(chunks)

def build_rag_prompt(query, retrieved_chunks):

    context = build_context(retrieved_chunks)
    
    prompt = f"""
    You are a helpful Toastmasters assistant.

    Use the following context to answer the question. 
    If the answer is not clearly present, combine Toastmasters rules with general communication knowledge.
    Do NOT mention the word "chunk" or "retrieval".

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    return prompt

model = SentenceTransformer("all-MiniLM-L6-v2")

retriever = Retriever(
    index_path="../data/vectordb/vector_index.faiss",
    metadata_path="../data/vectordb/metadata.json",
    model=model,
    chunks_path="../data/chunks"
)

query = "How can I give a better report as a grammarian ?"

results = retriever.retrieve(query, top_k=5)
retrieved_chunks = []

for r in results:
    retrieved_chunks.append(r["text_info"])
    print(r["distance"], r["text"][:200], r["chunk_id"])

# reranker = Reranker()
# reranked = reranker.rerank(query, retrieved_chunks)

prompt = build_rag_prompt(query, retrieved_chunks)

localLLM = LocalLLMGenerator()

response = localLLM.generate(prompt)

print("\n\n\nResponse: ",response)

# localLLM.generate()