import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def infer_category(query):
    q = query.lower()
    if "grammar" in q or "grammarian" in q:
        return "Role"
    elif "evaluate" in q or "evaluation" in q:
        return "Eval"
    elif "timer" in q:
        return "Role"
    elif "ah-counter" in q or "ah counter" in q:
        return "Role"
    elif "presiding officer" in q:
        return "Role"
    elif "leadership" in q:
        return "Leadership"
    elif "contest" in q:
        return "Contest"
    else:
        return "Generic"


def vectorize_json_folder(folder_path, index_file="vector_index.faiss", metadata_file="metadata.json"):
    # Load a sentence-transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and compact

    all_texts = []
    all_metadata = []
    g_id = 0 # Helps find the right chunk

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for chunk in data:
                    all_texts.append(chunk["text"])
                    all_metadata.append({
                        "source_file": filename,
                        "chunk_id": chunk["id"],
                        "category": infer_category(filename),
                        "global_id": g_id
                    })
                    g_id += 1

    # Convert texts to embeddings
    print(f"Generating embeddings for {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    # embeddings = embeddings / np.linalg.norm(embeddings)
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Vector index saved as {index_file}")
    print(f"Metadata saved as {metadata_file}")
    return index, all_metadata, model
