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
    elif "all_roles" in q:
        return "Role"
    elif "presiding officer" in q:
        return "Role"
    elif "leadership" in q:
        return "Leadership"
    elif "contest" in q:
        return "Contest"
    else:
        return "Generic"


def vectorize_json_folder(folder_path, index_file="vector_index.faiss", metadata_file="metadata.json", 
                          update_chunks=True, create_category_indices=True):
    """
    Vectorize chunks and create FAISS indices.
    
    Args:
        folder_path: Path to folder containing chunk JSON files
        index_file: Path to save main index (for backward compatibility)
        metadata_file: Path to save metadata
        update_chunks: Whether to update chunk files with global_id
        create_category_indices: If True, creates separate indices per category for latency reduction
    """
    # Load a sentence-transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and compact

    all_texts = []
    all_metadata = []
    g_id = 0 # Helps find the right chunk
    
    # Group chunks by category for separate indices
    category_data = {}  # {category: {"texts": [], "metadata": [], "indices": []}}

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Process chunks for vectorization and update with global_id
            updated_data = []
            for chunk in data:
                # Add global_id to chunk
                chunk["global_id"] = g_id
                if update_chunks:
                    updated_data.append(chunk)
                
                category = infer_category(filename)
                
                # Add to vectorization data
                all_texts.append(chunk["text"])
                all_metadata.append({
                    "source_file": filename,
                    "chunk_id": chunk["id"],
                    "category": category,
                    "global_id": g_id
                })
                
                # Group by category for separate indices
                if create_category_indices:
                    if category not in category_data:
                        category_data[category] = {"texts": [], "metadata": [], "indices": []}
                    category_data[category]["texts"].append(chunk["text"])
                    category_data[category]["metadata"].append({
                        "source_file": filename,
                        "chunk_id": chunk["id"],
                        "category": category,
                        "global_id": g_id
                    })
                    category_data[category]["indices"].append(g_id)
                
                g_id += 1
            
            # Write updated chunks back to file if requested
            if update_chunks:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=2)

    # Convert texts to embeddings
    print(f"Generating embeddings for {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # Build main FAISS index (for backward compatibility)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save main index and metadata
    faiss.write_index(index, index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Main vector index saved as {index_file}")
    print(f"Metadata saved as {metadata_file}")
    
    # Create separate indices per category for latency reduction
    category_indices = {}
    if create_category_indices:
        print("\nCreating category-specific indices for latency reduction...")
        base_dir = os.path.dirname(index_file)
        
        for category, cat_data in category_data.items():
            if len(cat_data["texts"]) == 0:
                continue
                
            print(f"  Creating index for category '{category}' ({len(cat_data['texts'])} chunks)...")
            
            # Generate embeddings for this category
            cat_embeddings = model.encode(cat_data["texts"], show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(cat_embeddings)
            
            # Create index for this category
            cat_index = faiss.IndexFlatIP(dim)
            cat_index.add(cat_embeddings)
            
            # Save category index
            cat_index_file = os.path.join(base_dir, f"vector_index_{category}.faiss")
            faiss.write_index(cat_index, cat_index_file)
            
            # Save category metadata
            cat_metadata_file = os.path.join(base_dir, f"metadata_{category}.json")
            with open(cat_metadata_file, "w", encoding="utf-8") as f:
                json.dump(cat_data["metadata"], f, ensure_ascii=False, indent=2)
            
            category_indices[category] = {
                "index": cat_index,
                "index_file": cat_index_file,
                "metadata": cat_data["metadata"],
                "metadata_file": cat_metadata_file,
                "global_id_to_local_idx": {gid: idx for idx, gid in enumerate(cat_data["indices"])}
            }
            
            print(f"    Saved: {cat_index_file} ({len(cat_data['texts'])} vectors)")
        
        print(f"\nCreated {len(category_indices)} category-specific indices")
    
    return index, all_metadata, model, category_indices
