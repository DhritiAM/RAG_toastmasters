import faiss
import json
import numpy as np

class Retriever:
    def __init__(self, index_path, metadata_path, model, chunks_path):
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.model = model   # sentence-transformer
        self.chunks_path = chunks_path
        
    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        similarities, ids = self.index.search(query_emb, top_k)

        print("\n\n ids:",ids)
        print("\n\n similarities:",similarities)
        
        results = []
        for i, idx in enumerate(ids[0]):

            with open(self.chunks_path+"/"+self.metadata[idx]["source_file"], "r", encoding="utf-8") as f:
                chunks_file = json.load(f)
                text = chunks_file[self.metadata[idx]["chunk_id"]]["text"]

            results.append({
                "text": self.metadata[idx]["source_file"],
                "chunk_id": self.metadata[idx]["chunk_id"],
                "global_id": self.metadata[idx]["global_id"],
                "score": float(similarities[0][i]),
                "text_info": text
            })
        return results



