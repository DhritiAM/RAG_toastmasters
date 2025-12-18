import faiss
import json
import numpy as np
import os
from typing import Optional, List, Dict

class Retriever:
    def __init__(self, index_path, metadata_path, model, chunks_path, filters: Optional[Dict] = None):
        # Load main index (for backward compatibility and fallback)
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.model = model   # sentence-transformer
        self.chunks_path = chunks_path
        
        # Setup filtering
        self.filters = filters or {}
        self.filter_enabled = self.filters.get('enable', False)
        
        # Load category-specific indices for latency reduction
        self.category_indices = {}
        self.category_metadata = {}
        if self.filter_enabled:
            self._load_category_indices(index_path, metadata_path)
    
    def _infer_category_from_query(self, query: str) -> Optional[str]:
        """
        Infer category from query keywords.
        Returns category name or None if no match.
        """
        q = query.lower()
        
        # Contest-related keywords
        if any(kw in q for kw in ["contest", "speech contest", "evaluation contest", 
                                   "table topics contest"]):
            return "Contest"
        
        # Role-related keywords
        if any(kw in q for kw in ["grammarian", "timer", "ah counter", "ah-counter", "general evaluator",
                                   "toastmaster", "tmod", "table topics master", "topicsmaster",
                                   "sergeant at arms", "saa", "presiding officer", "role", "roles"]):
            return "Role"
        
        # Leadership-related keywords
        if any(kw in q for kw in ["president", "vpe", "vice president education", "secretary", "treasurer",
                                   "vice president membership", "vice president public relations",
                                   "executive committee", "club officer", "leadership", "officer"]):
            return "Leadership"
        
        # Evaluation-related keywords
        if any(kw in q for kw in ["evaluation", "evaluator", "feedback", "evaluate", "evaluating"]):
            return "Eval"
        
        return None  # No specific category detected
    
    def _load_category_indices(self, index_path: str, metadata_path: str):
        """Load category-specific indices if they exist for latency reduction"""
        base_dir = os.path.dirname(index_path)
        categories = ["Role", "Eval", "Leadership", "Contest", "Generic"]
        
        for category in categories:
            cat_index_path = os.path.join(base_dir, f"vector_index_{category}.faiss")
            cat_metadata_path = os.path.join(base_dir, f"metadata_{category}.json")
            
            if os.path.exists(cat_index_path) and os.path.exists(cat_metadata_path):
                try:
                    self.category_indices[category] = faiss.read_index(cat_index_path)
                    with open(cat_metadata_path, "r", encoding="utf-8") as f:
                        self.category_metadata[category] = json.load(f)
                    print(f"Loaded category index for '{category}' ({len(self.category_metadata[category])} chunks)")
                except Exception as e:
                    print(f"Warning: Could not load category index for '{category}': {e}")
        
        if self.category_indices:
            print(f"Category-specific indices loaded: {len(self.category_indices)} categories available for latency reduction")
        
    def retrieve(self, query, top_k=5, apply_filters: bool = True):
        """
        Retrieve chunks for a query, optionally applying metadata filters.
        
        LATENCY REDUCTION: If category-specific indices are available and filtering is enabled,
        this method will search only the relevant category's index (much smaller), significantly
        reducing search latency compared to searching the full index.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            apply_filters: Whether to apply metadata filters (default: True)
        """
        # Detect category from query if filtering is enabled
        detected_category = None
        use_category_index = False
        
        if apply_filters and self.filter_enabled:
            detected_category = self._infer_category_from_query(query)
            # Use category index if available and category detected
            if detected_category and detected_category in self.category_indices:
                use_category_index = True
                if self.filter_enabled:
                    print(f"\nDetected category: {detected_category} - Using category-specific index for latency reduction")
        
        # Encode query
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        
        # Search category-specific index (LATENCY REDUCTION) or main index
        if use_category_index:
            # Search only the relevant category's index - MUCH FASTER!
            cat_index = self.category_indices[detected_category]
            cat_metadata = self.category_metadata[detected_category]
            similarities, ids = cat_index.search(query_emb, top_k)
            metadata_to_use = cat_metadata
            print(f"  Searched {len(cat_metadata)} vectors (category index) vs {len(self.metadata)} (full index)")
        else:
            # Fall back to main index (no category detected or filtering disabled)
            similarities, ids = self.index.search(query_emb, top_k)
            metadata_to_use = self.metadata
            if apply_filters and self.filter_enabled and detected_category:
                print(f"  Searched full index ({len(self.metadata)} vectors) - category index not available")
            elif not apply_filters or not self.filter_enabled:
                print(f"  Searched full index ({len(self.metadata)} vectors) - filtering disabled")

        print("\n\n ids:",ids)
        print("\n\n similarities:",similarities)
        
        results = []
        for i, idx in enumerate(ids[0]):
            meta_entry = metadata_to_use[idx]
            global_id = meta_entry["global_id"]
            
            chunk_file_path = os.path.join(self.chunks_path, meta_entry["source_file"])
            with open(chunk_file_path, "r", encoding="utf-8") as f:
                chunks_file = json.load(f)
                text = chunks_file[meta_entry["chunk_id"]]["text"]

            results.append({
                "text": meta_entry["source_file"],
                "chunk_id": meta_entry["chunk_id"],
                "global_id": global_id,
                "score": float(similarities[0][i]),
                "text_info": text
            })
        
        return results



