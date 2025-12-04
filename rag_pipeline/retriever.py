import faiss
import json
import numpy as np
from typing import Optional, List, Dict, Set

class Retriever:
    def __init__(self, index_path, metadata_path, model, chunks_path, filters: Optional[Dict] = None):
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.model = model   # sentence-transformer
        self.chunks_path = chunks_path
        
        # Setup filtering
        self.filters = filters or {}
        self.filter_enabled = self.filters.get('enable', False)
        
        # Pre-compute category mappings for faster filtering
        if self.filter_enabled:
            self.category_to_global_ids = self._compute_category_mappings()
    
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
    
    def _compute_category_mappings(self) -> Dict[str, Set[int]]:
        """Pre-compute which global_ids belong to each category"""
        category_map = {}
        for entry in self.metadata:
            category = entry.get('category')
            if category:
                if category not in category_map:
                    category_map[category] = set()
                category_map[category].add(entry['global_id'])
        return category_map
    
    def _get_allowed_global_ids_for_category(self, category: str) -> Set[int]:
        """Get allowed global_ids for a specific category"""
        if not self.filter_enabled or category is None:
            return None  # No filtering
        
        if self.category_to_global_ids and category in self.category_to_global_ids:
            return self.category_to_global_ids[category]
        
        return set()  # Category not found, return empty set
    
    def _matches_filter(self, global_id: int, allowed_ids: Optional[Set[int]]) -> bool:
        """Check if a global_id matches the current filter criteria"""
        if not self.filter_enabled or allowed_ids is None:
            return True
        
        return global_id in allowed_ids
        
    def retrieve(self, query, top_k=5, apply_filters: bool = True):
        """
        Retrieve chunks for a query, optionally applying metadata filters.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            apply_filters: Whether to apply metadata filters (default: True)
        """
        # Detect category from query if filtering is enabled
        allowed_ids = None
        detected_category = None
        if apply_filters and self.filter_enabled:
            detected_category = self._infer_category_from_query(query)
            allowed_ids = self._get_allowed_global_ids_for_category(detected_category)
            if self.filter_enabled and detected_category:
                print(f"\nDetected category from query: {detected_category}")
        
        # If filtering is enabled, we need to retrieve more candidates to account for filtering
        retrieve_k = top_k
        if apply_filters and self.filter_enabled and allowed_ids is not None:
            # Retrieve more candidates to ensure we get enough after filtering
            # Estimate: if filter reduces by 50%, retrieve 2x; if 90%, retrieve 10x
            filter_ratio = len(allowed_ids) / len(self.metadata) if self.metadata else 1.0
            if filter_ratio < 1.0:
                retrieve_k = min(int(top_k / max(filter_ratio, 0.1)), len(self.metadata))
        
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        similarities, ids = self.index.search(query_emb, retrieve_k)

        print("\n\n ids:",ids)
        print("\n\n similarities:",similarities)
        
        results = []
        for i, idx in enumerate(ids[0]):
            global_id = self.metadata[idx]["global_id"]
            
            # Apply filter if enabled
            if apply_filters and not self._matches_filter(global_id, allowed_ids):
                continue
            
            with open(self.chunks_path+"/"+self.metadata[idx]["source_file"], "r", encoding="utf-8") as f:
                chunks_file = json.load(f)
                text = chunks_file[self.metadata[idx]["chunk_id"]]["text"]

            results.append({
                "text": self.metadata[idx]["source_file"],
                "chunk_id": self.metadata[idx]["chunk_id"],
                "global_id": global_id,
                "score": float(similarities[0][i]),
                "text_info": text
            })
            
            # Stop once we have enough results
            if len(results) >= top_k:
                break
        
        if len(results) < top_k:
            print(f"Warning: Only retrieved {len(results)}/{top_k} results after filtering.")
        
        return results



