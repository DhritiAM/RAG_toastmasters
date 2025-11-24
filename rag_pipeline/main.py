from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict
import os
import yaml

try:
    from .retriever import Retriever
    from .generate import LocalLLMGenerator
    from .reranker import Reranker
    from .query_classifier import BroadQuestionClassifier
    from .core_lookup import CoreLookup
except ImportError:
    # Fallback to direct imports when running as a script
    from retriever import Retriever
    from generate import LocalLLMGenerator
    from reranker import Reranker
    from query_classifier import BroadQuestionClassifier
    from core_lookup import CoreLookup
import yaml


class RAG:
    """
    A complete RAG (Retrieval-Augmented Generation) pipeline for Toastmasters queries.
    
    This class orchestrates the entire pipeline:
    1. Query classification (broad vs specific)
    2. Retrieval from vector database (for specific queries)
    3. Reranking of retrieved chunks
    4. Generation of final response using LLM
    """
    
    def __init__(
        self,
        config_path: Optional[str] = "../config.yaml",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        chunks_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        core_knowledge_path: Optional[str] = None,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        verbose: Optional[bool] = None
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Configuration is loaded from config.yaml by default. Any explicit parameters
        will override the config file values. Set config_path=None to skip config loading
        and use only explicit parameters (with fallback to hardcoded defaults).
        
        Args:
            config_path: Path to config.yaml file (default: "../config.yaml"). 
                        Set to None to skip config loading entirely.
            index_path: Path to FAISS vector index (overrides config)
            metadata_path: Path to metadata JSON file (overrides config)
            chunks_path: Path to directory containing chunk JSON files (overrides config)
            embedding_model_name: Name of sentence transformer model (overrides config)
            reranker_model_name: Name of cross-encoder reranker model (overrides config)
            llm_model_name: Name of LLM model for generation (overrides config)
            core_knowledge_path: Path to core knowledge JSON file (overrides config)
            top_k_retrieve: Number of chunks to retrieve initially (overrides config)
            top_k_rerank: Number of top chunks to use after reranking (overrides config; ignored unless reranker enabled, unless explicitly provided)
            verbose: Whether to print debug information (overrides config)
            
        Raises:
            FileNotFoundError: If config_path is provided but file doesn't exist
            ValueError: If config file exists but is invalid YAML
            RuntimeError: If config file cannot be read
        """
        # Load config from YAML file
        config = {}
        if config_path is not None:
            # If config_path is explicitly provided, it must exist
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_path}. "
                    "Please provide a valid config file path or set config_path=None to use defaults."
                )
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    config = config_data.get('rag_pipeline', {})
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing config file {config_path}: {e}")
            except Exception as e:
                raise RuntimeError(f"Error loading config from {config_path}: {e}")
        # If config_path is None, config remains empty and defaults will be used
        
        # Use config values as defaults, but allow explicit overrides
        self.verbose = verbose if verbose is not None else config.get('verbose', False)
        self.top_k_retrieve = top_k_retrieve or config.get('top_k_retrieve', 10)
        reranker_config = config.get('reranker', {})
        self.reranker_enabled = reranker_config.get('enable', True)
        if top_k_rerank is not None:
            self.top_k_rerank = top_k_rerank
        elif self.reranker_enabled:
            self.top_k_rerank = reranker_config.get('top_k', 4)
        else:
            self.top_k_rerank = None
        
        # Set paths and model names
        index_path = index_path or config.get('index_path', "../data/vectordb/vector_index.faiss")
        metadata_path = metadata_path or config.get('metadata_path', "../data/vectordb/metadata.json")
        chunks_path = chunks_path or config.get('chunks_path', "../data/chunks")
        embedding_model_name = embedding_model_name or config.get('embedding_model', "all-MiniLM-L6-v2")
        reranker_model_name = reranker_model_name or config.get('reranker_model', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        llm_model_name = llm_model_name or config.get('llm_model', "phi3")
        core_knowledge_path = core_knowledge_path or config.get('core_knowledge_path', "../data/static/core_knowledge.json")
        
        # Store chunk settings for reference (used during ingestion)
        self.chunk_size = config.get('chunk_size', 400)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        
        # Store paths for external access (e.g., evaluation)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunks_path = chunks_path
        
        # Initialize embedding model
        if self.verbose:
            print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize query classifier
        self.query_classifier = BroadQuestionClassifier()
        
        # Initialize core lookup for broad questions
        self.core_lookup = CoreLookup(filepath=core_knowledge_path)
        
        # Initialize retriever
        if self.verbose:
            print("Initializing retriever...")
        self.retriever = Retriever(
            index_path=index_path,
            metadata_path=metadata_path,
            model=self.embedding_model,
            chunks_path=chunks_path
        )
        
        # Initialize reranker (optional)
        self.reranker = None
        if self.reranker_enabled:
            if self.verbose:
                print("Loading reranker model...")
            self.reranker = Reranker(model_name=reranker_model_name)
        elif self.verbose:
            print("Reranker disabled via config; skipping reranker initialization.")
        
        # Initialize LLM generator
        self.generator = LocalLLMGenerator(model_name=llm_model_name)
        
        if self.verbose:
            print("RAG pipeline initialized successfully!")
    
    def _build_context(self, chunks: List[str]) -> str:
        """Build context string from retrieved chunks."""
        return "\n\n".join(chunks)
    
    def _build_rag_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """Build the RAG prompt with context and query."""
        context = self._build_context(retrieved_chunks)
        
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
    
    def query(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Process a query through the entire RAG pipeline.
        
        Args:
            query: The user's question
            top_k_retrieve: Override default top_k for retrieval
            top_k_rerank: Override default top_k for reranking
            
        Returns:
            Dictionary containing:
                - response: The final answer
                - is_broad: Whether it was a broad question
                - retrieved_chunks: List of retrieved chunks (if not broad)
                - reranked_chunks: List of reranked chunks (if reranker used)
                - selected_chunks: Chunks ultimately fed to the generator
                - reranker_used: Whether reranking was applied
        """
        top_k_retrieve = top_k_retrieve or self.top_k_retrieve
        top_k_rerank = self.top_k_rerank if top_k_rerank is None else top_k_rerank
        
        # Step 1: Classify query (broad vs specific)
        broad_category = self.query_classifier.classify(query)
        
        if broad_category:
            # Handle broad questions with core lookup
            if self.verbose:
                print(f"Detected broad question category: {broad_category}")
            response = self.core_lookup.get(broad_category)
            
            return {
                "response": response,
                "is_broad": True,
                "broad_category": broad_category,
                "retrieved_chunks": None,
                "reranked_chunks": None
            }
        else:
            # Handle specific questions with full RAG pipeline
            if self.verbose:
                print("Processing specific question through RAG pipeline...")
            
            # Step 2: Retrieve relevant chunks
            if self.verbose:
                print(f"Retrieving top {top_k_retrieve} chunks...")
            results = self.retriever.retrieve(query, top_k=top_k_retrieve)
            
            retrieved_chunks = []
            for r in results:
                retrieved_chunks.append(r["text_info"])
                if self.verbose:
                    print(f"  Score: {r['score']:.4f}, Chunk ID: {r['chunk_id']}, Text: {r['text'][:100]}...")
            
            # Step 3: Rerank chunks (if enabled)
            reranker_used = self.reranker_enabled and self.reranker is not None
            if reranker_used:
                if self.verbose:
                    print(f"Reranking and selecting top {top_k_rerank} chunks...")
                reranked = self.reranker.rerank(query, retrieved_chunks)
                if top_k_rerank is not None:
                    selected_chunks = reranked[:top_k_rerank]
                else:
                    selected_chunks = reranked
            else:
                if self.verbose:
                    print("Reranker disabled; using retrieved chunks directly.")
                reranked = retrieved_chunks
                if top_k_rerank is not None:
                    selected_chunks = retrieved_chunks[:top_k_rerank]
                else:
                    selected_chunks = retrieved_chunks  # use all retrieved chunks when reranker is off
            
            # Step 4: Generate response
            if self.verbose:
                print("Generating response with LLM...")
            prompt = self._build_rag_prompt(query, selected_chunks)
            response = self.generator.generate(prompt)
            
            return {
                "response": response,
                "is_broad": False,
                "broad_category": None,
                "retrieved_chunks": retrieved_chunks,
                "reranked_chunks": selected_chunks if reranker_used else None,
                "selected_chunks": selected_chunks,
                "reranker_used": reranker_used
            }


# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAG(verbose=True)
    
    # Example query
    query = "What is the role of the club's Vice President Education (VPE)?"
    
    # Process query
    result = rag.query(query)
    
    # Display results
    print("\n" + "="*80)
    print("QUERY:", query)
    print("="*80)
    print("\nRESPONSE:")
    print(result["response"])
    print("\n" + "="*80)
    if not result["is_broad"]:
        print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
        print(f"Used top {len(result['reranked_chunks'])} chunks after reranking")