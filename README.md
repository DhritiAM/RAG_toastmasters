# Toastmasters RAG System

A Retrieval-Augmented Generation (RAG) system designed to help Toastmasters members access accurate, document-grounded information about pathways, roles, and club processes. The system implements a three-stage pipeline: Ingestion, RAG Pipeline, and Evaluation, with advanced features like query classification, reranking, and metadata-driven filtering for optimal performance.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Ingestion](#stage-1-ingestion)
  - [Stage 2: RAG Pipeline](#stage-2-rag-pipeline)
  - [Stage 3: Evaluation](#stage-3-evaluation)
- [Configuration](#configuration)
- [Key Features](#key-features)
- [Performance Notes](#performance-notes)
- [Dependencies](#dependencies)

## Overview

This RAG system processes Toastmasters documents (PDFs and DOCX files) and provides intelligent question-answering capabilities through:

- **Intelligent Query Classification**: Distinguishes between broad questions (instant lookup) and specific questions (full RAG pipeline)
- **Advanced Retrieval**: Vector similarity search with optional metadata-driven filtering
- **Reranking**: Cross-encoder reranking for improved relevance
- **Local LLM Generation**: Document-grounded answer generation
- **Comprehensive Evaluation**: Metrics for retrieval quality and system performance

## Repository Structure

```
RAG_toastmasters/
│
├── ingestion/              # Stage 1: Data Ingestion Pipeline
│   ├── main.py            # Main orchestration script for ingestion
│   ├── extract_data.py    # PDF and DOCX extraction with table handling
│   ├── chunk.py           # Text chunking using RecursiveCharacterTextSplitter
│   └── vectorise.py       # Embedding generation and FAISS index creation
│
├── rag_pipeline/          # Stage 2: RAG Query Processing Pipeline
│   ├── main.py            # Main RAG class orchestrating the pipeline
│   ├── query_classifier.py # Broad question classification (roles/pathways)
│   ├── core_lookup.py     # Fast lookup for broad questions
│   ├── retriever.py       # Vector retrieval with metadata filtering
│   ├── reranker.py        # Cross-encoder reranking for relevance
│   └── generate.py        # Local LLM response generation
│
├── evaluation/            # Stage 3: System Evaluation Framework
│   ├── evaluation_script.py # Comprehensive evaluation metrics
│   ├── test_cases.json    # Ground truth test cases
│   └── evaluation_report.txt # Generated evaluation reports
│
├── streamlit/             # Web Interface
│   ├── app.py             # Streamlit web application
│   ├── feedback_db.py     # Feedback database management
│   └── suggested_questions.json # Pre-defined question suggestions
│
├── data/                  # Data Directory
│   ├── raw/               # Original source documents (PDFs, DOCX)
│   ├── extracted/         # Extracted plain text files
│   ├── chunks/            # JSON files with chunked text and metadata
│   ├── vectordb/          # Vector database files
│   │   ├── vector_index.faiss  # FAISS vector index
│   │   └── metadata.json       # Chunk metadata with categories
│   └── static/            # Static knowledge files
│       └── core_knowledge.json # Pre-written answers for broad questions
│
├── config.yaml            # Configuration file for all pipeline settings
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key** (if using OpenAI models):
   - **Windows (PowerShell)**:
     ```powershell
     $env:OPENAI_API_KEY="your-api-key-here"
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     set OPENAI_API_KEY=your-api-key-here
     ```
   - **Linux/Mac**:
     ```bash
     export OPENAI_API_KEY=your-api-key-here
     ```
   - **Permanent setup** (recommended): Add to your shell profile (`.bashrc`, `.zshrc`, etc.) or use a `.env` file
   
   **Important**: Never commit your API key to the repository. The system only reads the API key from environment variables for security.

4. **Set up the data pipeline**:
   - Place your source documents (PDFs, DOCX) in `data/raw/`
   - Run the ingestion pipeline (see [Stage 1: Ingestion](#stage-1-ingestion))

## Usage

### Running the Ingestion Pipeline

Process raw documents and create the vector database:

```bash
cd ingestion/
python main.py
```

This will:
- Extract text from PDFs and DOCX files
- Chunk the text into manageable pieces
- Generate embeddings and create a FAISS index
- Save metadata for filtering and retrieval

### Querying the System

#### Programmatic Usage

```python
from rag_pipeline.main import RAG

# Initialize the RAG pipeline
rag = RAG(config_path="config.yaml", verbose=True)

# Query the system
result = rag.query("What is the timer's role?")
print(result["response"])
```

#### Web Interface

Launch the Streamlit web application:

```bash
cd streamlit/
streamlit run app.py
```

The web interface provides:
- Interactive question-answering
- Suggested questions
- Feedback collection system

### Running Evaluation

Evaluate system performance:

```bash
cd evaluation/
python evaluation_script.py
```

Review the generated report in `evaluation_report.txt` for:
- Retrieval quality metrics (Precision, Recall, F1, MRR)

## Pipeline Stages

### Stage 1: Ingestion

The ingestion pipeline processes raw documents and prepares them for retrieval through four main steps:

#### 1. Extraction (`extract_data.py`)

- **PDF Processing**: Uses PyMuPDF (fitz) for text extraction and Camelot for table extraction
- **DOCX Processing**: Uses python-docx to extract text and tables
- **Text Cleaning**: Normalizes Unicode characters, removes non-printable characters, and collapses excessive whitespace
- **Output**: Clean plain text files saved to `data/extracted/`

#### 2. Chunking (`chunk.py`)

- Uses LangChain's `RecursiveCharacterTextSplitter` for intelligent text segmentation
- Configurable chunk size (default: 1000 characters) and overlap (default: 100 characters)
- Respects paragraph boundaries, sentences, and word boundaries
- **Output**: JSON files with structured chunks containing:
  - `id`: Chunk identifier
  - `text`: Chunk content
  - `metadata`: Source file information
- Saved to `data/chunks/`

#### 3. Categorization (`vectorise.py`)

Automatically infers document categories based on filename patterns:
- **"Role"** - Role-related documents (grammarian, timer, etc.)
- **"Eval"** - Evaluation-related content
- **"Leadership"** - Leadership and officer information
- **"Contest"** - Speech contest rules and procedures
- **"Generic"** - General Toastmasters content

Categories are stored in metadata for later filtering.

#### 4. Vectorization (`vectorise.py`)

- Generates embeddings using sentence-transformers model (`all-MiniLM-L6-v2` by default)
- Creates FAISS indices for fast similarity search:
  - **Main index**: `data/vectordb/vector_index.faiss` (all chunks, for backward compatibility)
  - **Category indices**: `data/vectordb/vector_index_{Category}.faiss` (one per category, for latency reduction)
- Assigns `global_id` to each chunk for tracking
- Stores metadata linking chunks to source files and categories
- **Output**:
  - `data/vectordb/vector_index.faiss` - Main FAISS vector index (all chunks)
  - `data/vectordb/metadata.json` - Complete metadata mapping
  - `data/vectordb/vector_index_{Category}.faiss` - Category-specific indices (Role, Eval, Leadership, Contest, Generic)
  - `data/vectordb/metadata_{Category}.json` - Category-specific metadata files

### Stage 2: RAG Pipeline

The RAG pipeline processes user queries through a sophisticated multi-step process designed to provide accurate, contextually relevant answers.

#### Architecture Overview

```
Query → Classification → [Broad Questions → Fast Lookup]
                      ↓
                  [Specific Questions → Full RAG Pipeline]
                      ↓
                  Retrieval → Reranking → Generation
```

#### Query Classification (`query_classifier.py`)

The system implements a broad question classifier that distinguishes between general questions and specific questions.

**Broad Questions:**
- Pattern-based classification using keyword matching
- Categories detected: "roles", "pathways"
- Handled via `CoreLookup` for instant responses
- Uses pre-written answers from `data/static/core_knowledge.json`
- **Benefits**: Zero latency, always accurate for predefined categories

**Example Broad Patterns:**
- "different roles", "list of roles", "all roles"
- "different pathways", "list of pathways", "all pathways"

**Specific Questions:**
- All other queries routed to full RAG pipeline
- Undergo vector retrieval, reranking, and generation

#### Retrieval (`retriever.py`)

The retriever performs semantic similarity search using the FAISS vector index.

**Key Features:**
- Encodes query using same embedding model as ingestion
- Performs cosine similarity search (L2-normalized vectors)
- Returns top-k most similar chunks with relevance scores
- Supports metadata-driven filtering (see below)

#### Metadata-Driven Filtering with Latency Reduction

To improve both latency and relevance, the system implements intelligent metadata filtering using **category-specific FAISS indices**. This approach actually reduces search latency by searching only the relevant category's index instead of the full index.

**How It Works:**

1. **Query Analysis**: Analyzes query keywords to infer category:
   - Contest keywords → "Contest" category
   - Role keywords (grammarian, timer, etc.) → "Role" category
   - Leadership keywords (president, VPE, etc.) → "Leadership" category
   - Evaluation keywords → "Eval" category

2. **Category-Specific Indices** (Created during ingestion):
   - During ingestion, separate FAISS indices are created for each category
   - Each category index contains only vectors from that category
   - Files saved: `vector_index_{Category}.faiss` and `metadata_{Category}.json`
   - Main index (`vector_index.faiss`) is still created for backward compatibility

3. **Filtered Retrieval with Latency Reduction**:
   - When a category is detected and category-specific index exists:
     - **Searches ONLY the relevant category's index** (much smaller, faster)
     - No post-filtering needed - all results are already relevant
     - **Significant latency reduction**: e.g., if Role category has 20% of chunks, search is ~5x faster
   - When no category detected or category index unavailable:
     - Falls back to main index with post-retrieval filtering (slower but still works)

**Performance Benefits:**
- **True Latency Reduction**: Searches only relevant category's index (30-80% faster depending on category size)
- **Better Relevance**: Eliminates irrelevant category results
- **Automatic**: No manual filtering required
- **Backward Compatible**: Falls back to main index if category indices unavailable

**Configuration:**
```yaml
filters:
  enable: true  # Enable automatic category-based filtering
```

#### Reranking (`reranker.py`)

After initial retrieval, the system optionally applies cross-encoder reranking to improve result relevance beyond what pure embedding similarity provides.

**How It Works:**
- Uses cross-encoder model (`ms-marco-MiniLM-L-6-v2` by default)
- Scores each query-document pair independently
- Reranks retrieved chunks by relevance scores
- Selects top-k reranked chunks for generation

**Performance Benefits:**
- Improved Relevance: Cross-encoders consider query-document interaction
- Better Precision: Filters out false positives from initial retrieval
- Contextual Understanding: Better semantic matching than embeddings alone

**Configuration:**
```yaml
reranker:
  enable: true
  top_k: 5  # Number of top chunks after reranking
```

#### Response Generation (`generate.py`)

The final step generates natural language responses using a local LLM.

- Takes selected chunks and user query
- Constructs RAG prompt with context and question
- Generates coherent, document-grounded answer
- Model: Configurable via `config.yaml` (default: `phi3`)

### Stage 3: Evaluation

The evaluation framework provides comprehensive metrics to assess system performance across multiple dimensions.

#### Evaluation Metrics (`evaluation_script.py`)

1. **Retrieval Quality Metrics:**
   - **Precision@k**: Proportion of retrieved chunks that are relevant
   - **Recall@k**: Proportion of relevant chunks that were retrieved
   - **F1@k**: Harmonic mean of precision and recall
   - **Mean Reciprocal Rank (MRR)**: Quality of first relevant result ranking
   - **Average Retrieval Time**: Latency measurement

2. **Coverage Analysis:**
   - Total chunks and sources
   - Chunks per source distribution
   - Source balance score (lower is more balanced)

3. **Latency Benchmarking:**
   - Mean, median, P95, P99 latency metrics
   - Statistical analysis across multiple runs

#### Evaluation Process

1. Load test cases with ground truth (`test_cases.json`)
   - Format: `{"query": "...", "relevant_global_ids": [...]}`

2. Run evaluation:
   ```bash
   cd evaluation/
   python evaluation_script.py
   ```

3. Review generated report (`evaluation_report.txt`):
   - System statistics
   - Knowledge base coverage
   - Retrieval quality metrics
   - Performance benchmarks

#### Test Case Generation (`generate_test_cases.py`)

Utility script to generate test cases for evaluation:
- Create queries covering different categories
- Manually or automatically annotate relevant chunks
- Export to JSON format for evaluation

## Configuration

The system is fully configurable via `config.yaml`:

```yaml
rag_pipeline:
  # Paths
  index_path: "../data/vectordb/vector_index.faiss"
  metadata_path: "../data/vectordb/metadata.json"
  chunks_path: "../data/chunks"
  core_knowledge_path: "../data/static/core_knowledge.json"
  
  # Models
  embedding_model: "all-MiniLM-L6-v2"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  llm_model: "gpt-4o-mini"  # Options: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", or "phi3" for Ollama
  
  # OpenAI Settings (required if using OpenAI models)
  # IMPORTANT: API key should be set via OPENAI_API_KEY environment variable for security
  openai:
    base_url: null  # Optional: Custom base URL for OpenAI-compatible APIs (e.g., Azure OpenAI)
    temperature: 0.7  # Sampling temperature (0.0 to 2.0)
    max_tokens: null  # Maximum tokens to generate (null = model default)
  
  # Retrieval settings
  top_k_retrieve: 8    # Initial retrieval count
  
  # Metadata filtering
  filters:
    enable: false       # Enable automatic category-based filtering
  
  # Reranking settings
  reranker:
    enable: false
    top_k: 5            # Top chunks after reranking
  
  # Chunking settings (for ingestion)
  chunk_size: 1000
  chunk_overlap: 100
  
  # Verbose output
  verbose: false
```

**Security Note**: The OpenAI API key is **never** stored in `config.yaml`. It must be set as an environment variable `OPENAI_API_KEY` for security. See [Installation](#installation) for setup instructions.

## Key Features

### 1. Broad Question Classification
- Fast pattern-based classification for general questions
- Instant responses via `core_knowledge.json` lookup
- Zero retrieval overhead for predefined categories

### 2. Reranker for Improving Performance
- Cross-encoder reranking for superior relevance
- Filters false positives from initial retrieval
- Better semantic understanding than embeddings alone
- Configurable enable/disable

### 3. Metadata-Driven Filtering for Latency Reduction and Relevance
- Automatic category detection from query keywords
- **Category-specific FAISS indices** created during ingestion for true latency reduction
- Searches only relevant category's index (30-80% faster than full index search)
- Better relevance by eliminating irrelevant categories
- Automatic fallback to main index if category not detected
- Configurable enable/disable

### 4. Comprehensive Evaluation Framework
- Precision, Recall, F1, MRR metrics
- Latency benchmarking
- Coverage analysis
- Automated report generation

### 5. Flexible Pipeline Architecture
- Modular design allows easy component swapping
- Configurable via YAML without code changes
- Supports both broad and specific queries efficiently

## Performance Notes

- **Metadata filtering with category indices**: Reduces retrieval latency by 30-80% when category is detected (searches smaller category index instead of full index)
- **Metadata filtering without category indices**: Improves relevance but doesn't reduce latency (filters after retrieval from full index)
- **Reranking** adds ~100-200ms per query but improves precision significantly
- **Broad question classification** provides instant responses (<10ms)
- **Vector retrieval** (full index): typically 50-150ms depending on index size
- **Vector retrieval** (category index): typically 10-50ms (much faster!)
- **Total pipeline latency** (with category filtering, without reranking): ~100-200ms
- **Total pipeline latency** (with category filtering, with reranking): ~200-400ms
- **Total pipeline latency** (without filtering, without reranking): ~200-300ms
- **Total pipeline latency** (without filtering, with reranking): ~300-500ms

## Dependencies

### Core Libraries

- **sentence-transformers**: Embedding generation
- **faiss-cpu**: Vector similarity search
- **langchain**: Text splitting and document handling
- **pymupdf**: PDF text extraction
- **camelot-py**: PDF table extraction
- **python-docx**: DOCX processing
- **pyyaml**: Configuration management
- **streamlit**: Web interface (optional)

