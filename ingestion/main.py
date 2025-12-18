from extract_data import extract_pdf_text_and_tables, extract_docx_text_and_tables, clean_text
from chunk import chunk_text, save_chunks_to_json
import os
from pathlib import Path
import time
import yaml
from vectorise import vectorize_json_folder

# Import REPO_ROOT for path resolution
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import REPO_ROOT, resolve_path

def process_all_pdfs(data_folder=None, extracted_folder=None, chunks_folder=None, vectordb_folder=None, config_path=None):

    # Set default paths relative to REPO_ROOT
    if data_folder is None:
        data_folder = str(REPO_ROOT / "data/raw")
    if extracted_folder is None:
        extracted_folder = str(REPO_ROOT / "data/extracted")
    if chunks_folder is None:
        chunks_folder = str(REPO_ROOT / "data/chunks")
    if vectordb_folder is None:
        vectordb_folder = str(REPO_ROOT / "data/vectordb")
    if config_path is None:
        config_path = str(REPO_ROOT / "config.yaml")
    
    # Resolve all paths
    data_folder = str(resolve_path(data_folder))
    extracted_folder = str(resolve_path(extracted_folder))
    chunks_folder = str(resolve_path(chunks_folder))
    vectordb_folder = str(resolve_path(vectordb_folder))
    config_path = str(resolve_path(config_path))
    
    os.makedirs(extracted_folder, exist_ok=True)
    os.makedirs(chunks_folder, exist_ok=True)
    os.makedirs(vectordb_folder, exist_ok=True)

    # Load chunking settings from config
    chunk_size = 1000  # default
    chunk_overlap = 100  # default
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                rag_config = config_data.get('rag_pipeline', {})
                chunk_size = rag_config.get('chunk_size', 1000)
                chunk_overlap = rag_config.get('chunk_overlap', 100)
                print(f"Loaded chunking settings from config: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default chunking settings")

    files = [f for f in Path(data_folder).glob("*")]
    print(f" Found {len(files)} files in {data_folder}\n")

    for file in files:
        base_name = Path(file).stem
        txt_out = Path(extracted_folder) / f"{base_name}.txt"
        json_out = Path(chunks_folder) / f"{base_name}_chunks.json"

        print(f"\n Processing: {file.name}")

        # Step 1: Extract + clean text
        if(file.suffix.lower() == ".pdf"):
            text = extract_pdf_text_and_tables(file)
            text = clean_text(text)
        elif(file.suffix.lower() == ".docx"):
            text = extract_docx_text_and_tables(file)
        

        # Step 2: Save extracted text
        with open(txt_out, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)
        print(f" Saved extracted text → {txt_out}")

        # Step 3: Chunk + save JSON
        chunks = chunk_text(txt_out, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        save_chunks_to_json(chunks, json_out)

    print("\n All PDFs processed successfully!")

    # Step 4: Vectorise
    # This creates both the main index (for backward compatibility) and category-specific indices (for latency reduction)
    index, all_metadata, model, category_indices = vectorize_json_folder(
        Path(chunks_folder), 
        index_file=vectordb_folder+"/vector_index.faiss", 
        metadata_file=vectordb_folder+"/metadata.json",
        create_category_indices=True  # Enable category-specific indices for latency reduction
    )

if __name__ == "__main__":
    # Use default paths (relative to REPO_ROOT)
    process_all_pdfs()