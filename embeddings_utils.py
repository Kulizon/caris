import json
import os
import chromadb
import ollama
import requests
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "iconclass_db")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "iconclass_embeddings.jsonl")
DEFAULT_TXT_PATH = os.path.join(BASE_DIR, "data", "iconclass_clean.txt")
EMBEDDING_MODEL = "mxbai-embed-large"
BATCH_SIZE = 128

_CHROMA_CLIENT = None
_COLLECTION = None

def get_iconclass_collection(db_path=DEFAULT_DB_PATH):
    """Lazy initialization of the ChromaDB collection."""
    global _CHROMA_CLIENT, _COLLECTION
    if _COLLECTION is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=db_path)
        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(name="iconclass")
    return _COLLECTION

def embed_batch(documents, model=EMBEDDING_MODEL):
    """
    Generate embeddings for a list of documents.
    Returns a list of embeddings. Failed documents return None to maintain alignment.
    """
    try:
        response = ollama.embed(model=model, input=documents)
        return response["embeddings"]
    except Exception as e:
        print(f"\n[WARN] Batch embedding failed, switching to single mode: {e}")
        embeddings = []
        for doc in documents:
            try:
                r = ollama.embed(model=model, input=doc)
                embeddings.append(r["embeddings"][0])
            except Exception as inner:
                print(f"[ERROR] Skipping document: {doc[:100]}... Error: {inner}")
                embeddings.append(None)
        return embeddings
    
def get_iconclass_definitions():
    owner = "iconclass"
    repo = "data"

    paths = [
        "txt/en/txt_en_0_1.txt",
        "txt/en/txt_en_2_3.txt",
        "txt/en/txt_en_4.txt",
        "txt/en/txt_en_5_6_7_8.txt",
        "txt/en/txt_en_9.txt"
    ]

    all_lines = set()

    for path in paths:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref=main"

        response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            continue

        file_info = response.json()
        file_url = file_info["download_url"]

        file_data = requests.get(file_url)

        if file_data.status_code != 200:
            print(f"File download error: {file_data.status_code}")
            continue

        text = file_data.text

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            all_lines.add(line)

        print(f"Processed: {path}")

    print(f"\nUnique lines: {len(all_lines)}")

    # SORTING
    sorted_lines = sorted(all_lines)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(DEFAULT_DATA_PATH), exist_ok=True)

    # ===== TXT =====
    with open(DEFAULT_TXT_PATH, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(line + "\n")

    # ===== JSONL =====
    with open(DEFAULT_DATA_PATH, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            # split: 11A22|symbols ~ Divine Nature
            if "|" in line:
                code, description = line.split("|", 1)
            else:
                code = ""
                description = line

            obj = {
                "id": code,
                "text": description,
                "full_text": line
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved to {DEFAULT_DATA_PATH}")

def _process_upsert(collection, batch):
    """Helper to handle the embedding and upsert logic for a batch."""
    ids = [item["id"] for item in batch]
    documents = [item["text"] for item in batch]
    
    embeddings = embed_batch(documents)
    
    # Filter out failures while maintaining alignment
    valid_data = [(i, d, e) for i, d, e in zip(ids, documents, embeddings) if e is not None]
    
    if not valid_data:
        return

    v_ids, v_docs, v_embs = zip(*valid_data)
    
    collection.upsert(
        ids=list(v_ids),
        documents=list(v_docs),
        embeddings=list(v_embs),
        metadatas=[{"text": d} for d in v_docs]
    )

def create_vector_base(data_path=DEFAULT_DATA_PATH, db_path=DEFAULT_DB_PATH):
    """Reads the JSONL data and populates the vector database using streaming."""
    collection = get_iconclass_collection(db_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source data not found at {data_path}")

    def stream_data(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    print(f"Starting vector base creation from {data_path}")
    
    batch = []
    # Using tqdm for progress tracking. Note: total is unknown without reading file twice, 
    # but we can estimate or just show iterations.
    for item in tqdm(stream_data(data_path), desc="Processing Iconclass"):
        batch.append(item)
        if len(batch) >= BATCH_SIZE:
            _process_upsert(collection, batch)
            batch = []
            
    if batch:
        _process_upsert(collection, batch)

    print("\n[SUCCESS] Vector database created/updated!")

def get_iconclass_codes_embeddings(context_list, n_results=5):
    """
    Given a list of keywords/context strings, returns a flat list of 
    unique Iconclass codes from the vector database.
    """
    if not context_list:
        return []

    collection = get_iconclass_collection()
    
    # Batch generate embeddings for all context strings
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=context_list)
        query_embeddings = response["embeddings"]
    except Exception as e:
        print(f"[ERROR] Failed to embed context list: {e}")
        return []

    results = collection.query(
        query_embeddings=query_embeddings, 
        n_results=n_results
    )
    
    all_codes = set()
    if results["ids"]:
        for code_list in results["ids"]:
            for code in code_list:
                all_codes.add(str(code))
            
    return sorted(list(all_codes))


if __name__ == "__main__":
    # If run directly, assume we want to rebuild/update the base
    get_iconclass_definitions()
    create_vector_base()
