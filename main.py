import os
import json
import hashlib
import textwrap
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Google Cloud Libraries
from google.cloud import documentai_v1 as documentai
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Other Libraries
import chromadb
from tqdm import tqdm
import google.generativeai as genai
import uuid
import sys
import argparse


load_dotenv()

# --- Configuration ---
# Set Google Cloud credentials path
if os.path.exists('credentials.json'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
    # Extract project ID from credentials if not set in environment
    if not os.getenv("GOOGLE_CLOUD_PROJECT_ID"):
        try:
            with open('credentials.json', 'r') as f:
                creds = json.load(f)
                os.environ['GOOGLE_CLOUD_PROJECT_ID'] = creds.get('project_id', '')
        except Exception as e:
            print(f"Warning: Could not read project_id from credentials.json: {e}")

GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
DOCAI_PROCESSOR_ID = os.getenv("DOCUMENTAI_PROCESSOR_ID")
DOCAI_LOCATION = os.getenv("DOCAI_LOCATION")
VERTEXAI_LOCATION = os.getenv("VERTEXAI_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
MANIFEST_FILE = "processed_files.json"
METADATA_FILE = "metadata.json"

# --- Helper Functions ---

def get_file_hash(file_path: str) -> str:
    """Calculates the SHA-256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_manifest() -> Dict[str, str]:
    """Loads the manifest file."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_manifest(manifest: Dict[str, str]):
    """Saves the manifest file."""
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

def load_metadata() -> Dict[str, Any]:
    """Loads the metadata file."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"documents": {}, "metadata_version": "1.0", "last_updated": ""}

def get_document_metadata(filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Gets metadata for a document, with fallback to auto-detection."""
    if filename in metadata.get("documents", {}):
        return metadata["documents"][filename]
    
    # Fallback: auto-detect basic metadata
    return {
        "display_name": filename.replace(".pdf", "").replace("_", " "),
        "author": "Unknown",
        "topics": [],
        "document_type": "unknown",
        "year": "",
        "grouping": ["uncategorized"],
        "notes": "Auto-generated metadata - please update in metadata.json"
    }

def sync_metadata_to_chromadb(collection, metadata: Dict[str, Any]) -> int:
    """Syncs metadata.json changes to ChromaDB without affecting embeddings."""
    print("Syncing metadata to ChromaDB...")
    updated_count = 0
    
    try:
        # Get all documents in the collection
        all_docs = collection.get(include=["metadatas", "documents"])
        
        if not all_docs['ids']:
            print("No documents found in ChromaDB to sync.")
            return 0
        
        # Group by source document
        doc_chunks = {}
        for i, doc_id in enumerate(all_docs['ids']):
            current_meta = all_docs['metadatas'][i]
            source = current_meta.get('source', '')
            
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append({
                'id': doc_id,
                'metadata': current_meta,
                'document': all_docs['documents'][i]
            })
        
        # Update metadata for each source document
        for source_file, chunks in doc_chunks.items():
            doc_metadata = get_document_metadata(source_file, metadata)
            
            # Update each chunk's metadata
            for chunk in chunks:
                # Preserve chunk-specific metadata (page, chunk number)
                updated_metadata = {
                    'source': source_file,
                    'display_name': doc_metadata['display_name'],
                    'author': doc_metadata['author'],
                    'topics': doc_metadata['topics'],
                    'document_type': doc_metadata['document_type'],
                    'year': doc_metadata['year'],
                    'grouping': doc_metadata['grouping'],
                    'notes': doc_metadata['notes'],
                    # Preserve existing chunk-specific metadata
                    'page': chunk['metadata'].get('page'),
                    'chunk': chunk['metadata'].get('chunk')
                }
                
                # Remove None values
                updated_metadata = {k: v for k, v in updated_metadata.items() if v is not None}
                
                # Update the document in ChromaDB
                collection.update(
                    ids=[chunk['id']],
                    metadatas=[updated_metadata]
                )
                updated_count += 1
        
        print(f"Successfully updated metadata for {updated_count} document chunks.")
        return updated_count
        
    except Exception as e:
        print(f"Error syncing metadata: {e}")
        return 0

def check_metadata_sync_needed() -> bool:
    """Checks if metadata.json is newer than last sync."""
    if not os.path.exists(METADATA_FILE):
        return False
    
    # For now, always return True - we can add timestamp tracking later
    return True

def parse_group_mentions(query: str) -> Tuple[List[str], str]:
    """Parses @ mentions from query and returns (groups, clean_query)."""
    import re
    
    # Find all @ mentions: @group-name, @another-group
    mentions = re.findall(r'@([a-zA-Z0-9-_]+)', query)
    
    # Remove @ mentions from query to get clean search text
    clean_query = re.sub(r'@[a-zA-Z0-9-_]+\s*', '', query).strip()
    
    return mentions, clean_query

def get_available_groups(collection) -> Dict[str, int]:
    """Gets all available groups and their document counts."""
    try:
        # Get all documents with their metadata
        all_docs = collection.get(include=["metadatas"])
        
        group_counts = {}
        processed_sources = set()  # Track unique documents, not chunks
        
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', '')
            groupings = metadata.get('grouping', [])
            
            # Only count each source document once per group
            if source not in processed_sources:
                processed_sources.add(source)
                
                for group in groupings:
                    if group:
                        group_counts[group] = group_counts.get(group, 0) + 1
        
        return group_counts
    except Exception as e:
        print(f"Error getting available groups: {e}")
        return {}

def get_documents_in_group(collection, group_name: str) -> List[str]:
    """Gets all document names in a specific group."""
    try:
        # Query for documents in this group
        results = collection.get(
            where={"grouping": {"$in": [group_name]}},
            include=["metadatas"]
        )
        
        # Get unique document display names
        doc_names = set()
        for metadata in results['metadatas']:
            display_name = metadata.get('display_name', metadata.get('source', 'Unknown'))
            doc_names.add(display_name)
        
        return sorted(list(doc_names))
    except Exception as e:
        print(f"Error getting documents in group {group_name}: {e}")
        return []

# --- Text Extraction Functions ---

def process_document_ai_paginated(file_path: str) -> List[Tuple[str, int]]:
    """Processes a large PDF using the Document AI asynchronous batch mode."""
    gcs_input_blob = None
    gcs_output_blobs = []
    
    try:
        opts = {"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"}
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        storage_client = storage.Client()
        
        filename = os.path.basename(file_path)
        gcs_input_uri = f"gs://{GCS_BUCKET_NAME}/pdf_input/{filename}"
        gcs_output_uri_prefix = f"pdf_output/{filename}/"
        gcs_output_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_output_uri_prefix}"

        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        gcs_input_blob = bucket.blob(f"pdf_input/{filename}")
        gcs_input_blob.upload_from_filename(file_path)

        input_config = documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type="application/pdf")
        gcs_documents = documentai.GcsDocuments(documents=[input_config])
        input_configs = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=gcs_output_uri)
        )

        name = client.processor_path(GCP_PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)
        request = documentai.BatchProcessRequest(
            name=name,
            input_documents=input_configs,
            document_output_config=output_config,
        )

        operation = client.batch_process_documents(request)
        
        print(f"Waiting for Document AI batch processing of {filename} to complete...")
        operation.result(timeout=1800)

        output_bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
        gcs_output_blobs = list(output_bucket.list_blobs(prefix=gcs_output_uri_prefix))
        
        if not gcs_output_blobs:
            print(f"ERROR: Document AI operation for {filename} completed but produced no output files.")
            return []

        print(f"Found {len(gcs_output_blobs)} output files from Document AI.")
        pages = []
        total_pages_processed = 0
        
        for blob in gcs_output_blobs:
            if ".json" in blob.name:
                json_string = blob.download_as_string()
                document = documentai.Document.from_json(json_string, ignore_unknown_fields=True)
                
                print(f"Processing document chunk with {len(document.pages)} pages")
                total_pages_processed += len(document.pages)
                
                for page in document.pages:
                    page_text = "".join(
                        [document.text[segment.start_index:segment.end_index] for segment in page.layout.text_anchor.text_segments]
                    )
                    pages.append((page_text, page.page_number))

        print(f"Total pages processed: {total_pages_processed}")
        pages.sort(key=lambda x: x[1])
        return pages

    except Exception as e:
        print(f"An exception occurred while processing {file_path} with Document AI: {e}")
        return []
    
    finally:
        print("Cleaning up GCS files...")
        if gcs_input_blob:
            try:
                gcs_input_blob.delete()
            except Exception as e:
                print(f"Error deleting input blob: {e}")
        for blob in gcs_output_blobs:
            try:
                blob.delete()
            except Exception as e:
                print(f"Error deleting output blob: {e}")
        print("Cleanup complete.")


def upload_to_gcs(file_path: str, destination_blob_name: str):
    """Uploads a file to Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
    except Exception as e:
        print(f"Error uploading to GCS: {e}")

def transcribe_gcs_audio(gcs_uri: str) -> str:
    """Transcribes audio from a GCS URI using Google Cloud Speech-to-Text."""
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True
        )
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=900)
        
        transcripts = []
        for result in response.results:
            if result.alternatives:
                transcripts.append(result.alternatives[0].transcript)
        
        return " ".join(transcripts)
    except Exception as e:
        print(f"Error transcribing audio from {gcs_uri}: {e}")
        return ""

# --- Vector Store Management ---

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Splits a long text into smaller chunks."""
    chunks = []
    if not text: return []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def add_document_to_collection(collection, file_path: str, embedding_model, metadata: Dict[str, Any] = None):
    """Processes a single file, chunks it, and adds it to the vector store with enhanced metadata."""
    filename = os.path.basename(file_path)
    
    # Get enhanced metadata from metadata.json or fallback to auto-detection
    if metadata is None:
        metadata = load_metadata()
    doc_metadata = get_document_metadata(filename, metadata)
    
    # Log metadata source
    if filename in metadata.get("documents", {}):
        print(f"Using metadata.json data for: {doc_metadata['display_name']} by {doc_metadata['author']}")
        print(f"Groups: {', '.join(doc_metadata['grouping'])}")
    else:
        print(f"Using auto-detected metadata for: {filename} (consider adding to metadata.json)")
    
    if file_path.endswith(".pdf"):
        pages = process_document_ai_paginated(file_path)
        if not pages: return
        
        all_chunks, all_metadatas = [], []
        for page_text, page_num in pages:
            page_chunks = chunk_text(page_text)
            for i, chunk in enumerate(page_chunks):
                all_chunks.append(chunk)
                # Enhanced metadata combining JSON metadata with chunk-specific info
                chunk_metadata = {
                    "source": filename,
                    "display_name": doc_metadata['display_name'],
                    "author": doc_metadata['author'],
                    "topics": doc_metadata['topics'],
                    "document_type": doc_metadata['document_type'],
                    "year": doc_metadata['year'],
                    "grouping": doc_metadata['grouping'],
                    "notes": doc_metadata['notes'],
                    "page": page_num,
                    "chunk": i + 1
                }
                all_metadatas.append(chunk_metadata)

    else: # Handle TXT and MP3
        document_text = ""
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        elif file_path.endswith(".mp3"):
            blob_name = f"audio/{filename}"
            upload_to_gcs(file_path, blob_name)
            gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
            document_text = transcribe_gcs_audio(gcs_uri)
            # Clean up uploaded audio file
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(blob_name)
                blob.delete()
            except Exception as e:
                print(f"Warning: Could not delete audio file {blob_name}: {e}")
        
        if not document_text: return
        
        all_chunks = chunk_text(document_text)
        # Enhanced metadata for non-PDF files
        all_metadatas = []
        for i in range(len(all_chunks)):
            chunk_metadata = {
                "source": filename,
                "display_name": doc_metadata['display_name'],
                "author": doc_metadata['author'],
                "topics": doc_metadata['topics'],
                "document_type": doc_metadata['document_type'],
                "year": doc_metadata['year'],
                "grouping": doc_metadata['grouping'],
                "notes": doc_metadata['notes'],
                "chunk": i + 1
            }
            all_metadatas.append(chunk_metadata)

    # Process in smaller batches to avoid token limits
    batch_size = 100  # Smaller batches for large documents
    for i in tqdm(range(0, len(all_chunks), batch_size), desc=f"Embedding {filename}"):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        
        try:
            # Get embeddings from Vertex AI
            response = embedding_model.get_embeddings(batch_chunks)
            embeddings = [embedding.values for embedding in response]
            
            # Generate unique IDs to avoid conflicts
            ids = [f"{filename}_{uuid.uuid4().hex[:8]}_{i+j}" for j in range(len(batch_chunks))]
            
            collection.add(
                embeddings=embeddings,
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Warning: Vertex AI embedding failed for batch {i//250 + 1} of {filename}: {e}")
            print("Falling back to text-only storage (search will use text matching)...")
            
            # Fallback: store without embeddings (ChromaDB will use text search)
            try:
                ids = [f"{filename}_{uuid.uuid4().hex[:8]}_{i+j}" for j in range(len(batch_chunks))]
                collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=ids
                )
            except Exception as e2:
                print(f"Error storing documents without embeddings: {e2}")
                continue

# --- RAG and Q&A ---

def get_relevant_context(collection, query: str, embedding_model, group_filters: List[str] = None, n_results: int = 7) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Retrieves relevant context and metadata from the vector store with optional group filtering."""
    
    # Build where clause for group filtering
    where_clause = None
    if group_filters:
        # Filter to documents that have ANY of the specified groups
        where_clause = {
            "$or": [
                {"grouping": {"$in": [group]}} for group in group_filters
            ]
        }
        print(f"Filtering search to groups: {', '.join(group_filters)}")
    
    try:
        # First try embedding-based search
        query_embedding = embedding_model.get_embeddings([query])[0].values
        
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results, 
            where=where_clause,
            include=["documents", "metadatas"]
        )
        return results['documents'][0], results['metadatas'][0]
    except Exception as e:
        print(f"Embedding search failed: {e}")
        print("Falling back to text-based search...")
        
        # Fallback to text-based query
        try:
            results = collection.query(
                query_texts=[query], 
                n_results=n_results, 
                where=where_clause,
                include=["documents", "metadatas"]
            )
            return results['documents'][0], results['metadatas'][0]
        except Exception as e2:
            print(f"Text search also failed: {e2}")
            return [], []

def generate_response(context_docs: List[str], context_metadatas: List[Dict[str, Any]], query: str) -> str:
    """Generates a response using the Gemini model with footnote citations."""
    if not context_docs:
        return "No relevant context found to answer your question."
    
    # Create numbered context with citation mapping
    context_for_prompt = []
    citation_map = {}
    
    for i, (doc, meta) in enumerate(zip(context_docs, context_metadatas), 1):
        citation_key = f"Source {i}"
        citation_details = f"Source: {meta['source']}"
        if 'page' in meta:
            citation_details += f", Page: {meta['page']}"
        if 'chunk' in meta:
            citation_details += f", Part: {meta['chunk']}"
        
        citation_map[citation_key] = citation_details
        context_for_prompt.append(f"[{citation_key}]\n{doc}\n")

    prompt = f"""
You are a helpful assistant. Your task is to answer the user's question based ONLY on the provided context.
Do not use any prior knowledge.

IMPORTANT: Use NUMBERED FOOTNOTE citations in your response. When you reference information, use the format [1], [2], [3], etc. corresponding to the source numbers provided.

For example:
- "The author suggests three key principles [1]."
- "Advisory firms should focus on positioning [2][3]."

Do NOT use the full source information inline. Only use the numbers [1], [2], etc.

Here is the context:
---
{"---".join(context_for_prompt)}
---

Here is the user's question:
{query}

Answer (use numbered footnotes [1], [2], etc.):
"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return "Error: GEMINI_API_KEY environment variable not set."
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        
        if not response.text:
            return "No response generated."
        
        # Add footnotes at the end
        answer = response.text
        
        # Find which sources were actually cited in the response
        import re
        cited_numbers = set()
        # Find all citation patterns: [1], [1, 2], [1,2,3], etc.
        for match in re.finditer(r'\[([\d,\s]+)\]', answer):
            # Extract all numbers from the citation
            numbers_str = match.group(1)
            for num_str in re.findall(r'\d+', numbers_str):
                cited_numbers.add(int(num_str))
        
        # Only include footnotes for sources that were actually cited
        footnotes = "\n\n--- Sources ---\n"
        for i, (source_key, citation_details) in enumerate(citation_map.items(), 1):
            if i in cited_numbers:
                footnotes += f"[{i}] {citation_details}\n"
        
        return answer + footnotes
        
    except Exception as e:
        print(f"Error generating response with Gemini: {e}")
        return f"Error generating response: {str(e)}"

# --- Main Execution ---

def main():
    """Main function to run the RAG CLI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='JB Master RAG System')
    parser.add_argument('--sync-metadata', action='store_true', help='Sync metadata.json to ChromaDB')
    args = parser.parse_args()
    
    data_directory = "data"
    
    # --- Environment Variable Check ---
    required_vars = ["GOOGLE_CLOUD_PROJECT_ID", "DOCUMENTAI_PROCESSOR_ID", "DOCAI_LOCATION", "VERTEXAI_LOCATION", "GCS_BUCKET_NAME", "GEMINI_API_KEY"]
    missing_vars = [v for v in required_vars if not os.getenv(v)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease create a .env file with:")
        for var in missing_vars:
            print(f"{var}=your_value_here")
        print("\nOr set them as environment variables.")
        return
    
    # Validate project ID matches credentials
    if os.path.exists('credentials.json'):
        try:
            with open('credentials.json', 'r') as f:
                creds = json.load(f)
                cred_project = creds.get('project_id')
                if cred_project and cred_project != GCP_PROJECT_ID:
                    print(f"Warning: Project ID mismatch!")
                    print(f"  credentials.json: {cred_project}")
                    print(f"  Environment var:  {GCP_PROJECT_ID}")
                    print(f"Using project from credentials: {cred_project}")
                    # Update the environment variable to match credentials
                    os.environ['GOOGLE_CLOUD_PROJECT_ID'] = cred_project
        except Exception as e:
            print(f"Warning: Could not validate credentials.json: {e}")

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # --- Initialize Vertex AI ---
    # Get the current project ID (may have been updated by credential validation)
    final_project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    if not final_project_id:
        print("Error: No project ID available")
        return
        
    try:
        print(f"Initializing Vertex AI with project: {final_project_id}")
        vertexai.init(project=final_project_id, location=VERTEXAI_LOCATION)
        
        # Try embedding models in order of quality (best first)
        embedding_models_to_try = [
            "text-embedding-004",
            "textembedding-gecko@003", 
            "textembedding-gecko@002",
            "textembedding-gecko@001",
            "textembedding-gecko"
        ]
        
        embedding_model = None
        for model_name in embedding_models_to_try:
            try:
                print(f"Trying embedding model: {model_name}")
                embedding_model = TextEmbeddingModel.from_pretrained(model_name)
                # Test the model with a small query
                test_embedding = embedding_model.get_embeddings(["test"])
                print(f"✅ Successfully initialized embedding model: {model_name}")
                break
            except Exception as model_error:
                print(f"❌ Model {model_name} not available: {model_error}")
                continue
        
        if not embedding_model:
            raise Exception("No embedding models are available in your project/region")
    except Exception as e:
        print(f"Error initializing Vertex AI. Please check your authentication and project settings: {e}")
        print("\nTroubleshooting steps:")
        print(f"1. Verify project ID: {final_project_id}")
        print(f"2. Verify location: {VERTEXAI_LOCATION}")
        print("3. Add IAM permissions to your service account:")
        print("   - Vertex AI User (roles/aiplatform.user)")
        print("   - AI Platform Developer (roles/ml.developer)")
        print("4. Enable APIs:")
        print("   - https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
        print("   - https://console.cloud.google.com/apis/library/documentai.googleapis.com")
        print("5. Grant service account access in IAM:")
        print(f"   https://console.cloud.google.com/iam-admin/iam?project={final_project_id}")
        return

    try:
        # Use persistent client to save embeddings to disk
        client = chromadb.PersistentClient(path="./chroma_db")
        print("Using persistent ChromaDB storage at ./chroma_db/")
        
        # Get or create collection with persistent storage
        collection = client.get_or_create_collection(
            name="rag_collection_with_citations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Check if collection has documents
        doc_count = collection.count()
        if doc_count > 0:
            print(f"Found existing collection with {doc_count} documents")
        else:
            print("Empty collection - will process documents")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return
    
    # Handle metadata sync command
    if args.sync_metadata:
        metadata = load_metadata()
        updated_count = sync_metadata_to_chromadb(collection, metadata)
        print(f"Metadata sync complete. Updated {updated_count} document chunks.")
        return
    
    # Load metadata for normal operation
    metadata = load_metadata()
    
    # Auto-sync metadata if needed
    if check_metadata_sync_needed():
        print("Auto-syncing metadata changes...")
        sync_metadata_to_chromadb(collection, metadata)
    
    manifest = load_manifest()
    
    # --- Sync Deletions ---
    manifest_files = set(manifest.keys())
    current_files = set(os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f)))
    deleted_files = manifest_files - current_files
    
    if deleted_files:
        print(f"Found {len(deleted_files)} deleted files. Removing from vector store...")
        for file_path in deleted_files:
            collection.delete(where={"source": os.path.basename(file_path)})
            del manifest[file_path]

    # --- Sync New/Modified Files ---
    print("Scanning for new or modified files...")
    for filename in tqdm(os.listdir(data_directory), desc="Syncing files"):
        file_path = os.path.join(data_directory, filename)
        if not os.path.isfile(file_path): continue

        file_hash = get_file_hash(file_path)
        if manifest.get(file_path) == file_hash: continue

        print(f"\nProcessing new/modified file: {filename}")
        if file_path in manifest:
            collection.delete(where={"source": filename})

        try:
            add_document_to_collection(collection, file_path, embedding_model, metadata)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
        manifest[file_path] = file_hash

    save_manifest(manifest)
    print("Synchronization complete.")

    # --- Q&A Loop ---
    print("\n--- Q&A ---")
    print("Commands:")
    print("  list groups                    - Show all available groups")
    print("  list docs @group-name          - Show documents in a group")
    print("  @group-name your question      - Search within specific group(s)")
    print("  your question                  - Search all materials")
    print()
    
    while True:
        try:
            query = input("Ask a question (or type 'quit' to exit): ")
            if query.lower() == 'quit': break
            
            # Handle special commands
            if query.lower() == 'list groups':
                groups = get_available_groups(collection)
                if groups:
                    print("\nAvailable groupings:")
                    for group, count in sorted(groups.items()):
                        print(f"  @{group} ({count} documents)")
                else:
                    print("\nNo groups found.")
                print()
                continue
            
            if query.lower().startswith('list docs @'):
                group_name = query.split('@')[1].strip()
                docs = get_documents_in_group(collection, group_name)
                if docs:
                    print(f"\nDocuments in @{group_name}:")
                    for doc in docs:
                        print(f"  - {doc}")
                else:
                    print(f"\nNo documents found in group @{group_name}.")
                print()
                continue
            
            # Parse @ mentions for group filtering
            group_filters, clean_query = parse_group_mentions(query)
            
            # Validate groups exist
            if group_filters:
                available_groups = get_available_groups(collection)
                invalid_groups = [g for g in group_filters if g not in available_groups]
                if invalid_groups:
                    print(f"\nUnknown groups: {', '.join(invalid_groups)}")
                    print("Available groups:")
                    for group in sorted(available_groups.keys()):
                        print(f"  @{group}")
                    print()
                    continue
            
            context_docs, context_metadatas = get_relevant_context(collection, clean_query, embedding_model, group_filters)
            
            if not context_docs:
                if group_filters:
                    print(f"\n--- Answer ---\nCould not find any relevant context in groups {', '.join(group_filters)} for that question.")
                else:
                    print("\n--- Answer ---\nCould not find any relevant context to answer that question.")
                continue

            answer = generate_response(context_docs, context_metadatas, clean_query)
            
            print("\n--- Answer ---")
            print(answer)
            print("\n" + "="*80 + "\n")
        
        except Exception as e:
            print(f"An error occurred during Q&A: {e}")


if __name__ == "__main__":
    main()