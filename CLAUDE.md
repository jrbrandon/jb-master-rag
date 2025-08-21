# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based RAG (Retrieval-Augmented Generation) system that processes documents (PDF, TXT, MP3) and enables Q&A interactions using Google Cloud services and Gemini AI. The system extracts text from various file formats, creates embeddings using Google Vertex AI, stores them in ChromaDB for retrieval, and generates responses with proper citations.

## Development Environment Setup

### Prerequisites
- Python 3.9+ (virtual environment already set up in `venv/`)
- Google Cloud Project with enabled APIs:
  - Document AI API
  - Speech-to-Text API
  - Cloud Storage API
  - Vertex AI API
- Gemini API access

### Environment Configuration
Create a `.env` file in the project root with these required variables:
```
GOOGLE_CLOUD_PROJECT_ID=your-project-id
DOCUMENTAI_PROCESSOR_ID=your-processor-id
DOCAI_LOCATION=your-docai-location
VERTEXAI_LOCATION=your-vertexai-location
GCS_BUCKET_NAME=your-bucket-name
GEMINI_API_KEY=your-gemini-api-key
```

### Installation and Running
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## System Architecture

### Core Components

1. **Document Processing Pipeline** (`main.py:55-134`)
   - PDF processing via Google Document AI with pagination support
   - Audio transcription using Google Cloud Speech-to-Text
   - Text file reading for direct content extraction

2. **Vector Store Management** (`main.py:159-215`)
   - Text chunking with configurable size and overlap
   - Batch embedding generation using Vertex AI TextEmbedding model
   - ChromaDB collection management with detailed metadata

3. **RAG System** (`main.py:217-257`)
   - Semantic search for relevant context retrieval
   - Response generation using Gemini 1.5 Pro with citation requirements
   - Inline citation formatting with source references

4. **File Synchronization** (`main.py:292-320`)
   - Hash-based change detection using `processed_files.json`
   - Automatic deletion handling for removed files
   - Incremental processing for new/modified files

### Data Flow
1. Files placed in `data/` directory are automatically detected
2. Content is extracted based on file type (PDF → Document AI, MP3 → Speech-to-Text, TXT → direct read)
3. Text is chunked and embedded using Vertex AI
4. Embeddings stored in ChromaDB with metadata (source, page, chunk)
5. User queries retrieve relevant chunks and generate cited responses

## Key Technical Details

### Document Processing
- PDFs use Google Document AI batch processing with GCS staging
- Large files processed asynchronously with cleanup of temporary GCS objects
- Page-level metadata preservation for precise citations

### Embedding Strategy
- Uses `textembedding-gecko@003` model from Vertex AI
- Batch processing in groups of 250 (Vertex AI limit)
- Chunk size: 1000 characters with 200 character overlap

### Citation System
- Metadata includes source filename, page number (PDFs), and chunk index
- Responses include inline citations in format: [Source: filename, Page: X, Part: Y]
- Unique sources listed separately after each response

## File Structure
- `main.py`: Complete RAG system implementation
- `requirements.txt`: Python dependencies
- `processed_files.json`: File change tracking manifest
- `data/`: Directory for input documents (PDF, TXT, MP3)
- `credentials.json`: Google Cloud service account credentials
- `venv/`: Python virtual environment

## Common Operations

### Adding New Documents
Place files in the `data/` directory. Supported formats:
- PDF: Processed via Document AI
- TXT: Direct text extraction
- MP3: Transcribed via Speech-to-Text API

### Testing Changes
The system automatically detects file changes on startup and reprocesses as needed. No manual rebuild required.

### Debugging
- Check environment variables if initialization fails
- Verify GCS bucket permissions and Document AI processor setup
- Monitor ChromaDB collection state for vector store issues