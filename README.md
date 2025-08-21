# Multi-Document RAG System

A Python-based Retrieval-Augmented Generation (RAG) system that processes documents (PDF, TXT, MP3) and enables Q&A interactions using Google Cloud services and Gemini AI. The system extracts text from various file formats, creates embeddings using Google Vertex AI, stores them in ChromaDB for retrieval, and generates responses with proper citations.

## Quick Start

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Environment Setup

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

## Using the System

### Available Commands

When running the application, you can use these commands:

- `list groups` - Show all available document groups
- `list docs @group-name` - Show documents in a specific group
- `resync groups` - Update group metadata from metadata.json
- `@group-name your question` - Search within specific group(s)
- `your question` - Search all materials
- `quit` - Exit the application

### Adding Documents

1. Place files in the `data/` directory
2. Supported formats: PDF, TXT, MP3
3. System automatically detects and processes new files on startup

### Managing Document Groups

Groups are managed through the `metadata.json` file:

```json
{
  "documents": {
    "your-document.pdf": {
      "display_name": "Document Title",
      "author": "Author Name",
      "topics": ["topic1", "topic2"],
      "document_type": "book",
      "year": "2024",
      "grouping": ["group1", "group2"],
      "notes": "Description of the document"
    }
  }
}
```

#### To Add/Remove Groups:
1. Edit `metadata.json` to modify the `grouping` arrays
2. Run the application: `python main.py`
3. Type `resync groups` to update existing documents
4. Groups are now available for searching with `@group-name`

#### Group Search Examples:
- `@core-books what are the key principles?`
- `@foundational-reading @advanced-topics compare these concepts`
- `list docs @core-books`

## System Architecture

### Core Components
1. **Document Processing Pipeline** - PDF (Google Document AI), Audio (Speech-to-Text), Text files
2. **Vector Store Management** - Text chunking, embedding generation, ChromaDB storage
3. **RAG System** - Semantic search with citation generation
4. **Group Management** - Boolean metadata fields for efficient filtering

### Data Flow
1. Files in `data/` directory are automatically detected
2. Content extracted based on file type
3. Text chunked and embedded using Vertex AI
4. Embeddings stored in ChromaDB with metadata
5. User queries retrieve relevant chunks and generate cited responses

### Technical Details
- **Embeddings**: Uses `text-embedding-004` model from Vertex AI
- **Chunking**: 1000 characters with 200 character overlap
- **Citations**: Inline format with source filename, page, and chunk references
- **Group Filtering**: Boolean metadata fields for efficient ChromaDB queries

## File Structure

```
├── main.py                    # Complete RAG system implementation
├── requirements.txt           # Python dependencies
├── metadata.json             # Document metadata and group assignments
├── processed_files.json      # File change tracking
├── credentials.json          # Google Cloud service account credentials
├── .env                      # Environment variables
├── data/                     # Input documents directory
├── chroma_db/               # ChromaDB persistent storage
└── venv/                    # Python virtual environment
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify `credentials.json` and `.env` file setup
2. **Group Filter Errors**: Run `resync groups` after modifying `metadata.json`
3. **Missing Documents**: Check that files are in the `data/` directory
4. **No Search Results**: Verify document groups are correctly assigned in `metadata.json`

### Debugging Tips

- Check ChromaDB collection state: Run `list groups` to see available groups
- Verify file processing: System shows processing status on startup
- Monitor GCS permissions: Ensure bucket access for PDF processing
- Test embeddings: Try general queries before group-specific ones

## Development

### Adding New File Types
Extend the document processing pipeline in `main.py` around line 380.

### Modifying Chunk Size
Adjust `chunk_text()` function parameters (currently 1000 chars, 200 overlap).

### Customizing Citations
Modify `generate_response()` function citation format around line 600.