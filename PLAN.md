# Multi-Document RAG Refactor Action Plan

## Overview
Transform the single-document RAG system into a multi-document, group-filtered system with metadata management and @ mention querying.

## Phase 1: Metadata Foundation

### Task 1: Create metadata.json structure
- [x] Design JSON schema with arrays for groupings: `["core-psdw-books", "dcb-and-be"]`
- [x] Include fields: display_name, author, topics[], document_type, year, grouping[], notes
- [x] Create initial metadata.json for existing David Baker book

### Task 2: Implement metadata sync functionality
- [x] Function to read metadata.json and update ChromaDB document metadata
- [x] Preserve embeddings, only update metadata fields
- [x] Handle additions, updates, and removals

### Task 3: Modify document processing
- [x] Update `add_document_to_collection()` to read metadata from metadata.json
- [x] Fallback to auto-extraction if document not in metadata.json
- [x] Store enhanced metadata in ChromaDB chunks

## Phase 2: Query Enhancement

### Task 4: Add @ mention parsing
- [x] Parse queries for `@group-name` patterns
- [x] Extract group filters before sending to embedding search
- [x] Support multiple groups: `@group1 @group2 query text`

### Task 5: Implement special commands
- [x] `list groups` - show all groupings with document counts
- [x] `list docs @group-name` - show documents in specific group
- [x] Integrate into existing Q&A loop

### Task 6: Update query filtering
- [x] Modify `get_relevant_context()` to accept group filters
- [x] Filter ChromaDB queries by grouping metadata
- [x] Maintain global search when no @ mentions used

## Phase 3: Integration & Testing

### Task 7: Add command line options
- [x] `python main.py --sync-metadata` for manual sync
- [x] Auto-sync if metadata.json newer than last sync
- [x] Status reporting for sync operations

### Task 8: Test & validate
- [x] Convert existing David Baker content to new system
- [x] Test group filtering, @ mentions, special commands
- [x] Verify embeddings preserved during metadata updates

## Expected Timeline
- **Phase 1**: ~4-6 hours (core functionality)
- **Phase 2**: ~3-4 hours (query features) 
- **Phase 3**: ~2-3 hours (integration & testing)
- **Total**: ~1-2 days of focused work

## Example Usage After Implementation

### Metadata Structure
```json
{
  "documents": {
    "secret_tradecraft_david_baker.pdf": {
      "display_name": "Secret Tradecraft of Elite Advisors",
      "author": "David C. Baker",
      "topics": ["consulting", "positioning", "pricing"],
      "document_type": "book",
      "year": "2024",
      "grouping": ["core-psdw-books", "dcb-and-be", "foundational-reading"],
      "notes": "David's latest consulting methodology"
    }
  }
}
```

### Query Examples
```
> list groups
Available groupings:
- @core-psdw-books (3 documents)
- @dcb-and-be (8 documents)

> @core-psdw-books what are the key principles of positioning?
[Searches ONLY within core-psdw-books grouping]

> @dcb-and-be @practical-guides how do you handle difficult clients?
[Searches within BOTH dcb-and-be AND practical-guides]

> what is value-based pricing?
[Searches ALL materials - no @ mention = global search]
```

## Progress Tracking
- [x] Phase 1 Complete ✅
- [x] Phase 2 Complete ✅ 
- [x] Phase 3 Complete ✅
- [x] System Ready for Production Use 🎉

## 🎉 PROJECT COMPLETE! 

### ✅ **All 8 Tasks Completed Successfully**

**Phase 1: Metadata Foundation** ✅
- Task 1: ✅ metadata.json structure with multi-grouping support
- Task 2: ✅ Full metadata sync functionality 
- Task 3: ✅ Document processing integration with enhanced metadata

**Phase 2: Query Enhancement** ✅
- Task 4: ✅ @ mention parsing for group filtering
- Task 5: ✅ Special commands (list groups, list docs)
- Task 6: ✅ Group-based search filtering

**Phase 3: Integration & Testing** ✅
- Task 7: ✅ Command line options and auto-sync
- Task 8: ✅ Complete testing and validation

### 🚀 **System Now Ready for Production**

Your JB Master RAG system now supports:
- Multi-document management with manual metadata control
- Group-based querying (@core-psdw-books, @dcb-and-be, etc.)
- Rich metadata (author, topics, document_type, year, grouping, notes)
- Backwards compatibility with existing content
- Auto-sync and manual sync capabilities

---
*Created: January 21, 2025*
*Completed: January 21, 2025*
*Final Status: 8/8 tasks complete (100%) 🎯*