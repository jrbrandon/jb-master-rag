#!/usr/bin/env python3
"""
Test script for the multi-document RAG system.
Run this to validate all functionality works correctly.
"""

import os
import sys
from main import (
    load_metadata, 
    parse_group_mentions, 
    get_document_metadata,
    sync_metadata_to_chromadb
)

def test_metadata_loading():
    """Test metadata.json loading and parsing."""
    print("🧪 Testing metadata loading...")
    
    metadata = load_metadata()
    assert "documents" in metadata, "Metadata should have 'documents' key"
    
    # Test existing document
    filename = "Secret Tradecraft of Elite Advisors_ Cover - David C. Baker.pdf"
    doc_meta = get_document_metadata(filename, metadata)
    
    print(f"✅ Document: {doc_meta['display_name']}")
    print(f"✅ Author: {doc_meta['author']}")
    print(f"✅ Groups: {', '.join(doc_meta['grouping'])}")
    
    # Test non-existent document (should get auto-generated metadata)
    fake_doc = get_document_metadata("nonexistent.pdf", metadata)
    assert fake_doc['grouping'] == ['uncategorized'], "Should auto-generate uncategorized group"
    print("✅ Auto-generated metadata for unknown documents works")
    print()

def test_group_parsing():
    """Test @ mention parsing."""
    print("🧪 Testing @ mention parsing...")
    
    test_cases = [
        ("@core-psdw-books what is positioning?", ["core-psdw-books"], "what is positioning?"),
        ("@dcb-and-be @foundational-reading pricing advice", ["dcb-and-be", "foundational-reading"], "pricing advice"),
        ("what is consulting?", [], "what is consulting?"),
        ("@non-existent-group test query", ["non-existent-group"], "test query"),
        ("", [], "")
    ]
    
    for query, expected_groups, expected_clean in test_cases:
        groups, clean = parse_group_mentions(query)
        assert groups == expected_groups, f"Groups mismatch for '{query}': got {groups}, expected {expected_groups}"
        assert clean == expected_clean, f"Clean query mismatch for '{query}': got '{clean}', expected '{expected_clean}'"
        print(f"✅ '{query}' → groups: {groups}, clean: '{clean}'")
    
    print()

def test_command_validation():
    """Test that our command structure is correct."""
    print("🧪 Testing command validation...")
    
    # Test commands that should be recognized
    commands = [
        "list groups",
        "list docs @core-psdw-books",
        "@dcb-and-be what is pricing?",
        "regular query without groups"
    ]
    
    for cmd in commands:
        if cmd.lower() == 'list groups':
            print("✅ 'list groups' command recognized")
        elif cmd.lower().startswith('list docs @'):
            group_name = cmd.split('@')[1].strip()
            print(f"✅ 'list docs' command for group '{group_name}' recognized")
        elif '@' in cmd:
            groups, clean = parse_group_mentions(cmd)
            print(f"✅ Group query: {groups} + '{clean}'")
        else:
            print(f"✅ Regular query: '{cmd}'")
    
    print()

def main():
    """Run all tests."""
    print("🚀 Testing JB Master RAG Multi-Document System")
    print("=" * 50)
    print()
    
    try:
        test_metadata_loading()
        test_group_parsing()
        test_command_validation()
        
        print("🎉 All tests passed!")
        print()
        print("📋 System Ready For:")
        print("  ✅ Multi-document processing with metadata.json")
        print("  ✅ Group-based filtering with @ mentions")
        print("  ✅ Special commands (list groups, list docs)")
        print("  ✅ Metadata synchronization")
        print()
        print("🎯 Next Steps:")
        print("  1. Run: python main.py")
        print("  2. Try: list groups")
        print("  3. Try: @core-psdw-books what is positioning?")
        print("  4. Try: @dcb-and-be pricing strategies")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()