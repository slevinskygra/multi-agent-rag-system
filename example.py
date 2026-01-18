"""
Example script showing how to use the Multi-Agent RAG system programmatically.

This demonstrates:
- Setting up the system
- Ingesting documents
- Processing queries
- Batch processing
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
from rag_system import RAGSystem
from multi_agent import MultiAgentSystem


def example_basic_usage():
    """Basic usage example."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Initialize systems with unique database
    rag_system = RAGSystem(persist_directory="./example_db_1")
    multi_agent = MultiAgentSystem(rag_system)
    
    # Ingest a document
    print("\n1. Ingesting document...")
    result = rag_system.ingest_document("documents/gutenberg.org_cache_epub_1998_pg1998.txt.pdf")
    print(result)
    
    # Ask a question
    print("\n2. Asking question...")
    result = multi_agent.process_query(
        "can you summarise the document content? ",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("ANSWER:", result['answer'])
    print("SOURCES:", result['sources'])
    print("="*60)
    
    # Clean up
    print("\nCleaning up Example 1...")
    rag_system.clear_collection()


def example_batch_processing():
    """Example of processing multiple queries."""
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)
    
    # Initialize systems with unique database
    rag_system = RAGSystem(persist_directory="./example_db_2")
    multi_agent = MultiAgentSystem(rag_system)
    
    # Ingest documents
    print("\nIngesting documents...")
    result = rag_system.ingest_document("documents/gutenberg.org_cache_epub_1998_pg1998.txt.pdf")
    print(result)
    result = rag_system.ingest_document("documents/gutenberg.org_cache_epub_2701_pg2701.txt.pdf")
    print(result)
    
    # Batch process queries
    queries = [
        "Who is Zaratustra? Check retrieved data ",
        "Who is Moby? Check retrieved data ",
        "Summarise the content of retrieved documents",
    ]
    
    results = multi_agent.batch_process(queries)
    
    # Print summary
    print("\n\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    for i, (query, result) in enumerate(zip(queries, results), 1):
        print(f"\nQuery {i}: {query}")
        print(f"Category: {result['category']}")
        print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
        print(f"Answer : {result['answer'][:2000]}...")  # Show first 200 chars
    
    # Clean up
    print("\nCleaning up Example 2...")
    rag_system.clear_collection()


def example_folder_ingestion():
    """Example showing folder ingestion."""
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Folder Ingestion")
    print("="*60)
    
    # Initialize systems with unique database
    rag_system = RAGSystem(persist_directory="./example_db_3")
    multi_agent = MultiAgentSystem(rag_system)
    
    # Use the documents folder directly
    test_folder = "./documents"
    
    if not os.path.exists(test_folder):
        print(f"Warning: Folder {test_folder} not found, skipping this example")
        return
    
    # Ingest entire folder
    print("\n1. Ingesting entire folder:")
    result = rag_system.ingest_folder(test_folder)
    print(result)
    
    # Query the documents
    print("\n2. Asking question about folder contents:")
    result = multi_agent.process_query(
        "What are the main topics covered in these documents? ",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("ANSWER:", result['answer'][:2000])  # Show first 500 chars
    print("SOURCES:", result['sources'])
    print("="*60)
    
    # Clean up
    print("\nCleaning up Example 3...")
    rag_system.clear_collection()


def main():
    """Run all examples."""
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        return
    
    print("\n🚀 Multi-Agent RAG System Examples\n")
    print("These examples demonstrate different use cases:")
    print("1. Basic usage")
    print("2. Batch processing")
    print("3. Folder ingestion")
    print("\n" + "="*60 + "\n")
    
    try:
        example_basic_usage()
        example_batch_processing()
        example_folder_ingestion()
        
        print("\n\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error running examples: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
