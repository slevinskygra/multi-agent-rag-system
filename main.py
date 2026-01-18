"""
Smart Document Q&A System
Multi-Agent RAG System with CLI Interface

This application demonstrates:
- RAG (Retrieval Augmented Generation)
- Multi-Agent coordination
- Document ingestion and search
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path
from rag_system import RAGSystem
from multi_agent import MultiAgentSystem


def print_banner():
    """Print application banner."""
    print("\n" + "="*60)
    print("  SMART DOCUMENT Q&A SYSTEM")
    print("  Multi-Agent RAG System")
    print("="*60)
    print("\nThis system uses:")
    print("  • RAG for document retrieval")
    print("  • Multiple specialized agents:")
    print("    - Router Agent: Analyzes queries")
    print("    - Retriever Agent: Searches documents")
    print("    - Synthesizer Agent: Creates answers")
    print("\nCommands:")
    print("  ingest <filepath>      - Add a document to the system")
    print("  ingest-folder <path>   - Add all documents from a folder")
    print("  list                   - Show all documents")
    print("  stats                  - Show collection statistics")
    print("  clear                  - Clear all documents")
    print("  ask <question>         - Ask a question")
    print("  quit                   - Exit the system")
    print("="*60 + "\n")


def main():
    """Main application loop."""
    print_banner()
    
    # Initialize systems
    print("Initializing RAG system...")
    rag_system = RAGSystem(persist_directory="./chroma_db")
    
    print("Initializing multi-agent system...")
    try:
        multi_agent = MultiAgentSystem(rag_system)
        print("✓ System ready!\n")
    except Exception as e:
        print(f"\n✗ Error initializing system: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file with your ANTHROPIC_API_KEY")
        print("2. Installed all requirements: pip install -r requirements.txt\n")
        return
    
    # Main interaction loop
    while True:
        try:
            user_input = input("📝 You: ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Handle commands
            if command in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
                break
            
            elif command == 'ingest':
                if not args:
                    print("Usage: ingest <filepath>")
                    continue
                
                file_path = args.strip()
                if not os.path.exists(file_path):
                    print(f"Error: File '{file_path}' not found")
                    continue
                
                print(f"\n📥 Ingesting document: {file_path}")
                result = rag_system.ingest_document(file_path)
                print(result)
            
            elif command == 'ingest-folder':
                if not args:
                    print("Usage: ingest-folder <folder_path>")
                    continue
                
                folder_path = args.strip()
                if not os.path.exists(folder_path):
                    print(f"Error: Folder '{folder_path}' not found")
                    continue
                
                print(f"\n📂 Ingesting all documents from folder: {folder_path}")
                result = rag_system.ingest_folder(folder_path)
                print(result)
            
            elif command == 'list':
                docs = rag_system.list_documents()
                if docs:
                    print("\n📚 Documents in collection:")
                    for i, doc in enumerate(docs, 1):
                        print(f"  {i}. {doc}")
                else:
                    print("\n📚 No documents in collection yet")
                    print("Use 'ingest <filepath>' to add documents")
            
            elif command == 'stats':
                stats = rag_system.get_collection_stats()
                print("\n📊 Collection Statistics:")
                print(f"  Total chunks: {stats['total_chunks']}")
                print(f"  Embedding model: {stats['embedding_model']}")
                print(f"  Storage: {stats['persist_directory']}")
                print(f"  Status: {stats['status']}")
            
            elif command == 'clear':
                confirm = input("⚠️  Clear all documents? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    result = rag_system.clear_collection()
                    print(f"✓ {result}")
                else:
                    print("Cancelled")
            
            elif command == 'ask':
                if not args:
                    print("Usage: ask <your question>")
                    continue
                
                question = args.strip()
                result = multi_agent.process_query(question, verbose=True)
                
                print("\n" + "="*60)
                print("FINAL ANSWER")
                print("="*60)
                print(f"\n{result['answer']}")
                
                if result['sources']:
                    print(f"\n📚 Sources: {', '.join(result['sources'])}")
                
                print(f"\n🔄 Workflow: {result['workflow']}")
                print("="*60)
            
            else:
                # Treat as a question if not a command
                result = multi_agent.process_query(user_input, verbose=True)
                
                print("\n" + "="*60)
                print("FINAL ANSWER")
                print("="*60)
                print(f"\n{result['answer']}")
                
                if result['sources']:
                    print(f"\n📚 Sources: {', '.join(result['sources'])}")
                
                print(f"\n🔄 Workflow: {result['workflow']}")
                print("="*60)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
