"""
RAG System Component - Handles document storage and retrieval

This module manages:
- Document ingestion (PDF, TXT)
- Text chunking and embedding
- Vector storage using ChromaDB
- Semantic search and retrieval
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


class RAGSystem:
    """
    Retrieval Augmented Generation System
    
    Manages document ingestion, embedding, storage, and retrieval.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        
        # Initialize embeddings model (using a free, local model)
        print("Loading embedding model (this may take a moment on first run)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize or load the vector store
        self.vectorstore = None
        self._initialize_vectorstore()
        
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store."""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def ingest_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_path: Path to the document (PDF or TXT)
            metadata: Optional metadata to attach to the document
            
        Returns:
            Success message with document info
        """
        try:
            # Determine file type and load
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
                documents = loader.load()
            else:
                return f"Error: Unsupported file type '{file_extension}'. Supported: .pdf, .txt"
            
            # Add custom metadata
            doc_name = Path(file_path).name
            for doc in documents:
                doc.metadata['source'] = doc_name
                if metadata:
                    doc.metadata.update(metadata)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            
            return (f"Successfully ingested '{doc_name}'\n"
                   f"- Pages/Sections: {len(documents)}\n"
                   f"- Chunks created: {len(chunks)}\n"
                   f"- Chunk size: ~1000 chars with 200 char overlap")
        
        except Exception as e:
            return f"Error ingesting document: {str(e)}"
    
    def ingest_folder(self, folder_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest all supported documents from a folder into the RAG system.
        
        Args:
            folder_path: Path to the folder containing documents
            metadata: Optional metadata to attach to all documents
            
        Returns:
            Summary message with ingestion statistics
        """
        try:
            folder = Path(folder_path)
            
            if not folder.exists():
                return f"Error: Folder '{folder_path}' does not exist"
            
            if not folder.is_dir():
                return f"Error: '{folder_path}' is not a folder"
            
            # Supported file extensions
            supported_extensions = ['.pdf', '.txt']
            
            # Find all supported files in the folder
            files_to_ingest = []
            for ext in supported_extensions:
                files_to_ingest.extend(folder.glob(f'*{ext}'))
            
            if not files_to_ingest:
                return f"No supported files (.pdf, .txt) found in '{folder_path}'"
            
            # Track results
            successful = []
            failed = []
            total_chunks = 0
            
            # Ingest each file
            print(f"\nFound {len(files_to_ingest)} files to ingest:")
            for file_path in files_to_ingest:
                print(f"\nIngesting: {file_path.name}")
                result = self.ingest_document(str(file_path), metadata)
                
                if "Error" in result:
                    failed.append(file_path.name)
                    print(f"  ✗ Failed: {result}")
                else:
                    successful.append(file_path.name)
                    # Extract chunk count from result
                    if "Chunks created:" in result:
                        chunk_count = int(result.split("Chunks created:")[1].split("\n")[0].strip())
                        total_chunks += chunk_count
                    print(f"  ✓ Success")
            
            # Generate summary
            summary = f"\n{'='*60}\n"
            summary += f"FOLDER INGESTION SUMMARY\n"
            summary += f"{'='*60}\n"
            summary += f"Folder: {folder_path}\n"
            summary += f"Total files found: {len(files_to_ingest)}\n"
            summary += f"Successfully ingested: {len(successful)}\n"
            summary += f"Failed: {len(failed)}\n"
            summary += f"Total chunks created: {total_chunks}\n"
            
            if successful:
                summary += f"\n✓ Successfully ingested files:\n"
                for fname in successful:
                    summary += f"  - {fname}\n"
            
            if failed:
                summary += f"\n✗ Failed files:\n"
                for fname in failed:
                    summary += f"  - {fname}\n"
            
            summary += f"{'='*60}"
            
            return summary
            
        except Exception as e:
            return f"Error ingesting folder: {str(e)}"
    
    def retrieve_relevant_chunks(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        if self.vectorstore is None:
            return []
        
        # Perform similarity search
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def retrieve_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Retrieve relevant chunks with similarity scores.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.vectorstore is None:
            return {"status": "No collection initialized"}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "total_chunks": count,
            "persist_directory": self.persist_directory,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "status": "active" if count > 0 else "empty"
        }
    
    def list_documents(self) -> List[str]:
        """
        List all unique documents in the collection.
        
        Returns:
            List of document names
        """
        if self.vectorstore is None or self.vectorstore._collection.count() == 0:
            return []
        
        # Get all metadata
        collection = self.vectorstore._collection
        results = collection.get()
        
        # Extract unique source documents
        sources = set()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
        
        return sorted(list(sources))
    
    def clear_collection(self):
        """Clear all documents from the vector store."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            self._initialize_vectorstore()
            return "Collection cleared successfully"
        return "No collection to clear"


# Helper function to format retrieved documents
def format_retrieved_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents for display.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Formatted string of document contents
    """
    if not docs:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source} (Page: {page})\n"
            f"Content: {content}\n"
        )
    
    return "\n".join(formatted)
