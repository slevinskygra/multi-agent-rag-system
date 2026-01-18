"""
Multi-Agent System - Coordinates specialized agents for document Q&A

Agents:
1. Router Agent: Determines the best approach for each query
2. Retriever Agent: Searches documents using RAG
3. Synthesizer Agent: Combines information and creates answers
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from rag_system import RAGSystem, format_retrieved_docs


class RouterAgent:
    """
    Router Agent - Decides how to handle each query.
    
    This agent analyzes the user's question and decides:
    - Does it need document retrieval?
    - Is it a general question?
    - Does it need multiple documents?
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a Router Agent that analyzes user queries.

Your job is to categorize queries into ONE of these types:

1. DOCUMENT_SEARCH: Query asks about content, information, or summaries IN documents
   Examples: 
   - "What does the contract say about..."
   - "Find information about..."
   - "Summarize the documents"
   - "What are the main topics in these documents?"
   - "What is the content about?"
   - "Tell me about what's in the documents"
   - "What themes are covered?"
   - "Give me a summary of each document"

2. GENERAL_KNOWLEDGE: Query is about general knowledge, not specific documents
   Examples: 
   - "What is machine learning?"
   - "Explain quantum physics"
   - "How does RAG work?"

3. COLLECTION_INFO: Query asks ONLY about document metadata (names, count, list)
   Examples: 
   - "What documents do I have?" (just the names)
   - "List all documents" (just the filenames)
   - "How many documents are there?"
   - "Show me the collection stats"

IMPORTANT: If the query asks about content, topics, summaries, or what's IN the documents, use DOCUMENT_SEARCH, not COLLECTION_INFO.

Respond with ONLY the category name (DOCUMENT_SEARCH, GENERAL_KNOWLEDGE, or COLLECTION_INFO).
"""
    
    def route(self, query: str) -> str:
        """
        Determine the routing category for a query.
        
        Args:
            query: User's question
            
        Returns:
            Category: DOCUMENT_SEARCH, GENERAL_KNOWLEDGE, or COLLECTION_INFO
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Categorize this query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        category = response.content.strip()
        
        # Validate category
        valid_categories = ["DOCUMENT_SEARCH", "GENERAL_KNOWLEDGE", "COLLECTION_INFO"]
        if category not in valid_categories:
            # Default to document search if unclear
            return "DOCUMENT_SEARCH"
        
        return category


class RetrieverAgent:
    """
    Retriever Agent - Searches documents using RAG.
    
    This agent uses the RAG system to find relevant information
    from the document collection.
    """
    
    def __init__(self, llm, rag_system: RAGSystem):
        self.llm = llm
        self.rag_system = rag_system
    
    def retrieve(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        # Get relevant documents from RAG system
        docs = self.rag_system.retrieve_relevant_chunks(query, k=k)
        
        if not docs:
            return {
                "success": False,
                "message": "No relevant documents found in the collection.",
                "documents": [],
                "formatted_context": ""
            }
        
        # Format documents
        formatted_context = format_retrieved_docs(docs)
        
        return {
            "success": True,
            "message": f"Found {len(docs)} relevant document chunks",
            "documents": docs,
            "formatted_context": formatted_context
        }
    
    def retrieve_comprehensive(self, query: str, k_per_doc: int = 3) -> Dict[str, Any]:
        """
        Retrieve representative chunks from EACH document for comprehensive queries.
        Use this for queries like "summarize all documents" or "what are the topics".
        
        Args:
            query: User's question
            k_per_doc: Number of chunks to retrieve per document
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        # Get list of all documents
        doc_names = self.rag_system.list_documents()
        
        if not doc_names:
            return {
                "success": False,
                "message": "No documents found in the collection.",
                "documents": [],
                "formatted_context": ""
            }
        
        all_docs = []
        
        # For each document, retrieve representative chunks
        for doc_name in doc_names:
            # Create a query specific to this document
            doc_query = f"{query} in {doc_name}"
            docs = self.rag_system.retrieve_relevant_chunks(doc_query, k=k_per_doc)
            all_docs.extend(docs)
        
        if not all_docs:
            return {
                "success": False,
                "message": "Could not retrieve content from documents.",
                "documents": [],
                "formatted_context": ""
            }
        
        # Format documents
        formatted_context = format_retrieved_docs(all_docs)
        
        return {
            "success": True,
            "message": f"Retrieved content from {len(doc_names)} document(s)",
            "documents": all_docs,
            "formatted_context": formatted_context
        }


class SynthesizerAgent:
    """
    Synthesizer Agent - Creates comprehensive answers.
    
    This agent takes retrieved information and synthesizes it
    into a coherent, helpful answer.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a Synthesizer Agent that creates helpful answers based on retrieved documents.

Your job is to:
1. Read the provided document chunks carefully
2. Extract relevant information
3. Synthesize a clear, accurate answer
4. Cite which documents you used
5. Be honest if the documents don't contain the answer

Guidelines:
- Always cite your sources (e.g., "According to document X...")
- If information is not in the documents, say so clearly
- Combine information from multiple sources when relevant
- Be concise but thorough
"""
    
    def synthesize(self, query: str, context: str) -> str:
        """
        Synthesize an answer from retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document context
            
        Returns:
            Synthesized answer
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Question: {query}

Retrieved Documents:
{context}

Please provide a comprehensive answer based on the documents above.""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def answer_general(self, query: str) -> str:
        """
        Answer a general knowledge question.
        
        Args:
            query: User's question
            
        Returns:
            Answer
        """
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer questions clearly and concisely."),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class MultiAgentSystem:
    """
    Main Multi-Agent System orchestrator.
    
    Coordinates the Router, Retriever, and Synthesizer agents
    to answer user queries.
    """
    
    def __init__(self, rag_system: RAGSystem, api_key: str = None):
        """
        Initialize the multi-agent system.
        
        Args:
            rag_system: The RAG system instance
            api_key: Anthropic API key
        """
        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Initialize agents
        self.router = RouterAgent(self.llm)
        self.retriever = RetrieverAgent(self.llm, rag_system)
        self.synthesizer = SynthesizerAgent(self.llm)
        self.rag_system = rag_system
    
    def process_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            query: User's question
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            print("\n" + "="*60)
            print("MULTI-AGENT SYSTEM PROCESSING")
            print("="*60)
        
        # Step 1: Route the query
        if verbose:
            print("\n[STEP 1: ROUTER AGENT]")
            print(f"Analyzing query: '{query}'")
        
        category = self.router.route(query)
        
        if verbose:
            print(f"→ Category: {category}")
        
        # Step 2: Handle based on category
        if category == "COLLECTION_INFO":
            if verbose:
                print("\n[STEP 2: COLLECTION INFO]")
            
            docs = self.rag_system.list_documents()
            stats = self.rag_system.get_collection_stats()
            
            answer = f"Document Collection Status:\n\n"
            answer += f"Total chunks: {stats['total_chunks']}\n"
            answer += f"Documents in collection: {len(docs)}\n\n"
            
            if docs:
                answer += "Documents:\n"
                for i, doc in enumerate(docs, 1):
                    answer += f"{i}. {doc}\n"
            else:
                answer += "No documents loaded yet. Use the ingest command to add documents."
            
            return {
                "answer": answer,
                "category": category,
                "sources": [],
                "workflow": "Router → Collection Info"
            }
        
        elif category == "GENERAL_KNOWLEDGE":
            if verbose:
                print("\n[STEP 2: GENERAL KNOWLEDGE]")
                print("No document retrieval needed")
                print("\n[STEP 3: SYNTHESIZER AGENT]")
                print("Generating answer from general knowledge...")
            
            answer = self.synthesizer.answer_general(query)
            
            return {
                "answer": answer,
                "category": category,
                "sources": [],
                "workflow": "Router → Synthesizer (General Knowledge)"
            }
        
        else:  # DOCUMENT_SEARCH
            if verbose:
                print("\n[STEP 2: RETRIEVER AGENT]")
                print("Searching document collection...")
            
            # Detect if this is a comprehensive query (asking about all/multiple documents)
            comprehensive_keywords = [
                "all documents", "each document", "all the documents",
                "summarize the documents", "summarize all", "summary of each",
                "what are the topics", "what themes", "main topics",
                "overview of", "tell me about the documents"
            ]
            
            is_comprehensive = any(keyword in query.lower() for keyword in comprehensive_keywords)
            
            # Retrieve documents - use comprehensive retrieval for broad queries
            if is_comprehensive:
                if verbose:
                    print("→ Detected comprehensive query - retrieving from all documents")
                retrieval_result = self.retriever.retrieve_comprehensive(query, k_per_doc=60)
            else:
                retrieval_result = self.retriever.retrieve(query, k=80)
            
            if not retrieval_result["success"]:
                return {
                    "answer": retrieval_result["message"],
                    "category": category,
                    "sources": [],
                    "workflow": "Router → Retriever (No results)"
                }
            
            if verbose:
                print(f"→ {retrieval_result['message']}")
                print("\n[STEP 3: SYNTHESIZER AGENT]")
                print("Synthesizing answer from retrieved documents...")
            
            # Synthesize answer
            answer = self.synthesizer.synthesize(
                query,
                retrieval_result["formatted_context"]
            )
            
            # Extract sources
            sources = [doc.metadata.get('source', 'Unknown') 
                      for doc in retrieval_result["documents"]]
            sources = list(set(sources))  # Remove duplicates
            
            if verbose:
                print(f"→ Used sources: {', '.join(sources)}")
            
            return {
                "answer": answer,
                "category": category,
                "sources": sources,
                "workflow": "Router → Retriever → Synthesizer"
            }
    
    def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user questions
            
        Returns:
            List of results for each query
        """
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Processing Query {i}/{len(queries)}")
            print(f"{'='*60}")
            result = self.process_query(query, verbose=True)
            results.append(result)
        
        return results
