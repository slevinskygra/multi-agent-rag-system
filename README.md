#  Multi-Agent RAG System

A sophisticated document Q&A system that combines Retrieval Augmented Generation (RAG) with a multi-agent architecture to provide intelligent, context-aware answers from your documents. 

##  Features

- **Multi-Agent Architecture**: Coordinated specialized agents for optimal query processing
  - **Router Agent**: Intelligently categorizes queries
  - **Retriever Agent**: Searches documents using semantic similarity
  - **Synthesizer Agent**: Creates comprehensive, well-cited answers
  
- **Advanced Document Processing**:
  - Support for PDF and TXT files
  - Automatic chunking with overlap for context preservation
  - Semantic embeddings using sentence-transformers
  - ChromaDB vector storage for efficient retrieval

- **Flexible Interfaces**:
  - Interactive CLI for real-time queries
  - Programmatic API for batch processing
  - Folder ingestion for bulk document loading

- **Smart Query Routing**:
  - Automatic detection of document-based vs. general knowledge queries
  - Comprehensive retrieval for multi-document questions
  - Collection metadata queries without unnecessary retrieval

## 📋 Prerequisites

- Python 3.8+
- Anthropic API key
- Required packages (see `requirements.txt`)

##  Installation

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your API key**:

Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your_api_key_here
```

## 📁 Project Structure

```
multi-agent-rag/
├── rag_system.py          # RAG core functionality
├── multi_agent.py         # Multi-agent coordination
├── main.py                # Interactive CLI interface
├── example.py             # Programmatic usage examples
├── documents/             # Place your documents here
│   ├── gutenberg_org_cache_epub_1998_pg1998_txt.pdf
│   ├── gutenberg_org_cache_epub_2701_pg2701_txt.pdf
│   ├── gutenberg_org_cache_epub_6130_pg6130_txt.pdf
│   └── gutenberg_org_cache_epub_84_pg84_txt.pdf
├── .env                   # Your API key (create this)
└── README.md             # This file
```

##  Quick Start

### Interactive Mode

Run the interactive CLI:

```bash
python main.py
```

Available commands:
- `ingest <filepath>` - Add a single document
- `ingest-folder <path>` - Add all documents from a folder
- `list` - Show all loaded documents
- `stats` - Display collection statistics
- `clear` - Remove all documents
- `ask <question>` - Ask a question
- `quit` - Exit the system

**Example session:**
```bash
🔍 You: ingest-folder documents
🔍 You: What is the main theme of Thus Spoke Zarathustra?
🔍 You: Who is the protagonist in Moby Dick?
🔍 You: list
```

### Programmatic Usage

Run the example script to see all features in action:

```bash
python example.py
```

## 📚 Usage Examples

### Example 1: Basic Document Q&A

```python
from rag_system import RAGSystem
from multi_agent import MultiAgentSystem

# Initialize systems
rag_system = RAGSystem(persist_directory="./my_db")
multi_agent = MultiAgentSystem(rag_system)

# Ingest a document
result = rag_system.ingest_document("documents/gutenberg_org_cache_epub_1998_pg1998_txt.pdf")
print(result)

# Ask a question
result = multi_agent.process_query(
    "Can you summarise the document content?",
    verbose=True
)

print("ANSWER:", result['answer'])
print("SOURCES:", result['sources'])
```

**Expected Output:**
- Successfully ingests Nietzsche's "Thus Spoke Zarathustra"
- Provides a comprehensive summary focusing on philosophical themes
- Cites the source document
- Routes through: Router → Retriever → Synthesizer

### Example 2: Batch Processing Multiple Documents

```python
# Initialize with a separate database
rag_system = RAGSystem(persist_directory="./batch_db")
multi_agent = MultiAgentSystem(rag_system)

# Ingest multiple documents
rag_system.ingest_document("documents/gutenberg_org_cache_epub_1998_pg1998_txt.pdf")  # Zarathustra
rag_system.ingest_document("documents/gutenberg_org_cache_epub_2701_pg2701_txt.pdf")  # Moby Dick

# Process multiple queries
queries = [
    "Who is Zarathustra? Check retrieved data",
    "Who is Moby? Check retrieved data",
    "Summarise the content of retrieved documents"
]

results = multi_agent.batch_process(queries)

# Review results
for query, result in zip(queries, results):
    print(f"\nQuery: {query}")
    print(f"Category: {result['category']}")
    print(f"Answer: {result['answer'][:200]}...")
```

**Expected Output:**
- Query 1: Identifies Zarathustra as Nietzsche's prophet character
- Query 2: Identifies Moby Dick as the white whale from Melville's novel
- Query 3: Provides a combined summary of both philosophical and literary works
- All queries use DOCUMENT_SEARCH category with proper source citations

### Example 3: Folder Ingestion

```python
# Initialize with unique database
rag_system = RAGSystem(persist_directory="./folder_db")
multi_agent = MultiAgentSystem(rag_system)

# Ingest entire folder
result = rag_system.ingest_folder("./documents")
print(result)

# Query across all documents
result = multi_agent.process_query(
    "What are the main topics covered in these documents?",
    verbose=True
)

print("ANSWER:", result['answer'])
print("SOURCES:", result['sources'])
```

**Expected Output:**
```
FOLDER INGESTION SUMMARY
============================================================
Folder: ./documents
Total files found: 4
Successfully ingested: 4
Failed: 0
Total chunks created: 1,247

✓ Successfully ingested files:
  - gutenberg_org_cache_epub_1998_pg1998_txt.pdf (Zarathustra)
  - gutenberg_org_cache_epub_2701_pg2701_txt.pdf (Moby Dick)
  - gutenberg_org_cache_epub_6130_pg6130_txt.pdf (The Iliad)
  - gutenberg_org_cache_epub_84_pg84_txt.pdf (Frankenstein)
```

The system will then provide a comprehensive overview of themes across all four classic works, citing each source appropriately.

## 🏗️ Architecture

### Agent Workflow

```
User Query
    ↓
┌─────────────────┐
│  Router Agent   │  → Categorizes query
└─────────────────┘
    ↓
    ├─→ DOCUMENT_SEARCH
    │       ↓
    │   ┌─────────────────┐
    │   │ Retriever Agent │  → Searches vector DB
    │   └─────────────────┘
    │       ↓
    │   ┌──────────────────┐
    │   │Synthesizer Agent │  → Creates answer
    │   └──────────────────┘
    │
    ├─→ GENERAL_KNOWLEDGE
    │       ↓
    │   ┌──────────────────┐
    │   │Synthesizer Agent │  → Direct answer
    │   └──────────────────┘
    │
    └─→ COLLECTION_INFO
            ↓
        [Metadata retrieval]
```

### Key Components

**RAG System (`rag_system.py`)**:
- Document loading (PDF, TXT)
- Text chunking (1000 chars with 200 char overlap)
- Embedding generation (sentence-transformers/all-MiniLM-L6-v2)
- Vector storage (ChromaDB)
- Similarity search

**Multi-Agent System (`multi_agent.py`)**:
- **RouterAgent**: Query classification
- **RetrieverAgent**: Document retrieval with two modes:
  - Standard retrieval (k=80 chunks)
  - Comprehensive retrieval (60 chunks per document)
- **SynthesizerAgent**: Answer generation with strict document-only responses

## 🎛️ Configuration

### Retrieval Parameters

In `multi_agent.py`, you can adjust:

```python
# Standard retrieval
retrieval_result = self.retriever.retrieve(query, k=80)

# Comprehensive retrieval (for "summarize all" queries)
retrieval_result = self.retriever.retrieve_comprehensive(query, k_per_doc=60)
```

### Chunking Parameters

In `rag_system.py`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    length_function=len,
)
```

### Model Selection

In `multi_agent.py`:

```python
self.llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",  # Change model here
    temperature=0,                      # Adjust creativity (0-1)
    api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
)
```

## 📊 Sample Documents

The repository includes four classic literary works:

1. **Thus Spoke Zarathustra** (Nietzsche) - Philosophical discourse on the Übermensch
2. **Moby Dick** (Melville) - Epic tale of Captain Ahab's quest
3. **The Iliad** (Homer) - Ancient Greek epic of the Trojan War
4. **Frankenstein** (Mary Shelley) - Gothic novel on creation and responsibility

These demonstrate the system's ability to handle diverse literary and philosophical content.

## 🔧 Troubleshooting

### Database Permission Issues

If you encounter "readonly database" errors:

```bash
# Delete existing databases
rm -rf ./example_db_*
rm -rf ./chroma_db

# Or run the fix script
python fix_chroma_db.py
```

### Memory Issues with Large Documents

For very large documents, reduce chunk retrieval:

```python
# In multi_agent.py, reduce k values
retrieval_result = self.retriever.retrieve(query, k=40)  # Instead of 80
```

### API Rate Limits

Add delays between batch queries:

```python
import time
for query in queries:
    result = multi_agent.process_query(query)
    time.sleep(1)  # 1 second delay
```

## 🎯 Best Practices

1. **Document Preparation**:
   - Clean PDFs work best (avoid scanned images)
   - For large documents, consider splitting into chapters
   - Remove unnecessary headers/footers

2. **Query Formulation**:
   - Be specific about what you want
   - Use phrases like "Check retrieved data" to ensure document search
   - For summaries, ask about "main topics" or "key themes"

3. **Database Management**:
   - Use separate databases for different projects
   - Clear collections when switching document sets
   - Back up databases before major changes

4. **Performance Optimization**:
   - Ingest documents once, query multiple times
   - Use batch processing for related queries
   - Monitor chunk counts (aim for < 5000 for best performance)

## 📝 Common Query Patterns

**Document-specific questions:**
```
"What does [document] say about [topic]?"
"Who is [character] in the retrieved documents?"
"Summarize the main argument in [document]"
```

**Multi-document analysis:**
```
"Compare the themes across all documents"
"What are the common topics in these works?"
"Summarize each document separately"
```

**Metadata queries:**
```
"What documents do I have?"
"How many documents are loaded?"
"List all documents in the collection"
```

##  Contributing

Feel free to extend this system by:
- Adding new document loaders (DOCX, CSV, etc.)
- Implementing additional agents (SummaryAgent, ComparisonAgent)
- Enhancing the retrieval with re-ranking
- Adding visualization tools for results

## 📄 License

This project uses documents from Project Gutenberg, which are in the public domain.

##  Acknowledgments

- **Anthropic** for Claude API
- **LangChain** for RAG components
- **ChromaDB** for vector storage
- **Sentence Transformers** for embeddings
- **Project Gutenberg** for public domain texts

---


For questions or issues, please refer to the inline documentation in the code or run examples with `verbose=True` to see detailed processing steps.
