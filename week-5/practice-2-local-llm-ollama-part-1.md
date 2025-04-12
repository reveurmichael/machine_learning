# Comprehensive Tutorial: Local LLM Research Assistant with RAG (Part 1)

## Overview
This comprehensive tutorial guides students through building a sophisticated research assistant using local LLMs with Ollama, ChromaDB, and LangChain. This first 120-minute session focuses on the fundamentals of Retrieval-Augmented Generation (RAG) with local models, setting up the environment, building a pipeline for document processing, and creating a basic Q&A system.

## Learning Objectives
- Understand the architecture and limitations of large language models
- Master the concept of Retrieval-Augmented Generation (RAG)
- Implement vector embeddings for semantic search
- Build a functional research assistant that can answer questions from academic papers
- Develop skills in prompt engineering for task-specific applications

## Prerequisites
- Basic Python programming knowledge
- Familiarity with machine learning concepts
- Experience with neural networks (covered in weeks 1-4)
- Comfortable with installing Python packages and managing environments

---

## Part 0: Environment Setup (15 minutes)

### Required Software
- **Python 3.10+**: Essential for compatibility with all libraries
- **Ollama**: For running LLMs locally (https://ollama.com)
- **Required Python packages**:
  ```
  pip install langchain langchain-community chromadb pypdf sentence-transformers streamlit faiss-cpu
  ```

### Model Requirements
- Download required models:
  ```bash
  ollama pull llama3      # 8B parameter model, good balance of quality/speed
  ollama pull deepseek-coder  # Optional alternative model
  ```

### Data Preparation
- Create directories for your project:
  ```bash
  mkdir -p research-assistant/{pdfs,db,outputs}
  ```
- Download sample academic papers for testing:
  - "Attention Is All You Need" (Transformer paper)
  - Any ML papers relevant to your research interests

---

## Part 1: Understanding RAG and Local LLMs (20 minutes)

### The Challenge with LLMs
- LLMs have knowledge cutoffs and cannot access recent information
- They can hallucinate or generate false information when uncertain
- May lack domain-specific knowledge (especially in scientific fields)
- Privacy concerns with sending research data to commercial APIs

### The RAG Solution
- **Retrieval**: Find relevant information from trusted sources
- **Augmentation**: Enhance the LLM's context with this information
- **Generation**: Allow the LLM to synthesize the augmented information

### Why Local LLMs?
- **Privacy**: Keep sensitive research data on your own machine
- **Cost**: No API usage fees or token limits
- **Customization**: Greater control over the pipeline
- **Learning**: Better understand how LLMs work

### RAG Architecture Overview
1. **Document Processing**: Split documents into chunks
2. **Embedding**: Convert text chunks into vector representations
3. **Vector Storage**: Organize embeddings for efficient search
4. **Retrieval**: Find relevant chunks based on query similarity
5. **Context Augmentation**: Add retrieved information to LLM prompt
6. **LLM Generation**: Produce answers based on the augmented context

---

## Part 2: Document Processing Pipeline (30 minutes)

### Step 1: Loading PDF Documents
```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """Load a PDF file and extract text content."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
```

### Step 2: Document Chunking for Effective Retrieval
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into manageable chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### Step A: Experimentation - Finding Optimal Chunk Size
- **Exercise**: Experiment with different chunk sizes (500, 1000, 2000)
- **Analysis**: Discuss trade-offs between specificity and context
- **Data Structure**: Examine the metadata of chunks

### Step 3: Embedding Chunks with Sentence Transformers
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embeddings():
    """Initialize the embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
```

### Step 4: Creating and Persisting Vector Store
```python
from langchain_community.vectorstores import Chroma

def create_vectorstore(chunks, embeddings, persist_directory="db"):
    """Create a vector database from document chunks."""
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb
```

### Putting It All Together: Document Processing Pipeline
```python
def process_pdf(pdf_path, persist_dir="db"):
    """Complete pipeline for processing a PDF document."""
    # Load the PDF
    documents = load_pdf(pdf_path)
    
    # Chunk the documents
    chunks = chunk_documents(documents)
    
    # Create embeddings and vector store
    embeddings = create_embeddings()
    vectordb = create_vectorstore(chunks, embeddings, persist_dir)
    
    return vectordb
```

---

## Part 3: Building the RAG Q&A System (30 minutes)

### Step 1: Setting Up the Ollama LLM
```python
from langchain_community.llms import Ollama

def initialize_llm(model_name="llama3"):
    """Initialize the Ollama LLM with appropriate parameters."""
    llm = Ollama(
        model=model_name,
        temperature=0.1,  # Low temperature for more factual responses
        num_ctx=4096,     # Context window size
    )
    return llm
```

### Step 2: Creating a Retriever from Vector Store
```python
def setup_retriever(vectordb, search_kwargs={"k": 4}):
    """Set up a retriever from the vector store."""
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    return retriever
```

### Step 3: Building the RAG Chain
```python
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    """Build a question-answering chain using RAG."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" method puts all retrieved docs into context
        retriever=retriever,
        return_source_documents=True,  # Return source documents for verification
    )
    return qa_chain
```

### Step 4: Asking Questions with RAG
```python
def ask_question(qa_chain, question):
    """Ask a question and get response with sources."""
    result = qa_chain({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]
    
    return answer, source_docs
```

### Complete RAG Question-Answering Implementation
```python
# Load a paper and set up the RAG system
pdf_path = "pdfs/attention_is_all_you_need.pdf"
vectordb = process_pdf(pdf_path)

# Initialize LLM and QA chain
llm = initialize_llm("llama3")
retriever = setup_retriever(vectordb)
qa_chain = build_qa_chain(llm, retriever)

# Ask questions
question = "What is self-attention and why is it important?"
answer, sources = ask_question(qa_chain, question)
print(f"Answer: {answer}\n")
print("Sources:")
for i, doc in enumerate(sources):
    print(f"Source {i+1}: {doc.page_content[:100]}...\n")
```

---

## Part 4: Building a Basic Streamlit Interface (25 minutes)

### Creating a Simple Web UI with Streamlit
```python
import streamlit as st
import os

def create_rag_app():
    # App title and description
    st.title("Academic Paper Research Assistant")
    st.write("Upload a research paper and ask questions about its content.")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3", "deepseek-coder"]
    )
    
    # PDF uploader
    uploaded_pdf = st.sidebar.file_uploader("Upload Research Paper", type="pdf")
    
    # Processing parameters
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
    k_docs = st.sidebar.slider("Number of Documents to Retrieve", 1, 10, 4)
    
    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if uploaded_pdf:
        # Save the uploaded PDF
        pdf_path = f"pdfs/{uploaded_pdf.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        # Process the PDF if not already processed
        if st.session_state.qa_chain is None:
            with st.spinner("Processing document..."):
                # Initialize the LLM
                llm = initialize_llm(model_choice)
                
                # Process the PDF
                documents = load_pdf(pdf_path)
                chunks = chunk_documents(documents, chunk_size=chunk_size)
                embeddings = create_embeddings()
                vectordb = create_vectorstore(chunks, embeddings)
                
                # Set up the QA chain
                retriever = setup_retriever(vectordb, search_kwargs={"k": k_docs})
                qa_chain = build_qa_chain(llm, retriever)
                
                # Store in session state
                st.session_state.qa_chain = qa_chain
                st.sidebar.success("Document processed successfully!")
        
        # Question input
        st.subheader("Ask a question about the paper")
        question = st.text_input("Your question:")
        
        if st.button("Get Answer") and question:
            with st.spinner("Thinking..."):
                answer, sources = ask_question(st.session_state.qa_chain, question)
                
                # Display answer
                st.markdown("### Answer")
                st.write(answer)
                
                # Display sources
                st.markdown("### Sources")
                for i, doc in enumerate(sources):
                    with st.expander(f"Source {i+1}"):
                        st.write(doc.page_content)
                        st.text(f"Page: {doc.metadata.get('page', 'Unknown')}")

if __name__ == "__main__":
    create_rag_app()
```

### Running the Streamlit App
```bash
streamlit run app.py
```

---

## Part 5: Experimentation and Critical Analysis (20 minutes)

### Hands-on Exercise: Breaking the Model
1. Ask questions that require numerical reasoning
2. Request information that's spread across multiple sections
3. Ask about something not in the paper
4. Probe for interpretations vs. factual information

### Analyzing RAG Performance
- **Precision**: How accurate are the answers?
- **Relevance**: Is the system retrieving the right chunks?
- **Completeness**: Is important information being missed?
- **Hallucination**: Is the model making up information?

### Documentation Exercise
Create a table documenting:
- Question asked
- Answer received
- Quality assessment
- Hypothesis for improving the answer

---

## Part 6: Homework and Preparation for Session 2 (10 minutes)

### Homework Assignments
1. Experiment with different embedding models from Hugging Face
2. Try different chunking strategies
3. Test with different academic papers
4. Implement a basic citation mechanism

### Preview of Session 2
- Adding memory and conversation history
- Multi-document support
- Advanced prompting techniques
- Quantitative evaluation
- Building an autonomous research agent

---

## Additional Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [RAG Best Practices Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

## Troubleshooting Guide
- Common issues with Ollama installation
- Debugging vector search problems
- Memory management for large documents
- Performance optimization strategies