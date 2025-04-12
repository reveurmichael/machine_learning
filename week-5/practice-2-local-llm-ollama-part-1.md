# Comprehensive Tutorial: Local LLM Research Assistant with RAG (Part 1)

## Overview
This comprehensive tutorial guides students through building a sophisticated research assistant using local LLMs with Ollama, ChromaDB, and LangChain. This first 120-minute session focuses on the fundamentals of Retrieval-Augmented Generation (RAG) with local models, setting up the environment, building a pipeline for document processing, and creating a basic Q&A system.

In this tutorial, you'll learn how to create a system that can:
1. Process academic papers (PDFs)
2. Store their content in a vector database
3. Retrieve relevant information based on user queries
4. Generate accurate responses using a local language model

## Learning Objectives
- Understand the architecture and limitations of large language models
- Master the concept of Retrieval-Augmented Generation (RAG) and why it's essential for accurate information retrieval
- Implement vector embeddings for semantic search of document content
- Build a functional research assistant that can answer questions from academic papers
- Develop skills in prompt engineering for task-specific applications

## Prerequisites
- Basic Python programming knowledge (functions, classes, importing libraries)
- Familiarity with machine learning concepts (neural networks, embeddings)
- Experience with neural networks (covered in weeks 1-4)
- Comfortable with installing Python packages and managing environments
- A computer with at least 16GB RAM (recommended for running local LLMs)

---

## Part 0: Environment Setup (15 minutes)

### Required Software
- **Python 3.10+**: Essential for compatibility with all libraries used in this tutorial
- **Ollama**: A framework for running LLMs locally (install from https://ollama.com)
- **Required Python packages**:
  ```bash
  pip install langchain langchain-community chromadb pypdf sentence-transformers streamlit faiss-cpu
  ```
  
  Package explanation:
  - `langchain` & `langchain-community`: Framework for creating LLM applications
  - `chromadb`: Vector database for storing embeddings
  - `pypdf`: PDF parsing library
  - `sentence-transformers`: For creating text embeddings
  - `streamlit`: For building the web interface
  - `faiss-cpu`: Efficient similarity search library

### Model Requirements
- Download the required models using the following commands:
  ```bash
  # Download the main model we'll use (about 4GB)
  ollama pull llama3      # 8B parameter model, good balance of quality/speed
  
  # Optional alternative model for experimentation (about 7GB)
  ollama pull deepseek-coder  # Specialized for code understanding
  ```
  
  > **Note**: The first download may take several minutes depending on your internet connection.

### Data Preparation
- Create the necessary directory structure for your project:
  ```bash
  mkdir -p research-assistant/{pdfs,db,outputs}
  ```
  
  This creates:
  - `pdfs/`: Directory to store your academic papers
  - `db/`: Directory for the vector database files
  - `outputs/`: Directory for saving results
  
- Download sample academic papers for testing:
  - ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) (The original Transformer paper)
  - Any ML papers relevant to your research interests (from arxiv.org or other academic sources)
  
  > **Tip**: Save the papers in the `pdfs/` directory you created.

---

## Part 1: Understanding RAG and Local LLMs (20 minutes)

### The Challenge with LLMs
- **Knowledge Cutoffs**: LLMs like GPT-4 and Llama have knowledge cutoffs (e.g., training data only up to 2023) and cannot access recent information without updates
- **Hallucinations**: They can hallucinate or generate false information when uncertain about a topic
- **Domain Knowledge Gaps**: They may lack specialized domain-specific knowledge (especially in scientific fields)
- **Privacy Concerns**: Sending research data to commercial API services could compromise intellectual property or sensitive information

### The RAG Solution
- **Retrieval**: The system searches through your documents to find information relevant to a user's question
  - Example: When asked "What is self-attention?", it finds paragraphs in papers that explain this concept
  
- **Augmentation**: The retrieved information is added to the LLM's context window
  - Example: Adding paragraphs from the Transformer paper to the prompt sent to the LLM
  
- **Generation**: The LLM uses both its pre-trained knowledge and the augmented context to create a response
  - Example: Generating an explanation of self-attention based on the retrieved information

### Why Local LLMs?
- **Privacy**: Your data never leaves your machine, which is crucial for sensitive research
- **Cost**: No API usage fees or token limits, allowing unlimited experimentation
- **Customization**: Greater control over the entire pipeline, from document processing to response generation
- **Learning**: Better understand how LLMs work by directly interacting with the models

### RAG Architecture Overview
1. **Document Processing**: Split large documents (like academic papers) into smaller chunks that fit within the context window
2. **Embedding**: Convert text chunks into vector representations (numerical arrays) that capture semantic meaning
3. **Vector Storage**: Organize these embeddings in a database that enables efficient similarity search
4. **Retrieval**: When a question is asked, find chunks most similar to the query using vector similarity
5. **Context Augmentation**: Add the retrieved text chunks to the prompt sent to the LLM
6. **LLM Generation**: The LLM produces an answer based on both its pre-trained knowledge and the augmented context

---

## Part 2: Document Processing Pipeline (30 minutes)

### Step 1: Loading PDF Documents
```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Load a PDF file and extract text content.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects, each containing text from a page with metadata
    """
    # PyPDFLoader splits the PDF into pages and extracts text with metadata
    loader = PyPDFLoader(file_path)
    # Each document contains the text of one page with page number in metadata
    documents = loader.load()
    return documents
```

The `PyPDFLoader` creates Document objects that contain:
- `page_content`: The extracted text from each page
- `metadata`: Information like page number and source file

### Step 2: Document Chunking for Effective Retrieval
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into manageable chunks with overlap to maintain context.
    
    Args:
        documents (list): List of Document objects from the PDF loader
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of smaller Document chunks
    """
    # RecursiveCharacterTextSplitter tries to split on paragraph, then sentence, then word
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # Maximum chunk size in characters
        chunk_overlap=chunk_overlap,    # Overlap ensures context isn't lost between chunks
        length_function=len,            # Function to measure text length
        add_start_index=True,           # Adds character index in metadata for reference
    )
    # Split the documents into smaller chunks
    chunks = text_splitter.split_documents(documents)
    return chunks
```

Why chunking matters:
- LLMs have a limited context window (max tokens they can process)
- Large documents need to be broken into smaller pieces
- Overlapping chunks prevents information from being cut off at chunk boundaries
- The chunk size affects retrieval precision and context completeness

### Step 3: Experimentation - Finding Optimal Chunk Size
- **Exercise**: Experiment with different chunk sizes to find what works best for your documents
  - Small chunks (500 chars): More precise retrieval but may lack context
  - Medium chunks (1000 chars): Balanced approach for most use cases
  - Large chunks (2000 chars): More context but potentially less precise retrieval
  
- **Analysis Prompts**:
  - How does changing chunk size affect the quality of answers?
  - What are the trade-offs between large and small chunks?
  - How does chunk overlap impact context continuity?

- **Data Structure**: Examine the metadata of chunks to understand what information is preserved:
  ```python
  # Example code to inspect chunk metadata
  first_chunk = chunks[0]
  print(f"Content sample: {first_chunk.page_content[:100]}...")
  print(f"Metadata: {first_chunk.metadata}")
  ```

### Step 4: Embedding Chunks with Sentence Transformers
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embeddings():
    """
    Initialize the embedding model for converting text to vectors.
    
    Returns:
        HuggingFaceEmbeddings: The embedding model instance
    """
    # all-MiniLM-L6-v2 is a small, efficient model that balances quality and speed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
```

What are embeddings?
- Embeddings are dense vector representations of text
- They capture semantic meaning in a high-dimensional space
- Similar texts have embeddings that are close to each other in vector space
- This allows for semantic search rather than just keyword matching
- The model we're using creates 384-dimensional vectors for each text chunk

### Step 5: Creating and Persisting Vector Store
```python
from langchain_community.vectorstores import Chroma

def create_vectorstore(chunks, embeddings, persist_directory="db"):
    """
    Create a vector database from document chunks.
    
    Args:
        chunks (list): List of document chunks to embed
        embeddings (HuggingFaceEmbeddings): The embedding model to use
        persist_directory (str): Directory to save the database
        
    Returns:
        Chroma: Vector database instance containing the embedded chunks
    """
    # Create a Chroma database from the documents and embedding function
    vectordb = Chroma.from_documents(
        documents=chunks,              # The document chunks to embed
        embedding=embeddings,          # The embedding function
        persist_directory=persist_directory  # Where to save the database
    )
    # Save the database to disk for reuse
    vectordb.persist()
    return vectordb
```

ChromaDB explained:
- It's a vector database optimized for storing and retrieving embeddings
- Automatically handles the creation and indexing of vectors
- Enables efficient similarity search to find relevant document chunks
- Persisting the database means we don't have to re-embed documents each time

### Putting It All Together: Document Processing Pipeline
```python
def process_pdf(pdf_path, persist_dir="db"):
    """
    Complete pipeline for processing a PDF document.
    
    Args:
        pdf_path (str): Path to the PDF file
        persist_dir (str): Directory to save the vector database
        
    Returns:
        Chroma: Vector database containing the processed document
    """
    # Step 1: Load the PDF and extract text
    documents = load_pdf(pdf_path)
    print(f"Loaded {len(documents)} pages from the PDF")
    
    # Step 2: Chunk the documents into smaller pieces
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks from the document")
    
    # Step 3: Create embeddings and vector store
    embeddings = create_embeddings()
    print("Initialized embedding model")
    
    # Step 4: Create and persist vector database
    vectordb = create_vectorstore(chunks, embeddings, persist_dir)
    print(f"Created vector database with {len(chunks)} chunks")
    
    return vectordb
```

This pipeline transforms a PDF into a searchable vector database:
1. We load the PDF and extract text content
2. We split the content into manageable chunks
3. We create an embedding model to convert text to vectors
4. We store these vectors in ChromaDB for efficient retrieval

---

## Part 3: Building the RAG Q&A System (30 minutes)

### Step 1: Setting Up the Ollama LLM
```python
from langchain_community.llms import Ollama

def initialize_llm(model_name="llama3"):
    """
    Initialize the Ollama LLM with appropriate parameters.
    
    Args:
        model_name (str): Name of the Ollama model to use
        
    Returns:
        Ollama: Configured LLM instance
    """
    llm = Ollama(
        model=model_name,
        temperature=0.1,  # Low temperature for more factual, deterministic responses
        num_ctx=4096,     # Context window size in tokens (max context the model can use)
    )
    return llm
```

Understanding the parameters:
- `model`: The Ollama model to use (we downloaded llama3 earlier)
- `temperature`: Controls randomness (0.0-1.0)
  - Low values (0.1): More deterministic, factual responses
  - High values (0.7+): More creative, varied responses
- `num_ctx`: Maximum context window size in tokens (4096 is typical for llama3)

### Step 2: Creating a Retriever from Vector Store
```python
def setup_retriever(vectordb, search_kwargs={"k": 4}):
    """
    Set up a retriever from the vector store to find relevant document chunks.
    
    Args:
        vectordb (Chroma): The vector database
        search_kwargs (dict): Parameters for the search, k is the number of chunks to retrieve
        
    Returns:
        VectorStoreRetriever: Retriever object to find relevant chunks
    """
    # Create a retriever from the vector database
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    return retriever
```

What the retriever does:
- It's a wrapper around the vector database that simplifies document retrieval
- `k=4` means it will return the 4 most similar chunks to any query
- Increasing k provides more context but may include less relevant information
- The retriever uses similarity search to find chunks that match the query semantically

### Step 3: Building the RAG Chain
```python
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    """
    Build a question-answering chain using RAG.
    
    Args:
        llm (Ollama): The language model
        retriever (VectorStoreRetriever): The retriever for finding relevant chunks
        
    Returns:
        RetrievalQA: Chain that combines retrieval and question answering
    """
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                       # The language model
        chain_type="stuff",            # "stuff" method combines all retrieved docs into one prompt
        retriever=retriever,           # The retriever that finds relevant chunks
        return_source_documents=True,  # Return source documents for verification
    )
    return qa_chain
```

Understanding chain_type options:
- `"stuff"`: Simplest method that adds all retrieved documents to the context
  - Works well when retrieved documents are small and few in number
  - May fail if total context exceeds model's context window
- Other options (for advanced usage):
  - `"map_reduce"`: Process each document separately, then combine results
  - `"refine"`: Iteratively update answer with each document
  - `"map_rerank"`: Score responses from each document and return the best

### Step 4: Asking Questions with RAG
```python
def ask_question(qa_chain, question):
    """
    Ask a question and get response with sources.
    
    Args:
        qa_chain (RetrievalQA): The question-answering chain
        question (str): The question to ask
        
    Returns:
        tuple: (answer string, list of source documents)
    """
    # Run the chain with the query
    result = qa_chain({"query": question})
    
    # Extract the answer and source documents
    answer = result["result"]
    source_docs = result["source_documents"]
    
    return answer, source_docs
```

This function:
1. Sends the question to the QA chain
2. The chain retrieves relevant document chunks
3. These chunks are added to the prompt sent to the LLM
4. The LLM generates an answer based on the question and retrieved information
5. Both the answer and source documents are returned for verification

### Complete RAG Question-Answering Implementation
```python
def run_rag_example():
    """Example of running the complete RAG pipeline on a sample question."""
    # Load a paper and set up the RAG system
    pdf_path = "pdfs/attention_is_all_you_need.pdf"
    print(f"Processing {pdf_path}...")
    vectordb = process_pdf(pdf_path)
    
    # Initialize LLM and QA chain
    print("Initializing LLM...")
    llm = initialize_llm("llama3")
    retriever = setup_retriever(vectordb)
    qa_chain = build_qa_chain(llm, retriever)
    
    # Ask questions
    question = "What is self-attention and why is it important?"
    print(f"\nQuestion: {question}")
    print("Thinking...")
    answer, sources = ask_question(qa_chain, question)
    
    print(f"\nAnswer: {answer}\n")
    print("Sources:")
    for i, doc in enumerate(sources):
        print(f"Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):")
        print(f"{doc.page_content[:150]}...\n")
        
# You can run this function to test your implementation
# run_rag_example()
```

This example demonstrates:
1. Processing a research paper (the Transformer paper)
2. Setting up the retrieval system and language model
3. Asking a specific question about self-attention
4. Getting a response based on the paper's content
5. Viewing the source documents used to generate the answer

---

## Part 4: Building a Basic Streamlit Interface (25 minutes)

### Creating a Simple Web UI with Streamlit
```python
import streamlit as st
import os

def create_rag_app():
    """Create a Streamlit web application for the RAG system."""
    # App title and description
    st.title("Academic Paper Research Assistant")
    st.write("Upload a research paper and ask questions about its content. The assistant uses RAG technology to provide accurate answers based directly on the paper.")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3", "deepseek-coder"],
        help="Choose which local language model to use"
    )
    
    # PDF uploader
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload Research Paper", 
        type="pdf",
        help="Upload an academic paper in PDF format"
    )
    
    # Processing parameters with explanations
    st.sidebar.subheader("Advanced Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size", 
        500, 2000, 1000, 100,
        help="Size of text chunks in characters. Smaller chunks give more precise retrieval, larger chunks provide more context."
    )
    
    k_docs = st.sidebar.slider(
        "Number of Documents to Retrieve", 
        1, 10, 4,
        help="How many chunks to retrieve for each question. More chunks provide more context but might include irrelevant information."
    )
    
    # Initialize session state to store the QA chain
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    # If a PDF is uploaded, process it
    if uploaded_pdf:
        # Create directory if it doesn't exist
        os.makedirs("pdfs", exist_ok=True)
        
        # Save the uploaded PDF
        pdf_path = f"pdfs/{uploaded_pdf.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        # Display paper information
        st.write(f"📄 Loaded paper: **{uploaded_pdf.name}**")
        
        # Process the PDF if not already processed
        if st.session_state.qa_chain is None:
            with st.spinner("Processing document... This may take a minute or two."):
                # Initialize the LLM
                llm = initialize_llm(model_choice)
                
                # Process the PDF
                documents = load_pdf(pdf_path)
                st.write(f"📄 Extracted {len(documents)} pages from the document")
                
                chunks = chunk_documents(documents, chunk_size=chunk_size)
                st.write(f"✂️ Split into {len(chunks)} chunks")
                
                embeddings = create_embeddings()
                vectordb = create_vectorstore(chunks, embeddings)
                st.write("🧠 Created vector embeddings")
                
                # Set up the QA chain
                retriever = setup_retriever(vectordb, search_kwargs={"k": k_docs})
                qa_chain = build_qa_chain(llm, retriever)
                
                # Store in session state for reuse
                st.session_state.qa_chain = qa_chain
                st.sidebar.success("✅ Document processed successfully!")
        
        # Question input section
        st.subheader("Ask a question about the paper")
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the main contribution of this paper?"
        )
        
        # When the user asks a question
        if st.button("Get Answer") and question:
            with st.spinner("Thinking..."):
                answer, sources = ask_question(st.session_state.qa_chain, question)
                
                # Display answer
                st.markdown("### Answer")
                st.write(answer)
                
                # Display sources with metadata
                st.markdown("### Sources")
                for i, doc in enumerate(sources):
                    with st.expander(f"Source {i+1} (Page {doc.metadata.get('page', 'Unknown')})"):
                        st.write(doc.page_content)
                        
                        # Display additional metadata if available
                        if 'source' in doc.metadata:
                            st.text(f"File: {os.path.basename(doc.metadata['source'])}")
                        if 'start_index' in doc.metadata:
                            st.text(f"Position in document: character {doc.metadata['start_index']}")

if __name__ == "__main__":
    create_rag_app()
```

### Streamlit App Structure Explained

The Streamlit app provides:
1. **User Interface**: Clean, intuitive interface for uploading documents and asking questions
2. **Configuration Options**:
   - Choice of local LLM model
   - Document chunking parameters 
   - Number of chunks to retrieve
3. **Processing Pipeline**:
   - Loads and processes the uploaded PDF
   - Creates vector embeddings
   - Sets up the retrieval system
4. **Question Answering**:
   - Takes user questions
   - Retrieves relevant information
   - Generates answers with source citations
5. **Source Verification**:
   - Shows which parts of the document were used
   - Displays page numbers and other metadata
   - Allows users to verify the information

### Running the Streamlit App
```bash
# Save the code to a file named app.py
# Then run:
streamlit run app.py
```

Streamlit will:
1. Start a local web server (typically on port 8501)
2. Open your browser to the app interface
3. Allow you to interact with your RAG system through a user-friendly interface

---

## Part 5: Experimentation and Critical Analysis (20 minutes)

### Hands-on Exercise: Testing the Limits of RAG
Try the following types of questions to understand the strengths and limitations of your RAG system:

1. **Basic Factual Queries**:
   - "Who are the authors of the paper?"
   - "What dataset was used in the experiments?"

2. **Complex Reasoning Queries**:
   - "How does the performance of the model compare to previous approaches?"
   - "What are the limitations of the methodology described in the paper?"

3. **Numerical Reasoning**:
   - "What was the improvement in BLEU score compared to the baseline?"
   - "Calculate the total number of parameters in the model architecture."

4. **Cross-Document Questions**:
   - "How does this paper's approach differ from [another paper]?"
   - Questions that require information spread across multiple sections

5. **Out-of-Scope Questions**:
   - "What advances have been made in this field since this paper was published?"
   - "How would this approach work for [task not mentioned in paper]?"

### Analyzing RAG Performance
When evaluating your RAG system, consider these dimensions:

- **Precision**: How accurate are the answers?
  - Are factual details correct?
  - Does the system introduce errors not present in the source?
  
- **Relevance**: Is the system retrieving the right chunks?
  - Do the retrieved chunks contain the information needed?
  - Are irrelevant chunks being included?
  
- **Completeness**: Is important information being missed?
  - Does the answer cover all aspects mentioned in the paper?
  - Are important details omitted?
  
- **Hallucination**: Is the model making up information?
  - Does the answer contain details not found in the paper?
  - Is the model "filling in gaps" with its pre-trained knowledge?
  
- **Coherence**: Does the answer flow logically?
  - Is the response well-structured and easy to understand?
  - Does it integrate information from multiple chunks coherently?

### Documentation Exercise
Create a table documenting your experiments with the following columns:

| Question | Answer | Retrieved Chunks | Quality Assessment | Improvement Hypothesis |
|----------|--------|------------------|--------------------|-----------------------|
| What is self-attention? | [The answer...] | Chunks #2, #5, #7, #10 | Accurate but missing implementation details | Increase chunk size or change overlap |

For "Quality Assessment," rate each answer on:
- Factual correctness (1-5)
- Completeness (1-5)
- Clarity (1-5)
- Overall quality (1-5)

For "Improvement Hypothesis," suggest changes that might address any issues:
- Different chunk sizes
- Different retrieval parameters (k value)
- Different prompting strategies
- Different LLM models

---

## Part 6: Homework and Preparation for Session 2

### Homework Assignments

1. **Experiment with Different Embedding Models**
   - Try alternatives to the default model:
     ```python
     embeddings = HuggingFaceEmbeddings(
         model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
     )
     ```
   - Compare results and document differences in retrieval quality

2. **Implement a Basic Citation Mechanism**
   - Modify the `ask_question` function to include page numbers and section information
   - Format the response like:
     ```
     "According to page 4, section 'Methodology', the model achieves..."
     ```
   - Hint: Use the metadata from source documents to build citations

### Preview of Session 2: Advanced RAG Techniques

In the next session, we'll build on this foundation to create a more sophisticated research assistant:

- **Memory and Conversation History**
  - Enabling multi-turn conversations about papers
  - Maintaining context across multiple questions

- **Multi-Document Support**
  - Working with multiple papers simultaneously
  - Comparing and contrasting information across papers

- **Advanced Prompting Techniques**
  - Structured output formatting
  - Chain-of-thought reasoning for complex questions
  - Few-shot prompting for specialized tasks

- **Quantitative Evaluation**
  - Creating test sets for RAG evaluation
  - Measuring precision, recall, and F1 scores

- **Building an Autonomous Research Agent**
  - Using ReAct (Reasoning + Acting) agents
  - Implementing tools for literature search and analysis

---

## Additional Resources

### Documentation and Tutorials
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) - Comprehensive framework for LLM applications
- [Ollama GitHub Repository](https://github.com/ollama/ollama) - For running and customizing local LLMs
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database used in this tutorial
- [Sentence Transformers Documentation](https://www.sbert.net/) - For understanding embedding models
- [RAG Best Practices Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Advanced techniques for RAG systems

### Academic Papers
- ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) - Original RAG paper
- ["Improving Language Models by Retrieving from Trillions of Tokens"](https://arxiv.org/abs/2112.04426) - RETRO paper from DeepMind
- ["Internet-augmented language models through few-shot prompting"](https://arxiv.org/abs/2203.05115) - WebGPT paper

### Video Tutorials
- [Building RAG from Scratch](https://www.youtube.com/watch?v=qOCzLzFg5z4) - Step-by-step tutorial
- [Advanced RAG Techniques](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) - DeepLearning.AI course

---

## Troubleshooting Guide

### Common Issues with Ollama
- **Error**: "Failed to connect to Ollama server"
  - **Solution**: Ensure the Ollama service is running with `ollama serve`
  
- **Error**: "Out of memory"
  - **Solution**: Try a smaller model like "llama3:7b" or reduce the context window size

- **Slow Response Times**
  - **Solution**: Lower the number of retrieved documents or use a smaller model

### Debugging Vector Search Problems
- **Issue**: Irrelevant chunks being retrieved
  - **Solution**: Try a different embedding model or adjust chunk size
  
- **Issue**: Missing important information
  - **Solution**: Increase the number of chunks retrieved (k parameter)
  
- **Checking Embedding Quality**:
  ```python
  # Query your vector database directly to debug
  result = vectordb.similarity_search_with_score("your test query", k=5)
  for doc, score in result:
      print(f"Score: {score}, Content: {doc.page_content[:100]}...")
  ```

### Performance Optimization Strategies
- **Caching Embeddings**:
  ```python
  # Cache embeddings to disk
  import pickle
  
  # Save embeddings
  with open("cached_embeddings.pkl", "wb") as f:
      pickle.dump(document_embeddings, f)
      
  # Load embeddings
  with open("cached_embeddings.pkl", "rb") as f:
      document_embeddings = pickle.load(f)
  ```

- **Parallel Processing**:
  ```python
  # Process multiple PDFs in parallel
  from concurrent.futures import ProcessPoolExecutor
  
  def process_pdfs_parallel(pdf_paths, max_workers=4):
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
          results = list(executor.map(process_pdf, pdf_paths))
      return results
  ```