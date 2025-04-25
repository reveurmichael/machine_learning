# Building a Local LLM Social Network Assistant with RAG

## Overview
This practical tutorial will guide you through building a social network assistant using local LLMs with Ollama, ChromaDB, and LangChain. We'll take a step-by-step approach to set up all the necessary components and build a basic RAG (Retrieval-Augmented Generation) system that can answer questions about social profiles.

In this session, you'll learn how to:
1. Set up Ollama for running LLMs locally on your machine
2. Process social profile information with LangChain
3. Create a vector database of profile information with ChromaDB
4. Build a basic RAG system for answering questions
5. Create a simple interface with Streamlit

## What is RAG?

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by dynamically fetching relevant information from external knowledge sources and injecting it into the prompt, rather than relying solely on the model's static training data. This approach overcomes strict context window limits, reduces computational cost, ensures up-to-date and accurate responses, mitigates hallucinations, and allows secure, domain-specific customization without fine-tuning the base model. By combining retrieval and generation, RAG delivers more efficient, reliable, and scalable LLM applications across diverse use cases.

## Limitations of Feeding All Text Directly

Directly supplying every piece of text information to an LLM leads to several challenges:

- **Context Window Constraints**  
  Modern LLMs have finite context windows (e.g., 4K–32K tokens). Attempting to "pump in" all documents often exceeds these limits, forcing truncation or summarization that may omit critical details  ([RAG vs Large Context Window LLMs: When to use which one?](https://www.thecloudgirl.dev/blog/rag-vs-large-context-window)).

- **Increased Latency and Cost**  
  Passing excessively long inputs to an API increases both network latency and compute usage. This drives up per-inference costs and degrades user experience, especially when only a small fraction of the text is relevant  ([Long Context Models Explained: Do We Still Need RAG?](https://www.louisbouchard.ai/long-context-vs-rag/)).

- **Stale or Incomplete Knowledge**  
  An LLM's training cutoff means it lacks knowledge of events or documents added afterward. Without retrieval, the model cannot access evolving or private data, leading to outdated or incorrect outputs  ([Retrieval augmented generation: Keeping LLMs relevant and current](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)).

- **Higher Hallucination Risk**  
  When an LLM must infer details from its internal parameters alone, it may fabricate plausible-sounding but false information. Feeding more text directly does not eliminate this tendency unless the entire relevant corpus fits within the context window  ([Reduce AI Hallucinations With This Neat Software Trick](https://www.wired.com/story/reduce-ai-hallucinations-with-rag)).


## Benefits of RAG

### Efficient Context Handling

By narrowing the input to only the most relevant snippets, RAG avoids overwhelming the LLM with unnecessary data, making it more efficient than both naïvely extending context windows and brute-forcing full document ingestion  ([RAG vs Large Context Window LLMs: When to use which one?](https://www.thecloudgirl.dev/blog/rag-vs-large-context-window))  ([Long Context Models Explained: Do We Still Need RAG?](https://www.louisbouchard.ai/long-context-vs-rag/)).

### Freshness and Accuracy

Because RAG pulls directly from an external knowledge base, it can incorporate newly published or dynamically changing content, ensuring that the responses reflect the latest information rather than static training snapshots  ([Retrieval augmented generation: Keeping LLMs relevant and current](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)).

### Reduced Hallucinations

Grounding generation on actual retrieved passages constrains the model's creativity to documented facts, significantly lowering the rate of fabricated statements and improving trustworthiness  ([Reduce AI Hallucinations With This Neat Software Trick](https://www.wired.com/story/reduce-ai-hallucinations-with-rag)).

### Privacy and Security

Sensitive or proprietary data remains in the organization's secured storage and is only accessed at query time. Unlike fine-tuning, which embeds data permanently into model weights, RAG keeps private information under strict access control  ([RAG Vs Fine Tuning: How To Choose The Right Method](https://www.montecarlodata.com/blog-rag-vs-fine-tuning/))  ([Augment LLMs with RAGs or Fine-Tuning - Learn Microsoft](https://learn.microsoft.com/en-us/azure/developer/ai/augment-llm-rag-fine-tuning)).

### Domain-Specific Customization

RAG allows instant adaptation to new domains by indexing specialized corpora, without the need to retrain or fine-tune the LLM. This "chat over my data" paradigm offers highly tailored performance for specific tasks or industries  ([Retrieval Augmented Generation (RAG) for LLMs](https://www.promptingguide.ai/research/rag))  ([Ask HN: Is RAG the Future of LLMs? - Hacker News](https://news.ycombinator.com/item?id=40034972)).


## System Architecture

Here's the overall architecture of what we'll build:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Social Profile │────▶│  Document       │────▶│  Vector         │
│  Data           │     │  Processing     │     │  Database       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Streamlit      │◀───▶│  Query          │◀───▶│  Local LLM      │
│  Interface      │     │  Engine         │     │  (Ollama)       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites
- A computer with at least 8GB RAM (16GB recommended)

Let's get started!

## Part 1: Environment Setup 

### 1.1 Creating a Python Environment

First, let's create a project directory/folder:

```bash
# Create a project directory
mkdir -p social-network-assistant
cd social-network-assistant
```

### 1.2 Installing Required Packages

Now, let's install all the packages we'll need:

```bash
  pip install langchain langchain-community chromadb streamlit
```

or, if you are in China:

```bash
  pip install langchain langchain-community chromadb streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Package explanation:
- `langchain` & `langchain-community`: Framework for creating LLM applications
- `chromadb`: Vector database for storing embeddings
- `streamlit`: For building the web interface

### 1.3 Installing Ollama

Ollama is a tool that helps you run large language models locally. Let's install it:

#### For macOS:
1. Download the installer from [https://ollama.com/download/mac](https://ollama.com/download/mac)
2. Open the downloaded file and follow the installation instructions
3. Verify installation by opening Terminal and running:
   ```bash
   ollama --version
   ```

#### For Windows:
1. Download the installer from [https://ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the installer and follow the instructions
3. Verify installation by opening Command Prompt and running:
   ```bash
   ollama --version
   ```

#### For Linux:
Run the following command:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 1.4 Verifying Ollama Service

Let's make sure the Ollama service is running:

```bash
# Check if Ollama is running
ollama serve
```

You should see output indicating the service is running. You can also check by opening your browser and navigating to:
```
http://localhost:11434
```

You should see a simple page indicating Ollama is running.

### 1.5 Downloading LLM Models

Now, let's download the models we'll use. Modern models offer better performance and efficiency than older ones:

```bash
# Download a primary model (3-8GB depending on choice)
ollama pull llama3.1:8b      # Latest Llama model, good all-rounder

# DeepSeek models - excellent performance across various sizes
ollama pull deepseek-r1:7b      # Powerful 7B model with strong reasoning
ollama pull deepseek-coder:6.7b  # Specialized for coding tasks
ollama pull deepseek-lite:1.3b   # Extremely efficient small model

# Other recommended models
ollama pull mistral:7b       # Powerful and efficient 7B model
ollama pull phi3:3.8b        # Excellent smaller model (about 3GB)
ollama pull gemma:2b         # Very efficient 2B model

# For very resource-constrained systems
ollama pull phi2:2.7b        # Smaller but capable 2.7B model
ollama pull tinyllama:1.1b   # Extremely small 1.1B model
```

This will take several minutes depending on your internet connection and which model you choose. You'll see a progress bar as the models download.


### 1.6 Testing the Model

Let's make sure our model works:

```bash
# Test a simple query
ollama run deepseek-r1:7b "Hello, who are you?"
```

You should get a coherent response from the model. If you chose a different model, replace `deepseek-r1:7b` with your model's name.

### 1.7 Setting Up Project Structure

Let's create the directory structure for our project:

```bash
# Create project directories
mkdir -p social-network-assistant/profiles
```

This creates:
- `profiles/`: Directory to store social profiles

### 1.8 Preparing Sample Data

Let's copy the sample social network data:

```bash
# Copy the student database file to your profiles directory
cp student_database.md social-network-assistant/profiles/
```

---

## Part 2: Understanding RAG and Local LLMs  

### 2.1 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs by retrieving relevant information and adding it to the context. Here's how it works:

1. **Retrieval**: The system searches through your data to find information relevant to a question
2. **Augmentation**: The retrieved information is added to the prompt sent to the LLM
3. **Generation**: The LLM generates a response using both its pre-trained knowledge and the augmented context

This diagram illustrates the RAG process:

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │     │           │
│  User     │────▶│  Query    │────▶│  Vector   │────▶│  Document │
│  Question │     │  Analysis │     │  Search   │     │  Retrieval│
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └─────┬─────┘
                                                            │
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌─────▼─────┐
│           │     │           │     │           │     │           │
│  Response │◀────│  LLM      │◀────│  Prompt   │◀────│  Context  │
│  to User  │     │ Generation│     │  Creation │     │  Assembly │
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
```

RAG overcomes two key limitations of plain LLMs:
1. **Knowledge Cutoff**: LLMs only know information they were trained on (often outdated)
2. **Hallucination**: LLMs sometimes generate plausible-sounding but incorrect information

By retrieving real data from your profiles database, RAG ensures responses are factually accurate and up-to-date.

### 2.2 Why Use Local LLMs?

Using local LLMs like those provided by Ollama has several advantages:

- **Privacy**: Data never leaves your machine, critical for confidential profile information
- **No API costs**: Run as many queries as you want for free
- **Customization**: Full control over the model and parameters
- **No internet required**: Works offline
- **Lower latency**: Responses are generated without network delay

The main trade-off is that local models typically have lower capabilities than the largest cloud-based models, but recent advancements have significantly closed this gap.

### 2.3 RAG System Components

Our RAG system will have these key components:

1. **Document Processing**: Convert social profiles into chunks that fit within the context window
2. **Embedding**: Transform text chunks into vector representations (numerical encodings capturing semantic meaning)
3. **Vector Storage**: Organize these embeddings for efficient similarity search
4. **Retrieval**: Find chunks most similar to a query using vector similarity
5. **Context Augmentation**: Add retrieved text to the prompt
6. **LLM Generation**: Produce answers based on the augmented context

---

## Part 3: Document Processing Pipeline 

Let's build the document processing pipeline step by step:

### 3.1 Creating a Python Script

First, create a new file called `app.py` in your project directory:

```bash
touch social-network-assistant/app.py
```

Open this file in your favorite editor and add the following imports:

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import shutil
import time
import traceback

# Try to import chardet, which helps with file encoding detection
# Especially useful for Windows users facing encoding issues
try:
    import chardet
except ImportError:
    print(
        """
    =========================================================
    NOTE: For better file handling on Windows, install chardet:
          pip install chardet
    =========================================================
    """
    )
```

### 3.2 Loading Social Profile Data

Add this function to load profile data from text files:

```python
def load_profiles(file_path):
    """
    Load social profiles from text files and extract content.

    Args:
        file_path (str): Path to the profile file

    Returns:
        list: List of Document objects, each containing text with metadata
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            detailed_error = f"File not found: {file_path} (cwd: {os.getcwd()})"
            print(detailed_error)
            raise FileNotFoundError(detailed_error)

        # Print detailed file info
        file_size = os.path.getsize(file_path)
        print(f"Found file: {file_path} (Size: {file_size} bytes)")

        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            detailed_error = f"File is not readable: {file_path}"
            print(detailed_error)
            raise PermissionError(detailed_error)

        # Use explicit encoding and autodetect_encoding
        # This helps with Windows/Mac line ending and encoding differences
        loader = TextLoader(
            file_path,
            encoding="utf-8",  # Explicit encoding
            autodetect_encoding=True,  # Fallback to autodetection if needed
        )

        # Each document contains the text with metadata
        documents = loader.load()

        # Add source file to metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)

        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")

        # Try alternative approach for Windows if standard loader fails
        try:
            print("Attempting to load file using manual approach...")
            # Read the file manually with explicit encoding
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Create a Document manually
            from langchain.schema import Document

            documents = [
                Document(
                    page_content=text, metadata={"source": os.path.basename(file_path)}
                )
            ]
            print(
                f"Manually loaded {len(documents)} documents with length {len(text)} characters"
            )
            return documents
        except Exception as e2:
            print(f"Alternative loading also failed: {str(e2)}")
            raise Exception(
                f"Error loading {file_path}: Original error: {str(e)}; Fallback error: {str(e2)}"
            )
```

### 3.3 Chunking Documents

Add a function to split documents into smaller chunks:

```python
def chunk_documents(documents, chunk_size=1000, chunk_overlap=300):
    """
    Split profile data into manageable chunks with overlap to maintain context.

    Args:
        documents (list): List of Document objects from the loader
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        list: List of smaller Document chunks
    """
    # RecursiveCharacterTextSplitter tries to split on paragraph, then sentence, then word
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Maximum chunk size in characters
        chunk_overlap=chunk_overlap,  # Overlap ensures context isn't lost between chunks
        length_function=len,  # Function to measure text length
        add_start_index=True,  # Adds character index in metadata for reference
    )
    # Split the documents into smaller chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
```

The `chunk_overlap` parameter is important - it ensures that context isn't lost between chunks. By including some overlap, we maintain coherence when information spans chunk boundaries. Using a larger overlap of 300 characters (instead of the standard 200) helps ensure better context preservation, especially for complex profile information.

### 3.4 Creating Embeddings

Add a function to create embeddings from text:

```python
def create_embeddings(model_name="llama3:8b"):
    """
    Create an embedding model to convert text into vector representations.
    Uses Ollama's built-in embedding capability to avoid TensorFlow dependencies.

    Returns:
        OllamaEmbeddings: The embedding model
    """
    # Initialize the Ollama embedding model - completely avoids TensorFlow
    embeddings = OllamaEmbeddings(
        model=model_name,  # Using the same model for embeddings and generation
        show_progress=True,
    )
    print(f"Ollama embedding model created using {model_name}")
    return embeddings
```

Embeddings transform text into numerical vectors where semantically similar texts are close to each other in the vector space. This allows us to find related content through vector similarity. We're using Ollama's built-in embedding capabilities which eliminates dependencies on TensorFlow or other external embedding libraries, making it much more compatible with macOS and other platforms.

### 3.5 Creating a Vector Database

Add a function to create and store vector embeddings:

```python
def create_vectorstore(chunks, embeddings, persist_directory):
    """
    Create a vector database from embedded social profile chunks.

    Args:
        embeddings: The embedding model
        chunks: List of document chunks to embed
        persist_directory: Directory to save the database

    Returns:
        Chroma: The vector database
    """
    # Create a new vector database
    print(f"Creating vector database with {len(chunks)} chunks...")
    vectordb = Chroma.from_documents(
        documents=chunks,  # The document chunks to embed
        embedding=embeddings,  # The embedding model
        persist_directory=persist_directory,  # Where to save the database
    )

    # Persist the database to disk for reuse
    vectordb.persist()

    print(f"Vector database created and saved to {persist_directory}")
    return vectordb
```

ChromaDB is an efficient vector database that allows for quick similarity searches. By persisting it to disk, we can reuse it without reprocessing the documents each time.

### 3.6 Processing Documents Pipeline

Now, let's create a function that combines all these steps:

```python
def process_documents(file_path, db_directory, model_name="llama3:8b"):
    """Process documents from loading to vector storage."""
    try:
        # 1. Check if file exists before trying to load
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Profile file not found: {file_path}. Make sure the file exists at this path."
            )

        # Use normalized path with correct OS-specific separators
        file_path = os.path.normpath(file_path)
        print(f"Loading profiles from: {file_path}")

        # Check file encoding to help diagnose issues
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read(10000)  # Read first 10000 bytes
                result = chardet.detect(raw_data)
                print(
                    f"Detected file encoding: {result['encoding']} with confidence {result['confidence']}"
                )
        except ImportError:
            print("chardet not installed, skipping encoding detection")
        except Exception as e:
            print(f"Error checking file encoding: {e}")

        # 2. Load the document
        documents = load_profiles(file_path)

        # 3. Chunk the document
        chunks = chunk_documents(documents)

        # 4. Create embeddings
        embeddings = create_embeddings(model_name=model_name)

        # 5. Create and persist vector database
        vectordb = create_vectorstore(chunks, embeddings, db_directory)

        return vectordb
    except Exception as e:
        print(f"Error in process_documents: {str(e)}")
        traceback.print_exc()

        # Re-raise to let the caller handle it
        raise Exception(f"Error processing documents: {str(e)}")
```

This function sets up a database specific to the model being used, ensuring that embeddings are optimized for each model.

---

## Part 4: Building a Basic RAG System 

Now let's build the components needed for our RAG system:

### 4.1 Connecting to the Local LLM

Add a function to connect to Ollama:

```python
def create_llm(
    model_name="qwen2.5:3b",
    temperature=0.5,
    num_ctx=4096,
    num_predict=1024,
    repeat_penalty=1.1,
):
    """
    Create a connection to the local Ollama LLM.

    Args:
        model_name (str): Name of the model to use
        temperature (float): Controls randomness (0.0-1.0)
        num_ctx (int): Context window size
        num_predict (int): Maximum tokens to generate
        repeat_penalty (float): Penalty for repetition

    Returns:
        Ollama: The configured LLM
    """
    # Initialize the LLM with our preferred settings
    print(f"Connecting to Ollama with model {model_name}...")
    llm = Ollama(
        model=model_name,  # Model we downloaded earlier
        temperature=temperature,  # Controls randomness
        num_ctx=num_ctx,  # Context window size
        num_predict=num_predict,  # Maximum tokens to generate
        repeat_penalty=repeat_penalty,  # Discourage repetition
    )
    print("LLM connection established")
    return llm
```

The parameters are important to understand:
- `temperature`: Controls randomness (lower = more deterministic, higher = more creative)
- `num_ctx`: Maximum context size in tokens (affects how much text can be processed)
- `repeat_penalty`: Discourages repetitive text generation

### 4.2 Creating a Prompt Template

Add a function to create a prompt template:

```python
def create_qa_prompt():
    """
    Create a prompt template for the social intelligence Q&A system.

    Returns:
        PromptTemplate: The configured prompt template
    """
    # Define the template with placeholders for context and question
    template = """
    You are an expert social network assistant that helps users connect with elite individuals.
    Your role is to provide accurate and detailed information about people in elite social circles
    based EXCLUSIVELY on the profile information provided in the context below.
    
    Use ONLY the context below to answer the question. If the information is not in the context,
    say you don't know - DO NOT make up facts or connections that aren't supported by the context.
    Always maintain discretion and privacy when discussing elite individuals.
    
    When providing information about individuals:
    1. Always include their full name and ID (if available in the context)
    2. Be specific about their background, achievements, and interests
    3. Mention any connections they have with other individuals in the network
    4. Include quantitative details when available (wealth, age, properties, etc.)
    
    When suggesting networking approaches, focus on genuine common interests and values, not exploitation.
    Provide specific, actionable insights that would help the user establish meaningful connections.
    If appropriate, suggest potential conversation starters or shared interests.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """

    # Create the prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],  # The variables to be filled in
    )

    return prompt
```

The prompt template is crucial - it defines how the LLM will respond and what information it will use. Our template instructs the model to:
1. Only use information from the provided context
2. Admit when it doesn't know something
3. Maintain privacy and discretion
4. Focus on genuine connections rather than exploitation
5. Include specific details about individuals (names, IDs, background, connections)
6. Provide structured and comprehensive responses about profiles

### 4.3 Building a Question-Answering Chain

Add a function to create the QA chain:

```python
def create_qa_chain(llm, vectordb, prompt):
    """
    Create a question-answering chain that retrieves relevant profiles
    and generates answers based on the retrieved information.

    Args:
        llm: The language model
        vectordb: The vector database of profile chunks
        prompt: The prompt template

    Returns:
        RetrievalQA: The complete QA chain
    """
    # Create the QA chain with the retriever
    print("Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # The language model
        chain_type="stuff",  # "Stuff" all context into one prompt
        retriever=vectordb.as_retriever(  # Convert vector database to retriever
            search_kwargs={"k": 8}  # Retrieve top 8 most relevant chunks
        ),
        chain_type_kwargs={"prompt": prompt},  # Use our custom prompt
    )

    print("QA chain created")
    return qa_chain
```

The `chain_type="stuff"` parameter means we'll combine all retrieved chunks into a single prompt. This works well for most cases, but for very large documents, other chain types like "map_reduce" might be more appropriate. By retrieving 8 chunks instead of fewer, we ensure more comprehensive coverage of relevant profile information.

### 4.4 Creating a Simple Query Function

Add a function to query our system:

```python
def ask_question(qa_chain, question):
    """
    Ask a question to the RAG system.

    Args:
        qa_chain: The QA chain
        question (str): The question to ask

    Returns:
        dict: Dictionary containing the answer and source documents
    """
    print(f"Processing question: {question}")

    # Get the raw retriever to access the relevant documents
    retriever = qa_chain.retriever

    # Retrieve the relevant documents
    docs = retriever.get_relevant_documents(question)

    # Log the number of documents retrieved
    print(f"Retrieved {len(docs)} relevant document chunks")

    # Now get the answer from the chain
    result = qa_chain.invoke({"query": question})

    # Return both the answer and the documents
    return {"result": result["result"], "source_docs": docs}
```

## Handling Multiple Models with Model-Specific Databases

To improve the system's flexibility and avoid database conflicts, we've implemented model-specific vector databases. This means each LLM model (e.g., qwen2.5:3b, llama3.2, etc.) gets its own dedicated database directory.

### Key Benefits:

1. **No Permission Conflicts**: Each model's database is completely separate, avoiding file locking issues when switching between models
2. **Optimized Embeddings**: Each database contains embeddings generated by its specific model, ensuring better semantic representation
3. **Independent Rebuilding**: You can rebuild one model's database without affecting others
4. **No Automatic Database Creation**: The system won't automatically create databases - you need to explicitly click "Rebuild Knowledge Base"

Here's how the code creates model-specific databases:

```python
# Get model-specific database directory
db_directory = f"./chroma_db_{model_choice.replace(':', '_')}"

# Option to rebuild the database
if st.sidebar.button("🔄 Rebuild Knowledge Base"):
    with st.spinner(f"Rebuilding knowledge base for model {model_choice}..."):
        # Process documents to create model-specific database
        vectordb = process_documents(
            profile_path, 
            db_directory,
            model_name=model_choice  # Use the specific model for embeddings
        )
        
        # Store which model is currently loaded
        st.session_state.current_model = model_choice
```

When selecting a model from the dropdown, the system checks if a database exists for that model:

```python
# Check if we need to load a different model's database
if "current_model" not in st.session_state or st.session_state.current_model != model_choice:
    if "qa_chain" in st.session_state:
        del st.session_state["qa_chain"]  # Release previous model's QA chain

# Initialize or load the system
if "qa_chain" not in st.session_state:
    # Check if database exists for the current model
    if os.path.exists(db_directory):
        # Load existing database for this model
        ...
    else:
        # Prompt user to create database instead of doing it automatically
        st.sidebar.warning(f"No knowledge base found for model {model_choice}. Please click 'Rebuild Knowledge Base' to create one.")
```

This approach ensures a more robust experience when working with multiple models, particularly in environments where different team members might be using different models for experimentation.

## Part 5: Creating a Simple Streamlit Interface

Let's create a simple web interface using Streamlit:

### 5.1 Building the Streamlit App

Add the Streamlit interface code:

```python
def build_streamlit_app():
    """Create a Streamlit web interface for the social network assistant."""
    st.set_page_config(
        page_title="Social Network Assistant", page_icon="👥", layout="wide"
    )

    st.title("Elite Social Network Assistant")
    st.write(
        "Ask questions about individuals in exclusive social circles to help you connect."
    )

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # Get available models from Ollama with better Windows support
    available_models = []
    try:
        import subprocess
        import platform
        import os

        # Check if we're on Windows and try different paths
        if platform.system() == "Windows":
            try:
                # Try with executable in the PATH
                result = subprocess.run(
                    ["ollama.exe", "list"], capture_output=True, text=True
                )
            except FileNotFoundError:
                # Try common Windows installation paths
                ollama_paths = [
                    os.path.expanduser(
                        "~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
                    ),
                    "C:\\Program Files\\Ollama\\ollama.exe",
                    "C:\\Ollama\\ollama.exe",
                ]

                for path in ollama_paths:
                    if os.path.exists(path):
                        result = subprocess.run(
                            [path, "list"], capture_output=True, text=True
                        )
                        break
                else:
                    # If no path works, raise error
                    raise FileNotFoundError("Ollama executable not found on Windows")
        else:
            # For Linux/Mac
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Skip the header line
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        available_models.append(model_name)

        # If no models found but command succeeded, show a message
        if not available_models and result.returncode == 0:
            st.sidebar.warning(
                "No models found in Ollama. Please run 'ollama pull qwen2.5:3b' or another model."
            )
            available_models = ["qwen2.5:3b"]  # Default as fallback

    except Exception as e:
        st.sidebar.error(f"Error getting models: {str(e)}")
        st.sidebar.info(
            "Make sure Ollama is installed and running. Try running 'ollama pull qwen2.5:3b' from the command line."
        )
        available_models = ["qwen2.5:3b"]  # Default as fallback

    st.sidebar.subheader("Model Selection")
    model_choice = st.sidebar.selectbox("Select LLM Model", available_models)

    # LLM Parameter Configuration
    st.sidebar.subheader("LLM Parameters")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Controls randomness: Lower is more deterministic, higher is more creative",
    )

    num_ctx = st.sidebar.slider(
        "Context Size",
        min_value=1024,
        max_value=8192,
        value=4096,
        step=1024,
        help="Maximum context window size in tokens",
    )

    num_predict = st.sidebar.slider(
        "Max Generation Length",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Maximum number of tokens to generate",
    )

    repeat_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=1.5,
        value=1.1,
        step=0.1,
        help="Penalty for repeating tokens: Higher values discourage repetition",
    )

    # Get model-specific database directory
    db_directory = f"./chroma_db_{model_choice.replace(':', '_')}"

    # Option to rebuild the database
    if st.sidebar.button("🔄 Rebuild Knowledge Base"):
        # Delete the existing database
        with st.spinner(f"Rebuilding knowledge base for model {model_choice}..."):
            # Clear session state to release any database connections
            if "qa_chain" in st.session_state:
                del st.session_state["qa_chain"]

            # Wait a moment for connections to fully close
            time.sleep(1)

            try:
                if os.path.exists(db_directory):
                    # Try to change permissions before deleting
                    for root, dirs, files in os.walk(db_directory):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o755)  # rwx r-x r-x
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o644)  # rw- r-- r--

                    # Remove the directory
                    shutil.rmtree(db_directory)

                # Process documents
                # Use os.path.join for cross-platform path handling
                profile_dir = "profiles"
                profile_file = "student_database.md"
                profile_path = os.path.join(profile_dir, profile_file)

                # Add more debugging info
                st.sidebar.info(f"Current working directory: {os.getcwd()}")
                st.sidebar.info(f"Looking for file: {profile_path}")
                st.sidebar.info(
                    f"Profiles directory exists: {os.path.exists(profile_dir)}"
                )
                st.sidebar.info(f"Profile file exists: {os.path.exists(profile_path)}")

                # Verify the file exists
                if not os.path.exists(profile_path):
                    # Try alternative path construction approach
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    alt_profile_path = os.path.join(base_dir, profile_dir, profile_file)
                    st.sidebar.info(f"Trying alternative path: {alt_profile_path}")

                    if os.path.exists(alt_profile_path):
                        st.sidebar.info(f"Found file at alternative path")
                        profile_path = alt_profile_path
                    else:
                        st.sidebar.error(f"Profile file not found at either path")
                        st.sidebar.info(
                            "Make sure the student_database.md file exists in the 'profiles' directory."
                        )
                        return

                vectordb = process_documents(
                    profile_path, db_directory, model_name=model_choice
                )

                # Create LLM and QA chain
                llm = create_llm(
                    model_name=model_choice,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    repeat_penalty=repeat_penalty,
                )
                prompt = create_qa_prompt()
                qa_chain = create_qa_chain(llm, vectordb, prompt)

                st.session_state.qa_chain = qa_chain
                st.session_state.current_model = model_choice
                st.sidebar.success(
                    f"✅ Knowledge base for {model_choice} rebuilt successfully"
                )
            except Exception as e:
                st.sidebar.error(f"Error rebuilding knowledge base: {str(e)}")
                st.sidebar.info(
                    "Try restarting the application or check if the profile file exists"
                )

    # Check if we need to load a different model's database
    load_new_model = False
    if (
        "current_model" not in st.session_state
        or st.session_state.current_model != model_choice
    ):
        if "qa_chain" in st.session_state:
            del st.session_state["qa_chain"]
        load_new_model = True

    # Initialize or load the system
    if "qa_chain" not in st.session_state:
        # Check if database exists for the current model
        if os.path.exists(db_directory):
            with st.spinner(f"Loading knowledge base for model {model_choice}..."):
                # Load the existing vector database
                embeddings = create_embeddings(model_name=model_choice)
                vectordb = Chroma(
                    persist_directory=db_directory, embedding_function=embeddings
                )

                # Create LLM and QA chain
                llm = create_llm(
                    model_name=model_choice,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    repeat_penalty=repeat_penalty,
                )
                prompt = create_qa_prompt()
                qa_chain = create_qa_chain(llm, vectordb, prompt)

                st.session_state.qa_chain = qa_chain
                st.session_state.current_model = model_choice
                st.sidebar.success(f"✅ Loaded knowledge base for {model_choice}")
        else:
            # Do not automatically create a database - inform the user
            st.sidebar.warning(
                f"No knowledge base found for model {model_choice}. Please click 'Rebuild Knowledge Base' to create one."
            )
            if "question" in st.session_state:
                del st.session_state["question"]
            return

    # Example questions in the sidebar
    st.sidebar.title("Example Questions")
    examples = [
        "Who are the tech entrepreneurs in this social group?",
        "Which individuals are interested in quantum computing?",
        "Who is passionate about LeetCode?",
        "Who would be good connections for someone interested in AI?",
        "How could I approach 赵俊凯 (Zhao Junkai) based on his interests?",
    ]

    # Create a container for the question input at the top so it stays in view
    input_container = st.container()

    # Create a container for displaying the answer below
    answer_container = st.container()

    # Use the sidebar for example questions
    for example in examples:
        if st.sidebar.button(example):
            # Just set the value to be used in the input field
            st.session_state.question_text = example

    # Initialize session state for question text if it doesn't exist
    if "question_text" not in st.session_state:
        st.session_state.question_text = ""

    # Display the input field in the input container at the top
    with input_container:
        # Use a form to require explicit submission
        with st.form(key="question_form"):
            question = st.text_input(
                "Your question:",
                value=st.session_state.question_text,
                key="question_input",
            )
            submit_button = st.form_submit_button("Send")

    # Process the question when the form is submitted
    if submit_button and question and "qa_chain" in st.session_state:
        # Store the current question for reference
        st.session_state.question_text = question

        with answer_container:
            with st.spinner("Finding relevant information..."):
                response = ask_question(st.session_state.qa_chain, question)
                answer = response["result"]
                docs = response["source_docs"]

                # Display the answer in a nice box
                st.markdown("### Answer")
                st.markdown(
                    f"""
                <div style="background-color: #f0f7fb; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db;">
                {answer}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Display the source documents if the user wants to see them
                with st.expander("Show Source Information"):
                    st.markdown("### Retrieved Profile Chunks")
                    st.write(
                        f"Retrieved {len(docs)} relevant chunks from the profiles database."
                    )

                    for i, doc in enumerate(docs):
                        st.markdown(f"#### Chunk {i+1}")
                        source = doc.metadata.get("source", "Unknown source")
                        st.markdown(f"**Source**: {source}")
                        st.text_area(f"Content {i+1}", doc.page_content, height=200)
```

### 5.2 Main Function

Finally, add the main function to run everything:

```python
def main():
    """Main function to run the application."""
    try:
        # Change to the project directory if needed
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        print(f"Working directory set to: {os.getcwd()}")

        # Ensure profiles directory exists
        profiles_dir = os.path.join(script_dir, "profiles")
        if not os.path.exists(profiles_dir):
            print(f"Creating profiles directory: {profiles_dir}")
            os.makedirs(profiles_dir, exist_ok=True)

        # Check if sample profile exists, create it if not
        sample_profile_path = os.path.join(profiles_dir, "student_database.md")
        if not os.path.exists(sample_profile_path):
            print(f"Creating sample profile at: {sample_profile_path}")
            with open(sample_profile_path, "w", encoding="utf-8") as f:
                f.write(
                    """# Student Database - Sample Profiles
                
## Student 1
- **Name**: John Smith
- **ID**: JS001
- **Major**: Computer Science
- **Interests**: Machine Learning, Web Development, Gaming
- **Projects**: Personal website, ML classifiers

## Student 2
- **Name**: Jane Doe
- **ID**: JD002
- **Major**: Data Science
- **Interests**: AI, Statistics, Basketball
- **Projects**: Sentiment analysis, Sports analytics dashboard
                """
                )
            print("Sample profile created.")

        # Run the Streamlit app
        build_streamlit_app()
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

### 5.3 Running the Streamlit App

Save your file and run the Streamlit app:

```bash
cd social-network-assistant
streamlit run app.py
```

This will open a browser window with your Streamlit app. You should be able to ask questions about the profiles in the database and get relevant answers.

---

## Part 6: Testing and Using the System

### 6.1 Example Questions to Try

Here are some questions you can ask your social network assistant:

1. "Who are the tech entrepreneurs in this group?"
2. "Which individuals have interests in quantum computing?"
3. "What hobbies do the finance professionals have?"
4. "Who would be good connections for someone interested in AI?"
5. "How could I approach 赵俊凯 (Zhao Junkai) based on his interests?"
6. "What connections exist between individuals in the finance and technology sectors?"
7. "Which individuals have the most impressive art collections?"
8. "Who speaks multiple languages, and which languages do they speak?"

### 6.2 System Diagram Review

Let's review what we've built:

```
┌───────────────────────┐
│                       │
│  Social Profile Data  │
│  (student_database.md)│
│                       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│                       │     │                       │
│  Document Processing  │────▶│  Text Chunking        │
│  (TextLoader)         │     │  (CharacterSplitter)  │
│                       │     │                       │
└───────────────────────┘     └───────────┬───────────┘
                                          │
                                          ▼
┌───────────────────────┐     ┌───────────────────────┐
│                       │     │                       │
│  Vector Database      │◀────│  Embedding Creation   │
│  (Model-Specific      │     │  (Ollama Embeddings)  │
│   ChromaDB)           │     │                       │
│                       │     │                       │
└───────────┬───────────┘     └───────────────────────┘
            │
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│                       │     │                       │
│  Query Retrieval      │────▶│  Context Assembly     │
│  (Similarity Search)  │     │  (Retrieved Chunks)   │
│                       │     │                       │
└───────────────────────┘     └───────────┬───────────┘
                                          │
                                          ▼
┌───────────────────────┐     ┌───────────────────────┐
│                       │     │                       │
│  Response Generation  │◀────│  Local LLM            │
│  (Final Answer)       │     │  (Ollama)             │
│                       │     │                       │
└───────────┬───────────┘     └───────────────────────┘
            │
            ▼
┌───────────────────────┐
│                       │
│  Streamlit User       │
│  Interface            │
│                       │
└───────────────────────┘
```

## Troubleshooting and FAQs

### Common Issues

1. **"ModuleNotFoundError"**: Make sure you've installed all required packages:
   ```bash
   pip install langchain langchain-community chromadb sentence-transformers streamlit faiss-cpu
   ```

2. **Ollama connection errors**: Ensure the Ollama service is running:
   ```bash
   # Check if Ollama is running
   ollama serve
   ```

3. **Windows-specific Ollama issues**: If you see `Error getting models: [WinError 2] The system cannot find the file specified` or `Error loading profiles/student_database.md` on Windows:
   - Make sure Ollama is installed correctly on Windows
   - Add the Ollama installation directory to your system PATH
   - Install additional dependencies: `pip install chardet`
   - Try the enhanced code below for better Windows compatibility:
   
   ```python
   # Better Windows support for finding Ollama
   import platform
   import os
   import subprocess
   
   if platform.system() == "Windows":
       try:
           # Try with executable in the PATH
           result = subprocess.run(['ollama.exe', 'list'], capture_output=True, text=True)
       except FileNotFoundError:
           # Try common Windows installation paths
           ollama_paths = [
               os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
               "C:\\Program Files\\Ollama\\ollama.exe",
               "C:\\Ollama\\ollama.exe"
           ]
           
           for path in ollama_paths:
               if os.path.exists(path):
                   result = subprocess.run([path, 'list'], capture_output=True, text=True)
                   break
           else:
               # If no path works, use default models
               available_models = ["qwen2.5:3b"]
   
   # Enhanced TextLoader for Windows compatibility
   def load_profiles(file_path):
       """Load social profiles with enhanced Windows compatibility."""
       try:
           # Use normalized path with OS-specific separators
           file_path = os.path.normpath(file_path)
           
           # TextLoader with explicit encoding settings
           loader = TextLoader(
               file_path,
               encoding="utf-8",  # Explicit encoding
               autodetect_encoding=True  # Fallback to autodetection
           )
           
           documents = loader.load()
           return documents
       except Exception as e:
           print(f"Error loading file: {str(e)}")
           # Fallback to manual reading (Windows compatibility)
           try:
               with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                   text = f.read()
               from langchain.schema import Document
               return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
           except Exception as e2:
               raise Exception(f"Failed to load file: {str(e)} / {str(e2)}")
   
   # Always use os.path.join for paths
   profile_path = os.path.join("profiles", "student_database.md")
   ```

4. **Memory issues**: If you encounter memory errors, try:
   - Reducing the chunk size (e.g., 500 instead of 1000)
   - Using a smaller model (e.g., deepseek-lite:1.3b)
   - Closing other applications to free up memory

5. **Slow responses**: This is normal for local LLMs. For faster responses:
   - Use a more powerful computer if available
   - Try a smaller model like deepseek-lite:1.3b which balances quality and speed
   - Be patient - the first response is usually slower as the model loads

6. **App struggling to find information about profiles**: If the system doesn't seem to find relevant information, try these solutions:
   - Use the "Rebuild Knowledge Base" button to recreate the vector database with current settings
   - Check the "Show Source Information" expander to see which chunks are being retrieved
   - Make your questions more specific, including names, interests, or other specific details
   - Try different models as some are better at information retrieval than others
   - Adjust the prompt template to better match your specific use case
   - Increase the number of chunks retrieved (k value) for more comprehensive context

7. **Database permissions error when rebuilding knowledge base**: If you see an error like "InternalError: Query error: Database error: error returned from database: (code: 1032) attempt to write a readonly database", try these solutions:
   - Make sure no other processes are accessing the database
   - Restart the Streamlit application
   - If the error persists, manually delete the appropriate model's database directory when the app is not running:
     ```bash
     # For example, to delete qwen2.5:3b's database
     rm -rf ./chroma_db_qwen2.5_3b
     ```
   - Use the improved rebuild code that properly handles permissions:

   ```python
   if st.sidebar.button("🔄 Rebuild Knowledge Base"):
       with st.spinner(f"Rebuilding knowledge base for model {model_choice}..."):
           import shutil
           import time
           
           # Get model-specific database directory
           db_directory = f"./chroma_db_{model_choice.replace(':', '_')}"
           
           # Clear session state to release any database connections
           if "qa_chain" in st.session_state:
               del st.session_state["qa_chain"]
               
           # Wait a moment for connections to fully close
           time.sleep(1)
           
           try:
               if os.path.exists(db_directory):
                   # Try to change permissions before deleting
                   for root, dirs, files in os.walk(db_directory):
                       for dir in dirs:
                           os.chmod(os.path.join(root, dir), 0o755)
                       for file in files:
                           os.chmod(os.path.join(root, file), 0o644)
                   
                   # Remove the directory
                   shutil.rmtree(db_directory)
               
               # Process documents
               profile_path = "profiles/student_database.md"
               vectordb = process_documents(profile_path, db_directory, model_name=model_choice)
               
               # Create new QA chain
               llm = create_llm(model_name=model_choice)
               prompt = create_qa_prompt()
               qa_chain = create_qa_chain(llm, vectordb, prompt)
               
               st.session_state.qa_chain = qa_chain
               st.session_state.current_model = model_choice
               st.sidebar.success(f"✅ Knowledge base for {model_choice} rebuilt successfully")
           except Exception as e:
               st.sidebar.error(f"Error rebuilding knowledge base: {str(e)}")
               st.sidebar.info("Try restarting the application or check file permissions")
   ```

### FAQs

1. **Can I use different profile data?**
   Yes! Just place your text files in the profiles directory and update the profile_path in the code.

2. **How can I improve answer quality?**
   - Use the sidebar sliders to adjust LLM parameters:
     - Lower temperature (0.1-0.3) for more factual responses
     - Higher temperature (0.6-0.9) for more creative answers
     - Increase context size for more comprehensive information processing
     - Adjust repetition penalty if responses seem too repetitive
   - Modify the prompt template to be more specific
   - Retrieve more context chunks (increase k value)
   - Use a larger model if your hardware supports it
   - Increase chunk overlap to preserve more context between chunks

4. **How do I add more profiles?**
   Add more text files to the profiles directory, then re-run the document processing steps (or use the Rebuild Knowledge Base button).

5. **Is this system storing my queries?**
   No, all processing happens locally, and queries are not stored unless you explicitly add code to do so.

6. **Which model should I use for my computer?**
   - High-end systems (16GB+ RAM): deepseek-r1:7b, llama3.1:8b, or mistral:7b
   - Mid-range systems (8GB RAM): deepseek-coder:6.7b or phi3:3.8b
   - Low-end systems (4GB RAM): deepseek-lite:1.3b or gemma:2b

7. **I'm getting TensorFlow errors, especially on macOS**
   We've completely removed TensorFlow dependencies by using Ollama's built-in embedding capabilities. This approach is much more compatible with macOS and eliminates the need for external embedding libraries. If you're still encountering issues, make sure you have Ollama installed and running correctly.

8. **How can I see which profile information was used to answer my question?**
   Use the "Show Source Information" expander below each answer to see exactly which chunks of text were retrieved from the database. This can help you understand why certain answers were given and troubleshoot if information seems incorrect or missing.


## Configurable LLM Parameters

The enhanced version of our social network assistant includes user-configurable LLM parameters in the sidebar, allowing you to fine-tune the model's behavior without changing code:

1. **Temperature** (0.0-1.0): Controls the randomness of the model's output
   - Lower values (0.1-0.3): More deterministic, factual responses
   - Medium values (0.4-0.6): Balanced between creativity and consistency
   - Higher values (0.7-1.0): More creative, diverse, and sometimes surprising outputs

2. **Context Size** (1024-8192): Sets the maximum number of tokens the model can process
   - Larger values allow more profile information to be processed at once
   - Adjust based on your computer's RAM capabilities

3. **Max Generation Length** (256-2048): Controls how many tokens the model can generate
   - Shorter values provide quicker, more concise answers
   - Longer values allow for more detailed explanations

4. **Repetition Penalty** (1.0-1.5): Prevents the model from repeating the same phrases
   - Higher values strongly discourage repetition
   - Use higher values if you notice the model getting stuck in loops

## Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [DeepSeek Models Documentation](https://github.com/deepseek-ai/deepseek-LLM)
