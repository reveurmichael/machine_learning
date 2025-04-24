# Building a Local LLM Social Network Assistant (Part 1)

## Overview
This practical tutorial will guide you through building a social network assistant using local LLMs with Ollama, ChromaDB, and LangChain. We'll take a step-by-step approach to set up all the necessary components and build a basic RAG (Retrieval-Augmented Generation) system that can answer questions about social profiles.

In this session, you'll learn how to:
1. Set up Ollama for running LLMs locally on your machine
2. Process social profile information with LangChain
3. Create a vector database of profile information with ChromaDB
4. Build a basic RAG system for answering questions
5. Create a simple interface with Streamlit

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

---

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
ollama pull deepseek:7b      # Powerful 7B model with strong reasoning
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

#### Model Selection Guide

| Model | Size | RAM Required | Performance | Best For |
|-------|------|--------------|------------|----------|
| llama3.1:8b | ~8GB | 16GB+ | Excellent | General purpose, complex reasoning |
| deepseek:7b | ~7GB | 16GB | Excellent | Strong reasoning, detailed responses |
| deepseek-coder:6.7b | ~7GB | 16GB | Excellent | Programming and technical content |
| mistral:7b | ~7GB | 16GB | Very Good | Balanced performance and efficiency |
| phi3:3.8b | ~4GB | 8GB+ | Good | Good performance on limited hardware |
| deepseek-lite:1.3b | ~1.5GB | 4GB+ | Good | Best small model performance |
| gemma:2b | ~2GB | 6GB+ | Fair | Basic tasks on limited hardware |
| phi2:2.7b | ~3GB | 6GB+ | Fair | Basic tasks on limited hardware |
| tinyllama:1.1b | ~1GB | 4GB+ | Basic | Very constrained environments |

Choose the model that best fits your hardware capabilities. For this tutorial, any of these models will work, but larger models generally provide better responses.

### 1.6 Testing the Model

Let's make sure our model works:

```bash
# Test a simple query
ollama run deepseek:7b "Hello, who are you?"
```

You should get a coherent response from the model. If you chose a different model, replace `deepseek:7b` with your model's name.

### 1.7 Setting Up Project Structure

Let's create the directory structure for our project:

```bash
# Create project directories
mkdir -p social-network-assistant/{profiles,db,outputs}
```

This creates:
- `profiles/`: Directory to store social profiles
- `db/`: Directory for the vector database files
- `outputs/`: Directory for saving results

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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
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
    # TextLoader extracts text from the file
    loader = TextLoader(file_path)
    
    # Each document contains the text with metadata
    documents = loader.load()
    
    # Add source file to metadata
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
    
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents
```

### 3.3 Chunking Documents

Add a function to split documents into smaller chunks:

```python
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
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
        chunk_size=chunk_size,          # Maximum chunk size in characters
        chunk_overlap=chunk_overlap,    # Overlap ensures context isn't lost between chunks
        length_function=len,            # Function to measure text length
        add_start_index=True,           # Adds character index in metadata for reference
    )
    # Split the documents into smaller chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
```

The `chunk_overlap` parameter is important - it ensures that context isn't lost between chunks. By including some overlap, we maintain coherence when information spans chunk boundaries.

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
        show_progress=True
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
        documents=chunks,            # The document chunks to embed
        embedding=embeddings,        # The embedding model
        persist_directory=persist_directory  # Where to save the database
    )
    
    # Persist the database to disk for reuse
    vectordb.persist()
    
    print(f"Vector database created and saved to {persist_directory}")
    return vectordb
```

ChromaDB is an efficient vector database that allows for quick similarity searches. By persisting it to disk, we can reuse it without reprocessing the documents each time.

Once you've implemented all these functions, your project directory structure should look like:

```
social-network-assistant/
├── app.py                 # Main application file
├── chroma_db/             # Vector database (created after running)
├── profiles/              # Directory for social profile data
│   └── student_database.md # Sample profile data
└── outputs/               # Directory for any output files
```

### 3.6 Processing Documents Pipeline

Now, let's create a function that combines all these steps:

```python
def process_documents(file_path, db_directory="./chroma_db", model_name="llama3:8b"):
    """Process documents from loading to vector storage."""
    # 1. Load the document
    documents = load_profiles(file_path)
    
    # 2. Chunk the document
    chunks = chunk_documents(documents)
    
    # 3. Create embeddings
    embeddings = create_embeddings(model_name=model_name)
    
    # 4. Create and persist vector database
    vectordb = create_vectorstore(chunks, embeddings, db_directory)
    
    return vectordb
```

---

## Part 4: Building a Basic RAG System 

Now let's build the components needed for our RAG system:

### 4.1 Connecting to the Local LLM

Add a function to connect to Ollama:

```python
def create_llm(model_name="llama3.2:1b"):
    """
    Create a connection to the local Ollama LLM.
    
    Args:
        model_name (str): Name of the model to use
        
    Returns:
        Ollama: The configured LLM
    """
    # Initialize the LLM with our preferred settings
    print(f"Connecting to Ollama with model {model_name}...")
    llm = Ollama(
        model=model_name,              # Model we downloaded earlier
        temperature=0.1,               # Low temperature for more factual responses
        num_ctx=4096,                  # Context window size
        num_predict=1024,              # Maximum tokens to generate
        repeat_penalty=1.1             # Discourage repetition
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
    Your role is to provide accurate information about high-profile people in exclusive social circles.
    
    Use ONLY the context below to answer the question. If the information is not in the context,
    say you don't know - DO NOT make up facts or connections that aren't supported by the context.
    Always maintain discretion and privacy when discussing elite individuals.
    
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
        input_variables=["context", "question"]  # The variables to be filled in
    )
    
    return prompt
```

The prompt template is crucial - it defines how the LLM will respond and what information it will use. Our template instructs the model to:
1. Only use information from the provided context
2. Admit when it doesn't know something
3. Maintain privacy and discretion
4. Focus on genuine connections rather than exploitation

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
        llm=llm,                                  # The language model
        chain_type="stuff",                       # "Stuff" all context into one prompt
        retriever=vectordb.as_retriever(          # Convert vector database to retriever
            search_kwargs={"k": 5}                # Retrieve top 5 most relevant chunks
        ),
        chain_type_kwargs={"prompt": prompt}      # Use our custom prompt
    )
    
    print("QA chain created")
    return qa_chain
```

The `chain_type="stuff"` parameter means we'll combine all retrieved chunks into a single prompt. This works well for most cases, but for very large documents, other chain types like "map_reduce" might be more appropriate.

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
        str: The answer
    """
    print(f"Processing question: {question}")
    result = qa_chain.invoke({"query": question})
    
    return result["result"]
```

---

## Part 5: Creating a Simple Streamlit Interface

Let's create a simple web interface using Streamlit:

### 5.1 Building the Streamlit App

Add the Streamlit interface code:

```python
def build_streamlit_app():
    """Create a Streamlit web interface for the social network assistant."""
    st.set_page_config(
        page_title="Social Network Assistant",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("Elite Social Network Assistant")
    st.write("Ask questions about individuals in exclusive social circles to help you connect.")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3.2:1b", "llama3.2:latest", "deepseek:7b", "deepseek-coder:6.7b", "deepseek-lite:1.3b", 
         "mistral:7b", "phi3:3.8b", "gemma:2b"]
    )
    
    # Initialize or load the system
    if 'qa_chain' not in st.session_state:
        # Check if database exists
        if os.path.exists("./chroma_db"):
            with st.spinner("Loading existing knowledge base..."):
                # Load the existing vector database
                embeddings = create_embeddings(model_name=model_choice)
                vectordb = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings
                )
                
                # Create LLM and QA chain
                llm = create_llm(model_name=model_choice)
                prompt = create_qa_prompt()
                qa_chain = create_qa_chain(llm, vectordb, prompt)
                
                st.session_state.qa_chain = qa_chain
                st.sidebar.success("✅ Loaded existing knowledge base")
        else:
            # Process documents if database doesn't exist
            with st.spinner("Processing profiles and building knowledge base..."):
                profile_path = "profiles/student_database.md"
                
                if os.path.exists(profile_path):
                    # Process documents
                    vectordb = process_documents(
                        profile_path,
                        "./chroma_db",
                        model_name=model_choice
                    )
                    
                    # Create LLM and QA chain
                    llm = create_llm(model_name=model_choice)
                    prompt = create_qa_prompt()
                    qa_chain = create_qa_chain(llm, vectordb, prompt)
                    
                    st.session_state.qa_chain = qa_chain
                    st.sidebar.success("✅ Knowledge base created successfully")
                else:
                    st.error(f"Profile file not found: {profile_path}")
                    return
    
    # Example questions in the sidebar
    st.sidebar.title("Example Questions")
    examples = [
        "Who are the tech entrepreneurs in this social group?",
        "Which individuals are interested in quantum computing?",
        "What hobbies do the finance professionals have?",
        "Who would be good connections for someone interested in AI?",
        "How could I approach 赵俊凯 (Zhao Junkai) based on his interests?"
    ]
    
    for example in examples:
        if st.sidebar.button(example):
            st.session_state.question = example
    
    # Input area for questions
    if "question" not in st.session_state:
        st.session_state.question = ""
    
    question = st.text_input(
        "Your question:",
        value=st.session_state.question,
        key="question_input"
    )
    
    # Generate answer when question is submitted
    if question:
        with st.spinner("Finding relevant information..."):
            answer = ask_question(st.session_state.qa_chain, question)
            
            # Display the answer in a nice box
            st.markdown("### Answer")
            st.markdown(f"""
            <div style="background-color: #f0f7fb; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db;">
            {answer}
            </div>
            """, unsafe_allow_html=True)
```

### 5.2 Main Function

Finally, add the main function to run everything:

```python
def main():
    """Main function to run the application."""
    # Change to the project directory if needed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the Streamlit app
    build_streamlit_app()

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
│  (ChromaDB)           │     │  (HuggingFace)        │
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

### 6.3 Next Steps

In [Part 2](../week-6/practice-2-local-llm-ollama-part-2.md), we'll explore more advanced features including:
- Building sophisticated interfaces with both Streamlit and Gradio
- Implementing conversation memory for multi-turn interactions
- Creating a complete application with more features and a better user experience

---

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

3. **Memory issues**: If you encounter memory errors, try:
   - Reducing the chunk size (e.g., 500 instead of 1000)
   - Using a smaller model (e.g., deepseek-lite:1.3b)
   - Closing other applications to free up memory

4. **Slow responses**: This is normal for local LLMs. For faster responses:
   - Use a more powerful computer if available
   - Try a smaller model like deepseek-lite:1.3b which balances quality and speed
   - Be patient - the first response is usually slower as the model loads

### FAQs

1. **Can I use different profile data?**
   Yes! Just place your text files in the profiles directory and update the profile_path in the code.

2. **How can I improve answer quality?**
   - Adjust the temperature (lower for more factual, higher for more creative)
   - Modify the prompt template to be more specific
   - Retrieve more context chunks (increase k value)
   - Use a larger model if your hardware supports it

3. **Can I deploy this online?**
   This is designed for local use due to the Ollama dependency. For deployment, you'd need to adapt it to use API-based models.

4. **How do I add more profiles?**
   Add more text files to the profiles directory, then re-run the document processing steps.

5. **Is this system storing my queries?**
   No, all processing happens locally, and queries are not stored unless you explicitly add code to do so.

6. **Which model should I use for my computer?**
   - High-end systems (16GB+ RAM): deepseek:7b, llama3.1:8b, or mistral:7b
   - Mid-range systems (8GB RAM): deepseek-coder:6.7b or phi3:3.8b
   - Low-end systems (4GB RAM): deepseek-lite:1.3b or gemma:2b

7. **I'm getting TensorFlow errors, especially on macOS**
   We've completely removed TensorFlow dependencies by using Ollama's built-in embedding capabilities. This approach is much more compatible with macOS and eliminates the need for external embedding libraries. If you're still encountering issues, make sure you have Ollama installed and running correctly.

---

## Conclusion

Congratulations! You've built a basic RAG system that can answer questions about social profiles using a local LLM. You've learned:

1. How to set up Ollama for local LLM inference
2. How to process documents and create vector embeddings
3. How to implement a basic RAG system
4. How to build a simple Streamlit interface

In [Part 2](../week-6/practice-2-local-llm-ollama-part-2.md), we'll explore more advanced features including:
- Building sophisticated interfaces with both Streamlit and Gradio
- Implementing conversation memory for multi-turn interactions
- Creating a complete application with more features and a better user experience

## Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [DeepSeek Models Documentation](https://github.com/deepseek-ai/deepseek-LLM)
