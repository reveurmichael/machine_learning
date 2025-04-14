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
  pip install langchain langchain-community chromadb sentence-transformers streamlit faiss-cpu
  ```

Package explanation:
- `langchain` & `langchain-community`: Framework for creating LLM applications
- `chromadb`: Vector database for storing embeddings
- `sentence-transformers`: For creating text embeddings
- `streamlit`: For building the web interface
- `faiss-cpu`: Efficient similarity search library

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

Now, let's download the models we'll use:

```bash
# Download the main model (about 4GB)
ollama pull llama3.1:8b

# Optional: Download an alternative model (about 4GB)
ollama pull deepseek-r1:7b
```

This will take several minutes depending on your internet connection. You'll see a progress bar as the models download.

### 1.6 Testing the Model

Let's make sure our model works:

```bash
# Test a simple query
ollama run llama3.1:8b "Hello, who are you?"
```

You should get a coherent response from the model.

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
cp week-5/student_database.md social-network-assistant/profiles/
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

### 2.2 Why Use Local LLMs?

Using local LLMs like those provided by Ollama has several advantages:

- **Privacy**: Data never leaves your machine
- **No API costs**: Run as many queries as you want for free
- **Customization**: Full control over the model and parameters
- **No internet required**: Works offline

### 2.3 RAG System Components

Our RAG system will have these key components:

1. **Document Processing**: Convert social profiles into chunks that fit within the context window
2. **Embedding**: Transform text chunks into vector representations
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
from langchain_community.embeddings import HuggingFaceEmbeddings
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
def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
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

### 3.4 Creating Embeddings

Add a function to create embeddings from text:

```python
def create_embeddings():
    """
    Create an embedding model to convert text into vector representations.
    
    Returns:
        HuggingFaceEmbeddings: The embedding model
    """
    # Initialize the embedding model - using a smaller model for efficient local processing
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # A lightweight but effective embedding model
        model_kwargs={"device": "cpu"}   # Use CPU for compatibility
    )
    print("Embedding model created")
    return embeddings
```

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

### 3.6 Processing Documents Pipeline

Now, let's create a function that combines all these steps:

```python
def process_documents(file_path, db_directory="db"):
    """Process documents from loading to vector storage."""
    # 1. Load the document
    documents = load_profiles(file_path)
    
    # 2. Chunk the document
    chunks = chunk_documents(documents)
    
    # 3. Create embeddings
    embeddings = create_embeddings()
    
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
def create_llm(model_name="llama3.1:8b"):
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
        ["llama3.1:8b", "deepseek-r1:7b"]
    )
    
    # Initialize or load the system
    if 'qa_chain' not in st.session_state:
        # Check if database exists
        if os.path.exists("social-network-assistant/db"):
            with st.spinner("Loading existing knowledge base..."):
                # Load the existing vector database
                embeddings = create_embeddings()
                vectordb = Chroma(
                    persist_directory="social-network-assistant/db",
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
                profile_path = "social-network-assistant/profiles/student_database.md"
                
                if os.path.exists(profile_path):
                    # Process documents
                    vectordb = process_documents(
                        profile_path,
                        "social-network-assistant/db"
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
│  (Final Answer)       │     │  (Ollama llama3.1:8b) │
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

In Part 2 of this tutorial, we'll explore:
1. More advanced RAG techniques
2. Building a more sophisticated Streamlit interface
3. Creating interactive visualizations with Streamlit
4. Adding a Gradio interface as an alternative
5. Enhancing the system with conversational capabilities

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
   - Using a smaller model (e.g., deepseek-r1:7b)
   - Closing other applications to free up memory

4. **Slow responses**: This is normal for local LLMs. For faster responses:
   - Use a more powerful computer if available
   - Be patient - the first response is usually slower as the model loads

### FAQs

1. **Can I use different profile data?**
   Yes! Just place your text files in the profiles directory and update the profile_path in the code.

2. **How can I improve answer quality?**
   - Adjust the temperature (lower for more factual, higher for more creative)
   - Modify the prompt template to be more specific
   - Retrieve more context chunks (increase k value)

3. **Can I deploy this online?**
   This is designed for local use due to the Ollama dependency. For deployment, you'd need to adapt it to use API-based models.

4. **How do I add more profiles?**
   Add more text files to the profiles directory, then re-run the document processing steps.

5. **Is this system storing my queries?**
   No, all processing happens locally, and queries are not stored unless you explicitly add code to do so.

---

## Conclusion

Congratulations! You've built a basic RAG system that can answer questions about social profiles using a local LLM. You've learned:

1. How to set up Ollama for local LLM inference
2. How to process documents and create vector embeddings
3. How to implement a basic RAG system
4. How to build a simple Streamlit interface

In Part 2, we'll explore more advanced features and build a more sophisticated application with enhanced functionality and better user interfaces using both Streamlit and Gradio.


## Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
