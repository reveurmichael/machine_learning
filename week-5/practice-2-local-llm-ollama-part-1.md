# Comprehensive Tutorial: Local LLM Social Network Assistant with RAG (Part 1)

## Overview
This comprehensive tutorial guides students through building a sophisticated social network assistant using local LLMs with Ollama, ChromaDB, and LangChain. This first 120-minute session focuses on the fundamentals of Retrieval-Augmented Generation (RAG) with local models, setting up the environment, building a pipeline for social data processing, and creating a basic Q&A system to help you connect with elite social circles.

In this tutorial, you'll learn how to create a system that can:
1. Process social profiles and background information
2. Store this information in a vector database
3. Retrieve relevant details based on user queries
4. Generate accurate insights for social networking using a local language model

## Learning Objectives
- Understand the architecture and limitations of large language models
- Master the concept of Retrieval-Augmented Generation (RAG) and why it's essential for accurate information retrieval
- Implement vector embeddings for semantic search of social data
- Build a functional social networking assistant that can answer questions about elite individuals
- Develop skills in prompt engineering for social intelligence applications

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
  - `pypdf`: Text parsing library
  - `sentence-transformers`: For creating text embeddings
  - `streamlit`: For building the web interface
  - `faiss-cpu`: Efficient similarity search library

### Model Requirements
- Download the required models using the following commands:
  ```bash
  # Download the main model we'll use (about 4GB)
  ollama pull llama3.1:8b      # 8B parameter model, good balance of quality/speed
  
  # Optional alternative model for experimentation (about 4GB)
  ollama pull deepseek-r1:7b  # Alternative model with different knowledge patterns
  ```
  
  > **Note**: The first download may take several minutes depending on your internet connection.

### Data Preparation
- Create the necessary directory structure for your project:
  ```bash
  mkdir -p social-network-assistant/{profiles,db,outputs}
  ```
  
  This creates:
  - `profiles/`: Directory to store social profiles and background information
  - `db/`: Directory for the vector database files
  - `outputs/`: Directory for saving results
  
- Prepare the student database for use:
  ```bash
  # Copy the student database file to your profiles directory
  cp week-5/student_database.md social-network-assistant/profiles/
  ```
  
  > **Important**: The student database contains detailed profiles of elite individuals who typically maintain a low profile. This makes this information especially valuable for networking purposes.

---

## Part 1: Understanding RAG and Local LLMs (20 minutes)

### The Challenge with Social Network Intelligence
- **Knowledge Cutoffs**: LLMs like GPT-4 and Llama have knowledge cutoffs and cannot access the latest social information or non-public details about elite circles
- **Hallucinations**: They can hallucinate or generate false information about influential people
- **Lack of Specific Social Context**: They may lack knowledge about private social circles and connections
- **Privacy Concerns**: Sending queries about high-profile individuals to commercial API services could compromise privacy or sensitive networking strategies

### The RAG Solution for Social Intelligence
- **Retrieval**: The system searches through your social data to find information relevant to a user's question
  - Example: When asked "Who are the tech entrepreneurs in this social group?", it finds profiles that mention tech ventures
  
- **Augmentation**: The retrieved information is added to the LLM's context window
  - Example: Adding specific background details about elite individuals to the prompt
  
- **Generation**: The LLM uses both its pre-trained knowledge and the augmented context to create a response
  - Example: Generating insights about how to connect with these individuals based on shared interests

### Why Local LLMs for Social Network Analysis?
- **Privacy**: Information about elite social connections never leaves your machine
- **Discretion**: Sensitive details about high-net-worth individuals remain private
- **Customization**: Greater control over the entire pipeline, from profile processing to strategy generation
- **Learning**: Better understand social dynamics by analyzing specific groups and networks

### RAG Architecture for Social Intelligence
1. **Profile Processing**: Convert social profiles and background information into smaller chunks that fit within the context window
2. **Embedding**: Convert text chunks into vector representations (numerical arrays) that capture social relationships and characteristics
3. **Vector Storage**: Organize these embeddings in a database that enables efficient similarity search
4. **Retrieval**: When a question is asked, find chunks most similar to the query using vector similarity
5. **Context Augmentation**: Add the retrieved text chunks to the prompt sent to the LLM
6. **LLM Generation**: The LLM produces networking strategies and social insights based on both its pre-trained knowledge and the augmented context

---

## Part 2: Social Profile Processing Pipeline (30 minutes)

### Step 1: Loading Social Profile Data
```python
from langchain_community.document_loaders import TextLoader
import os

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
    
    return documents
```

The `TextLoader` creates Document objects that contain:
- `page_content`: The extracted text from the profiles
- `metadata`: Information like source file

### Step 2: Document Chunking for Effective Social Data Retrieval
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    return chunks
```

Why chunking social profiles matters:
- LLMs have a limited context window (max tokens they can process)
- Detailed profiles need to be broken into smaller pieces
- Overlapping chunks prevents information from being cut off at chunk boundaries
- The chunk size affects retrieval precision and context completeness

### Step 3: Experimentation - Finding Optimal Profile Chunk Size
- **Exercise**: Experiment with different chunk sizes to find what works best for social data
  - Small chunks (500 chars): More precise retrieval of specific details (like interests or educational background)
  - Medium chunks (1000 chars): Balanced approach for most social networking questions
  - Large chunks (2000 chars): More context about individuals but potentially less precise retrieval
  
- **Analysis Prompts**:
  - How does changing chunk size affect the quality of social insights?
  - What are the trade-offs between large and small chunks for profile information?
  - How does chunk overlap impact understanding complete social connections?

- **Data Structure**: Examine the metadata of chunks to understand what information is preserved:
  ```python
  # Example code to inspect chunk metadata
  first_chunk = chunks[0]
  print(f"Profile sample: {first_chunk.page_content[:100]}...")
  print(f"Metadata: {first_chunk.metadata}")
  ```

### Step 4: Embedding Social Data with Sentence Transformers
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    return embeddings
```

This embedding model transforms profile text into numerical vectors that capture semantic meaning:
- Each profile chunk is converted to a vector of 384 dimensions
- Similar concepts have similar vector representations
- Social connections, interests, and characteristics become mathematically comparable
- These embeddings enable semantic search based on meaning, not just keywords

### Step 5: Creating a Vector Database for Social Profiles
```python
from langchain_community.vectorstores import Chroma

def create_vectordb(embeddings, chunks, persist_directory):
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
    vectordb = Chroma.from_documents(
        documents=chunks,            # The document chunks to embed
        embedding=embeddings,        # The embedding model
        persist_directory=persist_directory  # Where to save the database
    )
    
    # Persist the database to disk for reuse
    vectordb.persist()
    
    return vectordb
```

The vector database:
- Stores all profile chunks as vectors
- Enables efficient similarity search
- Persists to disk so you don't need to re-embed profiles each time
- Supports complex queries about social connections and attributes

---

## Part 3: Building a Social Q&A System (30 minutes)

### Step 1: Setting Up a Local LLM with Ollama
```python
from langchain_community.llms import Ollama

def create_llm(model_name="llama3.1:8b"):
    """
    Create a connection to the local Ollama LLM.
    
    Args:
        model_name (str): Name of the model to use
        
    Returns:
        Ollama: The configured LLM
    """
    # Initialize the LLM with our preferred settings
    llm = Ollama(
        model=model_name,              # Model we downloaded earlier
        temperature=0.1,               # Low temperature for more factual responses
        num_ctx=4096,                  # Context window size (varies by model)
        num_predict=1024,              # Maximum tokens to generate
        repeat_penalty=1.1             # Discourage repetition
    )
    return llm
```

Understanding the parameters:
- `model_name`: The Ollama model we're connecting to
- `temperature`: Controls randomness (lower = more factual/deterministic)
- `num_ctx`: Maximum context window size in tokens
- `num_predict`: Maximum tokens the model will generate in response
- `repeat_penalty`: Discourages the model from repeating itself

### Step 2: Creating a Social Intelligence Prompt Template
```python
from langchain.prompts import PromptTemplate

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

Analyzing the prompt components:
- **System role**: Defines the assistant as a social networking expert
- **Instructions**: Sets guidelines for using context and maintaining discretion
- **Ethical guidance**: Emphasizes genuine relationships over exploitation
- **Context**: Placeholder for retrieved profile information
- **Question**: The user's query about social connections

### Step 3: Building a Retrievable Q&A Chain
```python
from langchain.chains import RetrievalQA

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
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                                  # The language model
        chain_type="stuff",                       # "Stuff" all context into one prompt
        retriever=vectordb.as_retriever(          # Convert vector database to retriever
            search_kwargs={"k": 5}                # Retrieve top 5 most relevant chunks
        ),
        chain_type_kwargs={"prompt": prompt}      # Use our custom prompt
    )
    
    return qa_chain
```

Understanding the chain components:
- `llm`: The language model generating answers
- `chain_type="stuff"`: Combines all retrieved chunks into a single prompt
- `retriever`: Finds the most relevant profile chunks for each question
- `search_kwargs={"k": 5}`: Retrieves the 5 most relevant chunks
- `prompt`: Our custom social intelligence prompt template

### Step 4: Building a Simple Streamlit Interface
```python
import streamlit as st

def create_streamlit_app(qa_chain):
    """
    Create a simple Streamlit interface for the social intelligence assistant.
    
    Args:
        qa_chain: The question-answering chain
    """
    st.title("Elite Social Network Assistant")
    st.write("Ask questions about individuals in exclusive social circles to help you connect.")
    
    # Example questions
    st.sidebar.header("Example Questions")
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
    
    # Input field for user questions
    if "question" not in st.session_state:
        st.session_state.question = ""
    question = st.text_input("Your question:", value=st.session_state.question)
    
    # Generate answer when question is submitted
    if question:
        with st.spinner("Analyzing social data..."):
            # Run the question through the QA chain
            response = qa_chain.invoke({"query": question})
            
            # Display the answer
            st.header("Social Intelligence Insights")
            st.write(response["result"])
            
            # Show sources (optional)
            if st.checkbox("Show profile sources"):
                st.header("Source Profiles")
                source_docs = response.get("source_documents", [])
                for i, doc in enumerate(source_docs):
                    st.subheader(f"Profile {i+1}")
                    st.write(doc.page_content)
                    st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")

# In a complete application, you would call this function with the qa_chain
```

The Streamlit interface provides:
- A simple input field for user questions
- Example questions to help users get started
- Displays of social intelligence insights based on the LLM's responses
- Optional view of source profiles for transparency
- A clean, user-friendly design for interacting with the system

---

## Part 4: Putting It All Together (25 minutes)

### Complete Application Code
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

# Configuration
PROFILES_DIR = "social-network-assistant/profiles"
DB_DIR = "social-network-assistant/db"
PROFILE_FILE = "student_database.md"

def main():
    # 1. Load social profiles
    profile_path = os.path.join(PROFILES_DIR, PROFILE_FILE)
    documents = load_profiles(profile_path)
    
    # 2. Chunk profiles into smaller pieces
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    print(f"Created {len(chunks)} profile chunks")
    
    # 3. Create embeddings model
    embeddings = create_embeddings()
    
    # 4. Create or load vector database
    vectordb = create_vectordb(embeddings, chunks, DB_DIR)
    print(f"Vector database created with {vectordb._collection.count()} profile chunks")
    
    # 5. Create LLM
    llm = create_llm()
    
    # 6. Create prompt template
    prompt = create_qa_prompt()
    
    # 7. Create QA chain
    qa_chain = create_qa_chain(llm, vectordb, prompt)
    
    # 8. Launch Streamlit app
    create_streamlit_app(qa_chain)

if __name__ == "__main__":
    main()
```

### Running the Application
To run the complete social network assistant:
1. Make sure all required packages are installed
2. Ensure the Ollama service is running
3. Place your social profile data in the profiles directory
4. Run the application with:
   ```bash
   streamlit run app.py
   ```

### Example Interactions
Let's explore some meaningful interactions with the social network assistant:

**Question**: "Who are the technology entrepreneurs in this network?"
```
Based on the provided context, there are several technology entrepreneurs in this network:

1. 赵俊凯 (Zhao Junkai) - A 27-year-old tech entrepreneur with a personal fortune of $2.3 billion. He founded Quantum Nexus Technologies (valued at over $4.5 billion) after developing a revolutionary quantum encryption algorithm at age 19.

2. 段轩 (Duan Xuan) - A 32-year-old entrepreneur who has established himself in artificial intelligence with a net worth exceeding $1.7 billion. His work focuses on emotional intelligence algorithms for advanced AI systems.

3. 雷宇杨 (Lei Yuyang) - A 31-year-old entrepreneur at the forefront of robotics and biomechanical engineering. As founder and CEO of Nexus Bionics, he has transformed the company from a startup to a global enterprise valued at $11.2 billion.

4. 陈佚 (Chen Yi) - A 30-year-old entrepreneur who has established herself as an authority on digital infrastructure and smart city design. She founded Synaptic Urban Systems, overseeing projects worth over $85 billion.

These individuals operate in different technology sectors ranging from quantum computing and encryption to artificial intelligence, robotics, and smart infrastructure.
```

**Question**: "How could I approach 赵俊凯 based on his interests?"
```
Based on the profile information, here are some thoughtful approaches for connecting with 赵俊凯 (Zhao Junkai) based on his genuine interests:

1. Classical Music Appreciation: As an accomplished concert pianist who has performed with the Shanghai Symphony Orchestra, engaging with him about classical music, particularly piano compositions, could create a meaningful connection.

2. Quantum Computing Discussions: Given his expertise in quantum encryption and his company Quantum Nexus Technologies, discussing developments in quantum computing could be highly relevant. He funds quantum computing research at thirteen universities worldwide, showing his commitment to this field.

3. Literature and Rare Manuscripts: With a personal library of over 15,000 rare manuscripts and first editions (focusing on ancient Chinese scientific texts and early Western works on mathematics and physics), approaching him through discussions about rare books or mathematical/scientific manuscripts might resonate.

4. The Game of Go: He engages in high-stakes games of Go with fellow student 段轩, sometimes lasting for days. If you have knowledge or interest in Go, this could be a point of connection.

5. Educational Initiatives: He dedicates substantial resources to education initiatives, so discussing innovations in education, particularly related to quantum computing, might align with his philanthropic interests.

For the most authentic approach, focus on areas where you have genuine knowledge or interest that overlaps with his passions, rather than attempting to manufacture a connection. Given his relatively low public profile, a thoughtful introduction through mutual connections in the technology or classical music spheres might be most appropriate.
```

**Question**: "What connections exist between individuals in finance and technology sectors?"
```
Based on the provided profiles, there are several significant connections between individuals in the finance and technology sectors:

1. 乔正蓬 (Qiao Zhengpeng), a finance expert, and 王嘉浩 (Wang Jiahao), an architect with technology integration:
   - Qiao serves as a silent investor in several of Wang's most ambitious architectural projects.

2. 乔正蓬 (Qiao Zhengpeng) and 刘璐 (Liu Lu), a cybersecurity expert:
   - They engage in regular chess matches with complex side wagers on market movements that have occasionally resulted in nine-figure wealth transfers.
   - Liu's financial empire relies heavily on Liu's security protocols.
   - Their relationship is characterized by intense chess matches described as "watching two grandmasters playing a game no one else can see."

3. Boniface Pierre (finance) and 赵俊凯 (Zhao Junkai, technology):
   - Pierre's quantum computing cluster was developed in collaboration with Zhao, powering his trading operations.

4. Boniface Pierre (finance) and Douik Ahmed (quantum communications):
   - They have a productive collaboration combining Pierre's quantum financial models with Ahmed's network technology to create trading systems with near-zero latency.

These connections demonstrate how individuals in finance often collaborate with or invest in technology experts, while technology experts provide security, infrastructure, or innovative solutions that benefit financial operations. The relationships typically involve both personal interactions (games, competitions) and professional collaborations.
```

### Next Steps for Improvement
To enhance your social network assistant:

1. **Add more social data**: Import additional profiles and information about elite individuals
2. **Improve profile parsing**: Add custom parsers for different profile formats (LinkedIn, resumes, etc.)
3. **Add relationship extraction**: Automatically detect and map connections between individuals
4. **Implement advanced RAG techniques**: Add hybrid search, reranking, and self-query capabilities
5. **Create a more sophisticated UI**: Add visualizations of social networks and relationship graphs
6. **Implement conversational memory**: Allow multi-turn conversations about social connections
7. **Add privacy controls**: Implement additional safeguards for sensitive information

In Part 2 of this tutorial, we'll explore these advanced features and take your social network assistant to the next level!

---

## Troubleshooting and FAQs

### Common Issues
1. **"ModuleNotFoundError"**: Make sure you've installed all required packages
   ```bash
   pip install langchain langchain-community chromadb pypdf sentence-transformers streamlit faiss-cpu
   ```

2. **Ollama connection errors**: Ensure the Ollama service is running
   ```bash
   # Check if Ollama is running
   ps aux | grep ollama
   
   # Restart Ollama if needed
   ollama serve
   ```

3. **Memory issues with large profiles**: Try reducing chunk size or using a more memory-efficient embedding model

4. **Slow responses**: Consider using a smaller LLM or optimize retrieval parameters for speed

### Frequently Asked Questions
1. **Can I use GPT-4 instead of a local model?**
   Yes, but you'll lose the privacy benefits. Modify the `create_llm()` function to use OpenAI instead.

2. **How do I add my own social data?**
   Create text files with profiles and place them in the profiles directory. Modify the loader to match your data format.

3. **Can this system work with other languages?**
   Yes, both the embedding model and Llama 3.1 support multiple languages. For best results, use profile data in the same language as your queries.

4. **How do I evaluate the quality of my social assistant?**
   Create a set of test questions with known answers from your profiles and measure the system's accuracy and relevance.

5. **Is this ethical to use?**
   This tool should be used responsibly for legitimate networking and understanding social dynamics, not for manipulation or privacy invasion. Focus on creating genuine connections based on shared interests and values.