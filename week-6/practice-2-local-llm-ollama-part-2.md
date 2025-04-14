# Building a Local LLM Social Network Assistant (Part 2)

## Overview
In this tutorial, we'll build upon the foundation established in Part 1 by creating interactive interfaces for our local LLM social network assistant using Streamlit and Gradio. These interfaces will make it easier to interact with our RAG system and provide a more user-friendly experience for exploring social profiles and networking opportunities.

## Learning Objectives
- Create a comprehensive Streamlit interface for the social network assistant
- Build an alternative interface using Gradio
- Implement basic conversation memory for multi-turn interactions
- Add social profile management features
- Enhance the user experience with visualization and advanced features

## Prerequisites
- Completion of Part 1 of the tutorial
- Functional RAG system with Ollama and ChromaDB
- Basic understanding of Python web applications
- The social profile dataset from Part 1

## Part 1: Building a Streamlit Interface

Streamlit provides an easy way to build interactive web applications for machine learning and data science projects. Let's create an enhanced interface for our social network assistant, breaking down each component step by step.

### Step 1: Setting Up the Project Structure

Building on our project structure from Part 1, let's enhance it with additional directories for our interface components:

```
social-network-assistant/
├── app/
│   ├── streamlit_app.py    # Enhanced Streamlit application
│   ├── gradio_app.py       # Alternative Gradio interface
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── profile_processor.py
│   │   └── retrieval.py
├── profiles/               # Social profile data (from Part 1)
├── db/                     # Vector database storage (from Part 1)
├── outputs/                # Saved results and analytics
└── requirements.txt        # Project dependencies
```

Create the new directories using the following commands:

```bash
cd social-network-assistant
mkdir -p app/utils
touch app/utils/__init__.py
```

### Step 2: Create the Requirements File

Let's update our requirements.txt file with all necessary dependencies for our enhanced interface:

```
langchain>=0.0.267
langchain-community>=0.0.6
chromadb>=0.4.13
sentence-transformers>=2.2.2
streamlit>=1.27.0
gradio>=3.50.2
ollama>=0.1.5
pandas>=2.0.0
matplotlib>=3.7.0
networkx>=3.1
plotly>=5.14.0
```

Install these dependencies using:

```bash
pip install -r requirements.txt
```

### Step 3: Create the Utility Functions

Let's create utility functions for profile processing and retrieval. First, create `profile_processor.py`:

```python
# social-network-assistant/app/utils/profile_processor.py
import os
from typing import List, Dict, Any
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_and_process_profile(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Load and process a social profile document into chunks.
    
    Args:
        file_path: Path to the profile file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    # Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Profile document processed into {len(chunks)} chunks")
    return chunks

def add_profile_to_vectorstore(
    file_path: str, 
    db_path: str, 
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> int:
    """
    Process a social profile document and add it to the vector store.
    
    Args:
        file_path: Path to the profile file
        db_path: Path to the vector database
        embedding_model_name: Name of the embedding model
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Number of chunks added to the database
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Process the document
    chunks = load_and_process_profile(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not chunks:
        return 0
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Add to or create the vector store
    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
    
    # Persist the vector store
    db.persist()
    
    return len(chunks)
```

Now create the retrieval utility in `retrieval.py`:

```python
# social-network-assistant/app/utils/retrieval.py
from langchain.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, Optional

def get_available_models():
    """
    Get a list of available models for Ollama, including very small models.
    
    Returns:
        List of model names
    """
    return [
        # Regular models
        "llama3.1:8b", 
        "mistral:7b", 
        "llama2:7b",
        
        # Smaller models
        "deepseek-coder:1.5b",
        "phi2:2.7b",
        "phi3:3.8b",
        "tinyllama:1.1b",
        "gemma:2b",
        
        # Larger models
        "mixtral:8x7b",
        "llama2:13b"
    ]

def create_social_network_prompt():
    """
    Create a prompt template specifically for social networking assistance.
    
    Returns:
        The formatted prompt string
    """
    return """
    You are an expert social network assistant that helps users connect with elite individuals.
    Your role is to provide accurate information about high-profile people in exclusive social circles.
    
    Use ONLY the context below to answer the question. If the information is not in the context,
    say you don't know - DO NOT make up facts or connections that aren't supported by the context.
    Always maintain discretion and privacy when discussing elite individuals.
    
    When suggesting networking approaches, focus on genuine common interests and values, not exploitation.
    Provide specific, actionable insights that would help the user establish meaningful connections.
    If appropriate, suggest potential conversation starters or shared interests.
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """

def create_retrieval_chain(
    db_path: str,
    model_name: str,
    temperature: float = 0.2,
    memory: Optional[ConversationBufferMemory] = None,
    search_kwargs: Dict[str, Any] = {"k": 3}
):
    """
    Create a conversational retrieval chain for social networking assistance.
    
    Args:
        db_path: Path to the vector database
        model_name: Name of the LLM model to use
        temperature: Temperature parameter for the LLM
        memory: Conversation memory to use
        search_kwargs: Keyword arguments for the retriever
        
    Returns:
        A conversational retrieval chain
    """
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Initialize the LLM
    llm = ChatOllama(model=model_name, temperature=temperature)
    
    # Create a memory if none is provided
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    # Create a conversational chain with memory
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    
    from langchain.prompts import PromptTemplate
    
    # Create a custom prompt
    prompt = PromptTemplate.from_template(create_social_network_prompt())
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain
```

### Step 4: Building the Streamlit App - Basic Structure

Now let's create an enhanced Streamlit application for our social network assistant in `streamlit_app.py`:

```python
# social-network-assistant/app/streamlit_app.py
import streamlit as st
import os
import sys
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from app.utils.profile_processor import add_profile_to_vectorstore
from app.utils.retrieval import get_available_models, create_retrieval_chain
from langchain.memory import ConversationBufferMemory

# Set page configuration
st.set_page_config(
    page_title="Social Network Assistant",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
PROFILES_DIR = Path(__file__).parent.parent / "profiles"
DB_PATH = Path(__file__).parent.parent / "db"

# Create directories if they don't exist
PROFILES_DIR.mkdir(exist_ok=True)
```

### Step 5: Setting Up Session State and Cache

Add session state management to maintain conversation history and system settings:

```python
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
# Initialize system settings in session state
if "settings" not in st.session_state:
    st.session_state.settings = {
        "model_name": "llama3.1:8b",
        "temperature": 0.1,  # Lower temperature for more factual responses
        "db_path": str(DB_PATH),
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "k_documents": 5     # Retrieve more documents for better context
    }
```

### Step 6: Creating the App Header and Sidebar

Let's add the header and a comprehensive sidebar with configuration options:

```python
# App header with custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">👥 Elite Social Network Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover connections and insights about high-profile individuals</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection section
    st.subheader("Model Selection")
    
    # Get available models and group them
    available_models = get_available_models()
    
    # Create model categories
    small_models = ["deepseek-coder:1.5b", "tinyllama:1.1b", "gemma:2b", "phi2:2.7b"]
    medium_models = ["phi3:3.8b", "llama2:7b", "llama3.1:8b", "mistral:7b"]
    large_models = ["llama2:13b", "mixtral:8x7b"]
    
    # Create tabs for model size categories
    model_tabs = st.tabs(["Small (1-3B)", "Medium (4-8B)", "Large (10B+)"])
    
    with model_tabs[0]:
        for model in small_models:
            if st.button(f"📦 {model}", key=f"model_{model}", help="Small models are fast but less capable"):
                st.session_state.settings["model_name"] = model
                st.success(f"Model changed to {model}")
                
    with model_tabs[1]:
        for model in medium_models:
            if st.button(f"🚀 {model}", key=f"model_{model}", help="Medium models balance speed and quality"):
                st.session_state.settings["model_name"] = model
                st.success(f"Model changed to {model}")
    
    with model_tabs[2]:
        for model in large_models:
            if st.button(f"🧠 {model}", key=f"model_{model}", help="Large models are slower but more powerful"):
                st.session_state.settings["model_name"] = model
                st.success(f"Model changed to {model}")
    
    # Current model indicator
    st.info(f"Current model: **{st.session_state.settings['model_name']}**")
    
    # Advanced parameters
    st.subheader("Advanced Parameters")
    
    # Temperature setting
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.settings["temperature"], 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    if temperature != st.session_state.settings["temperature"]:
        st.session_state.settings["temperature"] = temperature
    
    # Number of documents to retrieve
    k_documents = st.slider(
        "Profiles to retrieve", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.settings["k_documents"], 
        step=1,
        help="Number of profile chunks to retrieve for each query"
    )
    if k_documents != st.session_state.settings["k_documents"]:
        st.session_state.settings["k_documents"] = k_documents
    
    # Chunk size and overlap
    with st.expander("Profile Processing Settings"):
        chunk_size = st.number_input(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=st.session_state.settings["chunk_size"],
            step=50,
            help="Size of text chunks in characters"
        )
        if chunk_size != st.session_state.settings["chunk_size"]:
            st.session_state.settings["chunk_size"] = chunk_size
            
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=500, 
            value=st.session_state.settings["chunk_overlap"],
            step=10,
            help="Overlap between consecutive chunks in characters"
        )
        if chunk_overlap != st.session_state.settings["chunk_overlap"]:
            st.session_state.settings["chunk_overlap"] = chunk_overlap
    
    # Database path
    with st.expander("Database Settings"):
        db_path = st.text_input(
            "Vector Database Path", 
            value=st.session_state.settings["db_path"],
            help="Path to the ChromaDB directory"
        )
        if db_path != st.session_state.settings["db_path"]:
            st.session_state.settings["db_path"] = db_path
```

### Step 7: Add Profile Management

Let's add a profile management section to the sidebar:

```python
# Profile Management Section
with st.sidebar:
    st.header("📚 Profile Management")
    
    # Upload new profile
    uploaded_file = st.file_uploader(
        "Upload Social Profile (Text/Markdown)", 
        type=["txt", "md"],
        help="Upload a new profile document to add to your network"
    )
    
    # Process uploaded profile
    if uploaded_file is not None:
        # Create a unique filename
        file_path = PROFILES_DIR / uploaded_file.name
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Add button to process the profile
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button(
                "Process Profile", 
                type="primary",
                help="Add the profile to your network database"
            )
            
        with col2:
            advanced_process = st.checkbox(
                "Advanced Processing",
                help="Configure chunking parameters before processing"
            )
        
        # If advanced processing is selected, show chunking parameters
        if advanced_process:
            custom_chunk_size = st.slider(
                "Custom Chunk Size", 
                min_value=100, 
                max_value=2000, 
                value=st.session_state.settings["chunk_size"],
                step=50
            )
            
            custom_chunk_overlap = st.slider(
                "Custom Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=st.session_state.settings["chunk_overlap"],
                step=10
            )
        
        # Process profile when button is clicked
        if process_btn:
            with st.spinner("Processing profile... This may take a minute."):
                try:
                    chunk_size = custom_chunk_size if advanced_process else st.session_state.settings["chunk_size"]
                    chunk_overlap = custom_chunk_overlap if advanced_process else st.session_state.settings["chunk_overlap"]
                    
                    # Process the profile
                    chunk_count = add_profile_to_vectorstore(
                        file_path=str(file_path),
                        db_path=st.session_state.settings["db_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    st.success(f"✅ Profile processed into {chunk_count} chunks and added to the network database!")
                except Exception as e:
                    st.error(f"❌ Error processing profile: {str(e)}")
    
    # List existing profiles
    st.subheader("Existing Profiles")
    
    if PROFILES_DIR.exists():
        profiles = list(PROFILES_DIR.glob("*.md")) + list(PROFILES_DIR.glob("*.txt"))
        if profiles:
            for profile in profiles:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"👤 {profile.name}")
                with col2:
                    if st.button("Delete", key=f"delete_{profile.name}"):
                        try:
                            os.remove(profile)
                            st.success(f"Deleted {profile.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {str(e)}")
        else:
            st.info("No profiles found. Upload a text or markdown file to get started.")
```

### Step 8: Implement the Chat Interface

Now let's implement the main chat interface:

```python
# Main chat interface
main_container = st.container()
with main_container:
    # Create tabs for different functions
    tabs = st.tabs(["Chat", "Network Visualization", "Profile Explorer"])
    
    with tabs[0]:  # Chat Tab
        # Check if database exists
        db_exists = os.path.exists(st.session_state.settings["db_path"])
        
        if not db_exists:
            st.warning(f"⚠️ Vector database not found at {st.session_state.settings['db_path']}. Please upload and process a profile first.")
            
            # Provide instructions for setting up
            with st.expander("How to set up the knowledge base"):
                st.markdown("""
                ### Getting Started
                
                1. **Upload a profile**: Use the sidebar to upload a text/markdown profile
                2. **Process the profile**: Click the 'Process Profile' button to add it to your network database
                3. **Start chatting**: Once profiles are processed, you can ask questions about the individuals
                
                ### Example Profile Format
                
                ```
                # Jane Smith
                
                ## Background
                CEO of Tech Innovations Inc, former VP at Google
                
                ## Interests
                Quantum computing, AI ethics, mountain climbing
                
                ## Education
                PhD in Computer Science, Stanford University
                MBA, Harvard Business School
                
                ## Contact
                Email: jane.smith@example.com
                ```
                """)
        else:
            # Display chat messages
            message_container = st.container()
            with message_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Initialize the retrieval chain
            @st.cache_resource(show_spinner=False)
            def get_chain():
                return create_retrieval_chain(
                    db_path=st.session_state.settings["db_path"],
                    model_name=st.session_state.settings["model_name"],
                    temperature=st.session_state.settings["temperature"],
                    memory=st.session_state.conversation_memory,
                    search_kwargs={"k": st.session_state.settings["k_documents"]}
                )
            
            # Initialize the chain
            try:
                chain = get_chain()
            except Exception as e:
                st.error(f"Error initializing the model: {str(e)}")
                st.info("Try selecting a different model or checking your Ollama installation.")
                chain = None
            
            # Example questions
            with st.expander("Example Questions"):
                example_questions = [
                    "Who are the tech entrepreneurs in this social group?",
                    "Which individuals are interested in quantum computing?",
                    "What hobbies do the finance professionals have?",
                    "Who would be good connections for someone interested in AI?",
                    "How could I approach someone based on their interests?",
                    "Which individuals speak multiple languages?",
                    "What connections exist between people in finance and technology?",
                    "Who has the most impressive art collection?"
                ]
                
                for question in example_questions:
                    if st.button(question, key=f"q_{question[:20]}"):
                        # Set as current question
                        st.session_state.current_question = question
                        # Add to chat
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
            
            # Input for new question
            if chain is not None:
                if "current_question" in st.session_state:
                    # Clear after being used once
                    current_q = st.session_state.current_question
                    st.session_state.current_question = ""
                else:
                    current_q = ""
                
                if prompt := st.chat_input("Ask about social connections...", value=current_q):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message in chat (append to existing messages)
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        try:
                            # Add thinking animation
                            thinking_text = "Thinking"
                            for i in range(5):
                                message_placeholder.markdown(thinking_text + "." * (i % 4))
                                time.sleep(0.2)
                            
                            # Get response from chain
                            with st.spinner("Searching profiles and generating response..."):
                                response = chain({"question": prompt})
                                answer = response["answer"]
                                source_docs = response.get("source_documents", [])
                            
                            # Display answer
                            message_placeholder.markdown(answer)
                            
                            # Display sources if available
                            if source_docs:
                                with st.expander("View Profile Sources", expanded=False):
                                    for i, doc in enumerate(source_docs):
                                        source = doc.metadata.get("source", "Unknown")
                                        st.markdown(f"**Source {i+1}:** {source}")
                                        st.markdown(f"**Content excerpt:** {doc.page_content[:200]}...")
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                        except Exception as e:
                            message_placeholder.markdown(f"Error: {str(e)}")
                            st.error("Something went wrong with the LLM response. Please try again or switch to a different model.")
    
    with tabs[1]:  # Network Visualization Tab
        st.header("Social Network Visualization")
        st.info("This feature visualizes connections between individuals in your profile database.")
        
        # Placeholder for visualization
        # In a real implementation, you would extract entities and relationships
        # from your profiles and create a network graph
        
        if os.path.exists(st.session_state.settings["db_path"]):
            st.write("Generating network visualization...")
            
            # Create a simple example graph (in a real app, extract this from your profiles)
            try:
                # Create a sample graph for demonstration
                G = nx.Graph()
                G.add_nodes_from(["Jane Smith", "John Doe", "Alice Chen", "Bob Taylor", "Emma Wilson", "Michael Brown"])
                G.add_edges_from([
                    ("Jane Smith", "John Doe", {"relationship": "colleague"}),
                    ("Jane Smith", "Alice Chen", {"relationship": "mentor"}),
                    ("John Doe", "Bob Taylor", {"relationship": "friend"}),
                    ("Alice Chen", "Emma Wilson", {"relationship": "business partner"}),
                    ("Emma Wilson", "Michael Brown", {"relationship": "investor"}),
                    ("Michael Brown", "Jane Smith", {"relationship": "classmate"})
                ])
                
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(G)
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
                nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
                nx.draw_networkx_labels(G, pos, font_size=12)
                
                # Add edge labels (relationships)
                edge_labels = {(u, v): d["relationship"] for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
                
                plt.axis('off')
                st.pyplot(fig)
                
                # Add explanation
                st.markdown("""
                **Network Legend:**
                - **Nodes**: Individuals in the network
                - **Edges**: Relationships between individuals
                - **Edge Labels**: Type of relationship
                
                This visualization helps you understand how people are connected and identify potential networking opportunities.
                """)
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
        else:
            st.warning("No profile database found. Please upload and process profiles first.")
    
    with tabs[2]:  # Profile Explorer Tab
        st.header("Profile Explorer")
        st.info("Browse and search through the profiles in your database.")
        
        # Placeholder for profile explorer
        if os.path.exists(st.session_state.settings["db_path"]):
            # In a real implementation, you would extract and display profile information
            # from your database
            
            # Example implementation with sample data
            sample_profiles = [
                {"name": "Jane Smith", "occupation": "CEO", "company": "Tech Innovations Inc", "interests": ["Quantum computing", "AI ethics", "Mountain climbing"]},
                {"name": "John Doe", "occupation": "Investment Banker", "company": "Global Finance", "interests": ["Stock markets", "Tennis", "Fine dining"]},
                {"name": "Alice Chen", "occupation": "Research Scientist", "company": "BioTech Solutions", "interests": ["Genomics", "Chess", "Piano"]},
                {"name": "Bob Taylor", "occupation": "Marketing Director", "company": "Creative Media", "interests": ["Digital marketing", "Photography", "Surfing"]},
                {"name": "Emma Wilson", "occupation": "Venture Capitalist", "company": "Future Fund", "interests": ["Emerging technologies", "Yoga", "Contemporary art"]},
            ]
            
            # Convert to DataFrame for display
            df = pd.DataFrame(sample_profiles)
            
            # Add search filter
            search_term = st.text_input("Search profiles:", placeholder="Enter name, occupation, interest, etc.")
            
            # Filter data based on search term
            if search_term:
                filtered_df = df[
                    df["name"].str.contains(search_term, case=False) |
                    df["occupation"].str.contains(search_term, case=False) |
                    df["company"].str.contains(search_term, case=False) |
                    df["interests"].apply(lambda x: any(search_term.lower() in item.lower() for item in x))
                ]
            else:
                filtered_df = df
            
            # Display profiles
            st.write(f"Showing {len(filtered_df)} of {len(df)} profiles")
            
            for i, row in filtered_df.iterrows():
                with st.container():
                    st.subheader(row["name"])
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Occupation:** {row['occupation']}")
                        st.write(f"**Company:** {row['company']}")
                    with col2:
                        st.write("**Interests:**")
                        st.write(", ".join(row["interests"]))
                    st.divider()
        else:
            st.warning("No profile database found. Please upload and process profiles first.")
```

### Step 9: Add Advanced Features and Conversation Management

Let's add additional features and conversation management:

```python
# Add a button to clear the conversation at the bottom
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        st.rerun()

# Advanced features section
with st.expander("Advanced Features"):
    feature_tabs = st.tabs(["Conversation Settings", "Chat History"])
    
    with feature_tabs[0]:
        st.subheader("Conversation Settings")
        
        # Memory type (for future expansion)
        memory_type = st.selectbox(
            "Memory Type",
            ["Buffer Memory (All History)", "Window Memory (Limited)", "Summary Memory (Compressed)"],
            index=0,
            disabled=True
        )
        st.info("Additional memory types will be available in a future update.")
        
        # System message customization
        st.subheader("System Message")
        system_message = st.text_area(
            "Customize the system message",
            value="""You are an expert social network assistant that helps users connect with elite individuals.
Your role is to provide accurate information about high-profile people in exclusive social circles.""",
            height=100
        )
        if st.button("Apply System Message"):
            st.info("System message updated. This will take effect in the next conversation.")
 
    with feature_tabs[1]:
        st.subheader("Chat History")
        if st.session_state.messages:
            # Display conversation count and option to download
            st.write(f"Current conversation has {len(st.session_state.messages)} messages")
            
            # Option to export conversation history
            if st.button("Export Chat History"):
                # This would need to be implemented
                st.info("Export functionality coming in future update")
        else:
            st.info("No chat history yet. Start a conversation to see messages here.")
```

### Step 10: Running the Complete Streamlit App

To run the complete Streamlit app, navigate to the project directory and use:

```bash
cd social-network-assistant
streamlit run app/streamlit_app.py
```

This will start a local web server and open the app in your browser.

## Part 2: Implementing Conversation Memory

Our Streamlit app already includes basic conversation memory using LangChain's `ConversationBufferMemory`. This allows the model to remember previous exchanges within the session.

Let's explore what's happening:

1. We initialize the memory in session state:
   ```python
   if "conversation_memory" not in st.session_state:
       st.session_state.conversation_memory = ConversationBufferMemory(
           memory_key="chat_history",
           return_messages=True
       )
   ```

2. We use this memory when creating our retrieval chain:
   ```python
   chain = ConversationalRetrievalChain.from_llm(
       llm=llm,
       retriever=retriever,
       memory=st.session_state.conversation_memory,
       return_source_documents=True
   )
   ```

The `ConversationBufferMemory` stores all previous exchanges, allowing the model to reference past questions and answers. This enables follow-up questions and maintains context throughout the conversation.

For example, a user could ask:
1. "Who are the tech entrepreneurs in this network?"
2. "What are their interests?"
3. "Which of them speak multiple languages?"

With conversation memory, the model understands that the second and third questions refer to the tech entrepreneurs mentioned in the first question, creating a more natural and helpful interaction.

## Part 3: Building a Gradio Interface

Gradio is another popular library for creating web interfaces for machine learning models. Let's create an alternative interface for our social network assistant.

Create a new file called `gradio_app.py`:

```python
# social-network-assistant/app/gradio_app.py
import gradio as gr
import os
import sys
from pathlib import Path

# Add parent directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from app.utils.retrieval import get_available_models, create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Function to initialize the retrieval chain
def get_retrieval_chain(db_path, model_name, temperature):
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Initialize the LLM
    from langchain.chat_models import ChatOllama
    llm = ChatOllama(model=model_name, temperature=temperature)
    
    # Create a conversational chain with memory
    retriever = db.as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    return chain

# Default parameters
default_db_path = "db"
default_model = "llama3.1:8b"
default_temperature = 0.1

# Initialize the chain
try:
    chain = get_retrieval_chain(default_db_path, default_model, default_temperature)
    chain_initialized = True
except Exception as e:
    print(f"Error initializing chain: {str(e)}")
    chain_initialized = False

# Function to process queries
def process_query(message, history):
    if not chain_initialized:
        return "Error: Chain not initialized. Please check your database path and Ollama installation."
    
    # Get response from chain
    try:
        response = chain({"question": message})
        answer = response["answer"]
        source_docs = response.get("source_documents", [])
        
        # Format sources if available
        if source_docs:
            sources = "\n\nProfile Sources:\n"
            for i, doc in enumerate(source_docs):
                sources += f"{i+1}. {doc.metadata.get('source', 'Unknown')}\n"
            answer += sources
        
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Elite Social Network Assistant")
    gr.Markdown("Ask questions about individuals in exclusive social circles to help you connect.")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Ask about social connections...", show_label=False)
            clear = gr.Button("Clear Chat")
        
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")
            model_dropdown = gr.Dropdown(
                ["llama3.1:8b", "mistral:7b", "llama2:7b", "deepseek-coder:1.5b", "tinyllama:1.1b"], 
                label="Model", 
                value=default_model
            )
            temp_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=default_temperature, 
                step=0.1, 
                label="Temperature"
            )
            db_path_input = gr.Textbox(
                value=default_db_path,
                label="Database Path"
            )
            update_btn = gr.Button("Update Configuration")
            
            # Example questions
            gr.Markdown("## Example Questions")
            example_questions = [
                "Who are the tech entrepreneurs in this network?",
                "Which individuals are interested in quantum computing?",
                "What hobbies do the finance professionals have?",
                "Who would be good connections for someone interested in AI?"
            ]
            
            for question in example_questions:
                example_btn = gr.Button(question)
                example_btn.click(lambda q=question: q, None, msg)
    
    # Set up interactions
    msg.submit(process_query, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Function to update configuration
    def update_config(model, temp, db_path):
        global chain, chain_initialized
        try:
            chain = get_retrieval_chain(db_path, model, float(temp))
            chain_initialized = True
            return "Configuration updated successfully!"
        except Exception as e:
            chain_initialized = False
            return f"Error updating configuration: {str(e)}"
    
    update_btn.click(
        update_config,
        [model_dropdown, temp_slider, db_path_input],
        gr.Textbox(label="Status")
    )

# Launch the app
if __name__ == "__main__":
    if os.path.exists(default_db_path):
        demo.launch()
    else:
        print(f"Error: Vector database not found at {default_db_path}")
        print("Please make sure you've processed profiles and created a vector database first.")
```

### Running the Gradio App

To run the Gradio app, use the following command:

```bash
cd social-network-assistant
python app/gradio_app.py
```

## Part 4: Comparing Streamlit and Gradio

Both Streamlit and Gradio offer different advantages for building interfaces for your social network assistant:

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| Learning curve | Easier for beginners | Slightly steeper |
| Customization | Extensive | Focused on ML interfaces |
| Session state | Built-in | Manual implementation |
| UI components | Rich library | ML-focused components |
| Deployment | Multiple options | Hugging Face Spaces integration |
| Community | Large, active | Growing, ML-focused |

### When to choose Streamlit:
- For data-rich applications with visualizations of social networks
- When you need complex session state management for conversation history
- For multi-page applications with different views of your social data
- When rapid prototyping is a priority

### When to choose Gradio:
- For dedicated chat interfaces focused on profile queries
- When you want easy Hugging Face integration
- For simple interfaces where the focus is purely on question-answering
- When you prioritize minimal code for deployment

## Part 5: Enhancing Your Interface

Here are some ideas to further enhance your social network assistant interface:

1. **Profile Management**:
   - Add functionality to upload and manage social profiles directly
   - Implement profile categories or tags for better organization
   - Add analytics on profile engagement and query frequency

2. **Advanced Social Network Features**:
   - Implement entity extraction to automatically identify people, organizations, and interests
   - Create relationship graphs to visualize connections between individuals
   - Add filters for finding connections based on specific criteria

3. **User Experience Improvements**:
   - Add profile cards with rich information for each individual
   - Implement contact management for saving important connections
   - Create a dark/light mode toggle for different preferences

4. **Visualization**:
   - Add network graphs showing relationship strength and types
   - Visualize common interests across the network
   - Create industry and interest distribution charts

## Conclusion

In this tutorial, we've built two interactive interfaces for our local LLM social network assistant using Streamlit and Gradio. These interfaces make it easier to interact with our RAG system and provide a more user-friendly experience for exploring social profiles and networking opportunities.

Both frameworks offer unique advantages, and you can choose the one that best fits your needs or implement both for different use cases.

Remember that the key to a good social network assistant interface is making the complex information accessible and useful. Focus on clear presentation of profile information, intuitive ways to explore connections, and helpful features for users looking to expand their network.

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Network Analysis with NetworkX](https://networkx.org/documentation/stable/)
