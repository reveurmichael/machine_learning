# Building a Local LLM Social Network Assistant - Part 2

## Overview

In [Part 1](../week-6/practice-local-llm-ollama-part-1.md) of this tutorial, we built a basic RAG system that can answer questions about social profiles using a local LLM. Now in Part 2, we'll build upon that foundation by creating interactive user interfaces using Streamlit and Gradio. These interfaces will make our social network assistant more accessible and user-friendly.

## Learning Objectives

- Create an interactive Streamlit interface for our social network assistant
- Build a Gradio interface with chat functionality
- Implement basic conversation memory for multi-turn interactions

## Prerequisites

- Completed [Part 1](../week-6/practice-local-llm-ollama-part-1.md) of the tutorial

## Part 1: Creating an Advanced Streamlit Interface

Let's enhance our Streamlit interface with more features and better organization.

```python
# app.py
import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="Social Network Assistant",
    page_icon="👥",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Application header
st.title("📱 Social Network Assistant")
st.subheader("Ask questions about your social network profiles")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Get available models from Ollama
    available_models = []
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip the header line
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        available_models.append(model_name)
        if not available_models:
            available_models = ["llama3.2:latest"]
    except Exception as e:
        st.sidebar.error(f"Error getting models: {e}")
        available_models = ["llama3.2:latest"]
    
    model_name = st.selectbox("Select LLM Model", available_models, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Option to rebuild the database
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Rebuilding knowledge base..."):
            import shutil
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            
            # Re-import necessary modules for rebuilding
            from langchain_community.document_loaders import TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Process documents function
            def process_documents(file_path, db_directory="./chroma_db"):
                # Load documents
                loader = TextLoader(file_path)
                documents = loader.load()
                
                # Chunk documents with improved overlap
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=300,  # Increased overlap for better context
                    length_function=len,
                    add_start_index=True,
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create embeddings
                embeddings = OllamaEmbeddings(model=model_name)
                
                # Create vector store
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_directory
                )
                vectordb.persist()
                
                return vectordb
            
            # Process the documents
            profile_path = "profiles/student_database.md"
            if os.path.exists(profile_path):
                process_documents(profile_path)
                st.success("✅ Knowledge base rebuilt successfully")
            else:
                st.error(f"Profile file not found: {profile_path}")
    
    st.header("About")
    st.markdown("""
    This application uses a local LLM to answer questions about social network profiles.
    - Built with Ollama, LangChain, and Streamlit
    - Uses RAG (Retrieval-Augmented Generation)
    - All processing happens locally
    """)

# Initialize the QA system
@st.cache_resource
def initialize_qa_system(model_name, temperature):
    # Load embeddings using the same model for consistency
    embeddings = OllamaEmbeddings(model=model_name)
    
    # Load vector store
    vector_db = Chroma(
        collection_name="social_profiles",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialize the LLM
    llm = Ollama(model=model_name, temperature=temperature)
    
    # Create prompt template - enhanced for better profile information extraction
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
    
    Context: {context}
    
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create the QA chain with increased chunk retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 8}),  # Increased from 3 to 8 for better coverage
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

# Main application
qa_chain = initialize_qa_system(model_name, temperature)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about social profiles"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain({"query": prompt})
            answer = response["result"]
            st.markdown(answer)
            
            # Display sources if available with improved display
            if "source_documents" in response:
                with st.expander("Show Source Information"):
                    st.markdown("### Retrieved Profile Chunks")
                    st.write(f"Retrieved {len(response['source_documents'])} relevant chunks from the profiles database.")
                    
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"#### Chunk {i+1}")
                        source = doc.metadata.get("source", "Unknown source")
                        st.markdown(f"**Source**: {source}")
                        st.text_area(f"Content {i+1}", doc.page_content, height=200)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
```

## Part 2: Building a Gradio Interface

Gradio offers another way to create interactive interfaces for our AI applications. Let's build a chat interface using Gradio:

```python
# gradio_app.py
import gradio as gr
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Get available Ollama models
def get_available_models():
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        available_models = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip the header line
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        available_models.append(model_name)
        if not available_models:
            available_models = ["llama3.2:latest"]
        return available_models
    except Exception:
        return ["llama3.2:latest"]

# Get the first available model
available_models = get_available_models()
model_name = available_models[0]

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model=model_name)
vector_db = Chroma(
    collection_name="social_profiles",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialize the LLM
llm = Ollama(model=model_name, temperature=0.7)

# Enhanced prompt template for better profile retrieval
prompt_template = """
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

Context: {chat_history}
{context}

Question: {question}
Answer:
"""

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the conversational chain with improved retrieval
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 8}),  # Increased from 3 to 8
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
)

# Function to rebuild the knowledge base
def rebuild_knowledge_base(progress=gr.Progress()):
    progress(0, desc="Starting rebuild process")
    try:
        import shutil
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Remove existing database
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        progress(0.3, desc="Loading documents")
        
        # Load documents
        profile_path = "profiles/student_database.md"
        if not os.path.exists(profile_path):
            return "Error: Profile file not found"
            
        loader = TextLoader(profile_path)
        documents = loader.load()
        
        progress(0.5, desc="Chunking documents")
        
        # Chunk documents with improved overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,  # Increased overlap for better context
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        
        progress(0.7, desc="Creating embeddings")
        
        # Create embeddings
        embeddings = OllamaEmbeddings(model=model_name)
        
        progress(0.8, desc="Building vector database")
        
        # Create vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        vectordb.persist()
        
        progress(1.0, desc="Complete")
        
        # Reset global variables to use the new database
        global vector_db, qa, memory
        vector_db = Chroma(
            collection_name="social_profiles",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 8}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
        )
        
        return "Knowledge base rebuilt successfully!"
    except Exception as e:
        return f"Error rebuilding knowledge base: {str(e)}"

# Function to process chat interactions
def respond(message, chat_history):
    response = qa({"question": message})
    chat_history.append((message, response["answer"]))
    return "", chat_history

# Create the Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Social Network Assistant")
    gr.Markdown("Ask questions about social profiles in your network")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask a question about social profiles", scale=3)
                clear = gr.Button("Clear", scale=1)
            
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=model_name,
                label="Select Model"
            )
            rebuild_btn = gr.Button("🔄 Rebuild Knowledge Base")
            rebuild_output = gr.Textbox(label="Status")
            
            rebuild_btn.click(rebuild_knowledge_base, outputs=rebuild_output)
            
            gr.Markdown("### Example Questions")
            example_questions = [
                "Who are the tech entrepreneurs in this group?",
                "Which individuals are interested in quantum computing?",
                "Who is passionate about LeetCode?",
                "How could I approach 赵俊凯 based on his interests?",
                "Which individuals speak multiple languages?"
            ]
            
            for question in example_questions:
                gr.Button(question).click(
                    lambda q: (q, ""), 
                    [lambda example=question: example], 
                    [msg, None],
                    queue=False
                )

# Launch the app
if __name__ == "__main__":
    demo.launch()
```

## Part 3: Adding Conversation Memory

Let's enhance our Streamlit interface with conversation memory to make interactions more natural:

```python
# app_with_memory.py
import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# Page configuration
st.set_page_config(
    page_title="Social Network Assistant",
    page_icon="👥",
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Application header
st.title("Social Network Assistant with Memory")

# Initialize the QA system with memory
@st.cache_resource
def initialize_qa_system():
    # Load embeddings
    model_name = "llama3.2:1b"
    embeddings = OllamaEmbeddings(model=model_name)
    
    # Load vector store
    vector_db = Chroma(
        collection_name="social_profiles",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialize the LLM
    llm = Ollama(model=model_name, temperature=0.7)
    
    # Create the QA chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory
    )
    
    return qa_chain

# Main application
qa_chain = initialize_qa_system()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about social profiles"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.experimental_rerun()
```

## Part 4: Understanding Conversation Memory

Conversation memory allows our assistant to remember previous interactions within a session. Here's how it works:

1. **Conversation Buffer Memory** stores the full history of conversations
2. The assistant can reference previous questions and its own answers
3. This enables follow-up questions like "Tell me more about that" or "What was their age again?"

The memory component maintains a record of:
- User queries
- System responses
- Context from retrieved documents

This makes the assistant more natural to interact with and enhances its ability to maintain context across multiple turns of conversation.

## Part 5: Creating a Complete Application

Let's combine everything we've learned to create a complete application:

```python
# complete_app.py
import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="Social Network Assistant",
    page_icon="👥",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Application header
st.title("📱 Social Network Assistant")

# Navigation
tab1, tab2 = st.tabs(["Chat Assistant", "About"])

with tab1:
    # Get available models from Ollama
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_models():
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            available_models = []
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip the header line
                    for line in lines[1:]:
                        if line.strip():
                            model_name = line.split()[0]
                            available_models.append(model_name)
            if not available_models:
                available_models = ["llama3.2:latest"]
            return available_models
        except Exception:
            return ["llama3.2:latest"]
    
    # Initialize the QA system with memory
    @st.cache_resource
    def initialize_qa_system(model_name, temperature=0.7):
        # Load embeddings
        embeddings = OllamaEmbeddings(model=model_name)
        
        # Load vector store
        vector_db = Chroma(
            collection_name="social_profiles",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize the LLM
        llm = Ollama(model=model_name, temperature=temperature)
        
        # Create prompt template
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
        
        Chat History:
        {chat_history}
        
        Context:
        {context}
        
        Question: {question}
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]
        )
        
        # Create the QA chain with memory and enhanced retrieval
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 8}),  # Increased from 3 to 8
            memory=st.session_state.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return qa_chain, vector_db

    # Function to rebuild knowledge base
    def rebuild_knowledge_base(model_name):
        with st.status("Rebuilding knowledge base...", expanded=True) as status:
            try:
                import shutil
                from langchain_community.document_loaders import TextLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                
                status.update(label="Removing old database...")
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                
                status.update(label="Loading documents...")
                profile_path = "profiles/student_database.md"
                if not os.path.exists(profile_path):
                    status.update(label="Error: Profile file not found", state="error")
                    return
                    
                loader = TextLoader(profile_path)
                documents = loader.load()
                
                status.update(label="Processing documents...")
                # Chunk documents with improved overlap
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=300,  # Increased overlap for better context
                    length_function=len,
                    add_start_index=True,
                )
                chunks = text_splitter.split_documents(documents)
                
                status.update(label="Creating embeddings...")
                embeddings = OllamaEmbeddings(model=model_name)
                
                status.update(label="Building vector database...")
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                vectordb.persist()
                
                # Clear session state to force reinitialization of the QA system
                if "qa_chain" in st.session_state:
                    del st.session_state["qa_chain"]
                    
                # Reset memory
                st.session_state.memory.clear()
                
                status.update(label="Knowledge base rebuilt successfully!", state="complete")
            except Exception as e:
                status.update(label=f"Error rebuilding knowledge base: {str(e)}", state="error")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Chat Settings")
        
        available_models = get_available_models()
        model_name = st.selectbox("Model", available_models, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        if st.button("🔄 Rebuild Knowledge Base"):
            rebuild_knowledge_base(model_name)
            
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.experimental_rerun()

    # Main application
    if "qa_chain" not in st.session_state:
        with st.spinner("Initializing assistant..."):
            st.session_state.qa_chain, st.session_state.vector_db = initialize_qa_system(model_name, temperature)
    
    # Chat interface
    st.subheader("Chat with your Social Network Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about your social network..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)
                
                # Display sources if available with enhanced display
                if "source_documents" in response:
                    with st.expander("Show Source Information"):
                        st.markdown("### Retrieved Profile Chunks")
                        st.write(f"Retrieved {len(response['source_documents'])} relevant chunks from the profiles database.")
                        
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"#### Chunk {i+1}")
                            source = doc.metadata.get("source", "Unknown source")
                            st.markdown(f"**Source**: {source}")
                            st.text_area(f"Content {i+1}", doc.page_content, height=200)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.header("About This Application")
    
    st.markdown("""
    ### Social Network Assistant
    
    This application helps you explore and understand your social connections using:
    
    - **Local LLM Processing**: All data stays on your machine
    - **RAG (Retrieval-Augmented Generation)**: Accurate, up-to-date information
    - **Interactive Interface**: Easy to use and explore
    
    ### Enhanced Features
    
    This application includes several optimizations for better performance:
    
    - **Improved Document Chunking**: Enhanced chunk overlap (300 characters) for better context
    - **Comprehensive Retrieval**: Retrieves 8 chunks per query for more complete information
    - **Specialized Prompt Engineering**: Customized for detailed social profile information
    - **Source Information**: View exactly which profile chunks were used to generate answers
    - **Dynamic Model Selection**: Automatically detects available Ollama models
    - **Conversation Memory**: Maintains context across multiple questions
    
    ### How It Works
    
    1. Your social profile data is processed and stored in a vector database
    2. When you ask a question, relevant profiles are retrieved
    3. A local LLM generates helpful answers based on the retrieved information
    4. Conversation memory helps maintain context across multiple questions
    
    ### Privacy
    
    All processing happens locally on your machine. No data is sent to external servers.
    """)
```
## Running the Application

To run any of these applications:

1. Make sure you have completed [Part 1](../week-6/practice-local-llm-ollama-part-1.md) of the tutorial and have a functional knowledge base
2. Save the code for your chosen interface in the `social-network-assistant` directory:
   - For the basic interface: `app.py`
   - For the Gradio interface: `gradio_app.py`
   - For the memory-enabled interface: `app_with_memory.py`
   - For the complete application: `complete_app.py`

3. Install the additional required packages:

```bash
pip install streamlit gradio
```

4. Start the Streamlit app:

```bash
cd social-network-assistant
streamlit run app.py  # Or whichever file you want to run
```

Or for the Gradio app:

```bash
cd social-network-assistant
python gradio_app.py
```

The application will automatically load your existing knowledge base from the `chroma_db` directory if it exists, or create a new one if needed.

## Final Project Structure

After completing both parts of the tutorial, your final directory structure should look like:

```
social-network-assistant/
├── app.py                 # Basic Streamlit interface 
├── gradio_app.py          # Gradio interface
├── app_with_memory.py     # Streamlit interface with memory
├── complete_app.py        # Complete application
├── chroma_db/             # Vector database
├── profiles/              # Directory for social profile data
│   └── student_database.md # Sample profile data
└── outputs/               # Directory for any output files
```

## Conclusion

In this two-part tutorial series, we've built a comprehensive social network assistant using local LLMs:

- In [Part 1](../week-6/practice-local-llm-ollama-part-1.md), we set up the foundation by building an optimized RAG system with Ollama, LangChain, and ChromaDB, including enhanced chunking, retrieval, and prompt engineering.
- In Part 2, we built upon those optimizations by creating interactive interfaces with Streamlit and Gradio, adding conversation memory, and implementing developer-friendly features like database rebuilding and source visualization.

The result is a powerful, privacy-focused assistant that can:
1. Answer detailed questions about social profiles using local LLM inference
2. Retrieve comprehensive information with optimized chunking and retrieval parameters
3. Maintain context across conversations using memory
4. Provide an intuitive interface with transparent source information
5. Adapt to available models on your system dynamically
6. Run entirely on your local machine without sending data to external services

This project demonstrates how modern local LLMs, sophisticated frameworks like LangChain, and optimization techniques can be combined to create practical AI applications that respect user privacy while delivering high-quality, detailed responses about social network profiles.

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Memory Types](https://python.langchain.com/docs/modules/memory/)
- [Ollama Model Library](https://ollama.com/library)

