# Building a Local LLM Social Network Assistant (Part 2)

## Overview
In this tutorial, we'll build upon the foundation established in Part 1 by creating interactive interfaces for our local LLM research assistant using Streamlit and Gradio. These interfaces will make it easier to interact with our RAG system and provide a more user-friendly experience.

## Learning Objectives
- Create a comprehensive Streamlit interface for the research assistant
- Build an alternative interface using Gradio
- Implement basic conversation memory
- Add simple document management features

## Prerequisites
- Completion of Part 1 of the tutorial
- Functional RAG system with Ollama and ChromaDB
- Basic understanding of Python web applications

## Part 1: Building a Streamlit Interface

Streamlit provides an easy way to build interactive web applications for machine learning and data science projects. Let's create an enhanced interface for our research assistant.

### Setting Up the Basic App

First, let's create a new file called `streamlit_app.py`:

```python
import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# App header
st.title("Local LLM Research Assistant")
st.markdown("Ask questions about your research papers using a local LLM")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Select LLM Model",
        ["llama3.1:8b", "mistral:7b"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    # Database path input
    db_path = st.text_input("Vector Database Path", "chroma_db")
    
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        # Add button to process the document
        if st.button("Process Document"):
            st.info("Processing document... This may take a minute.")
            # This would call your document processing function
            st.success("Document processed and added to the knowledge base!")

# Function to initialize the retrieval chain
@st.cache_resource
def get_retrieval_chain(db_path, model_name, temperature):
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Initialize the LLM
    llm = ChatOllama(model=model_name, temperature=temperature)
    
    # Create a conversational chain with memory
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.conversation_memory,
        return_source_documents=True
    )
    
    return chain

# Main chat interface
if os.path.exists(db_path):
    # Initialize the retrieval chain
    chain = get_retrieval_chain(db_path, model_name, temperature)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new question
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Get response from chain
            with st.spinner("Thinking..."):
                response = chain({"question": prompt})
                answer = response["answer"]
                source_docs = response.get("source_documents", [])
            
            # Display answer
            message_placeholder.markdown(answer)
            
            # Display sources if available
            if source_docs:
                st.markdown("**Sources:**")
                for i, doc in enumerate(source_docs):
                    st.markdown(f"{i+1}. {doc.metadata.get('source', 'Unknown')}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning(f"Vector database not found at {db_path}. Please make sure you've processed documents and created a vector database.")
    
    # Provide instructions for setting up
    with st.expander("How to setup the knowledge base"):
        st.markdown("""
        1. Make sure you've completed Part 1 of the tutorial
        2. Process your documents to create the vector database
        3. Ensure the database path is correct in the sidebar
        """)
```

### Running the Streamlit App

To run the Streamlit app, use the following command:

```bash
streamlit run streamlit_app.py
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

## Part 3: Building a Gradio Interface

Gradio is another popular library for creating web interfaces for machine learning models. Let's create an alternative interface using Gradio.

Create a new file called `gradio_app.py`:

```python
import gradio as gr
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
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
    llm = ChatOllama(model=model_name, temperature=temperature)
    
    # Create a conversational chain with memory
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    return chain

# Default parameters
default_db_path = "chroma_db"
default_model = "llama3.1:8b"
default_temperature = 0.2

# Initialize the chain
chain = get_retrieval_chain(default_db_path, default_model, default_temperature)

# Function to process queries
def process_query(message, history):
    # Get response from chain
    response = chain({"question": message})
    answer = response["answer"]
    source_docs = response.get("source_documents", [])
    
    # Format sources if available
    if source_docs:
        sources = "\n\nSources:\n"
        for i, doc in enumerate(source_docs):
            sources += f"{i+1}. {doc.metadata.get('source', 'Unknown')}\n"
        answer += sources
    
    return answer

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Research Assistant")
    gr.Markdown("Ask questions about your research papers using a local LLM")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Ask a question about your documents...", show_label=False)
            clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")
            model_dropdown = gr.Dropdown(
                ["llama3.1:8b", "mistral:7b", "llama2:13b"], 
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
                label="Vector Database Path"
            )
            update_btn = gr.Button("Update Configuration")
    
    # Set up interactions
    msg.submit(process_query, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Function to update configuration
    def update_config(model, temp, db_path):
        global chain
        chain = get_retrieval_chain(db_path, model, float(temp))
        return "Configuration updated!"
    
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
```

### Running the Gradio App

To run the Gradio app, use the following command:

```bash
python gradio_app.py
```

## Part 4: Comparing Streamlit and Gradio

Both Streamlit and Gradio offer different advantages for building interfaces:

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| Learning curve | Easier for beginners | Slightly steeper |
| Customization | Extensive | Focused on ML interfaces |
| Session state | Built-in | Manual implementation |
| UI components | Rich library | ML-focused components |
| Deployment | Multiple options | Hugging Face Spaces integration |
| Community | Large, active | Growing, ML-focused |

### When to choose Streamlit:
- For data-focused applications with visualizations
- When you need complex session state management
- For multi-page applications
- When rapid prototyping is a priority

### When to choose Gradio:
- For dedicated ML model interfaces
- When you want easy Hugging Face integration
- For simple chat or image processing interfaces
- When you prioritize minimal code

## Part 5: Enhancing Your Interface

Here are some ideas to further enhance your research assistant interface:

1. **Document Management**:
   - Add functionality to upload, process, and manage documents directly from the interface
   - Implement document categories or tags

2. **Advanced RAG Features**:
   - Add options to adjust retrieval parameters (k, similarity threshold)
   - Implement hybrid search with filters

3. **User Experience Improvements**:
   - Add loading indicators during processing
   - Implement error handling and user feedback
   - Create a dark/light mode toggle

4. **Visualization**:
   - Add visualizations of retrieved documents
   - Visualize the relevance of retrieved chunks

## Conclusion

In this tutorial, we've built two interactive interfaces for our local LLM research assistant using Streamlit and Gradio. These interfaces make it easier to interact with our RAG system and provide a more user-friendly experience.

Both frameworks offer unique advantages, and you can choose the one that best fits your needs or implement both for different use cases.

Remember that the key to a good research assistant interface is making the complex technology accessible and useful. Focus on clear presentation of information, intuitive interactions, and helpful feedback to the user.

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
