# Building a Local LLM Social Network Assistant - Part 2

## Overview

In [Part 1](../week-5/practice-2-local-llm-ollama-part-1.md) of this tutorial, we built a basic RAG system that can answer questions about social profiles using a local LLM. Now in Part 2, we'll build upon that foundation by creating interactive user interfaces using Streamlit and Gradio. These interfaces will make our social network assistant more accessible and user-friendly.

## Learning Objectives

- Create an interactive Streamlit interface for our social network assistant
- Build a Gradio interface with chat functionality
- Implement basic conversation memory for multi-turn interactions

## Prerequisites

- Completed [Part 1](../week-5/practice-2-local-llm-ollama-part-1.md) of the tutorial

## Part 1: Creating an Advanced Streamlit Interface

Let's enhance our Streamlit interface with more features and better organization.

```python
# app.py
import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    model_name = st.selectbox("Select LLM Model", 
                             ["deepseek:7b", "deepseek-coder:6.7b", "deepseek-lite:1.3b", 
                              "llama3.1:8b", "mistral:7b", "phi3:3.8b"], 
                             index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
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
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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
    Your role is to provide accurate information about high-profile people in exclusive social circles.
    
    Use ONLY the context below to answer the question. If the information is not in the context,
    say you don't know - DO NOT make up facts or connections that aren't supported by the context.
    Always maintain discretion and privacy when discussing elite individuals.
    
    When suggesting networking approaches, focus on genuine common interests and values, not exploitation.
    Provide specific, actionable insights that would help the user establish meaningful connections.
    If appropriate, suggest potential conversation starters or shared interests.
    
    Context: {context}
    
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
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
            
            # Display sources if available
            if "source_documents" in response:
                with st.expander("Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(f"*Content:* {doc.page_content[:200]}...")
                        if hasattr(doc.metadata, 'source'):
                            st.markdown(f"*Source:* {doc.metadata['source']}")
    
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    collection_name="social_profiles",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialize the LLM
llm = Ollama(model="deepseek:7b", temperature=0.7)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the conversational chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# Function to process chat interactions
def respond(message, chat_history):
    response = qa({"question": message})
    chat_history.append((message, response["answer"]))
    return "", chat_history

# Create the Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Social Network Assistant")
    gr.Markdown("Ask questions about social profiles in your network")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about social profiles")
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

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
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load vector store
    vector_db = Chroma(
        collection_name="social_profiles",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialize the LLM
    llm = Ollama(model="deepseek:7b", temperature=0.7)
    
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

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
    # Initialize the QA system with memory
    @st.cache_resource
    def initialize_qa_system():
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store
        vector_db = Chroma(
            collection_name="social_profiles",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize the LLM
        llm = Ollama(model="deepseek:7b", temperature=0.7)
        
        # Create the QA chain with memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            return_source_documents=True
        )
        
        return qa_chain, vector_db

    # Main application
    qa_chain, vector_db = initialize_qa_system()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Chat Settings")
        model_name = st.selectbox("Model", ["deepseek:7b", "deepseek-coder:6.7b", "deepseek-lite:1.3b", "llama3.1:8b", "mistral:7b", "phi3:3.8b"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.experimental_rerun()

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
                response = qa_chain({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)
                
                # Display sources if available
                if "source_documents" in response:
                    with st.expander("Sources"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}**")
                            st.markdown(f"*Content:* {doc.page_content[:200]}...")
                            if hasattr(doc.metadata, 'source'):
                                st.markdown(f"*Source:* {doc.metadata['source']}")
        
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
    
    ### How It Works
    
    1. Your social profile data is processed and stored in a vector database
    2. When you ask a question, relevant profiles are retrieved
    3. A local LLM generates helpful answers based on the retrieved information
    4. Conversation memory helps maintain context across multiple questions
    
    ### Privacy
    
    All processing happens locally on your machine. No data is sent to external servers.
    """)

## Running the Application

To run any of these applications:

1. Make sure you have completed [Part 1](../week-5/practice-2-local-llm-ollama-part-1.md) of the tutorial and have a functional knowledge base
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

- In [Part 1](../week-5/practice-2-local-llm-ollama-part-1.md), we set up the foundation by building a basic RAG system with Ollama, LangChain, and ChromaDB.
- In Part 2, we enhanced the system with interactive interfaces and conversation memory.

The result is a powerful, privacy-focused assistant that can:
1. Answer questions about social profiles using local LLM inference
2. Maintain context across conversations using memory
3. Provide an intuitive interface for users through Streamlit and Gradio
4. Run entirely on your local machine without sending data to external services

This project demonstrates how modern local LLMs like DeepSeek and sophisticated frameworks like LangChain can be combined to create practical AI applications that respect user privacy while delivering high-quality responses.

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangChain Memory Types](https://python.langchain.com/docs/modules/memory/)
- [Ollama Model Library](https://ollama.com/library)
- [DeepSeek Models Documentation](https://github.com/deepseek-ai/deepseek-LLM)

