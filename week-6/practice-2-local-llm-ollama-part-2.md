# Comprehensive Tutorial: Advanced Local LLM Research Assistant with RAG (Part 2)

## Overview
This second 120-minute session builds on the foundation established in Part 1, taking your research assistant to the next level with advanced RAG techniques, multi-document support, evaluation metrics, and the development of an autonomous research agent. We'll explore how to make your system more robust, accurate, and useful for academic research.

## Learning Objectives
- Implement advanced RAG techniques including hybrid search and re-ranking
- Build a multi-document knowledge base for comprehensive research
- Create a conversational interface with memory
- Develop an autonomous research agent using LangChain agents
- Evaluate and optimize your RAG system using quantitative metrics
- Connect RAG techniques to core machine learning concepts

## Prerequisites
- Completed Part 1 of the tutorial
- Functional RAG system from Part 1
- Understanding of vector embeddings and basic RAG architecture

---

## Part 1: Advanced RAG Techniques (20 minutes)

### Hybrid Search: Combining Semantic and Keyword Search
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

def create_hybrid_retriever(vectordb, documents, search_kwargs={"k": 4}):
    """Create a hybrid retriever that combines vector search with BM25."""
    # Vector retriever
    vector_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    
    # BM25 (keyword) retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = search_kwargs["k"]
    
    # Combine retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]  # Weights for each retriever
    )
    
    return ensemble_retriever
```

### Re-ranking Retrieved Documents for Improved Relevance
```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

def create_reranking_retriever(llm, base_retriever):
    """Create a retriever that re-ranks results using an LLM."""
    # Document compressor that uses LLM to extract relevant parts
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever
```

### Self-Query Retriever: Allowing the LLM to Structure the Query
```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

def create_self_query_retriever(llm, vectordb, embeddings):
    """Create a retriever that allows the LLM to structure the query."""
    # Define metadata fields that can be queried
    metadata_field_info = [
        AttributeInfo(
            name="page",
            description="The page number in the document",
            type="integer",
        ),
        AttributeInfo(
            name="section",
            description="The section title of the document",
            type="string",
        ),
    ]
    
    # Create self-query retriever
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectordb,
        document_contents="Academic research paper content",
        metadata_field_info=metadata_field_info,
        embeddings=embeddings
    )
    
    return retriever
```

---

## Part 2: Building a Multi-Document Knowledge Base (25 minutes)

### Handling Multiple Documents with Metadata
```python
def process_multiple_documents(pdf_directory, persist_dir="db"):
    """Process multiple PDF documents and create a unified vector store."""
    all_chunks = []
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        # Extract filename without extension as document title
        doc_title = os.path.splitext(pdf_file)[0]
        
        # Load and chunk the document
        documents = load_pdf(pdf_path)
        
        # Add document title to metadata
        for doc in documents:
            doc.metadata["source"] = doc_title
        
        chunks = chunk_documents(documents)
        all_chunks.extend(chunks)
    
    # Create embeddings and vector store
    embeddings = create_embeddings()
    vectordb = create_vectorstore(all_chunks, embeddings, persist_dir)
    
    return vectordb
```

### Creating a Document Index with Citations
```python
def create_document_index(pdf_directory):
    """Create an index of documents with metadata for citations."""
    document_index = {}
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        doc_title = os.path.splitext(pdf_file)[0]
        
        # Extract basic metadata
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Count pages
        page_count = len(documents)
        
        # Extract text for summarization
        full_text = " ".join([doc.page_content for doc in documents])
        
        # Store in index
        document_index[doc_title] = {
            "path": pdf_path,
            "page_count": page_count,
            "text_sample": full_text[:500],  # First 500 chars for preview
        }
    
    return document_index
```

### Building a Document Summarization Tool
```python
def summarize_document(llm, document_text, max_length=200):
    """Generate a concise summary of a document."""
    prompt = f"""
    Please provide a concise summary of the following academic text in about 200 words or less:
    
    {document_text[:10000]}  # Limit text length for LLM
    
    CONCISE SUMMARY:
    """
    
    summary = llm.invoke(prompt)
    return summary
```

### Document Browser Interface with Streamlit
```python
def document_browser_tab():
    """Streamlit interface for browsing documents in the knowledge base."""
    st.header("Document Knowledge Base")
    
    # Get document index
    pdf_directory = "pdfs"
    document_index = create_document_index(pdf_directory)
    
    # Display documents
    st.subheader("Available Documents")
    for doc_title, metadata in document_index.items():
        with st.expander(f"{doc_title} ({metadata['page_count']} pages)"):
            st.text(f"Path: {metadata['path']}")
            st.text("Text preview:")
            st.text(metadata['text_sample'] + "...")
            
            # Get summary button
            if st.button(f"Summarize {doc_title}"):
                with st.spinner("Generating summary..."):
                    # Load document text
                    loader = PyPDFLoader(metadata['path'])
                    documents = loader.load()
                    full_text = " ".join([doc.page_content for doc in documents])
                    
                    # Generate summary
                    llm = initialize_llm()
                    summary = summarize_document(llm, full_text)
                    st.markdown("### Summary")
                    st.write(summary)
```

---

## Part 3: Conversational Interface with Memory (20 minutes)

### Implementing Conversation Memory
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def create_conversational_rag(llm, retriever):
    """Create a conversational RAG system with memory."""
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    return conversation_chain
```

### Chat Interface with Streamlit
```python
def chat_interface_tab():
    """Streamlit interface for a conversational RAG system."""
    st.header("Research Assistant Chat")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize RAG system if not already done
    if "conversation_chain" not in st.session_state:
        # Check if we have a vector store
        if os.path.exists("db"):
            with st.spinner("Loading knowledge base..."):
                # Initialize embeddings and vectorstore
                embeddings = create_embeddings()
                vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
                
                # Set up retriever
                retriever = vectordb.as_retriever(search_kwargs={"k": 4})
                
                # Initialize LLM
                llm = initialize_llm()
                
                # Create conversational chain
                conversation_chain = create_conversational_rag(llm, retriever)
                
                st.session_state.conversation_chain = conversation_chain
                st.success("Research assistant ready!")
        else:
            st.warning("No knowledge base found. Please add documents in the Document Browser tab.")
            return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a research question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from conversational chain
        with st.spinner("Researching..."):
            response = st.session_state.conversation_chain.invoke(prompt)
            answer = response["answer"]
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
```

---

## Part 4: Building an Autonomous Research Agent (30 minutes)

### Defining Tools for the Research Agent
```python
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

def create_research_tools(llm, vectordb):
    """Create tools for the research agent."""
    # Document search tool
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Define the search tool
    search_tool = Tool(
        name="DocumentSearch",
        func=lambda q: retriever.get_relevant_documents(q),
        description="Useful for searching academic documents for specific information. Input should be a search query."
    )
    
    # Document summarization tool
    summarize_tool = Tool(
        name="Summarize",
        func=lambda text: summarize_document(llm, text),
        description="Summarizes a piece of text. Input should be the text to summarize."
    )
    
    # Question answering tool
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    qa_tool = Tool(
        name="QuestionAnswering",
        func=lambda q: qa_chain.run(q),
        description="Answers questions based on the academic documents. Input should be a question."
    )
    
    return [search_tool, summarize_tool, qa_tool]
```

### Building the Research Agent
```python
def create_research_agent(llm, tools):
    """Create an autonomous research agent."""
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent
```

### Research Tasks for the Agent
```python
def run_research_task(agent, task):
    """Run a research task using the agent."""
    # Example tasks:
    # "Summarize the key contributions of the paper"
    # "Extract the methodology section and explain it in simple terms"
    # "Compare how different papers approach the same problem"
    # "Generate three research questions that could extend this work"
    
    result = agent.run(task)
    return result
```

### Research Agent Interface
```python
def research_agent_tab():
    """Streamlit interface for the research agent."""
    st.header("Autonomous Research Agent")
    
    # Initialize the agent if not already done
    if "research_agent" not in st.session_state:
        if os.path.exists("db"):
            with st.spinner("Initializing research agent..."):
                # Load vectordb
                embeddings = create_embeddings()
                vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
                
                # Initialize LLM
                llm = initialize_llm(model_name="llama3")
                
                # Create tools and agent
                tools = create_research_tools(llm, vectordb)
                agent = create_research_agent(llm, tools)
                
                st.session_state.research_agent = agent
        else:
            st.warning("No knowledge base found. Please add documents first.")
            return
    
    # Research task templates
    task_templates = {
        "Summarize paper": "Summarize the key contributions and findings of the paper(s) in the knowledge base.",
        "Extract methodology": "Extract and explain the methodology section in simple terms.",
        "Generate research questions": "Generate three research questions that could extend this work.",
        "Compare approaches": "Compare how different papers in the knowledge base approach similar problems.",
        "Create literature review": "Create a brief literature review based on the papers in the knowledge base."
    }
    
    # Task selection
    selected_template = st.selectbox("Select a research task:", list(task_templates.keys()))
    
    # Custom task input
    task = st.text_area("Research Task:", value=task_templates[selected_template], height=100)
    
    # Run task
    if st.button("Run Research Task"):
        with st.spinner("Agent is working on your task..."):
            try:
                result = run_research_task(st.session_state.research_agent, task)
                st.markdown("### Research Results")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please try a simpler task or check your knowledge base.")
```

---

## Part 5: Evaluation and Metrics for RAG Systems (15 minutes)

### Implementing Basic Evaluation Metrics
```python
def evaluate_retrieval(retriever, test_questions, ground_truth):
    """Evaluate a retriever using precision and recall metrics."""
    results = {}
    
    for i, question in enumerate(test_questions):
        # Get retrieved documents
        retrieved_docs = retriever.get_relevant_documents(question)
        retrieved_content = [doc.page_content for doc in retrieved_docs]
        
        # Calculate metrics
        true_positives = sum(1 for content in retrieved_content if any(truth in content for truth in ground_truth[i]))
        
        # Precision: What fraction of retrieved documents are relevant
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        
        # Recall: What fraction of relevant documents are retrieved
        recall = true_positives / len(ground_truth[i]) if ground_truth[i] else 0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[question] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results
```

### Implementing Answer Correctness Evaluation
```python
def evaluate_answer_correctness(qa_chain, test_questions, ground_truth_answers):
    """Evaluate the correctness of answers using LLM judge."""
    llm_judge = initialize_llm("llama3")
    results = {}
    
    for i, question in enumerate(test_questions):
        # Get model answer
        model_answer = qa_chain.run(question)
        
        # Create evaluation prompt
        eval_prompt = f"""
        Question: {question}
        
        Ground Truth Answer: {ground_truth_answers[i]}
        
        Model Answer: {model_answer}
        
        Please evaluate the model's answer on a scale of 1-5, where:
        1: Completely incorrect or irrelevant
        2: Mostly incorrect but has some relevant information
        3: Partially correct with some errors
        4: Mostly correct with minor errors or omissions
        5: Completely correct and comprehensive
        
        Provide your rating and a brief explanation.
        """
        
        # Get evaluation from LLM judge
        evaluation = llm_judge.invoke(eval_prompt)
        
        results[question] = {
            "model_answer": model_answer,
            "ground_truth": ground_truth_answers[i],
            "evaluation": evaluation
        }
    
    return results
```

### Evaluation Interface
```python
def evaluation_tab():
    """Streamlit interface for RAG system evaluation."""
    st.header("RAG System Evaluation")
    
    # Sample evaluation questions
    sample_questions = [
        "What is the attention mechanism?",
        "How does the transformer architecture work?",
        "What are the limitations discussed in the paper?",
        "How does this compare to RNN-based approaches?",
        "What future work is suggested?"
    ]
    
    # Editable questions
    evaluation_questions = []
    for i in range(5):
        q = st.text_input(f"Evaluation Question {i+1}:", 
                         value=sample_questions[i] if i < len(sample_questions) else "")
        if q:
            evaluation_questions.append(q)
    
    # Ground truth answers (simplified)
    ground_truth = []
    for q in evaluation_questions:
        truth = st.text_area(f"Ground Truth for: {q}", height=100)
        if truth:
            ground_truth.append(truth)
    
    # Run evaluation
    if st.button("Run Evaluation") and len(evaluation_questions) == len(ground_truth):
        if "qa_chain" not in st.session_state:
            st.warning("Please set up the QA system first in the main tab.")
            return
        
        with st.spinner("Evaluating system..."):
            results = evaluate_answer_correctness(
                st.session_state.qa_chain, 
                evaluation_questions, 
                ground_truth
            )
            
            # Display results
            st.markdown("### Evaluation Results")
            for question, result in results.items():
                with st.expander(question):
                    st.write("**Model Answer:**")
                    st.write(result["model_answer"])
                    st.write("**Ground Truth:**")
                    st.write(result["ground_truth"])
                    st.write("**Evaluation:**")
                    st.write(result["evaluation"])
    elif st.button("Run Evaluation"):
        st.warning("Please provide ground truth answers for all questions.")
```

---

## Part 6: Machine Learning Connections and Advanced Concepts (15 minutes)

### Vector Similarity Metrics in Embeddings
```python
def visualize_embeddings(chunks, embeddings):
    """Visualize document embeddings using PCA."""
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import plotly.express as px
    
    # Get embeddings for chunks
    texts = [chunk.page_content for chunk in chunks]
    embedding_vectors = embeddings.embed_documents(texts)
    
    # Convert to numpy array
    embedding_array = np.array(embedding_vectors)
    
    # Reduce dimensions with PCA
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embedding_array)
    
    # Create a DataFrame for plotting
    import pandas as pd
    df = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2', 'PC3'])
    df['text'] = [text[:50] + "..." for text in texts]  # Truncate text for display
    df['page'] = [chunk.metadata.get('page', 'Unknown') for chunk in chunks]
    
    # Create 3D scatter plot
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', 
                        color='page', hover_data=['text'],
                        title='Document Embeddings Visualization')
    
    return fig
```

### Comparing Different Embedding Models
```python
def compare_embedding_models(chunks, model_names):
    """Compare different embedding models on the same text."""
    results = {}
    
    for model_name in model_names:
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Create a vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=f"db_{model_name.split('/')[-1]}"
        )
        
        # Set up retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        # Store in results
        results[model_name] = retriever
    
    return results
```

### ML Concepts Visualization Tab
```python
def ml_concepts_tab():
    """Streamlit interface for exploring ML concepts in RAG."""
    st.header("Machine Learning Concepts in RAG")
    
    st.subheader("Embedding Space Visualization")
    
    # Check if we have documents
    if os.path.exists("db"):
        # Load embeddings and chunks
        embeddings = create_embeddings()
        vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
        
        # Get documents for visualization
        docs = vectordb.get()
        doc_chunks = [Document(page_content=text, metadata={"page": page}) 
                     for text, page in zip(docs["documents"], docs["metadatas"])]
        
        # Limit to 100 chunks for visualization
        sample_chunks = doc_chunks[:100] if len(doc_chunks) > 100 else doc_chunks
        
        # Visualize embeddings
        fig = visualize_embeddings(sample_chunks, embeddings)
        st.plotly_chart(fig)
    else:
        st.warning("No document embeddings found. Please process documents first.")
    
    # Embedding models comparison
    st.subheader("Comparing Embedding Models")
    
    models = ["sentence-transformers/all-MiniLM-L6-v2", 
              "sentence-transformers/all-mpnet-base-v2",
              "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    
    selected_models = st.multiselect(
        "Select embedding models to compare:",
        models,
        default=["sentence-transformers/all-MiniLM-L6-v2"]
    )
    
    if st.button("Compare Selected Models") and selected_models:
        # Get sample documents
        if not os.path.exists("db"):
            st.warning("Please process documents first.")
            return
            
        vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
        docs = vectordb.get()
        doc_chunks = [Document(page_content=text, metadata={"page": page}) 
                     for text, page in zip(docs["documents"][:50], docs["metadatas"][:50])]
        
        # Compare models
        with st.spinner("Comparing embedding models..."):
            retrievers = compare_embedding_models(doc_chunks, selected_models)
            
            # Display comparison
            st.markdown("### Model Comparison Results")
            st.write("Try a test query to compare models:")
            
            test_query = st.text_input("Test Query:", value="What is attention in transformers?")
            
            if test_query:
                st.markdown("#### Retrieved Documents by Model")
                
                for model_name, retriever in retrievers.items():
                    with st.expander(f"Results from {model_name.split('/')[-1]}"):
                        docs = retriever.get_relevant_documents(test_query)
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Document {i+1}**")
                            st.write(doc.page_content[:300] + "...")
```

---

## Part 7: Building a Complete Streamlit App (25 minutes)

### Putting It All Together in a Multi-Tab Application
```python
import streamlit as st

def main():
    """Main Streamlit application for the advanced research assistant."""
    st.set_page_config(
        page_title="Advanced Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Advanced Research Assistant")
    st.write("A comprehensive tool for academic research using local LLMs")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3", "mistral", "deepseek-coder"]
    )
    
    # Embedding model selection
    embedding_model = st.sidebar.selectbox(
        "Select Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", 
         "sentence-transformers/all-mpnet-base-v2"]
    )
    
    # Tabs for different functionality
    tabs = st.tabs([
        "Document Processing", 
        "Research Q&A", 
        "Document Browser", 
        "Chat Interface", 
        "Research Agent",
        "Evaluation",
        "ML Concepts"
    ])
    
    # Document Processing Tab
    with tabs[0]:
        document_processing_tab()
    
    # Research Q&A Tab
    with tabs[1]:
        research_qa_tab()
    
    # Document Browser Tab
    with tabs[2]:
        document_browser_tab()
    
    # Chat Interface Tab
    with tabs[3]:
        chat_interface_tab()
    
    # Research Agent Tab
    with tabs[4]:
        research_agent_tab()
    
    # Evaluation Tab
    with tabs[5]:
        evaluation_tab()
    
    # ML Concepts Tab
    with tabs[6]:
        ml_concepts_tab()

def document_processing_tab():
    """Tab for processing documents and building the knowledge base."""
    st.header("Document Processing")
    
    # Upload multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload Research Papers", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, 10)
    
    with col2:
        embedding_strategy = st.selectbox(
            "Embedding Strategy",
            ["Default", "Paragraph-based", "Sliding Window"]
        )
        
    # Process button
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents..."):
            # Save uploaded files
            pdf_directory = "pdfs"
            os.makedirs(pdf_directory, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join(pdf_directory, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process all documents
            vectordb = process_multiple_documents(
                pdf_directory,
                persist_dir="db"
            )
            
            st.success(f"Successfully processed {len(uploaded_files)} documents!")
            
            # Show statistics
            st.subheader("Knowledge Base Statistics")
            st.write(f"Total documents: {len(uploaded_files)}")
            
            # Get total chunks
            total_chunks = len(vectordb.get()["documents"]) if hasattr(vectordb, "get") else "Unknown"
            st.write(f"Total chunks: {total_chunks}")
            
            # Document titles
            st.write("Documents in knowledge base:")
            for file in uploaded_files:
                st.write(f"- {file.name}")

def create_full_app():
    """Create and run the complete Streamlit application."""
    if __name__ == "__main__":
        main()

### Running the Complete Application
```bash
streamlit run advanced_research_assistant.py
```

### Adding User Authentication (Optional)
```python
import streamlit as st
import hashlib
import pickle
import os

def authenticate_user():
    """Simple authentication system for the research assistant."""
    # Initialize user database
    if not os.path.exists("users.pkl"):
        with open("users.pkl", "wb") as f:
            pickle.dump({}, f)
    
    # Load user database
    with open("users.pkl", "rb") as f:
        users = pickle.load(f)
    
    # Authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("Research Assistant Login")
        
        # Login/Register tabs
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username in users and users[username] == hashlib.sha256(password.encode()).hexdigest():
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Register"):
                if new_username in users:
                    st.error("Username already exists")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    users[new_username] = hashlib.sha256(new_password.encode()).hexdigest()
                    with open("users.pkl", "wb") as f:
                        pickle.dump(users, f)
                    st.success("Registration successful! Please login.")
        
        return False
    
    return True

# Modify main function to include authentication
def secure_main():
    if authenticate_user():
        main()

## Part 8: Advanced ML Techniques Integration (20 minutes)

### Implementing Advanced Reranking with Cross-Encoders
```python
from sentence_transformers import CrossEncoder

def rerank_with_cross_encoder(query, documents, top_k=3):
    """Rerank documents using a cross-encoder model."""
    # Initialize cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Prepare document pairs for reranking
    doc_pairs = [[query, doc.page_content] for doc in documents]
    
    # Get relevance scores
    scores = cross_encoder.predict(doc_pairs)
    
    # Sort documents by score
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k documents
    return [doc for doc, score in scored_docs[:top_k]]
```

### Using Topic Modeling to Organize Document Clusters
```python
from bertopic import BERTopic

def extract_topics_from_documents(documents):
    """Extract topics from document chunks using BERTopic."""
    # Extract text content
    texts = [doc.page_content for doc in documents]
    
    # Initialize BERTopic model
    topic_model = BERTopic()
    
    # Fit model and transform documents
    topics, probs = topic_model.fit_transform(texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Get topic representation for documents
    topic_docs = {}
    for i, (doc, topic) in enumerate(zip(documents, topics)):
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append((i, doc))
    
    return topic_model, topic_info, topic_docs
```

### Implementing Neural Information Extraction
```python
from transformers import pipeline

def extract_paper_entities(text):
    """Extract research-specific entities from paper text."""
    # Initialize NER pipeline
    ner = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")
    
    # Extract entities
    entities = ner(text)
    
    # Group entities
    grouped_entities = {}
    current_entity = None
    current_type = None
    
    for entity in entities:
        if entity["entity"].startswith("B-"):
            if current_entity:
                if current_type not in grouped_entities:
                    grouped_entities[current_type] = []
                grouped_entities[current_type].append(current_entity)
            
            current_entity = entity["word"]
            current_type = entity["entity"][2:]  # Remove "B-" prefix
        
        elif entity["entity"].startswith("I-") and current_entity:
            current_entity += " " + entity["word"]
    
    # Add the last entity
    if current_entity:
        if current_type not in grouped_entities:
            grouped_entities[current_type] = []
        grouped_entities[current_type].append(current_entity)
    
    return grouped_entities
```

### Advanced ML Interface Tab
```python
def advanced_ml_tab():
    """Streamlit interface for advanced ML features."""
    st.header("Advanced ML Features")
    
    # Topic modeling section
    st.subheader("Topic Modeling")
    
    if st.button("Extract Topics from Knowledge Base"):
        if not os.path.exists("db"):
            st.warning("No knowledge base found. Please process documents first.")
            return
            
        with st.spinner("Extracting topics..."):
            # Load documents
            embeddings = create_embeddings()
            vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
            docs = vectordb.get()
            doc_chunks = [Document(page_content=text, metadata={"page": page}) 
                         for text, page in zip(docs["documents"], docs["metadatas"])]
            
            # Extract topics
            topic_model, topic_info, topic_docs = extract_topics_from_documents(doc_chunks)
            
            # Display topics
            st.write("### Extracted Topics")
            for idx, row in topic_info.iterrows():
                if row["Topic"] != -1:  # Skip outlier topic
                    with st.expander(f"Topic {row['Topic']}: {row['Name']}"):
                        st.write(f"Count: {row['Count']} documents")
                        st.write("Top terms:")
                        st.write(", ".join(topic_model.get_topic(row['Topic'])[:10]))
                        
                        # Show sample documents
                        if row["Topic"] in topic_docs:
                            st.write("Sample documents:")
                            for i, (doc_idx, doc) in enumerate(topic_docs[row["Topic"]][:3]):
                                st.text(doc.page_content[:200] + "...")
    
    # Neural information extraction
    st.subheader("Information Extraction")
    
    sample_text = st.text_area(
        "Enter text to extract research entities:",
        height=200,
        placeholder="Paste a paragraph from a research paper..."
    )
    
    if st.button("Extract Entities") and sample_text:
        with st.spinner("Extracting entities..."):
            entities = extract_paper_entities(sample_text)
            
            st.write("### Extracted Entities")
            for entity_type, entity_list in entities.items():
                st.write(f"**{entity_type}**")
                for entity in entity_list:
                    st.write(f"- {entity}")
```

## Conclusion and Next Steps (5 minutes)

### Summary of What We've Learned
- We've built an advanced research assistant using local LLMs
- Implemented various RAG techniques for improved retrieval
- Added conversation memory for natural interaction
- Created an autonomous research agent
- Connected RAG to fundamental machine learning concepts
- Evaluated the system using quantitative metrics

### Potential Extensions
- **Fine-tuning**: Adapt the LLM to your specific research domain
- **Multi-modal**: Add support for images, tables, and charts in research papers
- **Collaborative features**: Allow multiple researchers to share annotations
- **Citation generation**: Automatically generate bibliographic references
- **Cross-lingual research**: Implement translation capabilities

### Resources for Further Learning
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [Ollama Model Library](https://ollama.com/library)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [RAG Research Papers](https://arxiv.org/abs/2005.11401)

---

## Appendix: Complete System Implementation

The complete code for this tutorial can be found in the accompanying GitHub repository. The repository includes:

1. Installation scripts for all dependencies
2. Jupyter notebooks with step-by-step implementations
3. The full Streamlit application
4. Sample PDFs for testing
5. Example evaluation datasets

Happy researching!