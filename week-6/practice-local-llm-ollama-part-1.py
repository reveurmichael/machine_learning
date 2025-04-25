import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st


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


def create_llm(model_name="qwen2.5:3b"):
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
        model=model_name,  # Model we downloaded earlier
        temperature=0.1,  # Low temperature for more factual responses
        num_ctx=4096,  # Context window size
        num_predict=1024,  # Maximum tokens to generate
        repeat_penalty=1.1,  # Discourage repetition
    )
    print("LLM connection established")
    return llm


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

    # Get available models from Ollama
    available_models = []
    try:
        import subprocess

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Skip the header line
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        available_models.append(model_name)
        if not available_models:
            available_models = ["qwen2.5:3b"]
    except Exception as e:
        st.sidebar.error(f"Error getting models: {e}")
        available_models = ["qwen2.5:3b"]

    model_choice = st.sidebar.selectbox("Select LLM Model", available_models)

    # Get model-specific database directory
    db_directory = f"./chroma_db_{model_choice.replace(':', '_')}"

    # Option to rebuild the database
    if st.sidebar.button("🔄 Rebuild Knowledge Base"):
        # Delete the existing database
        with st.spinner(f"Rebuilding knowledge base for model {model_choice}..."):
            import shutil
            import time

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
                profile_path = "profiles/student_database.md"
                vectordb = process_documents(
                    profile_path, db_directory, model_name=model_choice
                )

                # Create LLM and QA chain
                llm = create_llm(model_name=model_choice)
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
                    "Try restarting the application or check file permissions"
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
                llm = create_llm(model_name=model_choice)
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

    for example in examples:
        if st.sidebar.button(example):
            st.session_state.question = example

    # Input area for questions
    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_input(
        "Your question:", value=st.session_state.question, key="question_input"
    )

    # Generate answer when question is submitted
    if question and "qa_chain" in st.session_state:
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


def main():
    """Main function to run the application."""
    # Change to the project directory if needed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the Streamlit app
    build_streamlit_app()


if __name__ == "__main__":
    main()
