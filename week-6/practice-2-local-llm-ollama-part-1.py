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


def create_llm(model_name="llama3.2:latest"):
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
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
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
    result = qa_chain.invoke({"query": question})

    return result["result"]


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

    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        [
            "llama3.2:latest",
            "qwen2.5:3b",
            "deepseek:7b",
            "deepseek-coder:6.7b",
            "deepseek-lite:1.3b",
            "llama3.1:8b",
            "mistral:7b",
            "phi3:3.8b",
            "gemma:2b",
        ],
    )

    # Initialize or load the system
    if "qa_chain" not in st.session_state:
        # Check if database exists
        if os.path.exists("./chroma_db"):
            with st.spinner("Loading existing knowledge base..."):
                # Load the existing vector database
                embeddings = create_embeddings(model_name=model_choice)
                vectordb = Chroma(
                    persist_directory="./chroma_db", embedding_function=embeddings
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
                        profile_path, "./chroma_db", model_name=model_choice
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
    if question:
        with st.spinner("Finding relevant information..."):
            answer = ask_question(st.session_state.qa_chain, question)

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


def main():
    """Main function to run the application."""
    # Change to the project directory if needed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the Streamlit app
    build_streamlit_app()


if __name__ == "__main__":
    main()
