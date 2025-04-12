Here’s a **slide deck outline** for the tutorial: **"Build Your Own Local AI Research Assistant with Ollama + RAG"**. I’ll list each slide title and content, and can generate a Google Slides or PowerPoint version on request.

---

### **Slide 1: Title Slide**
**Title:**  
*Build Your Own Local AI Research Assistant*  
**Subtitle:**  
*Using Ollama + LLaMA 3.1 + DeepSeek + RAG*

---

### **Slide 2: Learning Objectives**
- Understand Retrieval-Augmented Generation (RAG)
- Use Ollama with local LLMs (LLaMA 3.1, DeepSeek)
- Load and chunk academic PDFs
- Build a local Q&A assistant
- Evaluate model responses with real papers

---

### **Slide 3: What is RAG?**
- RAG = Retrieval + Generation
- Solves hallucination by grounding answers in real documents
- Useful for academic reading, legal texts, proprietary corpora

**Diagram:**  
[Retrieval → Context + Query → LLM → Answer]

---

### **Slide 4: Architecture Overview**
**Left:** PDF Loader → Text Splitter → Embeddings → Vector Store (Chroma)  
**Right:** User Query → Retriever → Ollama (LLM) → Answer

---

### **Slide 5: Setup Checklist**
- Install Ollama: `ollama pull llama3`, `ollama pull deepseek-coder`
- Install Python packages:
  ```bash
  pip install langchain chromadb pypdf sentence-transformers
  ```

---

### **Slide 6: Step 1 – Load and Chunk PDFs**
Code example:
```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("paper.pdf")
documents = loader.load()
```

---

### **Slide 7: Step 2 – Vectorize with ChromaDB**
```python
from langchain.vectorstores import Chroma
db = Chroma.from_documents(chunks, embedding)
```

---

### **Slide 8: Step 3 – Connect Ollama**
```python
from langchain.llms import Ollama
llm = Ollama(model="llama3")
```

---

### **Slide 9: Step 4 – Run the QA Chain**
```python
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
print(qa.run("What is self-attention?"))
```

---

### **Slide 10: Live Demo / Activity**
**Student Activity Instructions:**
- Choose a paper (provided or your own)
- Try at least 3 deep questions
- Compare LLaMA 3.1 vs DeepSeek

---

### **Slide 11: Analysis and Discussion**
- Did the model find the right information?
- Any hallucinations?
- What would improve the assistant?

---

### **Slide 12: Bonus Challenges**
- Add a UI (Streamlit)
- Summarize full paper
- Use multiple papers in one vector store

---

### **Slide 13: Summary**
- RAG boosts LLMs with accurate info
- Ollama enables fast local experimentation
- Great tool for academic Q&A, thesis reading, and paper review

---

Would you like a **PDF export**, a **Google Slides link**, or a **PowerPoint (.pptx)** file? I can generate one right away.