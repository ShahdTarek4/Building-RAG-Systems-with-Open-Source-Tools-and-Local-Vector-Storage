# Building-RAG-Systems-with-Open-Source-Tools-and-Local-Vector-Storage

This project implements a full Retrieval-Augmented Generation (RAG) pipeline using open-source tools and local vector storage. 

---

## 📦 1. Setup Instructions

### ✅ Dependencies
- Python 3.10+
- Install all required libraries:
```bash
pip install -r requirements.txt
```

### ✅ Environment Variables
Create a `.env` file with the following keys:
```
GROQ_API_KEY=your_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=model
```

### ✅ Running the System
1. Place your documents in a `documents/` folder.
2. Run all modules using `unified_test.py`, or test parts separately:
```bash
python unified_test.py
```

---

## 🧠 2. System Architecture

### 📂 Modules:
- `pipeline.py`: Loads documents, splits them, and embeds using FAISS.
- `retrieval.py`: Implements both basic and MMR retrieval.
- `rag.py`: Combines retrieval with LLM-based generation using prompt templates and query rewriting.
- `evaluation.py`: Calculates precision, recall, F1 for retrieval and keyword-match scoring for answers.
- `config_test.py`: Tests different configurations (prompt, model, rewritten query) and compares results.

### 🔁 RAG Flow:
1. Load & preprocess documents ➝
2. Create vector embeddings ➝
3. Perform similarity search ➝
4. Construct prompt ➝
5. Generate answer with LLM

---

## 🔍 3. Retrieval Strategy Results

We implemented:
- **Basic similarity search** using `similarity_search()`
- **Advanced retrieval** with **Maximum Marginal Relevance (MMR)** using `max_marginal_relevance_search()`

**Observed Differences**:
- Basic retrieval retrieved topically similar chunks
- MMR retrieved more **diverse but still relevant** results, helpful for broad queries

---

## 📊 4. Evaluation Metrics

### Answer Quality (Keyword Match Score):
| Configuration              | Score (0–1) |
|---------------------------|-------------|
| Default                   | 0.67        |
| Alternative Prompt        | 0.75        |
| Second Embedding Model    | 0.58        |
| Query Rewriting (LLM)     | 0.83        |

### Retrieval Performance (Example):
| Metric     | Value |
|------------|-------|
| Precision  | 0.67  |
| Recall     | 0.67  |
| F1 Score   | 0.67  |

---

## ⚖️ 5. Strengths & Weaknesses

### ✅ Strengths:
- Modular design: Easily test new prompts/models
- Real document metadata used in evaluation
- Bonus features implemented (query rewriting, prompt templates)

### ⚠️ Challenges:
- Manual specification of relevant docs still needed for precision/recall
- Answer quality scoring is keyword-based (not semantic)
|

---

## 📁 7. Document Corpus

All documents used are located in the `/documents` folder. If submitting code without documents, use your own `.pdf`, `.docx`, or `.txt` files on topics such as AI, ethics, and machine learning.

---

## ⚙️ 8. Configurations & Saved Models

### Vector Stores Created:
- `faiss_store/` — using `all-MiniLM-L6-v2`
- `faiss_store_l12/` — using `paraphrase-MiniLM-L12-v2`

You can switch between these using the `model_name` and `store_path` parameters in `run_rag()`.

---
