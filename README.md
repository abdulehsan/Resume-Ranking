# Resume-Ranking-Using-BM25-BERT-word2vec

# 🧠 Resume Ranking with BM25, Word2Vec, BERT & Groq

This project is a powerful Streamlit-based web app that allows recruiters and HR professionals to **rank resumes against a job description** using multiple AI/IR models like **BM25**, **Word2Vec**, and **BERT** — with optional **Groq-powered LLM evaluation** for deeper reasoning.

## 🚀 Features

- 📄 Upload multiple resumes (PDF, DOCX, TXT)
- 📝 Paste a job description
- ⚙️ Choose ranking algorithm(s): `BM25`, `Word2Vec`, `BERT`
- 📊 View ranked results with similarity scores
- 🤖 Optional LLM evaluation with **Groq**: classifies resumes as Good/Bad fits and explains why
- Evaluate with LLM(gemini used)
---

## 💡 Algorithms

- **BM25**: Probabilistic IR model for ranking based on token overlap and frequency.
- **Word2Vec**: Uses average word embeddings for each resume and compares it with the job vector.
- **BERT**: Transformer-based deep semantic comparison.
- **Groq (LLM)**: Post-ranking layer using an LLM to reason about the fit.

---

## 📦 Dependencies

Make sure the following modules are available in your environment:

- `streamlit`
- `sklearn`
- `transformers` *(for BERT model, optional)*
- `groq` *(or whatever backend you're using for explanation)*
- Custom modules:
  - `extraction.py`
  - `applyprocessing.py`
  - `querypre.py`
  - `bma.py`
  - `word2vec.py`
  - `bert.py`
  - `explainwithllm.py`

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
