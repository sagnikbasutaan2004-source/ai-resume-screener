# AI Resume Screener

Command-line tool that ranks PDF resumes against a job description using semantic similarity (transformer embeddings).

## Features
- PDF text extraction with pdfplumber
- Text cleaning & lemmatization with NLTK
- Sentence embeddings with `all-MiniLM-L6-v2`
- Cosine similarity ranking

## Installation

```powershell
git clone https://github.com/YOURUSERNAME/ai-resume-screener.git
cd ai-resume-screener
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4