
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
=======
# ai-resume-screener

AI-powered resume screening tool using sentence transformers and cosine similarity

## Features
- Extracts text from PDF resumes using pdfplumber
- Preprocesses text with NLTK (tokenization, stopwords, lemmatization)
- Generates embeddings with sentence-transformers (all-MiniLM-L6-v2)
- Ranks candidates using cosine similarity

## How to use
1. Place resume PDF files in the `resumes/` folder
2. Edit the `job_description` string in `resume_matcher.py`
3. Run the script:


