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

   ```powershell
   python resume_matcher.py
