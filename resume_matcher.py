# resume_matcher.py
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

if __name__ == "__main__":
    print("Loading model (may take 1-2 min first time)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model ready.")

    job_description = """
    We are hiring a Data Scientist or Machine Learning Engineer with strong expertise in:
    - Python programming (pandas, numpy, scikit-learn)
    - Machine Learning algorithms (regression, classification, clustering, ensemble methods)
    - Deep Learning frameworks (PyTorch or TensorFlow)
    - Natural Language Processing (NLP), transformers, BERT, embeddings
    - Data preprocessing, feature engineering, model evaluation
    - Data visualization (Matplotlib, Seaborn, Tableau)
    - Experience with big data tools (Spark, SQL) and cloud platforms (AWS, GCP) is a plus
    - 3+ years of relevant professional experience preferred
    """

    job_clean = preprocess_text(job_description)
    job_embedding = model.encode(job_clean)

    # Process resumes
    resume_dir = "resumes"
    results = []

    if not os.path.exists(resume_dir):
        os.makedirs(resume_dir)
        print(f"Created '{resume_dir}' folder. Add PDF resumes and run again.")
    else:
        pdf_files = [f for f in os.listdir(resume_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print("No PDF files found in 'resumes' folder.")
        for filename in pdf_files:
            full_path = os.path.join(resume_dir, filename)
            raw_text = extract_text_from_pdf(full_path)
            if raw_text:
                clean_text = preprocess_text(raw_text)
                if clean_text:
                    emb = model.encode(clean_text)
                    score = cosine_similarity([job_embedding], [emb])[0][0]
                    preview = raw_text[:300].replace('\n', ' ').strip() + "..." if len(raw_text) > 300 else raw_text
                    results.append((filename, score, preview))
                else:
                    print(f"→ {filename}: No text after cleaning")

    # Show results
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\n" + "="*70)
        print("  Ranked Candidates - Similarity to Job Description")
        print("="*70 + "\n")
        
        for i, (name, score, preview) in enumerate(results, 1):
            print(f"{i}. {name}")
            print(f"   Score: {score:.4f}  ({score*100:.1f}%)")
            print(f"   Preview: {preview}\n")
    else:
        print("Nothing to rank yet — add PDFs to 'resumes' folder.")