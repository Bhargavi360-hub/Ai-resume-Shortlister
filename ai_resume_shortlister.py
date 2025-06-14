
# AI Resume Shortlister using Cosine Similarity
# Requirements: pandas, sklearn, spacy, PyMuPDF (fitz)

import fitz  # PyMuPDF
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load English tokenizer
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Load and preprocess job description
with open("job_description.txt", "r") as file:
    jd = preprocess(file.read())

# Extract and preprocess resumes
resume_dir = "resumes"
resume_texts = []
file_names = []

for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_dir, filename)
        text = extract_text_from_pdf(file_path)
        processed_text = preprocess(text)
        resume_texts.append(processed_text)
        file_names.append(filename)

# Compute similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([jd] + resume_texts)
similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# Rank resumes
results = pd.DataFrame({'Resume': file_names, 'Score': similarity})
results = results.sort_values(by='Score', ascending=False)
results.to_csv("shortlisted_resumes.csv", index=False)

print("Top resumes:")
print(results.head())
