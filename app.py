import streamlit as st
import spacy
import nltk
import pdfplumber
import docx
import subprocess
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from cryptography.fernet import Fernet

# Ensure spaCy model is installed
nlp = spacy.load("en_core_web_sm")

# Load NLP models
nlp = ensure_spacy_model()
nltk.download('punkt')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def encrypt_text(text, key):
    cipher = Fernet(key)
    return cipher.encrypt(text.encode())

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return " ".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].numpy()

def calculate_similarity(job_desc, resumes):
    job_embedding = bert_embedding(job_desc)
    resume_embeddings = [bert_embedding(res) for res in resumes]
    scores = [cosine_similarity(job_embedding, res_emb)[0][0] for res_emb in resume_embeddings]
    return scores

def extract_candidate_details(text):
    doc = nlp(text)
    entities = {"Name": [], "Education": [], "Skills": [], "Experience": []}
    for ent in doc.ents:
        if ent.label_ in ["PERSON"]:
            entities["Name"].append(ent.text)
        elif ent.label_ in ["ORG", "EDUCATION"]:
            entities["Education"].append(ent.text)
        elif ent.label_ in ["SKILL", "ABILITY"]:
            entities["Skills"].append(ent.text)
        elif ent.label_ in ["DATE", "TIME"]:
            entities["Experience"].append(ent.text)
    return entities

def main():
    st.set_page_config(page_title="AI Resume Screening", layout="wide")
    st.title("AI-powered Resume Screening & Ranking System")
    
    st.subheader("Enter Job Description")
    job_desc = st.text_area("Paste the job description here")
    
    st.subheader("Upload Resumes (PDF/DOCX format)")
    uploaded_files = st.file_uploader("Upload multiple resumes", accept_multiple_files=True, type=["pdf", "docx"])
    
    if st.button("Rank Candidates"):
        if job_desc and uploaded_files:
            processed_job_desc = preprocess_text(job_desc)
            resume_texts, candidate_details, candidate_names = [], [], []
            
            for file in uploaded_files:
                if file.type == "application/pdf":
                    text = extract_text_from_pdf(file)
                else:
                    text = extract_text_from_docx(file)
                
                if text:
                    processed_text = preprocess_text(text)
                    resume_texts.append(processed_text)
                    candidate_names.append(file.name)
                    candidate_details.append(extract_candidate_details(text))
            
            if resume_texts:
                scores = calculate_similarity(processed_job_desc, resume_texts)
                ranked_candidates = sorted(zip(candidate_names, scores, candidate_details), key=lambda x: x[1], reverse=True)
                
                st.subheader("Candidate Ranking")
                for i, (name, score, details) in enumerate(ranked_candidates, 1):
                    st.write(f"{i}. {name} - Score: {score:.2f}")
                    st.write(f"**Extracted Details:** {details}")
                    
                st.subheader("Data Visualization")
                df = pd.DataFrame(ranked_candidates, columns=["Candidate", "Score", "Details"])
                fig = px.bar(df, x="Candidate", y="Score", title="Resume Similarity Scores", color="Score")
                st.plotly_chart(fig)
                
        else:
            st.warning("Please provide both job description and resumes.")

if __name__ == "__main__":
    main()
