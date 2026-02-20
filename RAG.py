# ================================
# COMPANY RAG DEMO (SINGLE FILE)
# ================================

import PyPDF2
import numpy as np
import google.generativeai as genai
from numpy.linalg import norm

# -------------------------------
# STEP 1: CONFIGURE API KEY
# -------------------------------
# Replace with your own API key
genai.configure(api_key="AIzaSyDUIa-OOuz9GaMsGxSFExnp_w4-TDNiQwU")

llm = genai.GenerativeModel("gemini-2.5-flash-lite")

# -------------------------------
# STEP 2: READ PDF FILES
# -------------------------------
def read_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

docs = []
docs.append(read_pdf("company_rules.pdf"))
docs.append(read_pdf("company_legal.pdf"))

# -------------------------------
# STEP 3: CHUNKING
# -------------------------------
def chunk_text(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

# -------------------------------
# STEP 4: CREATE EMBEDDINGS
# -------------------------------
def get_embedding(text):
    emb = genai.embed_content(
        model="gemini-embedding-001",
        content=text
    )
    return np.array(emb["embedding"])

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# -------------------------------
# STEP 5: SIMILARITY SEARCH
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def retrieve_context(question):
    q_emb = get_embedding(question)
    scores = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
    best_index = np.argmax(scores)
    return chunks[best_index]

# -------------------------------
# STEP 6: RAG QUESTION ANSWERING
# -------------------------------
print("\n🏢 Company AI Assistant (RAG Demo)")
print("Type 'exit' to quit\n")

while True:
    question = input("Ask a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    context = retrieve_context(question)

    prompt = f"""
Answer the question using ONLY the information below.
If the answer is not present, say "Information not available".

Information:
{context}

Question:
{question}
"""

    response = llm.generate_content(prompt)

    print("\nAI Assistant:")
    print(response.text)
    print("-" * 50)