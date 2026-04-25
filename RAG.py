import PyPDF2
import numpy as np
import google.generativeai as genai
from numpy.linalg import norm

#configure API Key
genai.configure(api_key = "AIzaSyCU7hg7Lsx5K6h3M6aF3Lx3yxdhMoRiqCM")
llm = genai.GenerativeModel("gemini-2.5-flash-lite")

#Read PDF Files
def read_pdf(path):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

docs = []
docs.append(read_pdf("SRMCEM_Lucknow_Overview_Updated.pdf"))
# docs.append(read_pdf("company_legal.pdf"))

#Chunking
def chunk_text(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

#Embedding
def get_embedding(text):
    emb = genai.embed_content(
        model = 'gemini-embedding-001',
        content = text
    )
    return np.array(emb["embedding"])

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

#Similarity Search
def cosine_similarity(a, b):
    return np.dot(a, b)/(norm(a) * norm(b))

def retrieve_context(question):
    q_emb = get_embedding(question)
    scores = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
    best_index = np.argmax(scores)
    return chunks[best_index]

#RAG Question Answer
print("\n Company AI Assistant (RAG Demo)")
print("Type 'exit' to quit")

while True:
    question = input("Ask a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    context = retrieve_context(question)

    prompt = f"""
Answer the question using ONLY the information below
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

