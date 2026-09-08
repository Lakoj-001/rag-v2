import os 
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file")

client = genai.Client(api_key=api_key)

file_path = Path("data/notes.txt")
text = file_path.read_text(encoding="utf-8")
chunks = text.split("\n\n")

question = "How many books can students borrow, and how long can they book study rooms?"

def cosine_similarity(vector_a, vector_b):
    dot_product = 0 

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b


    length_a = sum(a * a for a in vector_a) ** 0.5
    length_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (length_a * length_b)


chunk_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks, 
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)

chunk_embeddings = chunk_response.embeddings

question_response = client.models.embed_content(
    model="gemini-embedding-001", 
    contents=question, 
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)


question_vector = question_response.embeddings[0].values

print("Number of chunks:", len(chunks))
print("Number of chunk embeddings:", len(chunk_embeddings))
print("Question vector dimensions:", len(question_vector))

best_score = float("-inf")
best_chunk = None 

for chunk, embedding in zip(chunks, chunk_embeddings):
    chunk_vector = embedding.values
    score = cosine_similarity(question_vector, chunk_vector)

    print("Similarity:", score)
    print("Chunk:", chunk)
    print()

    if score > best_score:
        best_score = score
        best_chunk = chunk

print("Selected chunk:", best_chunk)
print("Best similarity:", best_score)


context = best_chunk 
prompt = f"""
Answer the question using only the context below.
If the context does not contain the answer, say "I dont know based on the provided context."

Context: 
{context}

Question:
{question}

"""

print(prompt)

answer_response = client.interactions.create(
    model="gemini-3.8-flash", 
    input=prompt
)

print("Answer:", answer_response.output_text)










