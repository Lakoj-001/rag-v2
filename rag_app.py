import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file")
print("API key loaded")

client = genai.Client(api_key=api_key)

documents = [
    "Melbourne is the capital of Victoria",
    "Canberra is the capital of Australia", 
    "Python is a programming language",
]

question = "What is the capital of france?"

question_words = set(question.lower().replace("?", "").split())

best_score = 0
best_document = None

for document in documents:
    document_words = set(document.lower().replace(".", "").split())
    shared_words = question_words & document_words
    score = len(shared_words)

    print(f"SCORE:{score}, SHARED WORDS:{shared_words}, DOCUMENT:{document}")

    

    if score > best_score:
        best_score = score 
        best_document = document

print("\n\n")
print("Retrieved document:", best_document)
print("Best score", best_score)

print("\n\n")
context = best_document

prompt = f"""

Answer the question using only the context below.
If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

print(prompt)

response = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)

print(response.output_text)
