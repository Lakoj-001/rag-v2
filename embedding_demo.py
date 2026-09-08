import os 
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file")

client = genai.Client(api_key=api_key)


sentences = [
    "What is the borrowing limit?", 
    "Students can borrow up to four books at a time.", 
    "The cafe serves hot coffee"
]


response = client.models.embed_content(
    model="gemini-embedding-001", 
    contents=sentences
)

embeddings = response.embeddings

print("Number of embeddings:", len(embeddings))

first_vector = embeddings[0].values

print("First vector length:", len(first_vector))
print("First five numbers:", first_vector[:5])


def cosine_similarity(vector_a, vector_b):
    dot_product = 0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b


    length_a = sum(a * a for a in vector_a) ** 0.5
    length_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (length_a * length_b)


question_vector = embeddings[0].values
borrowing_vector = embeddings[1].values
coffee_vector = embeddings[2].values

borrowing_score = cosine_similarity(question_vector, borrowing_vector)
coffee_score = cosine_similarity(question_vector, coffee_vector)

print("Borrowing similarity:", borrowing_score)
print("Coffee similarity:", coffee_score)
































