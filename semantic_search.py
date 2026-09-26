import os 
import json
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from google.genai import types



load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file")

client = genai.Client(api_key=api_key)

data_folder = Path("data")
chunk_records = []

for file_path in sorted(data_folder.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    paragraphs = text.strip().split("\n\n")

    for paragraph_id, paragraph in enumerate(paragraphs, start=1):
        chunk_records.append({
            "text": paragraph, 
            "source": file_path.name, 
            "paragraph_id": paragraph_id
        })

for record in chunk_records:
    print(f"[{record['source']}:{record['paragraph_id']}]")
    print(record['text'])
    print()

print('Total chunks:', len(chunk_records))


raise SystemExit

test_cases = [
    {
        "question": "How many books can students borrow, and how long can they book study rooms?",
        "expected": "Four books and two hours, citing chunks 2 and 3"
    }, 
    {
        "question": "How long can students book study room?", 
        "expected": "Two hours, citing chunk 3"
    }, 
    {
        "question": "How much does it cost to book study rooms?",
        "expected": "Refusal without a citation: no price in the notes"
    }, 
    {
        "question": "What is the capital of France?", 
        "expected": "Refusal without a citation: no answer in the notes"
    }  

]
def cosine_similarity(vector_a, vector_b):
    dot_product = 0 

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b


    length_a = sum(a * a for a in vector_a) ** 0.5
    length_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (length_a * length_b)

embedding_model = "gemini-embedding-001"
cache_path = Path("data/embeddings.json")

cache_data = None

if cache_path.exists():
    cache_data = json.loads(
        cache_path.read_text(encoding="utf-8")
    )

if (
    cache_data is not None
    and cache_data["model"] == embedding_model
    and cache_data["chunks"] == chunks    
):
    chunk_embeddings = cache_data['embeddings']
    print("Loaded document embeddings from cache.")

else:
    chunk_response = client.models.embed_content(
        model=embedding_model,
        contents=chunks, 
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )

    chunk_embeddings = [
        embedding.values
        for embedding in chunk_response.embeddings
    ] 

    cache_data = {
        "model": embedding_model, 
        "chunks": chunks, 
        "embeddings": chunk_embeddings
    }

    cache_path.write_text(
        json.dumps(cache_data), 
        encoding='utf-8'
    )   
    print("Created and saved document embeddings")



for test_case in test_cases:
    question = test_case["question"]
    expected = test_case["expected"]

    print("\nQuestion", question)
    print("Expected", expected)

    question_response = client.models.embed_content(
        model=embedding_model, 
        contents=question, 
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )


    question_vector = question_response.embeddings[0].values

    print("Number of chunks:", len(chunks))
    print("Number of chunk embeddings:", len(chunk_embeddings))
    print("Question vector dimensions:", len(question_vector))

    scored_chunks = []

    for chunk_id, (chunk, chunk_vector) in enumerate(
        zip(chunks, chunk_embeddings), start=1
    ):
        score = cosine_similarity(question_vector, chunk_vector)

        print("Similarity:", score)
        print("Chunk:", chunk)
        print()

        scored_chunks.append((score, chunk_id, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    top_chunks = scored_chunks[:2]

    retrieved_ids = [
        chunk_id for score, chunk_id, chunk in top_chunks
    ]
    print("Retrieved IDs:", retrieved_ids)

    context = "\n\n".join(
        f"[Chunk: {chunk_id}]\n{chunk}"
        for score, chunk_id, chunk in top_chunks
    )

    print("Selected context:", context)


    prompt = f"""
    Answer the question using only the context below.
    If the context does not contain the answer, say "I dont know based on the provided context."
    Cite each factual claim using the supporting chunk label, such as [Chunk 2].
    Use only labels provided in the context.
    If you cannot answer from the context, give the refusal without a citation.

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

