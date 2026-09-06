documents = [
    "Melbourne is the capital of Victoria",
    "Canberra is the capital of Australia", 
    "Python is a programming language",
]

question = "What is the capital of Australia?"

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

Context:
{context}

Question:
{question}

Answer:
"""

print(prompt)

if context is None:
    answer = "I couldn't find relevant information"
else:
    answer = context

print("Simulated answer:", answer)


