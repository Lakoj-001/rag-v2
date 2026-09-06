from pathlib import Path

file_path = Path("data/notes.txt")
text = file_path.read_text(encoding="utf-8")

print("Source:", file_path.name)

chunks = text.split("\n\n")

print("Number of chunks", len(chunks))

for number, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {number}")
    print(chunk)


question = "How long can students book study rooms?"
question_words = set(question.lower().replace("?", "").split())

best_score = 0 
best_chunk = None

for chunk in chunks:
    chunk_words = set(chunk.lower().replace(".", "").split())
    shared_words = question_words & chunk_words
    score = len(shared_words)

    print('\n')
    print(f"Score: {score}, Shared words: {shared_words}")


    if score > best_score:
        best_score = score 
        best_chunk = chunk


print("Selected chunk:", best_chunk)


