import numpy as np

# Load the v3 index
corpus = np.load('src/rag_integration/company_corpus_chunks_3.npy', allow_pickle=True)
metadata = np.load('src/rag_integration/company_corpus_metadata_3.npy', allow_pickle=True)

print(f"Total chunks: {len(corpus)}\n")

# Find solution-related chunks
print("=" * 80)
print("CHUNKS CONTAINING 'solution' or 'vertical':")
print("=" * 80)
solution_chunks = []
for i, (chunk, meta) in enumerate(zip(corpus, metadata)):
    if 'solution' in chunk.lower() or 'vertical' in chunk.lower():
        solution_chunks.append((i, chunk, meta))
        print(f"\nChunk #{i}:")
        print(f"Context Type: {meta.get('context_type')}")
        print(f"JSON Path: {meta.get('json_path')}")
        print(f"Text: {chunk}")
        print("-" * 80)

print(f"\n\nFound {len(solution_chunks)} chunks with 'solution' or 'vertical'")

# Show first 10 chunks for overview
print("\n" + "=" * 80)
print("FIRST 10 CHUNKS (for overview):")
print("=" * 80)
for i in range(min(10, len(corpus))):
    print(f"\n#{i}: [{metadata[i].get('context_type')}] {corpus[i][:150]}...")
