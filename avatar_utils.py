import concurrent.futures
import queue
from avatar.speech import synthesize
from socketio_setup import socketio
import re

def chunk_string(text, max_chunk_size=300):

    '''    
    # Use regex to split by words, while keeping the words intact.
    cleaned_text = re.sub(r'[^a-zA-Z.,\s]', '', text)
    words = re.findall(r'\S+\s*', cleaned_text)
    
    chunks = []
    current_chunk = ""
    
    for word in words:
        # If adding the next word exceeds the max_chunk_size, finalize the current chunk.
        if len(current_chunk) + len(word) > max_chunk_size:
            chunks.append(current_chunk.strip() + ".")
            current_chunk = word
        else:
            current_chunk += word
    
    # Add the last chunk if there's any leftover content
    if current_chunk:
        chunks.append(current_chunk.strip() + ".")
    
    return chunks'''
    
    # Clean the text to remove any unwanted characters, but keep periods, commas, and spaces.
    cleaned_text = re.sub(r'[^a-zA-Z.,\s]', '', text)
    
    # Split the text into sentences based on periods, but keep the periods with the sentences.
    sentences = re.split(r'(?<=\.)\s*', cleaned_text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding the next sentence exceeds the max_chunk_size, finalize the current chunk.
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    
    # Add the last chunk if there's any leftover content
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_chunks_with_limit(chunks, max_threads=2):
    total_chunks = len(chunks)
    results = [None] * total_chunks  # Pre-allocate for storing results
    chunk_index = 0  # Keep track of the current chunk index

    while chunk_index < total_chunks:
        # Process chunks in batches, limited by max_threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit tasks for this batch
            futures = {}
            for i in range(min(max_threads, total_chunks - chunk_index)):
                futures[executor.submit(synthesize, chunks[chunk_index], chunk_index)] = chunk_index
                chunk_index += 1

            # Collect results and store in the correct order
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    print(f"Chunk {index} generated an exception: {exc}")
    
        # Emit results in the order they were processed in the current batch
        for i in range(chunk_index - len(futures), chunk_index):
            if results[i] is not None:
                socketio.emit("video_chunk", results[i])
