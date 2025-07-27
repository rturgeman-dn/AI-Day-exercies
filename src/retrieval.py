import wikipedia
import os
from openai import OpenAI
import faiss
import numpy as np
from typing import List
from dotenv import load_dotenv
import re


def get_kong_client():
    """
    Create an OpenAI client configured to use Kong API Gateway.
    
    Returns:
        OpenAI client configured for Kong
    """
    load_dotenv()
    
    kong_token = os.getenv("KONG_API_TOKEN")
    kong_base_url = os.getenv("KONG_BASE_URL")
    
    if not kong_token or not kong_base_url:
        raise ValueError("Kong configuration missing. Please set KONG_API_TOKEN and KONG_BASE_URL in .env file")
    
    client = OpenAI(
        api_key=kong_token,
        base_url=kong_base_url,
        default_headers={"apikey": kong_token}
    )
    
    return client


def get_wikipedia_chunks(query: str, max_chunks: int = 10) -> List[str]:
    """
    Fetch Wikipedia article content and split it into chunks.
    
    Args:
        query: Search term for Wikipedia
        max_chunks: Maximum number of chunks to return
        
    Returns:
        List of text chunks, each approximately 800 characters
    """
    try:
        # Search for the most relevant Wikipedia page
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return []
        
        # Get the page content
        page = wikipedia.page(search_results[0])
        content = page.content
        
        # Clean the content by removing excessive whitespace and special characters
        content = re.sub(r'\n+', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Split content into chunks of approximately 800 characters
        chunks = []
        chunk_size = 800
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            
            # Try to end chunks at sentence boundaries when possible
            if i + chunk_size < len(content):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.7:  # Only if period is in latter part of chunk
                    chunk = chunk[:last_period + 1]
            
            if chunk.strip():
                chunks.append(chunk.strip())
                
            if len(chunks) >= max_chunks:
                break
                
        return chunks
        
    except wikipedia.exceptions.DisambiguationError as e:
        # If multiple pages match, use the first suggestion
        try:
            page = wikipedia.page(e.options[0])
            content = page.content
            return get_wikipedia_chunks(content, max_chunks)
        except:
            return []
    except:
        return []


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Generate embeddings for text chunks using OpenAI's API.
    
    Args:
        chunks: List of text chunks to embed
        
    Returns:
        NumPy array of shape (num_chunks, 1536) with dtype float32
    """
    if not chunks:
        return np.array([], dtype=np.float32).reshape(0, 1536)
    
    embeddings = []
    
    client = get_kong_client()
    
    for chunk in chunks:
        try:
            # Get embedding using OpenAI's text-embedding-ada-002 model via Kong
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            # If embedding fails for a chunk, use zero vector
            print(f"Failed to embed chunk: {e}")
            embeddings.append([0.0] * 1536)
    
    # Convert to numpy array with the correct shape and dtype
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array


def retrieve_relevant_chunks(question: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> List[str]:
    """
    Retrieve the most relevant chunks for a given question using FAISS similarity search.
    
    Args:
        question: User's question to search against
        chunks: Original text chunks
        embeddings: Pre-computed embeddings for the chunks
        top_k: Number of top chunks to retrieve
        
    Returns:
        List of the most relevant text chunks
    """
    if not chunks or embeddings.shape[0] == 0:
        return []
    
    try:
        # Embed the question using Kong client
        client = get_kong_client()
        response = client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        )
        question_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to the index
        index.add(embeddings)
        
        # Search for similar chunks
        k = min(top_k, len(chunks))
        distances, indices = index.search(question_embedding, k)
        
        # Return the corresponding chunks
        relevant_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                relevant_chunks.append(chunks[idx])
        
        return relevant_chunks
        
    except Exception as e:
        print(f"Failed to retrieve relevant chunks: {e}")
        # Return first few chunks as fallback
        return chunks[:top_k]
