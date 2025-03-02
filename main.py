import ollama
import requests
from bs4 import BeautifulSoup
import logging
import re
import time
import os
import json
from typing import List, Dict, Tuple, Optional, Union
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import faiss
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WebRAG22')

class WebRAG22:
    def __init__(self, model="custom-llama-model-v2", chunk_size=500, chunk_overlap=100, 
                 index_dir="./faiss_indexes", use_gpu=False):
        """
        Initialize the WebRAG22 system using FAISS for vector search.
        
        Args:
            model: The Ollama model to use for embeddings and generation
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            index_dir: Directory to store FAISS indexes and metadata
            use_gpu: Whether to use GPU acceleration for FAISS (if available)
        """
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_chunks = []
        self.url = None
        self.embedding_dimension = 4096  # Default for most Ollama models
        self.index_dir = index_dir
        self.use_gpu = use_gpu
        
        # Initialize NLTK for better text processing
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Initialize FAISS index and metadata storage
        self.initialize_faiss()
        
        # Cache for embeddings to avoid redundant computation
        self._embedding_cache = {}

    def initialize_faiss(self):
        """Initialize FAISS index for vector similarity search."""
        # Create a flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Use GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info(f"Using GPU acceleration for FAISS with {faiss.get_num_gpus()} GPUs")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        # Initialize metadata storage
        # This will store the mapping between index positions and text chunks
        self.chunk_metadata = {
            'chunks': [],     # List of text chunks
            'urls': [],       # URL for each chunk
            'positions': [],  # Position of chunk in original document
            'titles': []      # Page title for each chunk
        }
        
        logger.info("FAISS index initialized successfully")

    def fetch_webpage(self, url):
        """
        Fetches the HTML content of a webpage with error handling and retries.
        
        Args:
            url: The URL to fetch
            
        Returns:
            The HTML content of the webpage
        """
        self.url = url
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                logger.info(f"Successfully fetched webpage: {url}")
                return response.text
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch the webpage after {max_retries} attempts: {url}")
                    raise

    def process_html(self, html_content):
        """
        Extracts and processes text from HTML using BeautifulSoup with improved parsing.
        
        Args:
            html_content: The HTML content to process
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract title
            title = soup.title.string if soup.title else "Untitled Page"
            self.page_title = title
            
            # Remove unwanted elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.extract()
            
            # Extract text from main content areas with higher priority
            main_content = []
            for tag in ['main', 'article', 'section', 'div.content', 'div.main']:
                content_areas = soup.select(tag)
                if content_areas:
                    for area in content_areas:
                        main_content.append(area.get_text())
            
            # If no main content areas found, use the whole body
            if not main_content:
                main_content = [soup.get_text()]
            
            # Clean the text
            raw_text = " ".join(main_content)
            clean_text = self._clean_text(raw_text)
            
            # Create chunks with semantic boundaries
            self.text_chunks = self._create_semantic_chunks(clean_text)
            
            logger.info(f"Processed HTML into {len(self.text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing HTML: {e}")
            raise

    def _clean_text(self, text):
        """
        Cleans text by removing excess whitespace and unnecessary characters.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-breaking spaces and other problematic characters
        text = text.replace('\xa0', ' ')
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _create_semantic_chunks(self, text):
        """
        Creates semantically meaningful chunks from text, respecting sentence boundaries.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size and we already have content,
            # save the current chunk and start a new one
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Calculate overlap - keep some sentences from the end
                overlap_size = 0
                overlap_sentences = []
                
                # Add sentences from the end until we reach desired overlap
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                # Start new chunk with overlapping sentences
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def get_embedding(self, text):
        """
        Gets embedding with caching for efficiency.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        # Use hash of text as cache key
        cache_key = hash(text)
        
        # Check if embedding is already in cache
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # If not, generate the embedding
        embedding = self._generate_embedding(text)
        
        # Store in cache
        self._embedding_cache[cache_key] = embedding
        return embedding

    def _generate_embedding(self, text):
        """
        Actually generates the embedding with retries.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = ollama.embeddings(model=self.model, prompt=text)
                return response["embedding"]
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get embedding after {max_retries} attempts")
                    raise

    def generate_embeddings(self, batch_size=20):
        """
        Generates embeddings for text chunks and adds them to the FAISS index.
        
        Args:
            batch_size: Number of chunks to process in each batch
        """
        try:
            # Get current index size before adding new embeddings
            start_idx = self.index.ntotal
            total_chunks = len(self.text_chunks)
            
            # Process in batches
            for i in range(0, total_chunks, batch_size):
                batch = self.text_chunks[i:i+batch_size]
                batch_size_actual = len(batch)
                
                # Generate embeddings for the batch
                embeddings = []
                for chunk in batch:
                    embedding = self.get_embedding(chunk)
                    embeddings.append(embedding)
                
                # Convert to numpy array for FAISS
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Add to FAISS index
                self.index.add(embeddings_array)
                
                # Update metadata
                for j in range(batch_size_actual):
                    chunk_idx = i + j
                    position = chunk_idx
                    
                    self.chunk_metadata['chunks'].append(batch[j])
                    self.chunk_metadata['urls'].append(self.url)
                    self.chunk_metadata['positions'].append(position)
                    self.chunk_metadata['titles'].append(self.page_title)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
            
            # Calculate how many embeddings were added
            embeddings_added = self.index.ntotal - start_idx
            logger.info(f"Added {embeddings_added} embeddings to FAISS index")
            
            # Save the updated index and metadata
            self.save_index()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def save_index(self):
        """Save the FAISS index and chunk metadata to disk."""
        try:
            # Create a CPU version of the index if it's on GPU
            index_to_save = self.index
            if self.use_gpu and faiss.get_num_gpus() > 0:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            
            # Generate a filename based on the URL
            url_hash = abs(hash(self.url)) % 10000  # Simple hash for filename
            index_path = os.path.join(self.index_dir, f"index_{url_hash}.faiss")
            metadata_path = os.path.join(self.index_dir, f"metadata_{url_hash}.json")
            
            # Save the index
            faiss.write_index(index_to_save, index_path)
            
            # Create a simple serializable copy of the metadata
            # This avoids potential circular references that cause recursion errors
            serializable_metadata = {
                'chunks': self.chunk_metadata['chunks'],
                'urls': self.chunk_metadata['urls'],
                'positions': self.chunk_metadata['positions'],
                'titles': self.chunk_metadata['titles']
            }
            
            # Save the metadata as JSON instead of pickle to avoid recursion issues
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved FAISS index and metadata for {self.url}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            # Don't re-raise to avoid cascading errors

    def load_index(self, url):
        """
        Load a previously saved index for a URL.
        
        Args:
            url: The URL to load the index for
            
        Returns:
            True if index was loaded, False otherwise
        """
        try:
            url_hash = hash(url) % 10000
            index_path = os.path.join(self.index_dir, f"index_{url_hash}.faiss")
            metadata_path = os.path.join(self.index_dir, f"metadata_{url_hash}.pkl")
            
            # Check if index and metadata exist
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                logger.info(f"No existing index found for {url}")
                return False
            
            # Load the index
            self.index = faiss.read_index(index_path)
            
            # Move to GPU if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            
            # Load the metadata
            with open(metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            
            # Set the URL
            self.url = url
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors for {url}")
            return True
        except Exception as e:
            logger.error(f"Error loading index for {url}: {e}")
            return False

    def search_similar_text(self, query, top_k=5, rerank=True):
        """
        Finds the most relevant text chunks using FAISS search and optional reranking.
        
        Args:
            query: The search query
            top_k: Number of results to return
            rerank: Whether to rerank results using additional criteria
            
        Returns:
            List of tuples (chunk_text, relevance_score, metadata)
        """
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # First, retrieve top_k * 2 results (to have more candidates for reranking)
            initial_k = top_k * 2 if rerank else top_k
            
            # Convert the embedding to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search with FAISS
            distances, indices = self.index.search(query_array, initial_k)
            
            if len(indices[0]) == 0:
                logger.warning(f"No similar chunks found for query: {query}")
                return []
            
            # Convert distances to scores (smaller distance = higher similarity)
            scores = 1.0 / (1.0 + distances[0])
            
            # Extract chunk texts and scores
            chunks_with_scores = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunk_metadata['chunks']):
                    continue  # Skip invalid indices
                
                chunk_text = self.chunk_metadata['chunks'][idx]
                score = float(scores[i])
                metadata = {
                    'position': self.chunk_metadata['positions'][idx],
                    'url': self.chunk_metadata['urls'][idx],
                    'title': self.chunk_metadata['titles'][idx]
                }
                
                chunks_with_scores.append((chunk_text, score, metadata))
            
            # Apply reranking if requested
            if rerank:
                chunks_with_scores = self._rerank_results(query, chunks_with_scores)
                # Limit to top_k after reranking
                chunks_with_scores = chunks_with_scores[:top_k]
            
            logger.info(f"Found {len(chunks_with_scores)} relevant chunks for query: {query}")
            return chunks_with_scores
        except Exception as e:
            logger.error(f"Error searching similar text: {e}")
            return []

    def _rerank_results(self, query, chunks_with_scores):
        """
        Reranks search results using additional criteria beyond vector similarity.
        
        Args:
            query: The original query
            chunks_with_scores: List of (chunk_text, score, metadata) tuples
            
        Returns:
            Reranked list of (chunk_text, score, metadata) tuples
        """
        # Extract keywords from the query
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        # For each chunk, compute:
        # 1. Keyword match score
        # 2. Position score (earlier chunks might be more important)
        reranked = []
        
        for chunk_text, vector_score, metadata in chunks_with_scores:
            # Compute keyword match score (percentage of query keywords in the chunk)
            chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
            keyword_match_count = len(query_keywords.intersection(chunk_words))
            keyword_score = keyword_match_count / len(query_keywords) if query_keywords else 0
            
            # Position score (earlier chunks might be more important in some cases)
            position = metadata.get('position', 0)
            max_position = 100  # Assuming there won't be more than 100 chunks
            position_score = 1 - (position / max_position) if max_position > 0 else 0
            
            # Combined score (can adjust weights as needed)
            combined_score = (
                vector_score * 0.7 +  # Vector similarity is most important
                keyword_score * 0.2 +  # Keyword matching
                position_score * 0.1   # Position in document
            )
            
            reranked.append((chunk_text, combined_score, metadata))
        
        # Sort by combined score in descending order
        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def ask_question(self, query):
        """
        Answers a question by retrieving relevant chunks and generating a response.
        
        Args:
            query: The question to answer
            
        Returns:
            Generated answer based on the retrieved chunks
        """
        # Retrieve relevant chunks
        relevant_chunks = self.search_similar_text(query, top_k=3, rerank=True)
        
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Extract just the text from the retrieved chunks
        context_texts = [chunk[0] for chunk in relevant_chunks]
        context = "\n\n".join(context_texts)
        
        # Generate a response using the context
        prompt = f"""
        Based on the following information, please answer the question.
        
        Question: {query}
        
        Information:
        {context}
        
        Answer:
        """
        
        try:
            # Generate answer using Ollama
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            generated_answer = response['message']['content'].strip()
            return generated_answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to just returning the most relevant chunk
            return f"I encountered an issue generating a response, but here's the most relevant information I found:\n\n{context_texts[0]}"


if __name__ == "__main__":
    import sys
    
    # Accept URL as command line argument or prompt
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter a webpage URL: ")
    
    try:
        # Initialize the QA system
        # Set use_gpu=True if you have a GPU with CUDA support
        qa_system = WebRAG22(use_gpu=False)
        
        # Check if there's an existing index for this URL
        if not qa_system.load_index(url):
            print("No existing index found. Creating a new one...")
            
            print("Fetching webpage...")
            html_content = qa_system.fetch_webpage(url)
            
            print("Processing HTML...")
            qa_system.process_html(html_content)
            
            print("Generating embeddings and storing in FAISS...")
            qa_system.generate_embeddings()
        else:
            print(f"Loaded existing index for {url}")
        
        print("\nYou can now ask questions about this webpage.")
        
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
                
            print("\nSearching for answer...")
            answer = qa_system.ask_question(query)
            print("\nAnswer:\n", answer)
            
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"Program terminated with error: {e}")
