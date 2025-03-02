import ollama
import requests
from bs4 import BeautifulSoup
import psycopg2
from pgvector.psycopg2 import register_vector

class WebpageQA:
    def __init__(self, model="crewai-llama-model", db_params=None):
        self.model = model
        self.text_data = []
        self.embeddings = []
        self.db_params = db_params or {
            'dbname': 'postgres',
            'user': 'postgres',
            'password': '123456',
            'host': 'localhost',
            'port': '5432'
        }
        self.conn = None
        self.cursor = None
        self._connect_db()
        self._setup_table()

    def _connect_db(self):
        """Connects to the PostgreSQL database and registers pgvector."""
        self.conn = psycopg2.connect(**self.db_params)
        self.cursor = self.conn.cursor()
        register_vector(self.conn)

    def _setup_table(self):
        """Sets up the table for storing embeddings."""
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS webpage_embeddings (
                id SERIAL PRIMARY KEY,
                text TEXT,
                embedding VECTOR(4096)  -- Adjust dimension based on your embeddings
            )
        """)
        self.conn.commit()

    def fetch_webpage(self, url):
        """Fetches the HTML content of a webpage."""
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch the webpage: {url}")

    def process_html(self, html_content):
        """Extracts text from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()  # Remove script and style elements
        self.text_data = [text.strip() for text in soup.get_text().split("\n") if text.strip()]

    def generate_embeddings(self):
        """Generates embeddings using Ollama and stores them in PostgreSQL."""
        for text in self.text_data:
            embedding = self.get_embedding(text)
            self.embeddings.append(embedding)
            self.cursor.execute(
                "INSERT INTO webpage_embeddings (text, embedding) VALUES (%s, %s)",
                (text, embedding)
            )
        self.conn.commit()

    def get_embedding(self, text):
        """Fetches embedding from Ollama."""
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def search_similar_text(self, query, top_k=2):
        """Finds the most relevant text snippets using pgvector."""
        query_embedding = self.get_embedding(query)
        # Convert the embedding to a string format
        query_embedding_str = f'[{",".join(map(str, query_embedding))}]'
        self.cursor.execute(
            "SELECT text FROM webpage_embeddings ORDER BY embedding <-> %s::vector LIMIT %s",
            (query_embedding_str, top_k)
        )
        results = self.cursor.fetchall()
        return [result[0] for result in results]

    def ask_question(self, query):
        """Returns the most relevant text snippets for a given question."""
        results = self.search_similar_text(query)
        return "\n".join(results)

    def __del__(self):
        """Closes the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    url = input("Enter a webpage URL: ")
    qa_system = WebpageQA(db_params={
        'dbname': 'postgres',
        'user': 'postgres',
        'password': '123456',
        'host': 'localhost',
        'port': '5432'
    })

    print("Fetching webpage...")
    html_content = qa_system.fetch_webpage(url)

    print("Processing HTML...")
    qa_system.process_html(html_content)

    print("Generating embeddings and storing in PostgreSQL...")
    qa_system.generate_embeddings()

    while True:
        query = input("Ask a question about the page (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = qa_system.ask_question(query)
        print("\nAnswer:\n", answer, "\n")
