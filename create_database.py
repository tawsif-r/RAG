from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

# Custom Hugging Face Inference Embeddings class
class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, api_key=os.environ["HF_TOKEN"], model="intfloat/multilingual-e5-large", provider="hf-inference"):
        self.client = InferenceClient(
            provider=provider,
            api_key=api_key,
        )
        self.model = model

    def embed_documents(self, texts):
        try:
            embeddings = []
            for text in texts:
                result = self.client.feature_extraction(text, model=self.model)
                embeddings.append(result.tolist())  # Convert numpy array to list if needed
            return embeddings
        except Exception as e:
            raise Exception(f"Error embedding documents: {str(e)}")

    def embed_query(self, text):
        try:
            result = self.client.feature_extraction(text, model=self.model)
            return result.tolist()  # Convert numpy array to list if needed
        except Exception as e:
            raise Exception(f"Error embedding query: {str(e)}")

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if len(chunks) > 10:  # Ensure chunk 10 exists
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    else:
        print("Not enough chunks to display chunk 10.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents
    db = Chroma.from_documents(
        chunks,
        HuggingFaceInferenceEmbeddings(api_key=os.environ['HF_TOKEN']),
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()