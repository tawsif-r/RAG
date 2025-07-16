from langchain.evaluation import load_evaluator
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

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
    # Get embedding for a word
    embedding_function = HuggingFaceInferenceEmbeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(
        prediction=words[0],
        prediction_b=words[1],
        embedding=embedding_function  # Pass the custom embedding function
    )
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

if __name__ == "__main__":
    main()