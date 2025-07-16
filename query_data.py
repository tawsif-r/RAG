import argparse
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

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

# Custom Hugging Face Chat class
class HuggingFaceChat:
    def __init__(self, api_key=os.environ["HF_TOKEN"], model="deepseek-ai/DeepSeek-R1-0528", provider="fireworks-ai"):
        self.client = InferenceClient(
            provider=provider,
            api_key=api_key,
        )
        self.model = model

    def predict(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500  # Adjust as needed
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB
    embedding_function = HuggingFaceInferenceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Generate response using Hugging Face chat model
    model = HuggingFaceChat()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()