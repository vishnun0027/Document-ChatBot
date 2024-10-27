from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve and set the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not found in the environment variables.")
os.environ["GROQ_API_KEY"] = groq_api_key

# Retrieve and set the Hugging Face API key
hf_api_key = os.getenv("HF_API_KEY")
if hf_api_key is None:
    raise ValueError("HF_API_KEY not found in the environment variables.")
os.environ["HF_API_KEY"] = hf_api_key

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Initialize the ChatGroq model with the specified model
llm = ChatGroq(model="llama3-8b-8192")

# Initialize the HuggingFaceInferenceAPIEmbeddings with the specified model
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)
