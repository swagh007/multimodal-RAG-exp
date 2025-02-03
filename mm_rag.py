import os
import chromadb
import base64
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from IPython.display import Image, display, Markdown
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Instantiate the ChromaDB CLient
chroma_client = chromadb.PersistentClient(path="./image_vdb")
# Instantiate the ChromaDB Image Loader
image_loader = ImageLoader()
# Instantiate CLIP embeddings
CLIP = OpenCLIPEmbeddingFunction()

# Create the image vector database
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function = CLIP, data_loader = image_loader)

def query_db(query, results):
    results = image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])
    return results


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-002",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_output_tokens=2048
)

parser = StrOutputParser()


image_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful fashion and styling assistant. Answer the user's question 
    using the given image context with direct references to parts of the images provided.
    Maintain a more conversational tone, don't make too many lists. 
    Use markdown formatting for highlights, emphasis, and structure."""),
    ("user", [
        {"type": "text", "text": "What are some ideas for styling {user_query}"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
    ]),
])


vision_chain = image_prompt | llm | parser

def format_prompt_inputs(data, user_query):
    inputs = {}


    inputs['user_query'] = user_query


    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]
    
    # Encode the first image
    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')
    
    # Encode the second image
    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')
    
    return inputs

display(Markdown("## FashionRAG is At Your Service!"))
display(Markdown("What would you like to style today?"))

query = input("\n")

# Running Retrieval and Generation
results = query_db(query, results=2)
prompt_input = format_prompt_inputs(results, query)
response = vision_chain.invoke(prompt_input)

print(response)

