import os
import chromadb
import base64
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
# from IPython.display import Image, display, Markdown
# from datasets import load_dataset
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
    ("system", """You are a helpful Fashion and Styling assistant. Answer the user's question based on their gender and 
    using the given image context with direct references to parts of the images provided to frame appropriate response.
    Maintain a more conversational tone, don't make too many lists. 
    Use markdown formatting for highlights, emphasis, and structure."""),
    ("user", [
        {"type": "text" , "text" : " Gender : {gender}"},
        {"type": "text", "text": "What are some ideas for styling {user_query}"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
    ]),
])


vision_chain = image_prompt | llm | parser

def format_prompt_inputs(data, user_query, gender):
    inputs = {}


    inputs['user_query'] = user_query

    inputs['gender'] = gender


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

# print("AI Fashion Stylist","\n")
# print("Enter your Query : ")

# query = input("\n")


# # Running Retrieval and Generation
# results = query_db(query, results=2)
# prompt_input = format_prompt_inputs(results, query,gender="Male")
# response = vision_chain.invoke(prompt_input)

# print(response)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    gender: str

@app.post("/style-query/")
async def style_query(request: QueryRequest):
    try:
        # Running Retrieval and Generation
        results = query_db(request.query, results=2)
        prompt_input = format_prompt_inputs(results, request.query, gender=request.gender)
        response = vision_chain.invoke(prompt_input)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))