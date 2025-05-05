from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Make sure COHERE_API_KEY is set in your environment
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("Please set COHERE_API_KEY environment variable")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React runs on port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# Make sure COHERE_API_KEY is set in your environment
# os.environ["COHERE_API_KEY"] = os.getenv("CO_API_KEY")

def setup_qa_pipeline(vector_store):
    """Set up the question-answering pipeline."""
    # Set up the LLM
    llm = Cohere(
        model="command-r-plus",
        temperature=0.7
    )
    
    # Create prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def query_document(qa_chain, question: str) -> dict:
    """Query the document with a question."""
    try:
        result = qa_chain({"query": question})
        return {
            "question": question,
            "answer": result['result'],
            # "source": result['source_documents'][0].page_content if result['source_documents'] else None
        }
    except Exception as e:
        return {"error": str(e)}

# Initialize the pipeline once when starting the server
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PGVector(
    connection_string="postgresql://barathraaj:Barath_2002@localhost:5432/godfather_db",
    embedding_function=embeddings,
    collection_name="godfather_embeddings"
)
qa_pipeline = setup_qa_pipeline(vector_store)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    result = query_document(qa_pipeline, request.question)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result