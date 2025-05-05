from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.schema import Document
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

def create_embeddings(text_chunks):
    """Create embeddings and store in PostgreSQL."""
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Database configuration
    CONNECTION_STRING = "postgresql://barathraaj:Barath_2002@localhost:5432/godfather_db"
    
    # Create PGVector instance
    vector_store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="godfather_embeddings",
        pre_delete_collection=True
    )
    
    # Convert text chunks to Document objects
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    # Add Document objects to vector store
    vector_store.add_documents(documents)
    return vector_store

def setup_qa_pipeline(vector_store, cohere_api_key):
    """Set up the question-answering pipeline."""
    # Set up the LLM
    llm = Cohere(
        model="command-r-plus",
        cohere_api_key=cohere_api_key,
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
            "source": result['source_documents'][0].page_content if result['source_documents'] else None
        }
    except Exception as e:
        return {"error": str(e)}

# # Usage example:
# if __name__ == "__main__":
#     # First time setup (run once):
#     vector_store = create_embeddings(text_chunks)  # text_chunks from your PDF processing
#     qa_pipeline = setup_qa_pipeline(vector_store, os.environ.get('CO_API_KEY'))
    
#     # Query examples:
#     result = query_document(qa_pipeline, "How does Don Vito Corleone die?")
#     print(f"Question: {result['question']}")
#     print(f"Answer: {result['answer']}")