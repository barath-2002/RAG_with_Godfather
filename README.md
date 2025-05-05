# The Godfather Q&A System

An interactive question-answering system built on "The Godfather" novel by Mario Puzo. The system uses LangChain, FastAPI, and React to create a conversational AI that can answer questions about the book's content.

<div align="center">
  <img src="Cover.jpg" alt="The Godfather Book Cover" width="300"/>
</div>

## Features

- PDF text extraction and processing
- Vector storage using PostgreSQL with pgvector
- Q&A capability using Cohere LLM
- FastAPI backend
- React frontend interface

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Node.js and npm
- Cohere API key

## Installation

### 1. Database Setup

```bash
# Install PostgreSQL and pgvector extension
# Login to PostgreSQL and create database
psql
CREATE DATABASE godfather_db;
\c godfather_db
CREATE EXTENSION vector;
```

### 2. Backend Setup

#### Clone the repository
```bash
git clone <repository-url>
cd <repository-directory>
```

#### Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r [requirements.txt](http://_vscodecontentref_/0)
```

### 3. Set up environment variables
echo "COHERE_API_KEY=your_cohere_api_key" > .env


### 4. Frontend Setup
```bash
cd godfather-qa-frontend
npm install
```


### 5. Running the Application
```bash
#### Start the FastAPI server
uvicorn app:app --reload

#### In a new terminal, from the godfather-qa-frontend directory
npm start
```

The application will be available at:

Frontend: http://localhost:3000
Backend API: http://localhost:8000
API Documentation: http://localhost:8000/docs