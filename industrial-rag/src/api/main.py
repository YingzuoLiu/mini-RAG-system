from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from rag_system.core import RAGSystem

app = FastAPI(title="Industrial RAG API")
rag_system = None

class Document(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class Query(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = RAGSystem()
    rag_system.setup_retrieval_chain()

@app.post("/documents/")
async def index_documents(documents: List[Document]):
    """索引新文档"""
    try:
        docs = [
            {
                "text": doc.text,
                "metadata": doc.metadata or {}
            }
            for doc in documents
        ]
        processed_docs = rag_system.create_documents([d["text"] for d in docs], [d["metadata"] for d in docs])
        rag_system.index_documents(processed_docs)
        return {"message": f"Successfully indexed {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(query: Query):
    """查询RAG系统"""
    try:
        result = rag_system.query(query.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload/")
async def upload_file(file: UploadFile = File(...)):
    """上传JSON文件进行索引"""
    try:
        content = await file.read()
        documents = json.loads(content.decode())
        docs = [Document(**doc) for doc in documents]
        return await index_documents(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))