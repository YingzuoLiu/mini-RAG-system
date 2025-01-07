import os
from typing import List, Dict, Any
from datetime import datetime
import weaviate
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

class RAGSystem:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.client = weaviate.Client("http://localhost:8080")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # 初始化检索QA链的提示模板
        self.template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        尽量使用自己的语言回答。

        上下文：{context}

        问题：{question}

        回答："""
        
        self.QA_PROMPT = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )

    def create_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> List[Dict]:
        """将原始文本转换为文档格式"""
        documents = []
        for i, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": metadata[i] if metadata else {"source": f"doc_{i}"},
                    "chunk_id": f"{i}_{j}",
                    "created_at": datetime.now().isoformat()
                }
                documents.append(doc)
        return documents

    def index_documents(self, documents: List[Dict]):
        """将文档索引到Weaviate"""
        # 确保类模式存在
        class_obj = {
            "class": "Document",
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["object"]},
                {"name": "chunk_id", "dataType": ["string"]},
                {"name": "created_at", "dataType": ["date"]}
            ]
        }
        
        try:
            self.client.schema.create_class(class_obj)
        except:
            pass  # 类已存在
            
        # 批量索引文档
        with self.client.batch as batch:
            batch.batch_size = 100
            for doc in documents:
                embedding = self.embedding_model.embed_query(doc["text"])
                
                properties = {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "chunk_id": doc["chunk_id"],
                    "created_at": doc["created_at"]
                }
                
                batch.add_data_object(
                    properties,
                    "Document",
                    vector=embedding
                )

    def setup_retrieval_chain(self, temperature: float = 0.0):
        """设置检索QA链"""
        vectorstore = Weaviate(
            client=self.client,
            embedding=self.embedding_model,
            index_name="Document",
            text_key="text"
        )
        
        # 使用OpenAI作为基础模型
        llm = OpenAI(
            temperature=temperature,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
        # 创建检索QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_PROMPT}
        )

    def query(self, question: str) -> Dict:
        """查询RAG系统"""
        if not hasattr(self, 'qa_chain'):
            raise ValueError("Please call setup_retrieval_chain() first")
            
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }