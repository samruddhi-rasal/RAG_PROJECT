from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import generate_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API running 🚀"}

@app.post("/ask")
def ask(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    return generate_answer(query.question)