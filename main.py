import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # Updated 2026 package

load_dotenv()
app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index"


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not INDEX_PATH.exists():
    raise RuntimeError(f"Index folder not found at {INDEX_PATH}")

vectorstore = FAISS.load_local(
    str(INDEX_PATH), 
    embeddings, 
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        response = rag_chain.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "online", "model": "llama-3.3-70b-versatile"}


if __name__ == "__main__":
    # Render provides a PORT environment variable. We MUST use it.
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)