import traceback

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.medical_workflow import run_medical_workflow

app = FastAPI()


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    subject_id: str | None = None
    hadm_id: str | None = None


@app.get("/")
def root():
    return {"message": "Medical Multi-Agent RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = run_medical_workflow(
            query=req.query,
            session_id=req.session_id,
            subject_id=req.subject_id,
            hadm_id=req.hadm_id,
        )

        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "route": "error",
            "found": False,
            "message": f"Medical multi-agent RAG service error: {str(e)}",
            "products": [],
            "sources": [],
            "agent_trace": {
                "error": str(e),
            },
        }