from fastapi import FastAPI, HTTPException
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from agents import run_agent
from typing import Dict
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str

# In-memory session store (dict of session_id: list[BaseMessage])
sessions: Dict[str, list[BaseMessage]] = {}

@app.post("/ask")
def ask(query: Query):
    history = sessions.get(query.session_id, [])
    answer, new_history = run_agent(query.question, query.session_id, history)
    sessions[query.session_id] = new_history
    return {"answer": answer}

@app.post("/clear_memory")
def clear_memory(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)