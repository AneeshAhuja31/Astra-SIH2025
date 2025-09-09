from graph import create_oceanographic_workflow
from fastapi import FastAPI
from fastapi.requests import Request
app = create_oceanographic_workflow()

@app.post("/chat")
async def chat(request:Request):
    pass
