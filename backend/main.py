from graph import create_oceanographic_workflow
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph_app = create_oceanographic_workflow()
    print("Graph Compiled!")
    yield
    print("App shutting down")

app = FastAPI(
    title="FloatChat",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_prompt = body.get("user_prompt", "")

        initial_state = {
            "user_prompt": user_prompt,
            "check_sql": False,
            "check_graph": False,
            "sql_query": "",
            "fetched_rows": {},
            "graph_data": {},
            "generated_answer": ""
        }

        final_state = app.state.graph_app.invoke(initial_state)

        response_object = {
            "generated_answer": final_state.get("generated_answer", ""),
            "graph_data": final_state.get("graph_data") or None
        }
        print("///////////////////////////////////////")
        print(response_object)
        print("///////////////////////////////////////")
        return JSONResponse(content=response_object)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )