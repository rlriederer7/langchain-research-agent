from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import chat_routes, document_routes, agent_routes, test_routes
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from core.database import initialize_checkpointer, cleanup_checkpointer

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_checkpointer()
    print("checkpointer initialized")
    yield
    cleanup_checkpointer()
    print("checkpointer cleaned up")


app = FastAPI(title='Research Agent', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_routes.router, prefix="/api/chat")
app.include_router(document_routes.router, prefix="/api/documents")
app.include_router(agent_routes.router, prefix="/api/agents")
app.include_router(test_routes.router, prefix="/api/test")


# Health checks
@app.get("/")
async def root():
    return {"message": "running!"}


@app.get("/health")
async def health():
    return {"status": "healthy enough for this"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
