from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ui.gradio_ui import launch_gradio
from ui.schemas import router as agent_router

app = FastAPI()

# Allow CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)

# We don't need to launch Gradio from here anymore
# The Gradio UI is now launched independently in its own container

@app.get("/ping")
def ping():
    return {"status": "ok"}
