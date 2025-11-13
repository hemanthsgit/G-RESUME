from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.letter import router as resume_router
from routes.test import router as video_analyzer
import uvicorn
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

app = FastAPI(title="AI-Powered Career Assistant API")


app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for frontend running on port 3001
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(resume_router)
app.include_router(video_analyzer)

@app.get("/")
def root():
    return {"message": "Welcome to the AI Career Assistant API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)
