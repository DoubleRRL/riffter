from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn
import re
from inference import generate_riff as local_generate_riff, generate_joke as local_generate_joke

app = FastAPI(title="Riffter API", description="AI-powered comedy riff generator")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RiffRequest(BaseModel):
    topic: str

    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        if len(v.strip()) > 500:
            raise ValueError('Topic must be less than 500 characters')
        # Basic sanitization - remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        return v

class JokeRequest(BaseModel):
    topic: str

    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        if len(v.strip()) > 500:
            raise ValueError('Topic must be less than 500 characters')
        # Basic sanitization - remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        return v

@app.get("/")
def read_root():
    return {"message": "Riffter API is running"}

@app.post("/riff")
def generate_riff(request: RiffRequest):
    """Generate a quick 1-2 sentence riff in Nick Mullen's style"""
    try:
        riff = local_generate_riff(request.topic)

        if not riff or len(riff.strip()) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate riff")

        return {"riff": riff}

    except Exception as e:
        # Fallback riff
        fallback_riff = f"Imagine if {request.topic} was actually a conspiracy theory about alien lizards running the government."
        return {"riff": fallback_riff}

@app.post("/joke")
def generate_joke(request: JokeRequest):
    """Generate a structured joke in Nick Mullen's style"""
    try:
        joke_structure = local_generate_joke(request.topic)

        # Validate structure has required keys
        required_keys = ["premise", "punchline", "initial_tag", "alternate_angle", "additional_tags"]
        for key in required_keys:
            if key not in joke_structure:
                joke_structure[key] = f"Generated {key} for {request.topic}"

        return {"joke": joke_structure}

    except Exception as e:
        # Fallback structure
        fallback_joke = {
            "premise": f"Premise about {request.topic}",
            "punchline": "This is where the punchline goes",
            "initial_tag": "First tag here",
            "alternate_angle": "Another angle on the premise",
            "additional_tags": ["Tag 1", "Tag 2", "Tag 3"]
        }
        return {"joke": fallback_joke}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
