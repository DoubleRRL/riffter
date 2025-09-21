from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Dict, Any, Literal
import uvicorn
import re
from src.generation.inference import (
    generate_riff as local_generate_riff,
    generate_joke as local_generate_joke,
    regenerate_joke_part as local_regenerate_joke_part
)

app = FastAPI(title="Riffter API", description="AI-powered comedy riff generator")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev server
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

class JokeRegenerationRequest(BaseModel):
    topic: str
    joke_context: Dict[str, Any]
    part_to_regenerate: str

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

@app.post("/regenerate_joke_part")
def regenerate_joke_part(request: JokeRegenerationRequest):
    """Regenerate a specific part of a structured joke."""
    try:
        new_content = local_regenerate_joke_part(
            topic=request.topic,
            joke_context=request.joke_context,
            part_to_regenerate=request.part_to_regenerate
        )
        return {"new_content": new_content}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate joke part: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
