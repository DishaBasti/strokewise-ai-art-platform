import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI-compatible client for Nebius
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArtRequest(BaseModel):
    description: str
    medium: str

class Step(BaseModel):
    instruction: str
    image_url: str

class ArtResponse(BaseModel):
    steps: List[Step]

def generate_instructions(description: str, medium: str) -> List[str]:
    prompt = f"""
    Create 5 clear, step-by-step instructions for drawing or painting a {medium} artwork of: {description}.
    Ensure each step builds upon the previous one.
    Format: Return only the numbered steps, one per line.
    """

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.6,
        top_p=0.9,
        extra_body={"top_k": 50}
    )

    content = response.choices[0].message.content.strip()
    return [step.strip() for step in content.split('\n') if step.strip()]

@app.post("/api/generate")
async def generate_art(request: ArtRequest) -> ArtResponse:
    print(f"Received request: description={request.description}, medium={request.medium}")

    try:
        instructions = generate_instructions(request.description, request.medium)
        steps = []

        for instruction in instructions:
            image_prompt = f"{request.description} in {request.medium} style - {instruction}"

            image_response = client.images.generate(
                model="stability-ai/sdxl",
                prompt=image_prompt,
                response_format="url",
                extra_body={
                    "response_extension": "png",
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 30,
                    "negative_prompt": "blurry, distorted, low quality, unrealistic",
                    "seed": -1
                }
            )

            image_url = image_response.data[0].url
            steps.append(Step(instruction=instruction, image_url=image_url))

        return ArtResponse(steps=steps)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating art: {str(e)}")

@app.get("/")
async def root():
    return {"status": "ok", "message": "StrokeWise API is running"}
