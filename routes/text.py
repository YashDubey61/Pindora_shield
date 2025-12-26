from fastapi import APIRouter, HTTPException
from models.schemas import TextInput, TextResponse, HealthCheck
from pindora import Pindora
import json

router = APIRouter(
    prefix="/api",
    tags=["pindora"],
    responses={404: {"description": "Not found"}},
)

@router.post("/drug_konsi_doge", response_model=TextResponse)
async def process_text(request: TextInput):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    Pindora_instance = Pindora()
    Pindora_instance.drug_discovery_pipeline(request.text)
    
    gen_mol = None
    with open("data/generated_molecules_new.json", "r", encoding="utf-8") as f:
        gen_mol = json.load(f)    

    return {
        "results": gen_mol,
        "status": "success",
        }
