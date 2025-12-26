from fastapi import APIRouter, HTTPException
from models.schemas import TextInput, TextResponse, HealthCheck
from pindora import Pindora
from util.generate_3d import generate_3d_from_smiles
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
    
    with open("data/generated_molecules_new.json", "r", encoding="utf-8") as f:
        gen_mol = json.load(f)    

    return {
        "results": gen_mol,
        "status": "success",
    }

# ✅ NEW 3D MODEL ENDPOINT
@router.post("/generate-3d")
async def generate_3d_endpoint(request: dict):
    smiles = request.get("input_smile")

    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string is required")

    try:
        path = generate_3d_from_smiles(smiles)
        return {
            "message": "3D model generated successfully",
            "file_path": path,
            "status": "success",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))