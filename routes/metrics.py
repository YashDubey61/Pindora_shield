from fastapi import APIRouter, HTTPException
from models.schemas import Generate3DInput

router = APIRouter(
    prefix="/metrics",
    tags=["pindora"],
    responses={404: {"description": "Not found"}},
)

@router.post("/metrics_data")
async def generate_3d_endpoint(request: Generate3DInput):
    if not request.input_smile or len(request.input_smile.strip()) == 0:
        raise HTTPException(status_code=400, detail="SMILES string is required")
    ic50_value, pchemb_value, association_score, target_symbol, max_phase = 0, 0, 0, "", 0
    return {
        "ic50_value": ic50_value,
        "pchemb_value": pchemb_value,
        "association_score": association_score,
        "target_symbol": target_symbol,
        "max_phase": max_phase,
        "status": "success",
    }
