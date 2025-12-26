from pydantic import BaseModel
from typing import Optional

class TextInput(BaseModel):
    text: str


class TextResponse(BaseModel):
    input_text: str
    status: str
    message: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    message: str