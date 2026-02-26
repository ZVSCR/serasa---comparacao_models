from pydantic import BaseModel, Field
from typing import Literal

class LLMClassification(BaseModel):
    justification: str = Field(
        description="Explicação detalhada do raciocínio antes da decisão"
    )
    prediction: Literal[0, 1] = Field(
        description="0 para fake, 1 para real"
    )
    label_name: Literal["fake", "real"]