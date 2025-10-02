
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
from app.model_loader import load_model, infer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API de Inferência de Crimes", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo uma única vez na inicialização
MODEL = load_model()


class PredictRequest(BaseModel):
    dia_semana: str = Field(..., description="Dia da semana (ex: segunda)")
    hora: int = Field(..., ge=0, le=23, description="Hora cheia 0-23")
    bairro: str = Field(..., min_length=1, description="Nome do bairro")

    @validator("dia_semana")
    def valida_dia_semana_nao_vazio(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("dia_semana não pode ser vazio")
        return v

    @validator("bairro")
    def valida_bairro_nao_vazio(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("bairro não pode ser vazio")
        return v


class PredictResponse(BaseModel):
    crime_previsto: str
    top5: List[Dict[str, Any]]


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Se o modelo não está carregado, não travar: retornar 503
    if MODEL is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")

    crime_previsto, top5 = infer(MODEL, payload.dict())
    # Se ainda assim não há resultado, tratar como indisponível
    if not crime_previsto and not top5:
        raise HTTPException(status_code=500, detail="falha ao inferir")

    return {"crime_previsto": crime_previsto, "top5": top5}
