"""API HTTP para enviar novos dados e receber a analise de churn.

Sobe com:  uvicorn sistematizacao.web:app --host 0.0.0.0 --port 8000
Docs em:   http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from sistematizacao.carregamento_inicial import ARQUIVO_MODELO
from sistematizacao.carregar_modelo import (
    LIMIAR_PADRAO,
    carregar_metadados,
    carregar_modelo,
    limpar_cache,
    prever,
)

Sim = Literal["Yes", "No"]
SimSemInternet = Literal["Yes", "No", "No internet service"]

app = FastAPI(
    title="Sistematizacao - Analise de Churn",
    description="Previsao de churn de clientes de telecom (modelo LogisticRegression tunada).",
    version="1.0.0",
)


class Cliente(BaseModel):
    """Um registro de cliente, no mesmo formato do dataset Telco Customer Churn."""

    SeniorCitizen: Annotated[int, Field(ge=0, le=1, examples=[0])]
    Partner: Sim = "No"
    Dependents: Sim = "No"
    tenure: Annotated[int, Field(ge=0, description="Meses de casa", examples=[2])]
    PhoneService: Sim = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: SimSemInternet = "No"
    OnlineBackup: SimSemInternet = "No"
    DeviceProtection: SimSemInternet = "No"
    TechSupport: SimSemInternet = "No"
    StreamingTV: SimSemInternet = "Yes"
    StreamingMovies: SimSemInternet = "Yes"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Sim = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: Annotated[float, Field(ge=0, examples=[95.5])]


class Previsao(BaseModel):
    probabilidade_churn: float
    churn: int
    rotulo: str


class RespostaLote(BaseModel):
    quantidade: int
    limiar: float
    previsoes: list[Previsao]


@app.get("/health", summary="Liveness/readiness")
def health() -> dict[str, object]:
    """Diz se o processo esta de pe e se o binario do modelo esta disponivel."""
    return {"status": "ok", "modelo_disponivel": ARQUIVO_MODELO.exists()}


@app.get("/modelo", summary="Metadados do modelo em uso")
def modelo() -> dict[str, object]:
    meta = carregar_metadados()
    if not meta:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda nao treinado. Rode a etapa de treinamento_avaliacao.",
        )
    return meta


@app.post("/prever", response_model=Previsao, summary="Analisa um cliente")
def prever_um(
    cliente: Cliente,
    limiar: Annotated[float, Query(ge=0, le=1)] = LIMIAR_PADRAO,
) -> Previsao:
    return prever_lote([cliente], limiar).previsoes[0]


@app.post("/prever/lote", response_model=RespostaLote, summary="Analisa varios clientes")
def prever_lote(
    clientes: list[Cliente],
    limiar: Annotated[float, Query(ge=0, le=1)] = LIMIAR_PADRAO,
) -> RespostaLote:
    if not clientes:
        raise HTTPException(status_code=422, detail="Envie ao menos um cliente.")
    try:
        resultado = prever([c.model_dump() for c in clientes], limiar=limiar)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RespostaLote(
        quantidade=len(resultado),
        limiar=limiar,
        previsoes=[Previsao(**linha) for linha in resultado.to_dict(orient="records")],
    )


@app.post("/modelo/recarregar", summary="Recarrega o binario do disco")
def recarregar() -> dict[str, str]:
    """Usado depois de um novo treino, sem precisar reiniciar o container."""
    limpar_cache()
    try:
        carregar_modelo()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "modelo recarregado"}


def main() -> None:
    import os  # noqa: PLC0415

    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "sistematizacao.web:app",
        host=os.getenv("HOST", "0.0.0.0"),  # noqa: S104
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
