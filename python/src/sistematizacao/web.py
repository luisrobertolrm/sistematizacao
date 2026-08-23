"""API HTTP para enviar novos dados e receber a analise de churn.

Sobe com:  uvicorn sistematizacao.web:app --host 0.0.0.0 --port 8000
Docs em:   http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sistematizacao.carregamento_inicial import ARQUIVO_MODELO
from sistematizacao.carregar_modelo import (
    carregar_limiar,
    carregar_metadados,
    carregar_modelo,
    limpar_cache,
    prever,
)
from sistematizacao.gerador_fake import (
    COLUNAS_ORIGINAIS,
    DIR_NOVOS,
    MAX_LINHAS,
    gerar_e_salvar,
)

Sim = Literal["Yes", "No"]
SimSemInternet = Literal["Yes", "No", "No internet service"]

app = FastAPI(
    title="Sistematizacao - Analise de Churn",
    description="Previsao de churn de clientes de telecom (modelo tunado + limiar calibrado).",
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


class ArquivoGerado(BaseModel):
    arquivo: str
    caminho: str
    linhas: int
    colunas: list[str]
    taxa_churn: float


class ArquivoNovo(BaseModel):
    arquivo: str
    linhas: int
    bytes: int


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


def _limiar_query(limiar: float | None) -> float:
    """None -> limiar calibrado em producao; senao usa o informado."""
    if limiar is None:
        try:
            return carregar_limiar()
        except FileNotFoundError:
            return 0.5
    return limiar


@app.post("/prever", response_model=Previsao, summary="Analisa um cliente")
def prever_um(
    cliente: Cliente,
    limiar: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> Previsao:
    return prever_lote([cliente], limiar).previsoes[0]


@app.post("/prever/lote", response_model=RespostaLote, summary="Analisa varios clientes")
def prever_lote(
    clientes: list[Cliente],
    limiar: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> RespostaLote:
    if not clientes:
        raise HTTPException(status_code=422, detail="Envie ao menos um cliente.")
    thr = _limiar_query(limiar)
    try:
        resultado = prever([c.model_dump() for c in clientes], limiar=thr)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RespostaLote(
        quantidade=len(resultado),
        limiar=thr,
        previsoes=[Previsao(**linha) for linha in resultado.to_dict(orient="records")],
    )


@app.post(
    "/dados/fake",
    response_model=ArquivoGerado,
    status_code=201,
    summary="Gera dados sinteticos e grava em dados/novos",
)
def gerar_dados_fake(
    linhas: Annotated[int, Query(ge=1, le=MAX_LINHAS)] = 100,
    semente: Annotated[int | None, Query(description="Fixa o sorteio p/ reproduzir")] = None,
    nome: Annotated[str | None, Query(description="Nome do arquivo; padrao usa timestamp")] = None,
) -> ArquivoGerado:
    """Grava um CSV com as mesmas 21 colunas do dataset original, na mesma ordem.

    As colunas sao sorteadas da distribuicao empirica do CSV original e o rotulo
    Churn sai da probabilidade do proprio modelo — serve para exercitar o
    pipeline ponta a ponta, nao para medir a qualidade do modelo.
    """
    try:
        destino, df = gerar_e_salvar(linhas=linhas, semente=semente, nome=nome)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao gravar: {exc}") from exc

    return ArquivoGerado(
        arquivo=destino.name,
        caminho=str(destino),
        linhas=len(df),
        colunas=COLUNAS_ORIGINAIS,
        taxa_churn=round(float((df["Churn"] == "Yes").mean()), 4),
    )


@app.get(
    "/dados/novos",
    response_model=list[ArquivoNovo],
    summary="Lista os CSVs ja gerados",
)
def listar_dados_novos() -> list[ArquivoNovo]:
    if not DIR_NOVOS.exists():
        return []
    return [
        ArquivoNovo(
            arquivo=csv.name,
            linhas=sum(1 for _ in csv.open(encoding="utf-8")) - 1,
            bytes=csv.stat().st_size,
        )
        for csv in sorted(DIR_NOVOS.glob("*.csv"))
    ]


@app.post("/modelo/recarregar", summary="Recarrega o binario do disco")
def recarregar() -> dict[str, str]:
    """Usado depois de um novo treino, sem precisar reiniciar o container."""
    limpar_cache()
    try:
        carregar_modelo()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "modelo recarregado"}


@app.post("/modelo/treinar", summary="Re-treina (original + novos) e gera um run versionado")
def treinar_modelo(
    *,
    incluir_novos: Annotated[bool, Query(description="Concatena dados/novos ao treino")] = True,
    promover: Annotated[bool, Query(description="Promove o run p/ producao ao final")] = True,
) -> dict[str, object]:
    """SINCRONO: roda o treino da etapa 7, versiona o run e (opcional) promove.

    Bloqueia por alguns minutos (Optuna + CV). Em producao, mover para tarefa
    em background para nao segurar o worker HTTP.
    """
    from sistematizacao.treinamento_avaliacao import treinar_publicar  # noqa: PLC0415

    try:
        resultado = treinar_publicar(incluir_novos=incluir_novos, promover_auto=promover)
    except (FileNotFoundError, ImportError) as exc:
        # o dataset base precisa estar visivel em DIR_DADOS (ou em CHURN_CSV):
        # sem ele nao ha "original + novos" para treinar
        raise HTTPException(
            status_code=503, detail=f"Dataset base indisponivel para treinar: {exc}"
        ) from exc
    if promover:
        limpar_cache()
        carregar_modelo()
    return {"status": "treinado", **resultado}


@app.post("/modelo/monitorar", summary="Drift do modelo, e-mail e retreino condicionado")
def monitorar_modelo(
    *,
    retreinar: Annotated[
        bool,
        Query(description="Se houver drift, dispara retreino e promove so se Acc/AUC ok"),
    ] = True,
) -> dict[str, object]:
    """Avalia o binario de producao; alarma e retreina apenas se houver drift."""
    from sistematizacao.monitoramento import ciclo_diario  # noqa: PLC0415

    try:
        return ciclo_diario(retreinar=retreinar)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ImportError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/modelo/promover/{run_id}", summary="Promove um run para producao (aprovacao humana)")
def promover_modelo(run_id: str) -> dict[str, str]:
    """Copia o binario do run para producao e recarrega, sem reiniciar o container."""
    from sistematizacao.runs import promover  # noqa: PLC0415

    try:
        promover(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    limpar_cache()
    carregar_modelo()
    return {"status": "promovido", "run_id": run_id}


@app.get("/modelo/runs", summary="Historico de treinamentos e avaliacoes")
def listar_runs() -> dict[str, object]:
    """Conteudo do MANIFESTO.json: run em producao + historico completo."""
    from sistematizacao.runs import ler_manifesto  # noqa: PLC0415

    return ler_manifesto()


@app.get(
    "/modelo/runs/{run_id}/eda/{arquivo}",
    summary="PNG da EDA de um run (revisao antes de promover)",
    response_class=FileResponse,
)
def eda_do_run(run_id: str, arquivo: str) -> FileResponse:
    """Serve um grafico de treinamentos/<run_id>/eda/, sem deixar escapar da pasta."""
    from sistematizacao.runs import DIR_TREINAMENTOS  # noqa: PLC0415

    pasta = (DIR_TREINAMENTOS / run_id / "eda").resolve()
    destino = (pasta / arquivo).resolve()
    if not destino.is_relative_to(pasta) or not destino.is_file():
        raise HTTPException(status_code=404, detail="Grafico nao encontrado.")
    return FileResponse(destino, media_type="image/png")


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
