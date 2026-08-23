"""Carregamento do binario do modelo e predicao sobre novos dados.

Suporta bundle {pipe, threshold, modelo} gravado pelo treinamento (curso_2)
e pipelines legados salvos diretamente.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from sistematizacao.carregamento_inicial import (
    ALVO,
    ARQUIVO_METADADOS,
    ARQUIVO_MODELO,
    COLUNAS_REMOVIDAS,
    aplicar_engenharia_atributos,
)

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

LIMIAR_PADRAO = 0.5


@lru_cache(maxsize=1)
def _carregar_objeto() -> Any:
    if not ARQUIVO_MODELO.exists():
        msg = (
            f"Modelo nao encontrado em {ARQUIVO_MODELO}. "
            "Rode `python -m sistematizacao.treinamento_avaliacao` antes."
        )
        raise FileNotFoundError(msg)
    return joblib.load(ARQUIVO_MODELO)


@lru_cache(maxsize=1)
def carregar_modelo() -> Pipeline:
    """Pipeline sklearn treinado."""
    obj = _carregar_objeto()
    if isinstance(obj, dict):
        return obj["pipe"]
    return obj


@lru_cache(maxsize=1)
def carregar_limiar() -> float:
    """Limiar calibrado na etapa 7 (Prec_churn); fallback metadados ou 0.5."""
    obj = _carregar_objeto()
    if isinstance(obj, dict) and "threshold" in obj:
        return float(obj["threshold"])
    meta = carregar_metadados()
    if meta.get("threshold") is not None:
        return float(meta["threshold"])
    return LIMIAR_PADRAO


@lru_cache(maxsize=1)
def carregar_metadados() -> dict[str, Any]:
    """Metadados gravados no treino (params, metricas, limiar, data)."""
    if not ARQUIVO_METADADOS.exists():
        return {}
    return json.loads(ARQUIVO_METADADOS.read_text(encoding="utf-8"))


def limpar_cache() -> None:
    """Descarta caches em memoria (util apos novo treino)."""
    _carregar_objeto.cache_clear()
    carregar_modelo.cache_clear()
    carregar_limiar.cache_clear()
    carregar_metadados.cache_clear()


def preparar_entrada(dados: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Aplica a mesma engenharia de atributos do treino aos dados novos."""
    if isinstance(dados, dict):
        df = pd.DataFrame([dados])
    elif isinstance(dados, list):
        df = pd.DataFrame(dados)
    else:
        df = dados.copy()

    if "tenure" in df.columns:
        df = aplicar_engenharia_atributos(df)

    return df.drop(columns=[*COLUNAS_REMOVIDAS, ALVO], errors="ignore")


def prever(
    dados: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    limiar: float | None = None,
) -> pd.DataFrame:
    """Devolve probabilidade de churn, classe prevista e rotulo legivel."""
    if limiar is None:
        limiar = carregar_limiar()
    X = preparar_entrada(dados)
    modelo = carregar_modelo()

    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= limiar).astype(int)

    return pd.DataFrame(
        {
            "probabilidade_churn": np.round(prob, 4),
            "churn": pred,
            "rotulo": np.where(pred == 1, "Churn", "Fica"),
        }
    )


def main() -> None:
    meta = carregar_metadados()
    print("Modelo:", meta.get("nome", "desconhecido"))
    print("Algoritmo:", meta.get("modelo", "?"))
    print("Limiar:", carregar_limiar())
    print("Treinado em:", meta.get("treinado_em", "?"))
    print("Params:", meta.get("params", {}))
    print("Metricas no teste:", meta.get("metricas_teste", {}))

    exemplo = {
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
    }
    print("\nPredicao de exemplo:")
    print(prever(exemplo))


if __name__ == "__main__":
    main()
