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
_MAPA_FAIXA_TEXTO = {0: "novo", 1: "intermediario", 2: "antigo"}
_MIN_PARTES_PRE = 2


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


def alinhar_entrada_ao_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta faixa_tenure e dtypes categoricos ao binario em producao.

    Modelos gravados antes da faixa numerica esperam 'novo'/'intermediario'/'antigo'.
    sklearn 1.9 + pandas 3 falha se categoricas ficarem em StringDtype.
    """
    try:
        pipe = carregar_modelo()
    except FileNotFoundError:
        return df

    pre = pipe.named_steps.get("pre")
    if pre is None or len(pre.transformers_) < _MIN_PARTES_PRE:
        return df

    cat_cols = list(pre.transformers_[1][2])
    df = df.copy()

    if "faixa_tenure" in cat_cols and "faixa_tenure" in df.columns:
        ohe = pre.named_transformers_["cat"]
        idx = cat_cols.index("faixa_tenure")
        cats = ohe.categories_[idx]
        faixa_numerica = pd.api.types.is_numeric_dtype(df["faixa_tenure"])
        if len(cats) and isinstance(cats[0], str) and faixa_numerica:
            df["faixa_tenure"] = df["faixa_tenure"].map(_MAPA_FAIXA_TEXTO)

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    return df


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

    df = alinhar_entrada_ao_modelo(df)
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
