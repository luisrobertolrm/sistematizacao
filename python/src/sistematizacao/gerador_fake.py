"""Geracao de dados sinteticos no mesmo formato do CSV original.

O arquivo gerado em `dados/novos/` tem exatamente as 21 colunas do
WA_Fn-UseC_-Telco-Customer-Churn.csv, na mesma ordem, incluindo as manias do
original: `TotalCharges` em branco quando `tenure` e 0 e `customerID` no
formato 4 digitos + 5 letras.

Quando o CSV original esta disponivel, cada coluna e sorteada da distribuicao
empirica dele (preserva as marginais). Sem ele, cai para distribuicoes fixas
declaradas aqui. As dependencias entre colunas (sem telefone -> sem multiplas
linhas, sem internet -> sem servicos de internet) sao aplicadas depois do
sorteio, porque amostrar coluna a coluna as quebraria.
"""

from __future__ import annotations

import string
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sistematizacao.carregamento_inicial import DIR_DADOS, localizar_csv

DIR_NOVOS = DIR_DADOS / "novos"

# ordem exata do cabecalho do CSV original
COLUNAS_ORIGINAIS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

SERVICOS_INTERNET = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# usadas so quando o CSV original nao esta ao alcance
DISTRIBUICOES_PADRAO: dict[str, dict[str, float]] = {
    "gender": {"Male": 0.505, "Female": 0.495},
    "SeniorCitizen": {"0": 0.838, "1": 0.162},
    "Partner": {"No": 0.517, "Yes": 0.483},
    "Dependents": {"No": 0.701, "Yes": 0.299},
    "PhoneService": {"Yes": 0.903, "No": 0.097},
    "MultipleLines": {"No": 0.481, "Yes": 0.422, "No phone service": 0.097},
    "InternetService": {"Fiber optic": 0.440, "DSL": 0.344, "No": 0.216},
    "OnlineSecurity": {"No": 0.497, "Yes": 0.287, "No internet service": 0.216},
    "OnlineBackup": {"No": 0.439, "Yes": 0.345, "No internet service": 0.216},
    "DeviceProtection": {"No": 0.439, "Yes": 0.345, "No internet service": 0.216},
    "TechSupport": {"No": 0.493, "Yes": 0.291, "No internet service": 0.216},
    "StreamingTV": {"No": 0.399, "Yes": 0.385, "No internet service": 0.216},
    "StreamingMovies": {"No": 0.395, "Yes": 0.389, "No internet service": 0.216},
    "Contract": {"Month-to-month": 0.550, "Two year": 0.241, "One year": 0.209},
    "PaperlessBilling": {"Yes": 0.592, "No": 0.408},
    "PaymentMethod": {
        "Electronic check": 0.336,
        "Mailed check": 0.229,
        "Bank transfer (automatic)": 0.219,
        "Credit card (automatic)": 0.216,
    },
}

TAXA_CHURN_PADRAO = 0.265
TENURE_MAX = 72
COBRANCA_MIN = 18.25
COBRANCA_MAX = 118.75
MAX_LINHAS = 100_000


def _identificadores(rng: np.random.Generator, n: int) -> list[str]:
    """customerID no formato do original (4 digitos + 5 letras), sem repetir."""
    letras = np.array(list(string.ascii_uppercase))
    vistos: set[str] = set()
    ids: list[str] = []
    while len(ids) < n:
        numero = rng.integers(1000, 10000)
        sufixo = "".join(rng.choice(letras, size=5))
        novo = f"{numero}-{sufixo}"
        if novo not in vistos:
            vistos.add(novo)
            ids.append(novo)
    return ids


def _amostrar_do_original(
    rng: np.random.Generator, n: int, base: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Sorteia cada coluna da distribuicao empirica do CSV original."""
    colunas: dict[str, np.ndarray] = {}
    for col in COLUNAS_ORIGINAIS:
        if col in {"customerID", "TotalCharges", "Churn"}:
            continue
        valores = base[col].to_numpy()
        colunas[col] = rng.choice(valores, size=n, replace=True)
    return colunas


def _amostrar_do_padrao(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Sorteia cada coluna das distribuicoes fixas declaradas no modulo."""
    colunas: dict[str, np.ndarray] = {}
    for col, dist in DISTRIBUICOES_PADRAO.items():
        chaves = list(dist)
        pesos = np.array(list(dist.values()), dtype=float)
        sorteado = rng.choice(chaves, size=n, p=pesos / pesos.sum())
        colunas[col] = sorteado.astype(int) if col == "SeniorCitizen" else sorteado

    colunas["tenure"] = rng.integers(0, TENURE_MAX + 1, size=n)
    colunas["MonthlyCharges"] = np.round(rng.uniform(COBRANCA_MIN, COBRANCA_MAX, size=n), 2)
    return colunas


def _coerir_dependencias(df: pd.DataFrame) -> pd.DataFrame:
    """Reimpoe as regras que o sorteio coluna-a-coluna quebra."""
    sem_telefone = df["PhoneService"] == "No"
    df.loc[sem_telefone, "MultipleLines"] = "No phone service"
    df.loc[~sem_telefone & (df["MultipleLines"] == "No phone service"), "MultipleLines"] = "No"

    sem_internet = df["InternetService"] == "No"
    for col in SERVICOS_INTERNET:
        df.loc[sem_internet, col] = "No internet service"
        df.loc[~sem_internet & (df[col] == "No internet service"), col] = "No"

    return df


def _total_charges(rng: np.random.Generator, df: pd.DataFrame) -> pd.Series:
    """TotalCharges ~ tenure x MonthlyCharges com ruido; em branco quando tenure=0.

    Os 11 brancos do CSV original sao exatamente as linhas com tenure=0 — a
    razao de o carregamento_inicial converter com `errors="coerce"` e derrubar
    os nulos. O gerado reproduz isso para exercitar o mesmo caminho.
    """
    ruido = rng.normal(1.0, 0.05, size=len(df)).clip(0.7, 1.6)
    total = (df["tenure"] * df["MonthlyCharges"] * ruido).round(2)
    return total.astype(str).where(df["tenure"] > 0, "")


def _sortear_churn(rng: np.random.Generator, df: pd.DataFrame) -> np.ndarray:
    """Churn tirado da probabilidade do modelo treinado; sem ele, da taxa base.

    Usar o modelo deixa o rotulo coerente com os atributos — bom para exercitar
    o pipeline ponta a ponta. Nao serve para medir qualidade do modelo: o dado
    ja nasce concordando com ele.
    """
    try:
        from sistematizacao.carregar_modelo import prever  # noqa: PLC0415

        prob = prever(df)["probabilidade_churn"].to_numpy()
    except (FileNotFoundError, ValueError, KeyError):
        prob = np.full(len(df), TAXA_CHURN_PADRAO)

    return np.where(rng.random(len(df)) < prob, "Yes", "No")


def gerar(linhas: int = 100, semente: int | None = None) -> pd.DataFrame:
    """Monta o DataFrame sintetico com as 21 colunas do original, na mesma ordem."""
    if linhas < 1 or linhas > MAX_LINHAS:
        msg = f"linhas deve estar entre 1 e {MAX_LINHAS}."
        raise ValueError(msg)

    rng = np.random.default_rng(semente)

    try:
        base = pd.read_csv(localizar_csv())
        colunas = _amostrar_do_original(rng, linhas, base)
        origem = "distribuicao empirica do CSV original"
    except (FileNotFoundError, IndexError, OSError):
        colunas = _amostrar_do_padrao(rng, linhas)
        origem = "distribuicoes fixas do modulo"

    df = pd.DataFrame(colunas)
    df.insert(0, "customerID", _identificadores(rng, linhas))
    df = _coerir_dependencias(df)
    df["TotalCharges"] = _total_charges(rng, df)
    df["Churn"] = _sortear_churn(rng, df)

    print(f"Gerou {linhas} linhas a partir da {origem}.")
    return df[COLUNAS_ORIGINAIS]


def salvar(df: pd.DataFrame, nome: str | None = None) -> Path:
    """Grava em dados/novos/ com o mesmo layout do CSV original."""
    DIR_NOVOS.mkdir(parents=True, exist_ok=True)
    if nome is None:
        carimbo = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        nome = f"clientes_fake_{carimbo}.csv"
    if not nome.endswith(".csv"):
        nome = f"{nome}.csv"

    destino = DIR_NOVOS / Path(nome).name  # so o nome: nao deixa escapar da pasta
    df.to_csv(destino, index=False)
    return destino


def gerar_e_salvar(
    linhas: int = 100, semente: int | None = None, nome: str | None = None
) -> tuple[Path, pd.DataFrame]:
    df = gerar(linhas, semente)
    return salvar(df, nome), df


def main() -> None:
    destino, df = gerar_e_salvar(linhas=200, semente=42)
    print(f"Arquivo: {destino}")
    print(f"Colunas identicas ao original: {list(df.columns) == COLUNAS_ORIGINAIS}")
    print(f"Taxa de churn no gerado: {(df['Churn'] == 'Yes').mean():.3f}")
    print(f"Brancos em TotalCharges (tenure=0): {(df['TotalCharges'] == '').sum()}")
    print(df.head())


if __name__ == "__main__":
    main()
