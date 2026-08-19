"""Geracao de dados sinteticos com Faker, no mesmo formato do CSV original.

O arquivo gerado em `dados/novos/` tem exatamente as 21 colunas do
WA_Fn-UseC_-Telco-Customer-Churn.csv, na mesma ordem, incluindo as manias do
original: `TotalCharges` em branco quando `tenure` e 0 e `customerID` no
formato 4 digitos + 5 letras.

Cada campo sai de um provider do Faker (`random_element` para as categoricas,
`pyfloat`/`random_int` para as numericas). As dependencias entre colunas
(sem telefone -> sem multiplas linhas, sem internet -> sem servicos de
internet) sao aplicadas depois do sorteio, porque sortear campo a campo as
quebraria.
"""

from __future__ import annotations

import string
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from faker import Faker

from sistematizacao.carregamento_inicial import DIR_DADOS

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

# dominios de cada categorica, iguais aos do CSV original
SIM_NAO = ["Yes", "No"]
CONTRATOS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
LINHAS_MULTIPLAS = ["Yes", "No", "No phone service"]
SERVICO_OU_SEM_INTERNET = ["Yes", "No", "No internet service"]
PAGAMENTOS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

TENURE_MIN = 0
TENURE_MAX = 72
COBRANCA_MIN = 20
COBRANCA_MAX = 120
TAXA_CHURN_PADRAO = 0.265
RUIDO_TOTAL_MIN = 0.85
RUIDO_TOTAL_MAX = 1.15
MAX_LINHAS = 100_000


def _linha(fake: Faker) -> dict[str, Any]:
    """Um cliente sintetico, campo a campo pelos providers do Faker."""
    return {
        "customerID": fake.bothify(text="####-?????", letters=string.ascii_uppercase),
        "gender": fake.random_element(["Male", "Female"]),
        "SeniorCitizen": fake.random_element([0, 1]),
        "Partner": fake.random_element(SIM_NAO),
        "Dependents": fake.random_element(SIM_NAO),
        "tenure": fake.random_int(min=TENURE_MIN, max=TENURE_MAX),
        "PhoneService": fake.random_element(SIM_NAO),
        "MultipleLines": fake.random_element(LINHAS_MULTIPLAS),
        "InternetService": fake.random_element(INTERNET),
        "OnlineSecurity": fake.random_element(SERVICO_OU_SEM_INTERNET),
        "OnlineBackup": fake.random_element(SERVICO_OU_SEM_INTERNET),
        "DeviceProtection": fake.random_element(SERVICO_OU_SEM_INTERNET),
        "TechSupport": fake.random_element(SERVICO_OU_SEM_INTERNET),
        "StreamingTV": fake.random_element(SERVICO_OU_SEM_INTERNET),
        "StreamingMovies": fake.random_element(SERVICO_OU_SEM_INTERNET),
        "Contract": fake.random_element(CONTRATOS),
        "PaperlessBilling": fake.random_element(SIM_NAO),
        "PaymentMethod": fake.random_element(PAGAMENTOS),
        "MonthlyCharges": fake.pyfloat(
            min_value=COBRANCA_MIN, max_value=COBRANCA_MAX, right_digits=2
        ),
    }


def _coerir_dependencias(df: pd.DataFrame) -> pd.DataFrame:
    """Reimpoe as regras que o sorteio campo-a-campo quebra."""
    sem_telefone = df["PhoneService"] == "No"
    df.loc[sem_telefone, "MultipleLines"] = "No phone service"
    df.loc[~sem_telefone & (df["MultipleLines"] == "No phone service"), "MultipleLines"] = "No"

    sem_internet = df["InternetService"] == "No"
    for col in SERVICOS_INTERNET:
        df.loc[sem_internet, col] = "No internet service"
        df.loc[~sem_internet & (df[col] == "No internet service"), col] = "No"

    return df


def _total_charges(fake: Faker, df: pd.DataFrame) -> pd.Series:
    """TotalCharges ~ tenure x MonthlyCharges com ruido; em branco quando tenure=0.

    Os 11 brancos do CSV original sao exatamente as linhas com tenure=0 — a
    razao de o carregamento_inicial converter com `errors="coerce"` e derrubar
    os nulos. O gerado reproduz isso para exercitar o mesmo caminho.
    """
    ruido = pd.Series(
        [
            fake.pyfloat(min_value=RUIDO_TOTAL_MIN, max_value=RUIDO_TOTAL_MAX, right_digits=3)
            for _ in range(len(df))
        ],
        index=df.index,
    )
    total = (df["tenure"] * df["MonthlyCharges"] * ruido).round(2)
    return total.astype(str).where(df["tenure"] > 0, "")


def _sortear_churn(fake: Faker, df: pd.DataFrame) -> list[str]:
    """Churn tirado da probabilidade do modelo treinado; sem ele, da taxa base.

    Usar o modelo deixa o rotulo coerente com os atributos — bom para exercitar
    o pipeline ponta a ponta. Nao serve para medir qualidade do modelo: o dado
    ja nasce concordando com ele.
    """
    try:
        from sistematizacao.carregar_modelo import prever  # noqa: PLC0415

        probabilidades = prever(df)["probabilidade_churn"].tolist()
    except (FileNotFoundError, ValueError, KeyError):
        probabilidades = [TAXA_CHURN_PADRAO] * len(df)

    return ["Yes" if fake.random.random() < p else "No" for p in probabilidades]


def gerar(linhas: int = 100, semente: int | None = None) -> pd.DataFrame:
    """Monta o DataFrame sintetico com as 21 colunas do original, na mesma ordem."""
    if linhas < 1 or linhas > MAX_LINHAS:
        msg = f"linhas deve estar entre 1 e {MAX_LINHAS}."
        raise ValueError(msg)

    fake = Faker()
    if semente is not None:
        fake.seed_instance(semente)  # mesma semente -> mesmo arquivo

    df = pd.DataFrame([_linha(fake) for _ in range(linhas)])
    df = _coerir_dependencias(df)
    df["TotalCharges"] = _total_charges(fake, df)
    df["Churn"] = _sortear_churn(fake, df)

    print(f"Gerou {linhas} linhas com Faker (semente={semente}).")
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
