"""Etapa 2 - Analise Exploratoria dos Dados (EDA).

Roda sobre os dados ja limpos e grava os graficos em `relatorios/`.
As conclusoes desta etapa alimentam COLUNAS_REMOVIDAS em carregamento_inicial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")  # backend sem tela: funciona em container / CI
import matplotlib.pyplot as plt

from sistematizacao.carregamento_inicial import (
    ALVO,
    DIR_RELATORIOS,
    carregar_bruto,
    limpar,
)

if TYPE_CHECKING:
    from pathlib import Path

COLUNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
COLUNAS_CATEGORICAS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "gender",
]


def _salvar(fig: plt.Figure, nome: str, dir_saida: Path = DIR_RELATORIOS) -> Path:
    dir_saida.mkdir(parents=True, exist_ok=True)
    destino = dir_saida / nome
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    print(f"  -> grafico salvo em {destino}")
    return destino


def visao_geral(df: pd.DataFrame) -> None:
    """2.1 Tipos, nulos e estatisticas descritivas."""
    print("Formato (linhas, colunas):", df.shape)
    print()
    df.info()
    print("\nEstatisticas das colunas numericas:")
    print(df.describe())


def distribuicao_alvo(df: pd.DataFrame, dir_saida: Path = DIR_RELATORIOS) -> None:
    """2.2 Desbalanceamento do alvo (~73% x ~27%).

    Consequencia: acuracia sozinha engana. Por isso a selecao usa F1 como
    metrica principal.
    """
    print("\nDistribuicao do Churn (contagem):")
    print(df[ALVO].value_counts())
    print("\nDistribuicao do Churn (proporcao):")
    print(df[ALVO].value_counts(normalize=True).round(3))

    fig, ax = plt.subplots(figsize=(5, 4))
    df[ALVO].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Distribuicao do alvo (0 = Fica, 1 = Churn)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Clientes")
    _salvar(fig, "01_distribuicao_alvo.png", dir_saida)


def numericas_por_churn(df: pd.DataFrame, dir_saida: Path = DIR_RELATORIOS) -> None:
    """2.3 Distribuicao das numericas separada por churn.

    Espera-se: quem churna concentra-se em tenure baixo (cliente novo).
    """
    fig, axes = plt.subplots(1, len(COLUNAS_NUMERICAS), figsize=(15, 4))
    for ax, col in zip(axes, COLUNAS_NUMERICAS, strict=True):
        df.loc[df[ALVO] == 0, col].plot(kind="hist", bins=30, alpha=0.6, ax=ax, label="Fica")
        df.loc[df[ALVO] == 1, col].plot(kind="hist", bins=30, alpha=0.6, ax=ax, label="Churn")
        ax.set_title(col)
        ax.legend()
    _salvar(fig, "02_numericas_por_churn.png", dir_saida)


def categoricas_por_churn(df: pd.DataFrame, dir_saida: Path = DIR_RELATORIOS) -> None:
    """2.4 Taxa de churn dentro de cada categoria."""
    for col in COLUNAS_CATEGORICAS:
        print("=" * 40)
        print(col)
        print(pd.crosstab(df[col], df[ALVO], normalize="index").round(3))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flat, COLUNAS_CATEGORICAS, strict=False):
        (
            pd.crosstab(df[col], df[ALVO], normalize="index")[1]
            .sort_values()
            .plot(kind="barh", ax=ax)
        )
        ax.set_title(f"Taxa de churn por {col}")
        ax.set_xlabel("proporcao de churn")
    axes.flat[-1].axis("off")  # sobra 1 subplot vazio (5 categorias em grade 2x3)
    _salvar(fig, "03_categoricas_por_churn.png", dir_saida)


def gerar_eda(df: pd.DataFrame, dir_saida: Path) -> None:
    """Gera os 3 PNGs da EDA a partir de um df ja limpo, salvando em dir_saida.

    Usado pelo treino versionado: a EDA precisa descrever o dataset que de fato
    treinou aquele run (original + novos), nao o dataset base.
    """
    distribuicao_alvo(df, dir_saida)
    numericas_por_churn(df, dir_saida)
    categoricas_por_churn(df, dir_saida)


def correlacoes(df: pd.DataFrame) -> None:
    """2.5 Correlacao entre numericas (inclui SeniorCitizen e o alvo)."""
    print("\nCorrelacao entre numericas:")
    print(df[[*COLUNAS_NUMERICAS, "SeniorCitizen", ALVO]].corr().round(2))


def main() -> None:
    df = limpar(carregar_bruto())

    visao_geral(df)
    distribuicao_alvo(df)
    numericas_por_churn(df)
    categoricas_por_churn(df)
    correlacoes(df)

    print(
        "\n--- Decisoes de feature vindas da EDA ---\n"
        "gender: taxa de churn praticamente igual entre os sexos -> remover (sem sinal).\n"
        "PaperlessBilling: churn ~2x maior na fatura digital -> manter (tem sinal).\n"
        "customerID: identificador, nao e atributo -> remover.\n"
        "TotalCharges: deriva de tenure x MonthlyCharges -> remover (redundante).\n"
        "tenure: efeito satura com o tempo -> vira faixa categorica (faixa_tenure)."
    )


if __name__ == "__main__":
    main()
