"""Etapa 1 - carregamento, limpeza e preparacao dos dados (Telco Customer Churn).

Este modulo concentra tudo o que as demais etapas precisam compartilhar:
caminhos do projeto, download/leitura do dataset, limpeza, engenharia de
atributos, divisao treino/teste e o pre-processador (ColumnTransformer).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- caminhos do projeto (todos sobrescreviveis por variavel de ambiente) ---
RAIZ = Path(__file__).resolve().parents[2]
DIR_MODELOS = Path(os.getenv("DIR_MODELOS", str(RAIZ / "modelos")))
DIR_DADOS = Path(os.getenv("DIR_DADOS", str(RAIZ / "dados")))
DIR_RELATORIOS = Path(os.getenv("DIR_RELATORIOS", str(RAIZ / "relatorios")))

ARQUIVO_MODELO = DIR_MODELOS / "modelo_churn.joblib"
ARQUIVO_METADADOS = DIR_MODELOS / "modelo_churn.json"

# --- constantes do problema ---
DATASET_KAGGLE = "blastchar/telco-customer-churn"
ALVO = "Churn"
SEED = 42
TAMANHO_TESTE = 0.2
N_DOBRAS = 5

# Decisoes vindas da EDA (etapa 2):
#   customerID    -> identificador, nao e atributo
#   gender        -> taxa de churn praticamente igual entre os sexos (sem sinal)
#   TotalCharges  -> redundante: deriva de tenure x MonthlyCharges
#   tenure        -> substituido pela faixa categorica (faixa_tenure)
COLUNAS_REMOVIDAS = ["customerID", "gender", "TotalCharges", "tenure"]

# faixas de tempo de casa usadas na engenharia de atributos
LIMITE_CLIENTE_NOVO = 9
LIMITE_CLIENTE_INTERMEDIARIO = 29
FAIXAS_TENURE = ["novo", "intermediario", "antigo"]


@dataclass
class Dados:
    """Pacote com tudo que as etapas seguintes consomem."""

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pre: ColumnTransformer
    cv: StratifiedKFold


def baixar_dataset() -> Path:
    """Baixa o dataset do Kaggle (import tardio: kagglehub e opcional)."""
    import kagglehub  # noqa: PLC0415

    caminho = Path(kagglehub.dataset_download(DATASET_KAGGLE))
    print(f"Dataset baixado em: {caminho}")
    return caminho


def localizar_csv() -> Path:
    """Resolve o CSV do dataset na ordem: env CHURN_CSV -> pasta dados -> Kaggle."""
    do_env = os.getenv("CHURN_CSV")
    if do_env:
        return Path(do_env)

    locais = sorted(DIR_DADOS.glob("*.csv"))
    if locais:
        return locais[0]

    baixados = sorted(baixar_dataset().glob("*.csv"))
    if not baixados:
        msg = "Nenhum CSV encontrado no dataset baixado."
        raise FileNotFoundError(msg)
    return baixados[0]


def carregar_bruto() -> pd.DataFrame:
    """Le o CSV cru, sem nenhum tratamento."""
    csv = localizar_csv()
    print(f"Lendo: {csv}")
    return pd.read_csv(csv)


def carregar_novos() -> list[pd.DataFrame]:
    """Le os CSVs de dados/novos/ (alimentados via POST /dados/fake).

    Mesmas 21 colunas do original -> concatenaveis antes da limpeza. Em
    producao, esta funcao seria trocada pela leitura da fonte real (banco).
    """
    dir_novos = DIR_DADOS / "novos"
    if not dir_novos.exists():
        return []
    return [pd.read_csv(csv) for csv in sorted(dir_novos.glob("*.csv"))]


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges, remove nulos e binariza o alvo."""
    df = df.copy()

    # TotalCharges vem como texto e tem alguns brancos -> converte e remove nulos
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # alvo binario: Yes -> 1, No -> 0 (idempotente: aceita reexecucao)
    if not pd.api.types.is_numeric_dtype(df[ALVO]):
        df[ALVO] = (df[ALVO] == "Yes").astype(int)

    print("Formato apos limpeza:", df.shape)
    return df


def aplicar_engenharia_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Cria faixa_tenure (novo / intermediario / antigo) a partir de tenure.

    A faixa substitui o tenure continuo: o efeito do tempo de casa sobre o churn
    e forte no inicio e satura depois, entao a versao categorica descreve melhor
    esse comportamento do que uma reta.
    """
    df = df.copy()
    df["faixa_tenure"] = pd.cut(
        df["tenure"],
        bins=[-np.inf, LIMITE_CLIENTE_NOVO, LIMITE_CLIENTE_INTERMEDIARIO, np.inf],
        labels=FAIXAS_TENURE,
    ).astype(str)
    return df


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Descarta as colunas decididas na EDA e separa atributos do alvo."""
    df = df.drop(columns=COLUNAS_REMOVIDAS, errors="ignore")
    X = df.drop(columns=[ALVO])
    y = df[ALVO]
    return X, y


def construir_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """Padroniza numericas e aplica one-hot nas categoricas."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            # padroniza p/ nenhuma coluna pesar mais so pela escala
            ("num", StandardScaler(), num_cols),
            # cada categoria vira coluna binaria (evita ordem falsa 1<2<3)
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def preparar_tudo(*, incluir_novos: bool = False) -> Dados:
    """Executa o pipeline completo de preparacao e devolve o pacote de dados.

    incluir_novos=True concatena dados/novos/ ao dataset original antes da
    limpeza (usado no re-treino via API). O default False mantem EDA,
    observacao e treino manual inalterados.
    """
    bruto = carregar_bruto()
    if incluir_novos:
        extras = carregar_novos()
        if extras:
            bruto = pd.concat([bruto, *extras], ignore_index=True)
            print(f"Incluidos {len(extras)} arquivo(s) de dados/novos -> {len(bruto)} linhas")
    df = aplicar_engenharia_atributos(limpar(bruto))
    X, y = separar_x_y(df)
    pre = construir_preprocessador(X)

    # separa o teste ANTES de qualquer fit (evita data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANHO_TESTE, random_state=SEED, stratify=y
    )

    # validacao cruzada estratificada no treino (mantem ~27% de churn por dobra)
    cv = StratifiedKFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEED)

    return Dados(df, X, y, X_train, X_test, y_train, y_test, pre, cv)


def main() -> None:
    dados = preparar_tudo()
    print("\nAtributos usados:", list(dados.X.columns))
    print("Treino:", dados.X_train.shape, "| Teste:", dados.X_test.shape)
    print("\nPrimeiras linhas:")
    print(dados.X.head())


if __name__ == "__main__":
    main()
