"""Sistematizacao da escolha de modelo - previsao de churn."""


def main() -> None:
    print(
        "Sistematizacao - previsao de churn\n"
        "\n"
        "Etapas disponiveis:\n"
        "  python -m sistematizacao.carregamento_inicial   # 1. carga, limpeza e preparacao\n"
        "  python -m sistematizacao.eda                    # 2. analise exploratoria\n"
        "  python -m sistematizacao.treinamento_avaliacao  # 7. tuning + avaliacao + binario\n"
        "  python -m sistematizacao.observacao             # 8. observacao em lotes\n"
        "  python -m sistematizacao.carregar_modelo        # carrega o binario e preve\n"
        "  python -m sistematizacao.web                    # API HTTP (porta 8000)"
    )
