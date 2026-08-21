"""Etapa 8 - observacao do modelo em uso (alimentacao incremental do binario).

Le o binario gravado pela etapa 7 e alimenta ele em lotes, medindo o
desempenho ACUMULADO a cada lote. Simula o comportamento em producao: os
dados chegam aos poucos e a metrica precisa ser acompanhada ao longo do tempo.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from sistematizacao.carregamento_inicial import (
    ALVO,
    preparar_tudo,
)
from sistematizacao.carregar_modelo import (
    carregar_metadados,
    carregar_modelo,
    preparar_entrada,
)
from sistematizacao.runs import (
    dir_avaliacao,
    ler_manifesto,
    novo_eval_id,
    registrar_avaliacao,
)

if TYPE_CHECKING:
    import pandas as pd

N_LOTES = 5


def observar_em_lotes(
    X: pd.DataFrame, y: pd.Series, n_lotes: int = N_LOTES
) -> list[dict[str, float]]:
    """Alimenta o binario em lotes e mede a metrica acumulada a cada lote."""
    modelo = carregar_modelo()
    meta = carregar_metadados()

    Xr = preparar_entrada(X).reset_index(drop=True)
    yr = y.reset_index(drop=True)
    lotes = np.array_split(np.arange(len(Xr)), n_lotes)

    prob: list[float] = []
    pred: list[int] = []
    real: list[int] = []
    historico: list[dict[str, float]] = []

    print(f"Modelo: {meta.get('nome', '?')}  |  {len(Xr)} amostras em {n_lotes} lotes\n")
    print(f"{'lote':<6}{'n_acum':<9}{'Acc':<9}{'F1_macro':<11}{'F1_churn':<11}{'AUC':<8}")

    for i, ids in enumerate(lotes, 1):
        Xb = Xr.iloc[ids]
        prob += modelo.predict_proba(Xb)[:, 1].tolist()
        pred += modelo.predict(Xb).tolist()
        real += yr.iloc[ids].tolist()

        linha = {
            "lote": i,
            "n_acumulado": len(real),
            "acc": accuracy_score(real, pred),
            "f1_macro": f1_score(real, pred, average="macro"),
            "f1_churn": f1_score(real, pred),
            "auc": roc_auc_score(real, prob),
        }
        historico.append(linha)
        print(
            f"{i:<6}{linha['n_acumulado']:<9}{linha['acc']:<9.4f}"
            f"{linha['f1_macro']:<11.4f}{linha['f1_churn']:<11.4f}{linha['auc']:<8.4f}"
        )

    print("\n=== Avaliacao final (binario, dados acumulados) ===")
    print(classification_report(real, pred, target_names=["Fica", "Churn"]))
    return historico


def salvar_historico(historico: list[dict[str, float]]) -> None:
    """Grava o acompanhamento em avaliacoes/<eval_id>/, ligado ao run em producao."""
    eval_id = novo_eval_id()
    destino = dir_avaliacao(eval_id)
    (destino / "observacao_lotes.json").write_text(
        json.dumps(historico, indent=2), encoding="utf-8"
    )

    run_producao = ler_manifesto().get("producao")
    registrar_avaliacao(
        eval_id,
        {
            "avaliado_em": datetime.now(UTC).isoformat(),
            "run_avaliado": run_producao,
            "f1_macro_final": historico[-1]["f1_macro"] if historico else None,
        },
    )
    print(f"Avaliacao salva em: {destino} (modelo em producao: {run_producao})")


def main() -> None:
    dados = preparar_tudo()
    historico = observar_em_lotes(dados.X_test, dados.y_test)
    salvar_historico(historico)

    primeiro, ultimo = historico[0], historico[-1]
    delta = ultimo["f1_macro"] - primeiro["f1_macro"]
    print(
        f"\nVariacao do F1_macro entre o 1o e o ultimo lote: {delta:+.4f} "
        f"({primeiro['f1_macro']:.4f} -> {ultimo['f1_macro']:.4f})"
    )
    print(f"Alvo monitorado: {ALVO}")


if __name__ == "__main__":
    main()
