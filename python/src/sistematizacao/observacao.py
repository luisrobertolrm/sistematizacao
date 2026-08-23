"""Etapa 8 - observacao do modelo em uso (alimentacao incremental do binario).

Alimenta o binario em lotes usando o limiar calibrado na etapa 7 e acompanha
Prec_churn, contatos sugeridos e demais metricas acumuladas.
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
    precision_score,
    roc_auc_score,
)

from sistematizacao.carregamento_inicial import ALVO, preparar_tudo
from sistematizacao.carregar_modelo import (
    carregar_limiar,
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
    """Alimenta o binario em lotes e mede metricas acumuladas a cada lote."""
    modelo = carregar_modelo()
    meta = carregar_metadados()
    thr = carregar_limiar()

    Xr = preparar_entrada(X).reset_index(drop=True)
    yr = y.reset_index(drop=True)
    lotes = np.array_split(np.arange(len(Xr)), n_lotes)

    prob: list[float] = []
    pred: list[int] = []
    real: list[int] = []
    historico: list[dict[str, float]] = []

    print(
        f"Modelo: {meta.get('nome', '?')}  |  limiar={thr:.3f}  |  "
        f"{len(Xr)} amostras em {n_lotes} lotes\n"
    )
    print(
        f"{'lote':<6}{'n_acum':<9}{'contatos':<10}{'Prec':<9}{'Acc':<9}"
        f"{'F1_macro':<11}{'F1_churn':<11}{'AUC':<8}"
    )

    for i, ids in enumerate(lotes, 1):
        Xb = Xr.iloc[ids]
        pb = modelo.predict_proba(Xb)[:, 1]
        pb_pred = (pb >= thr).astype(int)
        prob += pb.tolist()
        pred += pb_pred.tolist()
        real += yr.iloc[ids].tolist()

        prec = precision_score(real, pred, zero_division=0) if sum(pred) else 0.0
        linha = {
            "lote": i,
            "n_acumulado": len(real),
            "contatos": sum(pred),
            "prec": prec,
            "acc": accuracy_score(real, pred),
            "f1_macro": f1_score(real, pred, average="macro"),
            "f1_churn": f1_score(real, pred),
            "auc": roc_auc_score(real, prob),
        }
        historico.append(linha)
        print(
            f"{i:<6}{linha['n_acumulado']:<9}{linha['contatos']:<10}{linha['prec']:<9.4f}"
            f"{linha['acc']:<9.4f}{linha['f1_macro']:<11.4f}"
            f"{linha['f1_churn']:<11.4f}{linha['auc']:<8.4f}"
        )

    print("\n=== Avaliacao final (binario, dados acumulados) ===")
    prec_final = precision_score(real, pred, zero_division=0) if sum(pred) else 0.0
    print(f"Total contatos sugeridos: {sum(pred)}  |  Prec_churn={prec_final:.4f}")
    print(classification_report(real, pred, target_names=["Fica", "Churn"]))
    return historico


def salvar_historico(historico: list[dict[str, float]]) -> None:
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
            "prec_churn_final": historico[-1]["prec"] if historico else None,
            "f1_macro_final": historico[-1]["f1_macro"] if historico else None,
        },
    )
    print(f"Avaliacao salva em: {destino} (modelo em producao: {run_producao})")


def main() -> None:
    dados = preparar_tudo()
    historico = observar_em_lotes(dados.X_test, dados.y_test)
    salvar_historico(historico)

    primeiro, ultimo = historico[0], historico[-1]
    delta = ultimo["prec"] - primeiro["prec"]
    print(
        f"\nVariacao da Prec_churn entre o 1o e o ultimo lote: {delta:+.4f} "
        f"({primeiro['prec']:.4f} -> {ultimo['prec']:.4f})"
    )
    print(f"Alvo monitorado: {ALVO}")


if __name__ == "__main__":
    main()
