"""Etapa 7 - ajuste de hiperparametros da LogisticRegression (Optuna) + avaliacao.

Otimiza o F1 da classe Churn na validacao cruzada do treino, escolhe o melhor
trial que ainda respeita as metas de Acc e AUC, mede UMA vez no hold-out e
persiste o pipeline treinado como binario em `modelos/treinamentos/<run_id>/`.
Producao (`modelos/modelo_churn.joblib`) so e tocada por `runs.promover()`.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from sistematizacao.carregamento_inicial import (
    DIR_DADOS,
    SEED,
    Dados,
    preparar_tudo,
)
from sistematizacao.eda import gerar_eda
from sistematizacao.runs import (
    dir_treinamento,
    escrever_json_atomico,
    novo_run_id,
    promover,
    registrar_treinamento,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 40
NOME_MODELO = "LogisticRegression (tunada)"

# metas da rubrica usadas como filtro na selecao do trial
META_ACC = 0.80
META_AUC = 0.82


def build_lr(trial: optuna.Trial) -> LogisticRegression:
    """Espaco de busca: C (forca da regularizacao) e class_weight.

    class_weight="balanced" fica no espaco de proposito: o filtro de metas
    mostra que ele sobe o recall da classe Churn mas derruba a accuracy abaixo
    de 0.80, e por isso acaba descartado (a tensao central do problema).
    """
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-2, 100, log=True),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        max_iter=1000,
        random_state=SEED,
    )


def otimizar(dados: Dados, n_trials: int = N_TRIALS) -> optuna.Trial:
    """Roda o Optuna e devolve o melhor trial que respeita Acc>=0.80 e AUC>=0.82."""

    def objetivo(trial: optuna.Trial) -> float:
        pipe = Pipeline([("pre", dados.pre), ("clf", build_lr(trial))])
        sc = cross_validate(
            pipe,
            dados.X_train,
            dados.y_train,
            cv=dados.cv,
            scoring=["f1", "roc_auc", "accuracy"],
        )
        trial.set_user_attr("acc", sc["test_accuracy"].mean())
        trial.set_user_attr("auc", sc["test_roc_auc"].mean())
        return sc["test_f1"].mean()  # otimiza F1 da classe Churn na CV do treino

    est = optuna.create_study(direction="maximize")
    est.optimize(objetivo, n_trials=n_trials)

    ok = [
        t
        for t in est.trials
        if t.user_attrs.get("acc", 0) >= META_ACC and t.user_attrs.get("auc", 0) >= META_AUC
    ]
    melhor = max(ok, key=lambda t: t.value) if ok else est.best_trial
    if not ok:
        print("(atencao: nenhum trial respeitou as metas; usando o de maior F1)")
    print(
        "melhores params:",
        melhor.params,
        f"(CV: F1_churn={melhor.value:.4f}  "
        f"Acc={melhor.user_attrs['acc']:.4f}  AUC={melhor.user_attrs['auc']:.4f})",
    )
    return melhor


def treinar(dados: Dados, params: dict[str, Any]) -> Pipeline:
    """Treina o pipeline final no treino inteiro."""
    return Pipeline(
        [
            ("pre", dados.pre),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED, **params)),
        ]
    ).fit(dados.X_train, dados.y_train)


def avaliar(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Mede UMA vez no hold-out de 20% (numero final honesto)."""
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metricas = {
        "acc": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_churn": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }
    print("\n=== TESTE (hold-out 20%) ===")
    print(
        f"Acc={metricas['acc']:.4f}  F1_macro={metricas['f1_macro']:.4f}  "
        f"F1_churn={metricas['f1_churn']:.4f}  AUC={metricas['auc']:.4f}"
    )
    print()
    print(classification_report(y_test, y_pred, target_names=["Fica", "Churn"]))
    return metricas


def coeficientes(pipe: Pipeline, n: int = 10) -> None:
    """Explicabilidade: os pesos do modelo linear."""
    import pandas as pd  # noqa: PLC0415

    nomes = pipe.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=nomes).sort_values()
    print("\n=== Empurram pra CHURN ===")
    print(coef.tail(n)[::-1])
    print("\n=== Seguram o cliente (NAO-churn) ===")
    print(coef.head(n))


def salvar_run(
    pipe: Pipeline,
    params: dict[str, Any],
    metricas: dict[str, float],
    *,
    run_id: str,
    destino: Path,
    n_amostras: int,
    incluiu_novos: bool,
    arquivos_novos: list[str],
) -> None:
    """Grava modelo + metadados DENTRO de treinamentos/<run_id>/ (escrita atomica)."""
    destino.mkdir(parents=True, exist_ok=True)

    joblib_dst = destino / "modelo_churn.joblib"
    tmp = joblib_dst.with_name(joblib_dst.name + ".tmp")
    joblib.dump(pipe, tmp)
    os.replace(tmp, joblib_dst)  # noqa: PTH105

    escrever_json_atomico(
        destino / "metadados.json",
        {
            "run_id": run_id,
            "nome": NOME_MODELO,
            "params": params,
            "metricas_teste": metricas,
            "treinado_em": datetime.now(UTC).isoformat(),
            "n_amostras": n_amostras,
            "incluiu_novos": incluiu_novos,
            "arquivos_novos_usados": arquivos_novos,
            "colunas_entrada": list(pipe.feature_names_in_)
            if hasattr(pipe, "feature_names_in_")
            else [],
        },
    )
    print(f"\nRun salvo em: {destino}")


def treinar_publicar(
    *,
    incluir_novos: bool = True,
    promover_auto: bool = True,
    n_trials: int = N_TRIALS,
) -> dict[str, Any]:
    """Mesmo caminho do treino desta etapa, mas versionado por run.

    Grava treinamentos/<run_id>/ (modelo + metadados + eda). Se promover_auto,
    copia o run para producao. Devolve run_id + metricas.
    """
    run_id = novo_run_id()
    destino = dir_treinamento(run_id)

    arquivos_novos: list[str] = []
    if incluir_novos:
        dir_novos = DIR_DADOS / "novos"
        if dir_novos.exists():
            arquivos_novos = [csv.name for csv in sorted(dir_novos.glob("*.csv"))]

    dados = preparar_tudo(incluir_novos=incluir_novos)

    gerar_eda(dados.df, destino / "eda")  # EDA do dataset deste run

    melhor = otimizar(dados, n_trials)
    pipe = treinar(dados, melhor.params)
    metricas = avaliar(pipe, dados.X_test, dados.y_test)
    coeficientes(pipe)

    salvar_run(
        pipe,
        melhor.params,
        metricas,
        run_id=run_id,
        destino=destino,
        n_amostras=len(dados.X),
        incluiu_novos=incluir_novos,
        arquivos_novos=arquivos_novos,
    )
    registrar_treinamento(
        run_id,
        {
            "treinado_em": datetime.now(UTC).isoformat(),
            "metricas": metricas,
            "n_amostras": len(dados.X),
            "incluiu_novos": incluir_novos,
            "arquivos_novos_usados": arquivos_novos,
        },
    )

    if promover_auto:
        promover(run_id)

    return {
        "run_id": run_id,
        "promovido": promover_auto,
        "n_amostras": len(dados.X),
        "params": melhor.params,
        "metricas": metricas,
    }


def main() -> None:
    resultado = treinar_publicar(incluir_novos=False, promover_auto=True)
    print("\nrun:", resultado["run_id"], "| promovido:", resultado["promovido"])


if __name__ == "__main__":
    main()
