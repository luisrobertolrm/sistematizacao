"""Etapas 4–7 — selecao, tuning, avaliacao e limiar (notebook curso_2).

Fluxo:
  4) baseline sem Optuna (5 modelos, CV)
  5) Optuna em LR/RF/GB/SVM otimizando Precisao (Churn); vencedor = maior Prec
     entre modelos com Acc>=0.80 e AUC>=0.82 (F1_churn desempata)
  6) avaliacao do vencedor no hold-out
  7) calibracao de limiar para maximizar Prec_churn com Acc>=0.80

Persiste bundle {pipe, threshold, modelo} em modelos/treinamentos/<run_id>/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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

N_TRIALS = 30
META_ACC = 0.80
META_AUC = 0.82
META_F1_MACRO = 0.68

MODELOS_BASELINE: dict[str, Any] = {
    "LogisticRegression": LogisticRegression(random_state=SEED, max_iter=2000),
    "RandomForest": RandomForestClassifier(random_state=SEED),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    "SVM": SVC(random_state=SEED),
    "KNN": KNeighborsClassifier(),
}


@dataclass
class ResultadoModelo:
    nome: str
    f1: float
    prec: float
    f1_macro: float
    auc: float
    acc: float
    pipe: Pipeline
    params: dict[str, Any] | None = None


def _metricas_cv(scores: dict[str, np.ndarray]) -> dict[str, float]:
    return {k.removeprefix("test_"): float(v.mean()) for k, v in scores.items()}


def baseline_cv(dados: Dados) -> dict[str, ResultadoModelo]:
    """Etapa 4 — triagem inicial sem Optuna."""
    metricas = ["f1", "precision", "recall", "roc_auc", "accuracy", "f1_macro"]
    resultados: dict[str, ResultadoModelo] = {}
    print("=== Etapa 4: baseline (sem Optuna) ===")
    for nome, modelo in MODELOS_BASELINE.items():
        pipe = Pipeline([("pre", dados.pre), ("clf", modelo)])
        sc = cross_validate(pipe, dados.X_train, dados.y_train, cv=dados.cv, scoring=metricas)
        m = _metricas_cv(sc)
        resultados[nome] = ResultadoModelo(
            nome=nome,
            f1=m["f1"],
            prec=m["precision"],
            f1_macro=m["f1_macro"],
            auc=m["roc_auc"],
            acc=m["accuracy"],
            pipe=pipe,
        )
        print(
            f"{nome:20s} F1={m['f1']:.4f}  Prec={m['precision']:.4f}  "
            f"AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}"
        )
    melhor = max(resultados, key=lambda k: resultados[k].f1)
    print(f"\nMelhor baseline (F1_churn): {melhor} ({resultados[melhor].f1:.4f})")
    return resultados


def build_model_optuna(nome: str, trial: optuna.Trial):
    """Espaco de busca Optuna por algoritmo (etapa 5)."""
    if nome == "LogisticRegression":
        cw_choice = trial.suggest_categorical("class_weight", [None, "balanced", "custom"])
        if cw_choice == "custom":
            w1 = trial.suggest_float("w1", 1.1, 2.5)
            class_weight: Any = {0: 1, 1: w1}
        else:
            class_weight = cw_choice
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-2, 10, log=True),
            class_weight=class_weight,
            max_iter=2000,
            random_state=SEED,
        )
    if nome == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=SEED,
        )
    if nome == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            random_state=SEED,
        )
    if nome == "SVM":
        svc = SVC(
            C=trial.suggest_float("C", 1e-2, 100, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            random_state=SEED,
        )
        return CalibratedClassifierCV(svc, ensemble=False)
    msg = f"Modelo nao tunavel: {nome}"
    raise ValueError(msg)


def otimizar_todos(dados: Dados, n_trials: int = N_TRIALS) -> dict[str, ResultadoModelo]:
    """Etapa 5 — Optuna por modelo, objetivo = Precisao (Churn)."""
    tuning: dict[str, ResultadoModelo] = {}
    print("\n=== Etapa 5: Optuna (maximiza Prec_churn) ===")
    for nome in ["LogisticRegression", "RandomForest", "GradientBoosting", "SVM"]:
        est = optuna.create_study(direction="maximize")
        est.optimize(
            lambda t, nm=nome: cross_val_score(
                Pipeline([("pre", dados.pre), ("clf", build_model_optuna(nm, t))]),
                dados.X_train,
                dados.y_train,
                cv=dados.cv,
                scoring="precision",
            ).mean(),
            n_trials=n_trials,
        )
        pipe = Pipeline([("pre", dados.pre), ("clf", build_model_optuna(nome, est.best_trial))])
        sc = cross_validate(
            pipe,
            dados.X_train,
            dados.y_train,
            cv=dados.cv,
            scoring=["f1", "precision", "f1_macro", "roc_auc", "accuracy"],
        )
        m = _metricas_cv(sc)
        tuning[nome] = ResultadoModelo(
            nome=nome,
            f1=m["f1"],
            prec=m["precision"],
            f1_macro=m["f1_macro"],
            auc=m["roc_auc"],
            acc=m["accuracy"],
            pipe=pipe,
            params=est.best_params,
        )
        print(
            f"{nome:18s} Prec_churn={m['precision']:.4f}  F1_churn={m['f1']:.4f}  "
            f"F1_macro={m['f1_macro']:.4f}  AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}"
        )
        print(f"    params: {est.best_params}")
    return tuning


def bate_metas_rubrica(v: ResultadoModelo) -> bool:
    return v.acc >= META_ACC and v.auc >= META_AUC


def selecionar_vencedor(tuning: dict[str, ResultadoModelo]) -> ResultadoModelo:
    """Maior Prec_churn entre elegiveis; F1_churn desempata."""
    elegiveis = {k: v for k, v in tuning.items() if bate_metas_rubrica(v)}
    excluidos = {k: v for k, v in tuning.items() if not bate_metas_rubrica(v)}
    if excluidos:
        print("\n--- Excluidos (Acc/AUC abaixo do minimo) ---")
        for nome, v in excluidos.items():
            motivos = []
            if v.acc < META_ACC:
                motivos.append(f"Acc={v.acc:.4f}<{META_ACC}")
            if v.auc < META_AUC:
                motivos.append(f"AUC={v.auc:.4f}<{META_AUC}")
            print(f"  {nome:18s} Prec={v.prec:.4f}  ({'; '.join(motivos)})")
    pool = elegiveis or tuning
    nome = max(pool, key=lambda k: (pool[k].prec, pool[k].f1))
    vencedor = pool[nome]
    print(f"\nVencedor (CV): {nome}  Prec={vencedor.prec:.4f}  Acc={vencedor.acc:.4f}")
    return vencedor


def calibrar_limiar(pipe: Pipeline, dados: Dados) -> float:
    """Etapa 7 — limiar que maximiza Prec_churn com Acc>=META_ACC na CV."""
    oof_prob = cross_val_predict(
        pipe, dados.X_train, dados.y_train, cv=dados.cv, method="predict_proba"
    )[:, 1]
    melhor: tuple[float, float, float, float] | None = None
    for thr in np.arange(0.35, 0.85, 0.005):
        pred = (oof_prob >= thr).astype(int)
        if pred.sum() == 0:
            continue
        acc = accuracy_score(dados.y_train, pred)
        prec = precision_score(dados.y_train, pred, zero_division=0)
        if acc >= META_ACC:
            if melhor is None or prec > melhor[0]:
                melhor = (prec, thr, acc, recall_score(dados.y_train, pred, zero_division=0))
    if melhor:
        print(
            f"Limiar={melhor[1]:.3f}  CV Prec_churn={melhor[0]:.4f}  "
            f"Acc={melhor[2]:.4f}  Rec={melhor[3]:.4f}"
        )
        return melhor[1]
    print(f"Limiar=0.500 (default — nenhum limiar manteve Acc>={META_ACC} na CV)")
    return 0.5


def avaliar(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    limiar: float = 0.5,
    titulo: str = "TESTE (hold-out 20%)",
) -> dict[str, float]:
    """Mede UMA vez no hold-out (etapas 6 e 7)."""
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= limiar).astype(int)
    metricas = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec": recall_score(y_test, y_pred, zero_division=0),
        "f1_churn": f1_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc": roc_auc_score(y_test, y_prob),
        "n_contatos": int(y_pred.sum()),
    }
    bate = (
        metricas["acc"] >= META_ACC
        and metricas["f1_macro"] >= META_F1_MACRO
        and metricas["auc"] >= META_AUC
    )
    print(f"\n=== {titulo} ===")
    print(
        f"Acc={metricas['acc']:.4f}  Prec_churn={metricas['prec']:.4f}  "
        f"Rec={metricas['rec']:.4f}  F1_churn={metricas['f1_churn']:.4f}  "
        f"F1_macro={metricas['f1_macro']:.4f}  AUC={metricas['auc']:.4f}"
    )
    print(
        f"Contatos sugeridos: {metricas['n_contatos']} de {len(y_test)} "
        f"({100 * metricas['n_contatos'] / len(y_test):.1f}%)"
    )
    print(f"Rubrica A3: {'OK' if bate else 'revisar'}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Fica", "Churn"]))
    return metricas


def coeficientes(pipe: Pipeline, n: int = 10) -> None:
    """Explicabilidade quando o classificador expoe coef_."""
    import pandas as pd  # noqa: PLC0415

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "calibrated_classifiers_"):
        clf = clf.calibrated_classifiers_[0].estimator
    if not hasattr(clf, "coef_"):
        print("\n(modelo sem coeficientes lineares diretos)")
        return
    nomes = pipe.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(clf.coef_[0], index=nomes).sort_values()
    print("\n=== Empurram pra CHURN ===")
    print(coef.tail(n)[::-1])
    print("\n=== Seguram o cliente (NAO-churn) ===")
    print(coef.head(n))


def salvar_run(
    bundle: dict[str, Any],
    vencedor: ResultadoModelo,
    limiar: float,
    metricas: dict[str, float],
    baseline_melhor: str,
    *,
    run_id: str,
    destino: Path,
    n_amostras: int,
    incluiu_novos: bool,
    arquivos_novos: list[str],
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    joblib_dst = destino / "modelo_churn.joblib"
    tmp = joblib_dst.with_name(joblib_dst.name + ".tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, joblib_dst)  # noqa: PTH105

    pipe = bundle["pipe"]
    escrever_json_atomico(
        destino / "metadados.json",
        {
            "run_id": run_id,
            "nome": f"{vencedor.nome} (tunado + limiar)",
            "modelo": vencedor.nome,
            "params": vencedor.params,
            "threshold": limiar,
            "metricas_teste": metricas,
            "metricas_cv": {
                "prec_churn": vencedor.prec,
                "f1_churn": vencedor.f1,
                "f1_macro": vencedor.f1_macro,
                "acc": vencedor.acc,
                "auc": vencedor.auc,
            },
            "baseline_melhor_etapa4": baseline_melhor,
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
    """Pipeline completo (etapas 4–7) versionado por run."""
    run_id = novo_run_id()
    destino = dir_treinamento(run_id)

    arquivos_novos: list[str] = []
    if incluir_novos:
        dir_novos = DIR_DADOS / "novos"
        if dir_novos.exists():
            arquivos_novos = [csv.name for csv in sorted(dir_novos.glob("*.csv"))]

    dados = preparar_tudo(incluir_novos=incluir_novos)
    gerar_eda(dados.df, destino / "eda")

    baselines = baseline_cv(dados)
    baseline_melhor = max(baselines, key=lambda k: baselines[k].f1)

    tuning = otimizar_todos(dados, n_trials)
    vencedor = selecionar_vencedor(tuning)

    print("\n=== Etapa 6: vencedor da etapa 5 no hold-out (limiar=0.5) ===")
    clf = tuning[vencedor.nome].pipe.named_steps["clf"]
    pipe_fit = Pipeline([("pre", dados.pre), ("clf", clf)]).fit(dados.X_train, dados.y_train)
    avaliar(pipe_fit, dados.X_test, dados.y_test, limiar=0.5, titulo="Pre-limiar")

    limiar = calibrar_limiar(pipe_fit, dados)
    metricas = avaliar(
        pipe_fit,
        dados.X_test,
        dados.y_test,
        limiar=limiar,
        titulo="Final com limiar calibrado",
    )
    coeficientes(pipe_fit)

    bundle = {"pipe": pipe_fit, "threshold": limiar, "modelo": vencedor.nome}
    salvar_run(
        bundle,
        vencedor,
        limiar,
        metricas,
        baseline_melhor,
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
            "modelo": vencedor.nome,
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
        "modelo": vencedor.nome,
        "threshold": limiar,
        "n_amostras": len(dados.X),
        "params": vencedor.params,
        "metricas": metricas,
    }


def main() -> None:
    resultado = treinar_publicar(incluir_novos=False, promover_auto=True)
    print(
        "\nrun:", resultado["run_id"],
        "| modelo:", resultado["modelo"],
        "| limiar:", f"{resultado['threshold']:.3f}",
        "| promovido:", resultado["promovido"],
    )


if __name__ == "__main__":
    main()
