"""Monitoramento diario: drift do modelo, e-mail e retreino condicionado.

Pisos de alarme (estrategia de monitoramento do relatorio):
  Prec_churn < 0.65, AUC < 0.80, ou fracao de contatos longe de ~17%.
Retreino so promove se Acc >= 0.80 e AUC >= 0.82.
"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)

from sistematizacao.carregamento_inicial import (
    aplicar_engenharia_atributos,
    carregar_novos,
    limpar,
    preparar_tudo,
    separar_x_y,
)
from sistematizacao.carregar_modelo import (
    carregar_limiar,
    carregar_metadados,
    carregar_modelo,
    limpar_cache,
    preparar_entrada,
)
from sistematizacao.runs import ler_manifesto, promover
from sistematizacao.treinamento_avaliacao import META_ACC, META_AUC, treinar_publicar

PREC_MIN = 0.65
AUC_MIN_DRIFT = 0.80
FRACAO_CONTATOS_REF = 0.171
FRACAO_TOLERANCIA = 0.05
SMTP_PORTA_PADRAO = 587


def _lote_avaliacao() -> tuple[pd.DataFrame, pd.Series, str]:
    """Prefere dados/novos/ (rotulados); senao usa o hold-out do dataset base."""
    extras = carregar_novos()
    if extras:
        bruto = pd.concat(extras, ignore_index=True)
        df = aplicar_engenharia_atributos(limpar(bruto))
        X, y = separar_x_y(df)
        return X, y, "dados/novos"
    dados = preparar_tudo(incluir_novos=False)
    return dados.X_test, dados.y_test, "hold-out"


def metricas_producao(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Avalia o binario de producao (pipe + limiar) em um conjunto rotulado."""
    modelo = carregar_modelo()
    thr = carregar_limiar()
    Xr = preparar_entrada(X).reset_index(drop=True)
    yr = y.reset_index(drop=True)
    n = len(yr)
    if n == 0:
        return {
            "acc": 0.0,
            "prec": 0.0,
            "auc": 0.0,
            "f1_macro": 0.0,
            "n": 0.0,
            "n_contatos": 0.0,
            "fracao_contatos": 0.0,
        }
    prob = modelo.predict_proba(Xr)[:, 1]
    pred = (prob >= thr).astype(int)
    n_contatos = int(pred.sum())
    auc = 0.0
    if yr.min() != yr.max():
        auc = float(roc_auc_score(yr, prob))
    return {
        "acc": float(accuracy_score(yr, pred)),
        "prec": float(precision_score(yr, pred, zero_division=0)),
        "auc": auc,
        "f1_macro": float(f1_score(yr, pred, average="macro")),
        "n": float(n),
        "n_contatos": float(n_contatos),
        "fracao_contatos": n_contatos / n,
    }


def motivos_drift(m: dict[str, float]) -> list[str]:
    """Regras de alarme da estrategia de monitoramento."""
    motivos: list[str] = []
    if m["prec"] < PREC_MIN:
        motivos.append(f"Prec_churn={m['prec']:.4f}<{PREC_MIN}")
    if m["auc"] < AUC_MIN_DRIFT:
        motivos.append(f"AUC={m['auc']:.4f}<{AUC_MIN_DRIFT}")
    lo = FRACAO_CONTATOS_REF - FRACAO_TOLERANCIA
    hi = FRACAO_CONTATOS_REF + FRACAO_TOLERANCIA
    frac = m["fracao_contatos"]
    if frac < lo or frac > hi:
        motivos.append(f"fracao_contatos={frac:.3f} fora de {lo:.3f}-{hi:.3f} (ref ~17%)")
    return motivos


def _corpo_email(relatorio: dict[str, Any]) -> str:
    return json.dumps(relatorio, indent=2, ensure_ascii=False, default=str)


def enviar_email(assunto: str, corpo: str) -> bool:
    """Envia e-mail se SMTP_HOST e EMAIL_TO estiverem definidos. Senao so registra."""
    host = os.getenv("SMTP_HOST", "").strip()
    destinos = [e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()]
    if not host or not destinos:
        print("E-mail nao enviado: defina SMTP_HOST e EMAIL_TO.")
        print(assunto)
        print(corpo)
        return False

    msg = EmailMessage()
    msg["Subject"] = assunto
    msg["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "churn@localhost"))
    msg["To"] = ", ".join(destinos)
    msg.set_content(corpo)

    porta = int(os.getenv("SMTP_PORT", str(SMTP_PORTA_PADRAO)))
    usuario = os.getenv("SMTP_USER", "")
    senha = os.getenv("SMTP_PASSWORD", "")
    with smtplib.SMTP(host, porta, timeout=30) as smtp:
        smtp.starttls()
        if usuario:
            smtp.login(usuario, senha)
        smtp.send_message(msg)
    print(f"E-mail enviado para: {msg['To']}")
    return True


def _promove_se_rubrica(resultado: dict[str, Any]) -> bool:
    metricas = resultado.get("metricas") or {}
    acc = float(metricas.get("acc", 0))
    auc = float(metricas.get("auc", 0))
    if acc >= META_ACC and auc >= META_AUC:
        promover(str(resultado["run_id"]))
        limpar_cache()
        carregar_modelo()
        return True
    print(
        f"Candidato nao promovido: Acc={acc:.4f} AUC={auc:.4f} "
        f"(precisa Acc>={META_ACC} e AUC>={META_AUC})"
    )
    return False


def ciclo_diario(*, retreinar: bool = True) -> dict[str, Any]:
    """Avalia producao, alarma por e-mail se drift, e retreina so nesse caso."""
    X, y, origem = _lote_avaliacao()
    atuais = metricas_producao(X, y)
    motivos = motivos_drift(atuais)
    meta = carregar_metadados()
    relatorio: dict[str, Any] = {
        "avaliado_em": datetime.now(UTC).isoformat(),
        "run_producao": ler_manifesto().get("producao"),
        "modelo": meta.get("nome"),
        "origem_lote": origem,
        "metricas_atuais": atuais,
        "drift": bool(motivos),
        "motivos": motivos,
        "email_enviado": False,
        "retreinado": False,
        "promovido": False,
        "run_candidato": None,
        "metricas_candidato": None,
    }

    if not motivos:
        print("Sem drift: modelo de producao permanece.")
        return relatorio

    assunto = "Churn: drift detectado no modelo de producao"
    relatorio["email_enviado"] = enviar_email(assunto, _corpo_email(relatorio))

    if not retreinar:
        return relatorio

    print("Drift: disparando retreino (sem promover automaticamente).")
    resultado = treinar_publicar(incluir_novos=True, promover_auto=False)
    relatorio["retreinado"] = True
    relatorio["run_candidato"] = resultado.get("run_id")
    relatorio["metricas_candidato"] = resultado.get("metricas")
    relatorio["promovido"] = _promove_se_rubrica(resultado)
    enviar_email(
        "Churn: resultado do retreino apos drift",
        _corpo_email(relatorio),
    )
    return relatorio


def main() -> None:
    relatorio = ciclo_diario(retreinar=True)
    print(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
