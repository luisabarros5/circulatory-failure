#!/usr/bin/env python3
"""
02_audit_cluster_features_v2.py

Audita as features que o pipeline MiniRocket antigo selecionará para o clustering.

Decisão principal:
- mantém vasopressor_ativo;
- exclui IDs/tempo/labels/outcomes já blacklistados pelo common.py;
- exclui *_imputed, pois no dataset atual são MÁSCARAS de imputação;
- não altera o pipeline silenciosamente.

Salva um CSV com feature + status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory containing the local feature-store Parquet batches.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path for the feature audit manifest.",
    )
    args = ap.parse_args()

    bundle = Path(__file__).resolve().parents[1] / "core"
    feature_dir = args.feature_dir
    out = args.out

    if not bundle.exists():
        raise SystemExit(f"Bundle MiniRocket não encontrado: {bundle}")

    sys.path.insert(0, str(bundle))
    from common import select_feature_columns, DROP_COLS  # noqa: E402

    paths = sorted(
        p for p in feature_dir.glob("batch_*.parquet")
        if not p.name.endswith("_labels.parquet")
    )
    if not paths:
        raise SystemExit(f"Nenhum batch_*.parquet em {feature_dir}")

    schema = pq.read_schema(paths[0]).names
    selected = select_feature_columns(schema, drop_imputed=True)
    selected_set = set(selected)

    rows = []
    for c in schema:
        if c in selected_set:
            status = "INCLUDE"
            reason = "feature clínica/temporal"
        elif c.endswith("_imputed"):
            status = "EXCLUDE"
            reason = "máscara/flag de imputação"
        elif c in DROP_COLS:
            status = "EXCLUDE"
            reason = "ID, target ou outcome"
        elif c in {"charttime", "rel_charttime", "time", "timestamp"}:
            status = "EXCLUDE"
            reason = "timestamp"
        else:
            status = "EXCLUDE"
            reason = "filtrada pelo seletor"

        rows.append({"feature": c, "status": status, "reason": reason})

    manifest = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)

    print("=== AUDITORIA DE FEATURES ===")
    print(f"Arquivo-base: {paths[0]}")
    print(f"Colunas no schema: {len(schema)}")
    print(f"Features selecionadas: {len(selected)}")
    print(f"Máscaras *_imputed excluídas: {sum(c.endswith('_imputed') for c in schema)}")
    print(f"vasopressor_ativo presente no schema: {'vasopressor_ativo' in schema}")
    print(f"vasopressor_ativo entra no clustering: {'vasopressor_ativo' in selected_set}")
    print(f"falencia entra no clustering: {'falencia' in selected_set}")
    print(f"Manifesto salvo em: {out}")

    if "vasopressor_ativo" not in selected_set:
        raise SystemExit(
            "ERRO: vasopressor_ativo não entrou. Não prossiga antes de revisar."
        )

    forbidden_found = sorted(
        set(selected) & {
            "stay_id", "patientunitstayid", "patientunitstay_id",
            "falencia", "falencia_point", "falencia_any", "falencia_onset",
            "mortality",
        }
    )
    if forbidden_found:
        raise SystemExit(f"ERRO: colunas proibidas entraram: {forbidden_found}")

    print("\n[OK] Seleção coerente para a análise principal.")


if __name__ == "__main__":
    main()
