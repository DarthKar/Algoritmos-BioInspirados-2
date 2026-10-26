"""Scorecard final de la Parte 1, construida desde la regresión logística guardada."""
import json

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.pd_model.dataset import load_pd_dataset
from src.pd_model.scorecard import ScorecardParams, prob_to_score, scorecard_table
from src.utils import MODELS_DIR, RESULTS_DIR


# Scorecard final desde la logística: puntajes, tabla de puntos y bandas por decil.
def run_scorecard(params: ScorecardParams = ScorecardParams(), name: str = "pd_logistic",
                  mode: str = "matured", save: bool = True) -> dict:
    pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
    pre, model = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    df = load_pd_dataset(mode)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    Z = {s: pre.transform(p) for s, p in parts.items()}
    scores = {}
    for s, p in parts.items():
        pd_bad = model.predict_proba(Z[s])[:, 1]
        scores[s] = pd.DataFrame({"id": p["id"].to_numpy(), "target": p["target"].to_numpy(),
                                  "pd": pd_bad, "score": prob_to_score(pd_bad, params)})

    # Tabla de puntos en unidades originales + impacto de 1 desviación estándar (comparable entre variables)
    base, table = scorecard_table(model.coef_, model.intercept_[0], pre, params)
    std_orig = Z["train"].std(axis=0) * table["desv_est"].to_numpy()
    table["impacto_1sd"] = table["puntos_por_unidad"].abs() * std_orig
    table = table.sort_values("impacto_1sd", ascending=False).reset_index(drop=True)

    # Bandas por decil en test (banda 1 = puntajes más bajos = más riesgo)
    t = scores["test"].copy()
    t["banda"] = pd.qcut(t["score"], 10, labels=False, duplicates="drop") + 1
    bands = (t.groupby("banda")
             .agg(n=("target", "size"), score_min=("score", "min"), score_max=("score", "max"),
                  pd_media=("pd", "mean"), tasa_malos=("target", "mean"))
             .reset_index())

    tasa_train = float(scores["train"]["target"].mean())
    summary = {
        "parametros": {"base_score": params.base_score, "base_odds": params.base_odds, "pdo": params.pdo,
                       "factor": float(params.factor), "offset": float(params.offset)},
        "justificacion_base_odds": {"tasa_malos_train": tasa_train,
                                    "odds_train": (1 - tasa_train) / tasa_train},
        "puntos_base": float(base),
        "percentiles_puntaje": {s: scores[s]["score"].quantile([.01, .25, .5, .75, .99]).round(1).tolist()
                                for s in scores},
        "auc_test_con_puntaje": float(roc_auc_score(scores["test"]["target"], -scores["test"]["score"])),
    }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS_DIR / "pd_scorecard_tabla.csv", index=False)
        bands.to_csv(RESULTS_DIR / "pd_scorecard_bandas.csv", index=False)
        (RESULTS_DIR / "pd_scorecard_resumen.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(table.head(12).to_string())
    print(bands.to_string())
    return summary


if __name__ == "__main__":
    run_scorecard()