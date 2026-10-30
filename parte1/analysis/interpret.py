"""Variables de riesgo: SHAP sobre el MLP y contraste con la regresión logística."""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from parte1.data_prep.dataset import load_pd_dataset
from parte1.modeling.mlp import load_mlp
from parte1.common.utils import FIGURES_DIR, MODELS_DIR, SEED, records, update_analysis


# Nombre de la variable original a la que pertenece una columna one-hot o indicador.
def _group(name: str, categorical: list[str]) -> str:
    """Variable original a la que pertenece una columna preprocesada."""
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):]
    for c in categorical:
        if name.startswith(c + "_"):
            return c
    return name


# Envuelve la red para que SHAP reciba el logit desde un arreglo NumPy.
def _logit_fn(model):
    # Logit del MLP para un lote de filas ya preprocesadas.
    def f(X):
        with torch.no_grad():
            return model(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy()
    return f


# SHAP sobre el MLP y SHAP exacto de la logística; compara importancias.
def run_interpretability(n_background: int = 100, n_explain: int = 300, nsamples: int = 1000,
                         mode: str = "matured", save: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = load_pd_dataset(mode)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    pre_mlp, mlp = load_mlp("pd_mlp")
    pipe = joblib.load(MODELS_DIR / "pd_logistic.joblib")
    pre_lr, lr = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    names = list(pre_mlp.get_feature_names_out())
    categorical = next((cols for n, _, cols in pre_mlp.transformers_ if n == "cat"), [])

    bg_rows = train.iloc[rng.choice(len(train), n_background, replace=False)]
    ex_rows = test.iloc[rng.choice(len(test), n_explain, replace=False)]
    Z_bg, Z_ex = pre_mlp.transform(bg_rows), pre_mlp.transform(ex_rows)

    # MLP: SHAP sobre el logit (KernelExplainer, agnóstico al modelo)
    f = _logit_fn(mlp)
    explainer = shap.KernelExplainer(f, Z_bg)
    sv_mlp = np.asarray(explainer.shap_values(Z_ex, nsamples=nsamples)).reshape(len(Z_ex), -1)
    err = np.abs(explainer.expected_value + sv_mlp.sum(axis=1) - f(Z_ex)).max()
    print(f"Aditividad SHAP (error máximo en logit): {err:.2e}")

    # Logística: SHAP exacto (modelo lineal) con el mismo fondo y las mismas filas
    Zl_bg, Zl_ex = pre_lr.transform(bg_rows), pre_lr.transform(ex_rows)
    sv_lr = lr.coef_.ravel() * (Zl_ex - Zl_bg.mean(axis=0))

    groups = [_group(n, categorical) for n in names]

    # Importancia media |SHAP| sumando las columnas de una misma variable.
    def importance(sv):
        grouped = pd.DataFrame(sv, columns=names).T.groupby(groups).sum().T
        return grouped.abs().mean()

    out = pd.DataFrame({"mlp_shap_abs": importance(sv_mlp), "lr_shap_abs": importance(sv_lr)})
    out = out.sort_values("mlp_shap_abs", ascending=False)
    out["rango_mlp"] = out["mlp_shap_abs"].rank(ascending=False).astype(int)
    out["rango_lr"] = out["lr_shap_abs"].rank(ascending=False).astype(int)
    out.index.name = "variable"

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        update_analysis("shap_importancia", records(out.reset_index()))
        shap.summary_plot(sv_mlp, Z_ex, feature_names=names, max_display=15, show=False)
        plt.title(f"SHAP sobre el logit del MLP (test, {n_explain} clientes)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pd_mlp_shap.png", dpi=150)
        plt.close()
    print(out.head(15).to_string())
    return out


if __name__ == "__main__":
    run_interpretability()