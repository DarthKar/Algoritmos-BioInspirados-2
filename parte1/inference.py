"""Adaptador de inferencia para modelos sklearn y MLP PyTorch ya entrenados."""
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd


# Envuelve preprocesador + red para que la app la use como un modelo de sklearn.
class MLPAdapter:
    # Guarda el preprocesador, la red y las columnas de entrada esperadas.
    def __init__(self, preprocessor, network):
        self.preprocessor = preprocessor
        self.network = network
        self.input_columns = list(dict.fromkeys(
            c for name, _, cols in preprocessor.transformers_
            if name not in ("remainder", "drop") for c in cols
        ))

    # Devuelve [P(bueno), P(malo)] por fila, como sklearn.
    def predict_proba(self, X):
        from parte1.modeling.core import predict_proba
        frame = pd.DataFrame(X).reindex(columns=self.input_columns)
        transformed = self.preprocessor.transform(frame)
        p = np.asarray(predict_proba(self.network, transformed), dtype=float).reshape(-1)
        return np.column_stack((1.0 - p, p))


# Carga un modelo registrado en app_config.json (sklearn o MLP de PyTorch).
@lru_cache(maxsize=8)
def load_registered_model(path, kind="sklearn", preprocessor_path=None):
    if kind == "sklearn":
        return joblib.load(path)
    if kind != "pytorch_mlp":
        raise ValueError(f"Tipo de modelo no soportado: {kind}")
    if not preprocessor_path:
        raise ValueError("Falta preprocessor_path en app_config.json")
    import torch
    from parte1.modeling.core import MLP
    # Solo cargar checkpoints producidos por el equipo, nunca archivos de origen desconocido.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    conf = ckpt["config"]
    network = MLP(int(ckpt["n_inputs"]), list(conf["hidden"]), conf["activation"], float(conf["dropout"]))
    network.load_state_dict(ckpt["state_dict"])
    network.eval()
    preprocessor = joblib.load(preprocessor_path)
    if hasattr(preprocessor, "get_feature_names_out"):
        n = len(preprocessor.get_feature_names_out())
        if n != int(ckpt["n_inputs"]):
            raise ValueError(f"Dimensiones incompatibles: preprocesador={n}; MLP={ckpt['n_inputs']}")
    return MLPAdapter(preprocessor, network)