"""Comando único de la Parte 1: corre todo el experimento según config.yaml.

    python run.py                          # experimento completo (~40 min, sobre todo el MLP)
    python run.py --demo                   # demostración en menos de 2 minutos; salidas en demo/
    python run.py --etapas mlp figuras     # solo algunas etapas
    python run.py --set mlp.n_seeds=5 mlp.fixed.lr=0.01 scorecard.pdo=40
    python run.py --listar                 # muestra las etapas disponibles

Las etapas leen lo que dejaron las anteriores en models/ y results/, así que se
pueden repetir por separado sin rehacer todo.
"""
import argparse
import copy
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STAGES = ["datos", "eda", "logistica", "mlp", "figuras", "interpretabilidad", "scorecard",
          "clasificacion", "hipotesis", "comparacion", "informe"]


# Mezcla recursiva: los valores de `extra` reemplazan a los de `base`.
def merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# Aplica --set clave.subclave=valor (el valor se interpreta como YAML: números, listas, null).
def apply_overrides(cfg: dict, items: list[str]) -> dict:
    for item in items:
        key, _, raw = item.partition("=")
        node, *path = key.split(".")
        target = cfg.setdefault(node, {})
        for p in path[:-1]:
            if not isinstance(target.get(p), dict):   # p. ej. mlp.fixed es null en config.yaml
                target[p] = {}
            target = target[p]
        if path:
            target[path[-1]] = yaml.safe_load(raw)
        else:
            cfg[node] = yaml.safe_load(raw)
    return cfg


# Lee config.yaml, aplica el bloque demo si se pide y luego los --set.
def load_config(path: Path, demo: bool, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    demo_cfg = cfg.pop("demo", {})
    if demo:
        cfg = merge(cfg, demo_cfg)
    return apply_overrides(cfg, overrides)


# Ejecuta una etapa; las importaciones van aquí para que PD_SALIDA ya esté fijada.
def run_stage(stage: str, cfg: dict) -> None:
    mode, seed = cfg["datos"]["modo"], cfg["semilla"]
    if stage == "datos":
        from src.pd_model.dataset import build_pd_dataset, save_pd_dataset
        from src.utils import RAW_LOANS_CSV
        if not RAW_LOANS_CSV.exists():
            print(f"  {RAW_LOANS_CSV} no existe; se usa data/processed ya guardado.")
            return
        df, report = build_pd_dataset(mode)
        print(f"  {save_pd_dataset(df, report, mode)}: {len(df):,} filas")
    elif stage == "eda":
        from src.pd_model.eda import run_eda
        run_eda(mode)
    elif stage == "logistica":
        from src.pd_model.logistic import LogisticConfig, run_logistic_baseline
        r = run_logistic_baseline(LogisticConfig(mode=mode, c_grid=cfg["logistica"]["c_grid"], seed=seed))
        print(f"  C = {r['busqueda']['mejor_C']}; AUC test = {r['metricas']['test']['auc']:.4f}")
    elif stage == "mlp":
        from src.pd_model.mlp import MLPExperimentConfig, run_mlp_experiment
        m = cfg["mlp"]
        r = run_mlp_experiment(MLPExperimentConfig(mode=mode, n_trials=m["n_trials"], n_seeds=m["n_seeds"],
                                                   max_epochs=m["max_epochs"], patience=m["patience"],
                                                   seed=seed, fixed=m.get("fixed")))
        auc = next(x for x in r["semillas"]["resumen_test"] if x["metrica"] == "auc")
        print(f"  AUC test: media {auc['mean']:.4f} ± {auc['std']:.4f} en {m['n_seeds']} semillas")
    elif stage == "figuras":
        from src.pd_model.plots import save_figures, save_loss_figure
        save_figures("pd_logistic", "regresión logística")
        save_figures("pd_mlp", "MLP")
        save_loss_figure("pd_mlp", "MLP")
    elif stage == "interpretabilidad":
        from src.pd_model.interpret import run_interpretability
        run_interpretability(mode=mode, **cfg["interpretabilidad"])
    elif stage == "scorecard":
        from src.pd_model.build_scorecard import run_scorecard
        from src.pd_model.scorecard import ScorecardParams
        run_scorecard(ScorecardParams(**cfg["scorecard"]), mode=mode)
    elif stage == "clasificacion":
        from src.pd_model.classification import run_classification
        run_classification()
    elif stage == "hipotesis":
        from src.pd_model.hypothesis_check import run_hypothesis_check
        run_hypothesis_check(mode)
    elif stage == "comparacion":
        from src.pd_model.compare import run_comparison
        run_comparison(n_boot=cfg["comparacion"]["n_boot"])
    elif stage == "informe":
        from src.pd_model.report import run_report
        print(f"  {run_report()}")


# Lee argumentos, fija la carpeta de salida y corre las etapas en orden.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--demo", action="store_true", help="corrida corta (<2 min) con salidas en demo/")
    parser.add_argument("--etapas", nargs="+", choices=STAGES, help="etapas a correr (por defecto, todas)")
    parser.add_argument("--set", nargs="+", default=[], metavar="CLAVE=VALOR", help="cambia valores de la configuración")
    parser.add_argument("--listar", action="store_true", help="muestra las etapas y termina")
    args = parser.parse_args()
    if args.listar:
        print("\n".join(STAGES))
        return

    cfg = load_config(args.config, args.demo, args.set)
    if args.demo:
        os.environ["PD_SALIDA"] = "demo"   # antes de importar src.utils
    stages = args.etapas or [s for s in STAGES if not (args.demo and s == "datos")]

    from src.utils import OUTPUT_ROOT, set_seed
    set_seed(cfg["semilla"])
    print(f"Configuración: {yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=None)}")
    print(f"Salidas en: {OUTPUT_ROOT}")
    start = time.perf_counter()
    for stage in stages:
        t0 = time.perf_counter()
        print(f"\n=== {stage} ===", flush=True)
        run_stage(stage, cfg)
        print(f"  ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"\nListo en {time.perf_counter() - start:.1f} s.")


if __name__ == "__main__":
    main()
