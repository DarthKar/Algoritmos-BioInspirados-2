"""Comando único de la Parte 1: corre todo el experimento según config.yaml.

Desde la raíz del repositorio:

    python parte1/run.py                          # experimento completo (15-35 min, sobre todo el MLP)
    python parte1/run.py --demo                   # demostración en menos de 2 minutos; salidas en parte1/salidas_demo/
    python parte1/run.py --etapas mlp figuras     # solo algunas etapas
    python parte1/run.py --set mlp.n_seeds=5 mlp.fixed.lr=0.01 scorecard.pdo=40
    python parte1/run.py --listar                 # muestra las etapas disponibles

Las etapas leen lo que dejaron las anteriores en parte1/salidas/, así que se
pueden repetir por separado sin rehacer todo.
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))   # permite `python parte1/run.py` desde cualquier carpeta
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
        from parte1.preparacion.dataset import build_pd_dataset, save_pd_dataset
        from parte1.comun.utils import RAW_LOANS_CSV
        if not RAW_LOANS_CSV.exists():
            print(f"  {RAW_LOANS_CSV} no existe; se usa parte1/datos/procesados ya guardado.")
            return
        df, report = build_pd_dataset(mode, terms=tuple(cfg["datos"]["plazos"]))
        print(f"  {save_pd_dataset(df, report, mode)}: {len(df):,} filas")
    elif stage == "eda":
        from parte1.analisis.eda import run_eda
        run_eda(mode)
    elif stage == "logistica":
        from parte1.modelos.logistic import LogisticConfig, run_logistic_baseline
        r = run_logistic_baseline(LogisticConfig(mode=mode, c_grid=cfg["logistica"]["c_grid"], seed=seed))
        print(f"  C = {r['busqueda']['mejor_C']}; AUC test = {r['metricas']['test']['auc']:.4f}")
    elif stage == "mlp":
        from parte1.modelos.mlp import MLPExperimentConfig, run_mlp_experiment
        m = cfg["mlp"]
        r = run_mlp_experiment(MLPExperimentConfig(mode=mode, n_trials=m["n_trials"], n_seeds=m["n_seeds"],
                                                   max_epochs=m["max_epochs"], patience=m["patience"],
                                                   seed=seed, fixed=m.get("fixed")))
        auc = next(x for x in r["semillas"]["resumen_test"] if x["metrica"] == "auc")
        print(f"  AUC test: media {auc['mean']:.4f} ± {auc['std']:.4f} en {m['n_seeds']} semillas")
    elif stage == "figuras":
        from parte1.evaluacion.plots import save_figures, save_loss_figure
        save_figures("pd_logistic", "regresión logística")
        save_figures("pd_mlp", "MLP")
        save_loss_figure("pd_mlp", "MLP")
    elif stage == "interpretabilidad":
        from parte1.analisis.interpret import run_interpretability
        run_interpretability(mode=mode, **cfg["interpretabilidad"])
    elif stage == "scorecard":
        from parte1.puntaje.build import run_scorecard
        from parte1.puntaje.transform import ScorecardParams
        sc = dict(cfg["scorecard"])
        params = ScorecardParams(**{k: sc.pop(k) for k in ("base_score", "base_odds", "pdo")})
        run_scorecard(params, mode=mode, **sc)
    elif stage == "clasificacion":
        from parte1.evaluacion.classification import run_classification
        run_classification()
    elif stage == "hipotesis":
        from parte1.analisis.hypothesis_check import run_hypothesis_check
        run_hypothesis_check(mode)
    elif stage == "comparacion":
        from parte1.evaluacion.compare import run_comparison
        run_comparison(n_boot=cfg["comparacion"]["n_boot"])
    elif stage == "informe":
        from parte1.report import run_report
        print(f"  {run_report()}")


# Lee argumentos, fija la carpeta de salida y corre las etapas en orden.
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # Δ, → y ± también si la salida se redirige a un archivo
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
        os.environ["PD_SALIDA"] = "salidas_demo"   # antes de importar parte1.comun.utils
    stages = args.etapas or [s for s in STAGES if not (args.demo and s == "datos")]

    from parte1.comun.utils import OUTPUT_ROOT, set_seed
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
