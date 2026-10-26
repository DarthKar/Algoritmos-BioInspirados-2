"""Verifica scorecard y percentiles de la aplicacion, sin reentrenar modelos.
Ejecutar desde la raiz: python scripts/verificar_scorecard_percentiles.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pd_model.scorecard import ScorecardParams, prob_to_score, score_to_prob


def main():
    cfg = json.loads((ROOT / 'app_config.json').read_text(encoding='utf-8'))
    sc = cfg['scorecard']
    params = ScorecardParams(float(sc['base_score']), float(sc['base_odds']), float(sc['pdo']))
    if params.base_odds <= 0 or params.pdo <= 0:
        raise ValueError('base_odds y pdo deben ser positivos')

    checks = []
    def check(name, passed, description=''):
        ok = bool(passed)
        checks.append((name, ok, description))
        print(('CORRECTO' if ok else 'ERROR') + ' | ' + name + (f' | {description}' if description else ''))

    factor_expected = params.pdo / np.log(2)
    offset_expected = params.base_score - factor_expected * np.log(params.base_odds)
    check('factor y offset', np.isclose(params.factor, factor_expected, atol=1e-12) and np.isclose(params.offset, offset_expected, atol=1e-12),
          f'factor={params.factor:.10f}, offset={params.offset:.10f}')
    base_p = 1 / (1 + params.base_odds)
    base_score = float(prob_to_score([base_p], params)[0])
    check('odds de referencia producen puntaje base', np.isclose(base_score, params.base_score, atol=1e-8),
          f'PD={base_p:.10f}, score={base_score:.8f}')
    pd_double_good_odds = 1 / (1 + 2 * params.base_odds)
    doubled_score = float(prob_to_score([pd_double_good_odds], params)[0])
    check('duplicar odds buenas suma exactamente PDO', np.isclose(doubled_score-base_score, params.pdo, atol=1e-8),
          f'variacion={doubled_score-base_score:.8f}')
    ptest = np.array([1e-7, 0.001, 0.01, 0.05, 0.125, 0.25, 0.5, 0.75, 0.99, 1-1e-7])
    test_scores = prob_to_score(ptest, params)
    recovered = score_to_prob(test_scores, params)
    check('inversion puntaje-probabilidad', np.allclose(ptest, recovered, atol=1e-10, rtol=0),
          f'error maximo={np.max(np.abs(ptest-recovered)):.3g}')
    check('mayor PD implica menor puntaje', np.all(np.diff(test_scores) < 0))

    rows = []
    for model in cfg['models']:
        name = model['name']
        path = ROOT / model['predictions_path']
        split = model.get('reference_split', 'train')
        if not path.exists():
            check(f'{name}: predicciones disponibles', False, str(path))
            continue
        try:
            d = pd.read_parquet(path)
        except Exception as exc:
            check(f'{name}: lectura parquet', False, str(exc))
            continue
        if not {'pd','split'}.issubset(d.columns):
            check(f'{name}: columnas pd, split', False, f'columnas={list(d.columns)}')
            continue
        raw = d.loc[d['split'].eq(split), 'pd']
        ps = pd.to_numeric(raw, errors='coerce').to_numpy(dtype=float)
        check(f'{name}: referencia no vacia', len(ps)>0, f'split={split}, n={len(ps)}')
        if len(ps)==0: continue
        finite = np.isfinite(ps).all()
        bounded = bool(np.all((ps>=0)&(ps<=1)))
        check(f'{name}: probabilidades finitas y en [0,1]', finite and bounded)
        if not finite or not bounded: continue
        # Reproduce exactamente el calculo de la aplicacion con <=.
        scores = prob_to_score(ps, params)
        check(f'{name}: puntajes finitos', np.isfinite(scores).all())
        # En caso de recorte de probabilidad, puede haber empates en extremos.
        ix = np.unique(np.linspace(0, len(ps)-1, min(51,len(ps)), dtype=int))
        test_pd = np.r_[ps[ix], [base_p, .01, .5, .99]]
        scores_test = prob_to_score(test_pd, params)
        percentiles_app = np.array([100*np.mean(scores<=s) for s in scores_test])
        percentiles_manual = np.array([100*np.count_nonzero(scores<=s)/len(scores) for s in scores_test])
        check(f'{name}: percentil coincide con conteo empirico', np.allclose(percentiles_app,percentiles_manual,atol=1e-12))
        # Contraste independiente sobre PD: percentil alto significa menor riesgo.
        # Para scores no saturados, score<=s equivale a PD_referencia >= PD_usuario.
        clipped_ref = np.clip(ps, 1e-9, 1-1e-9)
        clipped_test = np.clip(test_pd, 1e-9, 1-1e-9)
        percentiles_pd = np.array([100*np.mean(clipped_ref>=p) for p in clipped_test])
        diff=float(np.max(np.abs(percentiles_app-percentiles_pd)))
        check(f'{name}: percentil consistente con orden inverso de PD', np.allclose(percentiles_app,percentiles_pd,atol=1e-8), f'diferencia maxima={diff:.3g} puntos porcentuales')
        pct_median = 100 * np.mean(scores <= float(np.median(scores)))
        check(f'{name}: percentil mediana aproximadamente 50', 49<=pct_median<=51,
              f'percentil mediana={pct_median:.3f}; posibles empates')
        # Diagnostico de empates: la app cuenta igualdad a favor de usuario.
        counts = pd.Series(scores).value_counts()
        ties = int(counts[counts>1].sum())
        print(f'  REFERENCIA {name}: n={len(ps)}; puntajes min/mediana/max={scores.min():.2f}/{np.median(scores):.2f}/{scores.max():.2f}; filas con puntaje empatado={ties}')
        for p,s,a,b in zip(test_pd, scores_test, percentiles_app, percentiles_pd):
            rows.append({'modelo':name,'split_referencia':split,'n_referencia':len(ps),'PD_usuario':p,'puntaje':s,'percentil_app':a,'percentil_comprobacion_pd':b,'diferencia':a-b})

    report = ROOT/'results'/'verificacion_scorecard_percentiles.csv'
    report.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(report,index=False, encoding='utf-8-sig')
    print('\nReporte:',report)
    passed=sum(ok for _,ok,_ in checks)
    print(f'RESUMEN: {passed}/{len(checks)} verificaciones correctas')
    if passed != len(checks):
        print('REVISAR: al menos una comprobacion ha fallado')
        raise SystemExit(1)
    print('CORRECTO: formula scorecard y calculo empirico de percentiles verificados para modelos disponibles.')
    print('NOTA: verifica por separado la representatividad de train, calibracion del modelo y significado comercial del score.')

if __name__=='__main__':
    main()
