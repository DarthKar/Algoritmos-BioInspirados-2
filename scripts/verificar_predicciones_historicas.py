"""Verifica las predicciones históricas de la MLP con el checkpoint actual.

Desde la raíz del proyecto:
    python scripts/verificar_predicciones_historicas.py
    python scripts/verificar_predicciones_historicas.py --tolerancia 1e-6

No reentrena ni modifica los archivos del modelo.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parte1.inference import load_registered_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tolerancia', type=float, default=1e-6,
                        help='Máxima diferencia absoluta tolerada entre probabilidades (por defecto 1e-6).')
    parser.add_argument('--lote', type=int, default=8192,
                        help='Préstamos por lote para limitar uso de memoria.')
    args = parser.parse_args()
    if not np.isfinite(args.tolerancia) or args.tolerancia < 0 or args.lote < 1:
        parser.error('La tolerancia debe ser finita y no negativa; el lote debe ser positivo.')

    dataset_path = ROOT / 'parte1/datos/procesados/pd_matured.parquet'
    preds_path = ROOT / 'parte1/salidas/resultados/pd_mlp_predicciones.parquet'
    model_path = ROOT / 'parte1/salidas/modelos/pd_mlp.pt'
    pre_path = ROOT / 'parte1/salidas/modelos/pd_mlp_preprocessor.joblib'
    for path in [dataset_path, preds_path, model_path, pre_path]:
        if not path.is_file():
            raise FileNotFoundError(f'Falta el archivo requerido: {path}')

    datos = pd.read_parquet(dataset_path)
    historicas = pd.read_parquet(preds_path)
    for name, frame, required in [
        ('dataset', datos, {'id', 'split', 'target'}),
        ('histórico', historicas, {'id', 'split', 'target', 'pd'}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'Faltan columnas en {name}: {sorted(missing)}')
        if frame[['id', 'split']].isna().any().any():
            raise ValueError(f'Hay claves nulas en {name}.')
        dup = frame.duplicated(['id', 'split'], keep=False)
        if dup.any():
            raise ValueError(f'Hay {int(dup.sum())} filas con clave (id, split) duplicada en {name}; no es seguro emparejarlas.')

    if datos['id'].dtype != historicas['id'].dtype:
        print(f'AVISO: tipos de id distintos: {datos["id"].dtype} vs {historicas["id"].dtype}.')

    # Comprobación estricta de cobertura antes de generar predicciones.
    claves_datos = datos[['id', 'split']]
    claves_hist = historicas[['id', 'split']]
    coverage = claves_datos.merge(claves_hist, on=['id', 'split'], how='outer', indicator=True, validate='one_to_one')
    faltantes = int((coverage['_merge'] == 'left_only').sum())
    sobrantes = int((coverage['_merge'] == 'right_only').sum())

    adapter = load_registered_model(str(model_path), 'pytorch_mlp', str(pre_path))
    columnas = list(adapter.input_columns)
    missing_inputs = sorted(set(columnas) - set(datos.columns))
    if missing_inputs:
        raise ValueError(f'El dataset no contiene variables requeridas por el modelo: {missing_inputs}')

    rows = []
    for split in ('train', 'val', 'test'):
        subset = datos.loc[datos['split'].eq(split)]
        for start in range(0, len(subset), args.lote):
            chunk = subset.iloc[start:start + args.lote]
            if chunk.empty:
                continue
            predicted = np.asarray(adapter.predict_proba(chunk[columnas])[:, 1], dtype=float)
            rows.append(pd.DataFrame({'id': chunk['id'].to_numpy(), 'split': split,
                                      'target_actual': chunk['target'].to_numpy(), 'pd_actual': predicted}))
    current = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['id', 'split', 'target_actual', 'pd_actual'])

    comparison = historicas.rename(columns={'target': 'target_historico', 'pd': 'pd_historica'}).merge(
        current, on=['id', 'split'], how='outer', indicator=True, validate='one_to_one'
    )
    matched = comparison['_merge'].eq('both')
    comparison['diferencia_absoluta'] = (comparison['pd_historica'] - comparison['pd_actual']).abs()
    comparison['target_coincide'] = comparison['target_historico'].eq(comparison['target_actual'])
    finite = np.isfinite(pd.to_numeric(comparison.loc[matched, 'pd_historica'], errors='coerce')).all() and np.isfinite(
        pd.to_numeric(comparison.loc[matched, 'pd_actual'], errors='coerce')).all()
    bad_targets = int((~comparison.loc[matched, 'target_coincide']).sum())
    bad_probs = int((comparison.loc[matched, 'diferencia_absoluta'] > args.tolerancia).sum())
    missing_probs = int(comparison.loc[matched, ['pd_historica', 'pd_actual']].isna().any(axis=1).sum())
    unexpected_splits = sorted(set(datos['split'].dropna()) - {'train', 'val', 'test'})
    report = comparison.loc[:, ['id', 'split', 'target_historico', 'target_actual',
                                'pd_historica', 'pd_actual', 'diferencia_absoluta', 'target_coincide', '_merge']]
    out = ROOT / 'parte1/salidas/resultados/verificacion_predicciones_historicas.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    print('\nVerificación del checkpoint MLP frente a predicciones históricas')
    print(f'Datos: {len(datos)} filas | Histórico: {len(historicas)} filas | Emparejadas: {int(matched.sum())}')
    print(f'Claves faltantes en histórico: {faltantes} | Claves sobrantes: {sobrantes}')
    for split in ('train', 'val', 'test'):
        p = comparison.loc[matched & comparison['split'].eq(split)]
        maximum = p['diferencia_absoluta'].max() if len(p) else float('nan')
        print(f'{split:>5}: {len(p):>7} coincidencias | error máximo={maximum:.10g} | errores>{args.tolerancia:g}: {int((p["diferencia_absoluta"] > args.tolerancia).sum())}')
    print(f'Etiquetas distintas: {bad_targets} | Probabilidades nulas: {missing_probs} | Valores finitos: {finite}')
    if unexpected_splits:
        print(f'Particiones inesperadas: {unexpected_splits}')
    print(f'Reporte: {out}')
    if faltantes or sobrantes or bad_targets or bad_probs or missing_probs or not finite or unexpected_splits:
        print('NO COINCIDEN: revisa las filas discrepantes en el CSV.')
        return 1
    print(f'CORRECTO: las {int(matched.sum())} predicciones históricas coinciden con el modelo guardado (tolerancia absoluta {args.tolerancia:g}).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
