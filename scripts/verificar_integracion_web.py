"""Comprueba que la app y PyTorch original calculan la misma PD.
Ejecutar desde la raiz: python scripts/verificar_integracion_web.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parte1.inference import load_registered_model
from parte1.modeling.core import MLP, predict_proba

DATA = ROOT / 'parte1/data/processed/pd_matured.parquet'
MODEL = ROOT / 'parte1/outputs/models/pd_mlp.pt'
PRE = ROOT / 'parte1/outputs/models/pd_mlp_preprocessor.joblib'
OUT = ROOT / 'parte1/outputs/results/verificacion_web_mlp.csv'

for path in (DATA, MODEL, PRE):
    if not path.is_file():
        raise FileNotFoundError(f'Falta el archivo requerido: {path}')

data = pd.read_parquet(DATA)
test = data.loc[data['split'].eq('test')].head(10).copy()
if test.empty:
    raise ValueError('No se encontraron filas de prueba (split=test).')

# Camino A: el mismo adaptador que utiliza Streamlit.
adapter = load_registered_model(str(MODEL), 'pytorch_mlp', str(PRE))
columns = adapter.input_columns
input_frame = test[columns].copy()
app_pd = adapter.predict_proba(input_frame)[:, 1]

# Camino B: carga directa del checkpoint con el codigo original del equipo.
checkpoint = torch.load(MODEL, map_location='cpu', weights_only=True)
config = checkpoint['config']
net = MLP(int(checkpoint['n_inputs']), list(config['hidden']),
          config['activation'], float(config['dropout']))
net.load_state_dict(checkpoint['state_dict'])
net.eval()
preprocessor = joblib.load(PRE)
original_pd = predict_proba(net, preprocessor.transform(input_frame))

error = np.abs(app_pd - original_pd)
result = pd.DataFrame({
    'fila_dataset': test.index,
    'probabilidad_original': original_pd,
    'probabilidad_app': app_pd,
    'diferencia_absoluta': error,
})
if 'id' in test:
    result.insert(1, 'id', test['id'].to_numpy())
OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT, index=False)
print('\nComparacion de 10 prestamos del conjunto de prueba:')
print(result.to_string(index=False, float_format=lambda n: f'{n:.10f}'))
print(f'\nDiferencia maxima: {error.max():.12g}')
print(f'Tabla guardada en: {OUT}')
if not np.allclose(app_pd, original_pd, rtol=0, atol=1e-7):
    raise SystemExit('ERROR: el adaptador no coincide con la inferencia directa.')
print('CORRECTO: adaptador web e inferencia directa coinciden (tolerancia 1e-7).')
print('\nPara comprobar el formulario manualmente:')
print('Usa el prestamo de la primera fila que aparece abajo, con estos valores:')
for k, v in input_frame.iloc[0].items():
    print(f'  {k} = {v}')
print(f'Probabilidad esperada: {app_pd[0]*100:.6f}%')
print('Nota: la interfaz muestra solo dos decimales porcentuales.')
