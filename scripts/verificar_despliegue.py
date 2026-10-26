"""Comprueba los archivos de la app antes de desplegar. Ejecutar desde la raíz."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
issues = []
notes = []

def check_file(relative, context):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        issues.append(f"FALTA ({context}): {relative}")
    else:
        print(f"OK ({context}): {relative} ({path.stat().st_size / 1024:.1f} KiB)")

check_file('web/app.py', 'entrada cloud')
check_file('app.py', 'entrada local')
check_file('src/pd_model/adapter.py', 'adaptador')
check_file('src/pd_model/scorecard.py', 'scorecard')
config_path = ROOT / 'app_config.json'
if config_path.is_file():
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for item in config.get('models', []):
        name = item.get('name','modelo')
        for field in ('path', 'preprocessor_path', 'data_path', 'predictions_path'):
            if item.get(field):
                check_file(item[field], name + ':' + field)
        if not item.get('predictions_path'):
            issues.append(f'Sin predicciones de referencia para {name}')
    for field in ('report_url', 'video_url'):
        if not config.get(field):
            notes.append(f'PENDIENTE: configura {field} en app_config.json cuando esté disponible')
else:
    issues.append('FALTA app_config.json')

print('\nAdvertencias:')
for n in notes:
    print(' -',n)
print('\nResultado:', 'LISTO PARA REVISIÓN DE DESPLIEGUE' if not issues else 'ERRORES')
for i in issues:
    print(' -',i)
if not issues:
    print('Nota: el chequeo es estructural; no comprueba instalación remota ni privacidad de los datos.')
sys.exit(1 if issues else 0)