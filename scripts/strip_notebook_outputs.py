import nbformat
from pathlib import Path
for ip in Path('src').glob('*.ipynb'):
    nb = nbformat.read(str(ip), as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.get('outputs'):
            cell['outputs'] = []
            changed = True
        if cell.get('execution_count'):
            cell['execution_count'] = None
            changed = True
    if changed:
        nbformat.write(nb, str(ip))
        print('stripped outputs from', ip)
