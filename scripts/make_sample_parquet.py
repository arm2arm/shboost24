import pandas as pd
import numpy as np
from pathlib import Path
from src import constants

out = Path('data/sample.parq')
out.parent.mkdir(exist_ok=True)
# create 100 rows with features from constants.features (use zeros or random)
cols = list(constants.features)
# add label met50
cols += ['met50']
N = 200
arr = {c: np.random.normal(size=N).astype('float32') for c in cols}
# make met50 in a reasonable range
arr['met50'] = np.random.uniform(-3, 0.5, size=N).astype('float32')
df = pd.DataFrame(arr)
df.to_parquet(out)
print('wrote', out)
