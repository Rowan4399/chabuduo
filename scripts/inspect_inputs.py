import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from traffic.algorithms.generation import Generation
from traffic.data import airports

root = Path.cwd()
for name in [
    'zurich_tcvae-v1', 'zurich_ddpm_unconditional-v1',
    'zurich_rwy14_flow162_216_tcvae-v1', 'zurich_rwy14_tcvae_vamp_k1000-v1',
]:
    folder = root / 'data/processed' / name
    print('ARTIFACT', name)
    meta = pd.read_parquet(folder / 'metadata.parquet')
    print(meta.head().to_string(), meta.columns.tolist())
    with np.load(folder / 'train.npz') as a:
        print({k: (a[k].shape, str(a[k].dtype)) for k in a.files})
        print('IDs', a['trajectory_id'][:3])
for p in (root / 'runs').rglob('generated.parquet'):
    if any(s in str(p) for s in [
        'resume700-best', 'best-epoch354',
        'best-after-resume100', 'steps-1000-eta-0',
    ]):
        f = pd.read_parquet(p)
        print('SAMPLE', str(p), f.flight_id.nunique(), f.columns.tolist())
        print(f.iloc[[0,1,198,199]].to_string())
        with np.load(p.with_name('samples.npz')) as a:
            print('NPZ', {k: (a[k].shape, str(a[k].dtype)) for k in a.files})
for p in (root / 'data/interim').glob('lszh*'):
    f = pd.read_parquet(p)
    print('CACHE', p.name, len(f), f.flight_id.nunique(), f.columns.tolist())
    print(f.iloc[:2].to_string())
airport = airports['LSZH']
print('AIRPORT', airport.latitude, airport.longitude)
print('GENERATION_SOURCE', inspect.getsourcefile(Generation))
print(inspect.getsource(Generation.build_traffic))
print('ENV', json.dumps({'numpy':np.__version__, 'pandas':pd.__version__}))
