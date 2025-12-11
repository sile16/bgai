#!/usr/bin/env python3
"""Generate tiered bearoff table for 6pip-14checker."""

import sys
import time
import json
import numpy as np
import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import bearoff

max_checkers = 14
print(f'Starting tiered bearoff generation for 6pip-{max_checkers}checker...')
print(f'n_positions = C({6 + max_checkers}, 6) = 38,760')
print()

start = time.time()
header_json, tier1_arr, tier2_arr = bearoff.generate_tiered_bearoff(max_checkers=max_checkers)
elapsed = time.time() - start

print(f'Generation complete in {elapsed:.1f}s')
print(f'Header: {header_json}')
print(f'Tier1 shape: {tier1_arr.shape}, dtype: {tier1_arr.dtype}')
print(f'Tier2 shape: {tier2_arr.shape}, dtype: {tier2_arr.dtype}')

# Save
os.makedirs('data/tiered', exist_ok=True)
np.save(f'data/tiered/bearoff_tiered_{max_checkers}_tier1.npy', tier1_arr)
np.save(f'data/tiered/bearoff_tiered_{max_checkers}_tier2.npy', tier2_arr)
with open(f'data/tiered/bearoff_tiered_{max_checkers}.json', 'w') as f:
    f.write(header_json)

print(f'Saved to data/tiered/')
print('DONE')
