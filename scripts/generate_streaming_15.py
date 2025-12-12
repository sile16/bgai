#!/usr/bin/env python3
"""Generate streaming bearoff table for 6pip-15checker with uint16 encoding."""

import sys
import time
import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import bearoff

max_checkers = 15
output_path = "data/bearoff_15_u16.bin"
header_path = "data/bearoff_15_u16.json"

print(f'Starting streaming bearoff generation for 6pip-{max_checkers}checker...')
print(f'n_positions = C({6 + max_checkers}, 6) = 54,264')
print(f'upper_triangle_entries = 54,264 * 54,265 / 2 = 1,472,317,980')
print(f'output size = 1.47B * 28 bytes = ~41 GB')
print()
print(f'Output: {output_path}')
print(f'Header: {header_path}')
print()

os.makedirs('data', exist_ok=True)

start = time.time()
bearoff.generate_streaming_bearoff_u16(
    max_checkers=max_checkers,
    output_path=output_path,
    header_path=header_path
)
elapsed = time.time() - start

print(f'Generation complete in {elapsed:.1f}s')
print('DONE')
