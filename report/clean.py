#!/usr/bin/env python3

from pathlib import Path
curr_path = Path('.')
if Path('./clean.py') not in curr_path.glob('*'):
    print('FATAL: clean.py can only be executed when cwd is the report directory')
    exit(-1)

for file in curr_path.glob('*'):
    print(file.suffix)