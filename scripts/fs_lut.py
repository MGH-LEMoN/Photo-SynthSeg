#!/usr/bin/env python

import re
import sys

## !!! INSTRUCTIONS !!! ## (You do this only once)
# 1. Place this script in your home directory
# 2. Run chmod +x /path/to/script
# 3. Create an alias in your ~/.bashrc file: alias fs_lut='/path/to/script'
# 4. source ~/.bashrc
# 5. Example: fs_lut <integer> | <list of integers>


def fs_lut():
    # Extract contents from look-up table
    LUT_FILE = '/usr/local/freesurfer/dev/FreeSurferColorLUT.txt'
    with open(LUT_FILE, 'r') as f:
        lines = f.read().splitlines()

    filter_lines = [line for line in lines if re.search('^[0-9]', line)]
    split_lines = [line.split()[:2] for line in filter_lines]
    idx_to_label = dict(split_lines)
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    return idx_to_label, label_to_idx


if __name__ == '__main__':
    lut, reverse_lut = fs_lut()

    for arg in sys.argv[1:]:
        print(f'{str(arg)} - {lut.get(str(arg), "Does Not Exist")}')
