# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# UniMatch V2: https://github.com/liheyoung/unimatch-v2
# --------------------------------------------------------


import glob
import os

import numpy as np
from PIL import Image


if __name__ == '__main__':
    datapath = '<your/loveda/path>'

    filepaths = glob.glob(os.path.join(datapath, '**', '*.png'), recursive=True)
    filepaths = [filepath for filepath in filepaths if 'masks_png' in filepath]

    for filepath in filepaths:
        mask = np.array(Image.open(filepath))
        mask[mask == 0] = 255
        mask -= 1
        mask[mask == 254] = 255

        mask = Image.fromarray(mask)
        mask.save(filepath)
