import numpy as np
import pandas as pd

import os
for dir_name, _, file_names in os.walk('/kaggle/input'):
    for file_name in file_names:
        print(os.path.join(dir_name, file_name))