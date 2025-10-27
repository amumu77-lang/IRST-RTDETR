import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info



def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = ''
    model = RTDETR(model_path)
    result = model.val(data='',
                      split='test',
                      imgsz=640,
                      batch=4,
                      project='runs/val',
                      name='exp',
                      )
