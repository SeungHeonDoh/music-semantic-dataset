import os
import csv
from collections import Counter
import random
import ast
import json
import torch
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from contextlib import contextmanager
from skmultilearn.model_selection import iterative_train_test_split
from .audio_utils import load_audio
from .constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def audioset_processor(audioset_path):
    print("WIP")
