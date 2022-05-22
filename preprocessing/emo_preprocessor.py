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
from preprocessing.audio_utils import load_audio
from .constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]


def emo_resampler(track_id):
    audio_path = os.path.join(DATASET, 'emo', 'clips_45seconds', track_id + ".mp3")
    src, _ = load_audio(
        path=audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    save_name = os.path.join(DATASET,'emo','npy', track_id.replace(".mp3",".npy"))
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))

def EMO_processor(emo_path):
    annotation = torch.load(os.path.join(emo_path, "annotation.pt"))
    # pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
    # pool.map(emo_resampler, list(annotation.keys()))