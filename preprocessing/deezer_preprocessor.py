import os
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
from preprocessing.audio_utils import load_audio
from .constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT, DEEZER_TAG_INFO

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]


def deezer_resampler(track_id):
    fname = str(track_id) + ".mp3"
    audio_path = os.path.join(DATASET, 'deezer', 'audio', fname)
    src, _ = load_audio(
        path=audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] > DATA_LENGTH: # too long case
        random_idx = random.randint(0, src.shape[-1]- DATA_LENGTH)
        src = src[random_idx:random_idx+DATA_LENGTH]
    save_name = os.path.join(DATASET,'deezer','npy', str(track_id) + ".npy")
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))

def get_track_split(deezer_path):
    df_train = pd.read_csv(os.path.join(deezer_path, "split","train_ground_truth.csv"), index_col=0)
    df_valid = pd.read_csv(os.path.join(deezer_path, "split","validation_ground_truth.csv"), index_col=0)
    df_test = pd.read_csv(os.path.join(deezer_path, "split","test_ground_truth.csv"), index_col=0)
    save_mp3s = [int(i.replace(".mp3","")) for i in os.listdir(os.path.join(deezer_path, "audio")) if ".mp3" in i]
    track_split = {
        "train_track": list([fname for fname in df_train.index if fname in save_mp3s]),
        "valid_track": list([fname for fname in df_valid.index if fname in save_mp3s]),
        "test_track": list([fname for fname in df_test.index if fname in save_mp3s]),
    }
    with open(os.path.join(deezer_path, f"track_split.json"), mode="w") as io:
        json.dump(track_split, io, indent=4)
    return track_split

def get_tag_info(df_label):
    tags = list(df_label.columns)
    print(df_label.sum(axis=1).mean())
    deezer_tag_info = {tag: DEEZER_TAG_INFO[tag] for tag in tags}
    deezer_tag_stats = {i:j for i,j in df_label.sum().sort_values(ascending=False).to_dict().items()}
    torch.save(tags, os.path.join(DATASET, "supervision", "deezer_tags.pt"))
    torch.save(deezer_tag_info, os.path.join(DATASET, "supervision", "deezer_tag_info.pt"))
    torch.save(deezer_tag_stats, os.path.join(DATASET, "supervision", "deezer_tag_stats.pt"))

def get_annotation(deezer_path, df_meta, df_label):
    results = {}
    for idx in df_meta.index:
        item_meta = df_meta.loc[idx]
        item_anno = df_label.loc[idx]
        tag = [tag for tag, binary in item_anno.to_dict().items() if binary]
        results[idx] = {
            "tag": tag,
            "title": item_meta['title'],
            "artist_name": item_meta['artist'],
            "release": item_meta['album'],
            "year": str(item_meta['release_date'])[:4],
            "track_id": idx,
            "binary": list(item_anno.values)
        }
    torch.save(results, os.path.join(deezer_path, "annotation.pt"))  

def DEEZER_processor(deezer_path):
    track_split = get_track_split(deezer_path)
    total_track = track_split['train_track'] + track_split['valid_track'] + track_split['test_track']
    df_meta = pd.read_csv(os.path.join(deezer_path, "metadata.csv"), index_col=0)
    df_label = pd.read_csv(os.path.join(deezer_path, "annotation.csv"), index_col=0)
    df_meta = df_meta.loc[total_track]
    df_label = df_label.loc[total_track]
    get_annotation(deezer_path, df_meta, df_label)
    get_tag_info(df_label)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count()-12)
    # pool.map(deezer_resampler, total_track)
    print("finish deezer extract", len(total_track))