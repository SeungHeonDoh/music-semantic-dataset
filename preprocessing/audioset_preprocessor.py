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


def audioset_resampler(track_id):
    audio_path = os.path.join(DATASET, 'audioset', 'audio', track_id)
    src, _ = load_audio(
        path=audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    save_name = os.path.join(DATASET,'audioset','full_npy', track_id.replace(".mp3",".npy"))
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3],
                'tag': row[5:],
            }
    return tracks

def get_split(audioset_path, split_type):
    train = read_file(os.path.join(audioset_path, "split-0", f"{split_type}-train.tsv"))
    validation = read_file(os.path.join(audioset_path, "split-0", f"{split_type}-validation.tsv"))
    test = read_file(os.path.join(audioset_path, "split-0", f"{split_type}-test.tsv"))
    track_split = {
        "train_track": list(train.keys()),
        "valid_track": list(validation.keys()),
        "test_track": list(test.keys())
    }
    with open(os.path.join(audioset_path, f"{split_type}_track_split.json"), mode="w") as io:
        json.dump(track_split, io, indent=4)
    return track_split

def get_tag_split(df_total, split_type=False):
    if split_type:
        tags = [i.split("---")[1] for i in set(flatten_list_of_list(df_total['tag']))]
        torch.save(tags, os.path.join(DATASET, "supervision", f"audioset_{split_type}_tags.pt"))
    else:
        audioset_tag_info = audioset_TAG_INFO.copy()
        all_tags = flatten_list_of_list(list(df_total['tag']))
        tag_statistics= {i.split("---")[1]:j for i,j in Counter(all_tags).most_common()}
        tags = list(tag_statistics.keys())
        print("number of tag",len(tags))
        torch.save(tags, os.path.join(DATASET, "supervision", "audioset_tags.pt"))
        torch.save(audioset_tag_info, os.path.join(DATASET, "supervision", "audioset_tag_info.pt"))
        torch.save(tag_statistics, os.path.join(DATASET, "supervision", "audioset_tag_stats.pt"))


def get_annotation(audioset_path, split_type=False):
    if split_type:
        train = read_file(os.path.join(audioset_path, "split-0", f"{split_type}-train.tsv"))
        validation = read_file(os.path.join(audioset_path, "split-0", f"{split_type}-validation.tsv"))
        test = read_file(os.path.join(audioset_path, "split-0", f"{split_type}-test.tsv"))
    else: 
        train = read_file(os.path.join(audioset_path, "split-0", "autotagging-train.tsv"))
        validation = read_file(os.path.join(audioset_path, "split-0", "autotagging-validation.tsv"))
        test = read_file(os.path.join(audioset_path, "split-0", "autotagging-test.tsv"))
    total = {}
    total.update(train)
    total.update(validation)
    total.update(test)
    annotation = {}
    for track_id, path_tags in total.items():
        annotation[track_id] = {
            "track_id": track_id,
            "path": path_tags['path'],
            "tag": [tag.split("---")[1] for tag in path_tags['tag']]
        }
    if split_type:
        torch.save(annotation, os.path.join(audioset_path, f"{split_type}_annotation.pt"))
    else:
        torch.save(annotation, os.path.join(audioset_path, "annotation.pt"))
    return pd.DataFrame(total).T
    

def audioset_processor(audioset_path):
    for split_type in ['autotagging_top50tags', 'autotagging_genre','autotagging_moodtheme','autotagging_instrument']:
        split_info = get_split(audioset_path, split_type=split_type)
        df_annotation = get_annotation(audioset_path, split_type=split_type)
        get_tag_split(df_annotation, split_type=split_type)
    track_split = get_split(audioset_path, split_type="autotagging")
    df_total = get_annotation(audioset_path)
    get_tag_split(df_total)
    mp3_path = list(df_total['path'])
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool.map(audioset_resampler, mp3_path)
    print("finish audioset extract", len(df_total))
