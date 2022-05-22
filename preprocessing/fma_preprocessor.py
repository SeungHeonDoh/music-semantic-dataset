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
from preprocessing.audio_utils import load_audio
from .constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT, FMA_TAG_INFO

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def fma_load(filepath):
    filename = os.path.basename(filepath)
    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)
    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)
        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])
        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')
        return tracks

def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def fma_resampler(track_id):
    root_path = os.path.join(DATASET,'fma','fma_large')
    audio_path = get_audio_path(root_path, track_id)
    fname = audio_path.split(root_path + "/")[1]
    try:
        src, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= MUSIC_SAMPLE_RATE,
            downmix_to_mono= True)
        if src.shape[-1] < DATA_LENGTH: # short case
            pad = np.zeros(DATA_LENGTH)
            pad[:src.shape[-1]] = src
            src = pad
        elif src.shape[-1] > DATA_LENGTH: # too long case
            src = src[:DATA_LENGTH]
        save_name = os.path.join(DATASET,'fma','npy', fname.replace(".mp3",".npy"))
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        np.save(save_name, src.astype(np.float32))
    except:
        save_name = os.path.join(DATASET,'fma','error', fname.replace(".mp3",".npy"))
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        np.save(save_name, track_id)
    
def get_track_split(tracks, target, fma_path):
    target_subset = tracks['set', 'subset'] <= target
    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'
    y_train = tracks.loc[target_subset & train, ('track', 'genre_top')]
    y_val = tracks.loc[target_subset & val, ('track', 'genre_top')]
    y_test = tracks.loc[target_subset & test, ('track', 'genre_top')]
    traget_track_split = {
        "train_track": list(y_train.index),
        "valid_track": list(y_val.index),
        "test_track": list(y_test.index),
    }
    return traget_track_split

def get_annotation(tracks, genres, fma_path):
    track_target = ['genres_all', 'genre_top', 'title']
    album_target = ['date_released','title']
    artist_target = ['name']
    df_track = tracks['track'][track_target]
    df_album = tracks['album'][album_target]
    df_artist = tracks['artist'][artist_target]
    df_album = df_album.rename(columns={"title": "release"})
    genres_name = []
    for i in df_track['genres_all']:
        genres_name.append(list(genres.loc[i]['title']))
    df_track['genres_all']= genres_name
    track_album = pd.merge(df_track, df_album, how='outer',on='track_id')
    df_fma = pd.merge(track_album, df_artist, how='outer',on='track_id')
    df_fma['genres_all'] = df_fma['genres_all'].map(lambda x: list(map(str.lower, x)))
    df_fma['genre_top'] =  df_fma['genre_top'].map(lambda x: x.lower())
    df_annotation = pd.DataFrame(index=df_fma.index, columns=['tag','genre','title','year','release','artist_name'])
    df_annotation['tag'] = df_fma['genres_all']
    df_annotation['genre'] = list(df_fma['genre_top'])
    df_annotation['title'] = df_fma['title']
    df_annotation['artist_name'] = df_fma['name']
    df_annotation['release'] = df_fma['release']
    df_annotation['year'] = df_fma['date_released'].map(lambda x: str(x.year))
    df_annotation['track_id'] = df_annotation.index
    df_annotation = df_annotation.fillna(0)
    df_filter = df_annotation[df_annotation['tag'].map(lambda d: len(d)) > 0]
    df_dict = df_filter.to_dict('index')
    torch.save(df_dict, os.path.join(fma_path, "annotation.pt"))
    return df_dict, df_filter

def get_tag_info(df_annotation, filtered_small):
    small_subset = filtered_small['train_track'] + filtered_small['valid_track']+ filtered_small['test_track']
    fma_tag_info = FMA_TAG_INFO.copy()
    all_tags = flatten_list_of_list(list(df_annotation['tag']))
    tag_statistics = {i:j for i,j in Counter(all_tags).most_common()}
    genre_list = list(set(df_annotation.loc[small_subset]['genre']))
    torch.save(list(set(all_tags)), os.path.join(DATASET, "supervision", "fma_tags.pt"))
    torch.save(genre_list, os.path.join(DATASET, "supervision", "fma_genres.pt"))
    torch.save(fma_tag_info, os.path.join(DATASET, "supervision", "fma_tag_info.pt"))
    torch.save(tag_statistics, os.path.join(DATASET, "supervision", "fma_tag_stats.pt"))

def FMA_processor(fma_path):
    tracks = fma_load(os.path.join(fma_path,"annotation/tracks.csv"))
    genres = fma_load(os.path.join(fma_path,"annotation/genres.csv"))
    small_track_split = get_track_split(tracks, "small", fma_path)
    large_track_split = get_track_split(tracks, "large", fma_path)
    total_track = large_track_split['train_track'] + large_track_split['valid_track']+ large_track_split['test_track']
    # pool = multiprocessing.Pool(multiprocessing.cpu_count()-12)
    # pool.map(fma_resampler, total_track)
    error_samples = []
    error_dir = os.path.join(DATASET,'fma','error')
    for dirs in os.listdir(error_dir):
        error_samples.extend(os.listdir(os.path.join(error_dir, dirs)))
    error_fnames = [int(i.split(".npy")[0]) for i in error_samples]
    tracks = tracks.drop(error_fnames, axis=0)
    df_dict, df_filter = get_annotation(tracks, genres, fma_path)
    filtered_small = {
        "train_track": [track_id for track_id in small_track_split['train_track'] if track_id in df_dict.keys()],
        "valid_track": [track_id for track_id in small_track_split['valid_track'] if track_id in df_dict.keys()],
        "test_track": [track_id for track_id in small_track_split['test_track'] if track_id in df_dict.keys()]
    }
    with open(os.path.join(fma_path, "small_track_split.json"), mode="w") as io:
        json.dump(filtered_small, io, indent=4)
    
    get_tag_info(df_filter, filtered_small)

    filtered_large = {
        "train_track": [track_id for track_id in large_track_split['train_track'] if track_id in df_dict.keys()],
        "valid_track": [track_id for track_id in large_track_split['valid_track'] if track_id in df_dict.keys()],
        "test_track": [track_id for track_id in large_track_split['test_track'] if track_id in df_dict.keys()]
    }
    with open(os.path.join(fma_path, "large_track_split.json"), mode="w") as io:
        json.dump(filtered_large, io, indent=4)
    track_list = filtered_large['train_track'] + filtered_large['valid_track']+ filtered_large['test_track']
    print("finish fma extraction", len(track_list))

