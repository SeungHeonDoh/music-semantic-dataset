import os
import pickle
import random
import sqlite3
import torch
import json
import pandas as pd
import numpy as np
import multiprocessing
from collections import Counter
from functools import partial
from contextlib import contextmanager
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from preprocessing.audio_utils import load_audio
from .constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def getMsdInfo(msd_path):
    con = sqlite3.connect(msd_path)
    msd_db = pd.read_sql_query("SELECT * FROM songs", con)
    msd_db = msd_db.set_index('track_id')
    return msd_db

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def msd_resampler(_id, path):
    try:
        src, _ = load_audio(
            path=os.path.join(DATASET,'msd','songs',path),
            ch_format= STR_CH_FIRST,
            sample_rate= MUSIC_SAMPLE_RATE,
            downmix_to_mono= True)
        if src.shape[-1] < DATA_LENGTH: # short case
            pad = np.zeros(DATA_LENGTH)
            pad[:src.shape[-1]] = src
            src = pad
        elif src.shape[-1] > DATA_LENGTH: # too long case
            src = src[:DATA_LENGTH]
        save_name = os.path.join(DATASET,'msd','npy', path.replace(".mp3",".npy"))
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        np.save(save_name, src.astype(np.float32))
    except:
        np.save(os.path.join(DATASET,'msd', "error", _id + ".npy"), _id) # check black case

def binary_df_to_list(binary, tags, indices, data_type):
    list_of_tag = []
    for bool_tags in binary:
        list_of_tag.append([tags[idx] for idx, i in enumerate(bool_tags) if i] )
    df_tag_list = pd.DataFrame(index=indices, columns=[data_type])
    df_tag_list.index.name = "track_id"
    df_tag_list[data_type] = list_of_tag
    df_tag_list['is_'+ data_type] = [True for i in range(len(df_tag_list))]
    return df_tag_list

def lastfm_processor(lastfm_path):
    """
    input: lastfm_path 
    return: pandas.DataFrame => index: msd trackid, columns: list of tag
            TRAAAAK128F9318786	[rock, alternative rock, hard rock]
            TRAAAAW128F429D538	[hip-hop]
    """
    lastfm_tags = open(os.path.join(lastfm_path, "50tagList.txt"),'r').read().splitlines()
    lastfm_tags = [i.lower() for i in lastfm_tags]
    torch.save(lastfm_tags, os.path.join(DATASET, "supervision", "lastfm_tags.pt"))
    # lastfm split and 
    train_list = pickle.load(open(os.path.join(lastfm_path, "filtered_list_train.cP"), 'rb'))
    test_list = pickle.load(open(os.path.join(lastfm_path, "filtered_list_test.cP"), 'rb'))
    msd_id_to_tag_vector = pickle.load(open(os.path.join(lastfm_path, "msd_id_to_tag_vector.cP"), 'rb'))
    total_list = train_list + test_list
    binary = [msd_id_to_tag_vector[msdid].astype(np.int16).squeeze(-1) for msdid in total_list]
    track_split = {
        "train_track": train_list[0:201680],
        "valid_track": train_list[201680:],
        "test_track": test_list,
    }
    lastfm_binary = pd.DataFrame(binary, index=total_list, columns=lastfm_tags)
    df_lastfm = binary_df_to_list(binary=binary, tags=lastfm_tags, indices=total_list, data_type="lastfm")
    return df_lastfm, track_split

def allmusic_processor(allmusic_path):
    """
    input: allmusic_path 
    return: pandas.DataFrame => index: msd trackid, columns: list of tag
            TRWYIGP128F1454835	[Pop/Rock, Electronic, Adult Alternative Pop/R...
            TRGFXIU128F1454832	[Pop/Rock, Electronic, Adult Alternative Pop/R...
    """
    df_all = pd.read_hdf(os.path.join(allmusic_path, 'ground_truth_assignments/AMG_Multilabel_tagsets/msd_amglabels_all.h5'))
    tag_stats, tag_dict = {}, {}
    for category in df_all.columns:
        df_all[category] = df_all[category].apply(NaN_to_emptylist)
        df_all[category] = df_all[category].map(lambda x: list(map(str.lower, x)))
        tag_stats[category[:-1]] = {i:j for i,j in Counter(flatten_list_of_list(df_all[category])).most_common()}
        for tag in set(flatten_list_of_list(df_all[category])):
            tag_dict[tag] = category[:-1]
    torch.save(list(tag_dict.keys()), os.path.join(DATASET, "supervision", "allmusic_tags.pt"))
    torch.save(tag_dict, os.path.join(DATASET, "supervision", "allmusic_tag_info.pt"))
    torch.save(tag_stats, os.path.join(DATASET, "supervision", "allmusic_tag_stats.pt"))

    tag_list = df_all['genres']+df_all['styles']+df_all['moods']+df_all['themes']
    df_allmusic = pd.DataFrame(index=df_all.index, columns=["allmusic"])
    df_allmusic["allmusic"] = tag_list
    df_allmusic['is_allmusic'] = [True for i in range(len(df_allmusic))]
    return df_allmusic

def msd500_processor(msd500_path):
    msd500_tags = pd.read_csv(os.path.join(msd500_path,"selected_tags.tsv"), sep='\t', header=None)
    msd500_map = {'mood':'mood', 'instrument':'instrument', 'activity':'theme', 
            'language':'language', 'location':'location', 'decade':'decade', 'genre':'genre'}
    msd500_tag_info = {i:msd500_map[j.split("/")[0]] for i,j in zip(msd500_tags[0], msd500_tags[1])}
    msd500_anno = pd.read_csv(os.path.join(msd500_path,"track_tags.tsv"), sep="\t", header=None)
    use_tag = list(msd500_tag_info.keys())
    msd500_anno = msd500_anno.set_index(2)
    msd500_anno = msd500_anno.loc[use_tag]
    item_dict = {i:[] for i in msd500_anno[0]}
    for _id, tag in zip(msd500_anno[0], msd500_anno.index):
        item = item_dict[_id].copy()
        item.append(tag)
        item_dict[_id] = list(set(item))

    df_msd500 = pd.DataFrame(index=item_dict.keys())
    df_msd500['msd500'] = item_dict.values()
    df_msd500['is_msd500'] = [True for i in range(len(df_msd500))]
    df_msd500.index.name = "track_id"
    msd500_tag_stat = {i:j for i,j in Counter(flatten_list_of_list(df_msd500['msd500'])).most_common()}
    torch.save(msd500_tag_info, os.path.join(DATASET, "supervision", "msd500_tag_info.pt"))
    torch.save(msd500_tags[0], os.path.join(DATASET, "supervision", "msd500_tags.pt"))
    torch.save(msd500_tag_stat, os.path.join(DATASET, "supervision", "msd500_tag_stats.pt"))

    return df_msd500

def _check_mp3_file(df_msd, id_to_path, MSD_id_to_7D_id):
    mp3_path, error_id = {}, []
    for msdid in df_msd.index:
        try:
            mp3_path[msdid] = id_to_path[MSD_id_to_7D_id[msdid]]
        except:
            error_id.append(msdid)
    df_msd = df_msd.drop(error_id)
    return df_msd, mp3_path

def MSD_processor(msd_path):
    meta_path = os.path.join(msd_path, "track_metadata.db")
    lastfm_path = os.path.join(msd_path, "lastfm_annotation")
    allmusic_path = os.path.join(msd_path, "allmusic_annotation")
    msd500_path = os.path.join(msd_path, "msd500_annotation")

    MSD_id_to_7D_id = pickle.load(open(os.path.join(lastfm_path, "MSD_id_to_7D_id.pkl"), 'rb'))
    id_to_path = pickle.load(open(os.path.join(lastfm_path, "7D_id_to_path.pkl"), 'rb'))
    lastfm_tags = [i.lower() for i in open(os.path.join(lastfm_path, "50tagList.txt"),'r').read().splitlines()]
    df_msdmeta = getMsdInfo(meta_path)
    df_lastfm, track_split = lastfm_processor(lastfm_path)
    df_msd500 = msd500_processor(msd500_path)
    df_allmusic = allmusic_processor(allmusic_path)
    merge_msd = pd.merge(df_lastfm, df_msd500, how='outer',on='track_id')
    df_tags = pd.merge(merge_msd, df_allmusic, how='outer',on='track_id')
    for column in df_tags.columns:
        if "is_" in column:
            df_tags[column] = df_tags[column].fillna(False)
        else:
            df_tags[column] = df_tags[column].apply(NaN_to_emptylist)
    df_tags['tag'] = df_tags['allmusic'] + df_tags['lastfm'] + df_tags['msd500']
    df_final = pd.merge(df_tags, df_msdmeta, how='left',on='track_id')
    target_col = ["tag","lastfm","msd500","allmusic","is_lastfm","is_msd500","is_allmusic","release","artist_name","year","title"]
    df_target = df_final[target_col]
    df_target, mp3_path = _check_mp3_file(df_target, id_to_path, MSD_id_to_7D_id)
    
    # with poolcontext(processes=multiprocessing.cpu_count()) as pool:
    #     pool.starmap(msd_resampler, zip(list(mp3_path.keys()),list(mp3_path.values())))
    # print("finish extract")

    # error_ids = [msdid.replace(".npy","") for msdid in os.listdir(os.path.join(msd_path, 'error'))]
    # df_target = df_target.drop(error_ids) # drop errors    
    df_target['track_id'] = df_target.index
    torch.save(df_target.to_dict('index'), os.path.join(msd_path, 'annotation.pt')) # 183M
    total_track = track_split['train_track'] + track_split['valid_track'] + track_split['test_track']
    extra_track = list(df_target.index.drop(total_track))
    random.shuffle(extra_track)
    split_ratio = int(len(extra_track) * 0.1)
    track_split['extra_valid_track'] = extra_track[:split_ratio]
    track_split['extra_train_track'] = extra_track[split_ratio:]

    with open(os.path.join(msd_path, "track_split.json"), mode="w") as io:
        json.dump(track_split, io, indent=4)
    
    track_list = track_split['train_track'] + track_split['valid_track']+ track_split['test_track']
    print("finish msd extraction", len(track_list), "extra_track: ", len(extra_track))