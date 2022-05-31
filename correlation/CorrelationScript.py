#!/usr/bin/env python

import os
import numpy as np
from mmsdk import mmdatasdk as md
import heapq

# we can see they are in the format of 'video_id[segment_no]', but the splits was specified with video_id only
# we need to use regex or something to match the video IDs...
import re
import torch
import torch.nn as nn
import pprint

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import tqdm.notebook
from collections import defaultdict

def align_train(all_videos=False):
    # define your different modalities - refer to the filenames of the CSD files
    visual_field = 'CMU_MOSEI_VisualFacet42'
    acoustic_field = 'CMU_MOSEI_COVAREP'
    if all_videos==True:
        hand_r_field = "hand_r_features_all"
        hand_l_field = "hand_l_features_all"
        
    else:
        hand_r_field = "hand_r_features"
        hand_l_field = "hand_l_features"
        
    pose_field = "pose_features_all"
    text_field = 'CMU_MOSEI_TimestampedWordVectors'

    features = [
        text_field,
        visual_field, 
        acoustic_field, 
        pose_field,
        hand_l_field,
        hand_r_field]

    
    is_male={'-tANM6ETl_M':True,
         '1LkYxsqRPZM':False,
         'N-NnCI6U52c':False,
         'OwYfPi9St0w':True,
         'Va54WZgPTdY':True,
         'XQOUZhWI1B0':True,
         'o2bNnLOEEC0':False,
         'pVs1daijYTw':True,
         'rWjPSlzUc-8':False,
         'tkzdanzsA0A':False}
    
    
    recipe = {feat: os.path.join("cmumosei", feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)

    remove = [name for name in list(dataset[visual_field].keys()) + list(dataset[acoustic_field].keys()) + list(dataset[text_field].keys()) + list(dataset[pose_field].keys()) if name not in list(dataset[pose_field].keys())]
    for v in remove:
        dataset.remove_id(v)

    # we define a simple averaging function that does not depend on intervals
    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features

    # first we align to words with averaging, collapse_function receives a list of functions
    #dataset.align(pose_features, collapse_functions=[avg])
    dataset.align(text_field, collapse_functions=[avg])

    label_field = 'CMU_MOSEI_Labels'
    # we add and align to lables to obtain labeled segments
    # this time we don't apply collapse functions so that the temporal sequences are preserved
    label_recipe = {label_field: os.path.join("cmumosei", label_field + '.csd')}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field)

    pattern = re.compile('(.*)\[.*\]')
    videos = set()
    for video in list(dataset[label_field].keys()):
        vid_id = re.search(pattern, video).group(1)
        videos.add(vid_id)

    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

    # construct a word2id mapping that automatically takes increment when new words are encountered
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['<unk>']
    PAD = word2id['<pad>']

    # place holders for the final train/dev/test dataset
    train = []
    male = []
    female = []
    # define a regular expression to extract the video ID out of the keys
    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0 # a counter to count how many data points went into some processing issues

    for segment in dataset[label_field].keys():
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features']
        _words = dataset[text_field][segment]['features']
        _visual = dataset[visual_field][segment]['features']
        _acoustic = dataset[acoustic_field][segment]['features']
        # Need this because some segments does not exist in pose
        if not segment in dataset[pose_field].keys():
            print(f"Havent found features for {segment}. Continouing")
            continue
        _pose = dataset[pose_field][segment]['features']
        _lhand = dataset[hand_l_field][segment]['features']
        _rhand = dataset[hand_r_field][segment]['features']
        
   
        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0] == _pose.shape[0] == _lhand.shape[0] == _rhand.shape[0]:
            num_drop += 1
            continue
        
        # remove nan values
        label = np.nan_to_num(label)
        _visual = np.nan_to_num(_visual)
        _acoustic = np.nan_to_num(_acoustic)
        _pose = np.nan_to_num(_pose)
        _lhand = np.nan_to_num(_lhand)
        _rhand = np.nan_to_num(_rhand)

        # remove speech pause tokens - this is in general helpful
        # we should remove speech pauses and corresponding visual/acoustic features together
        # otherwise modalities would no longer be aligned
        words = []
        visual = []
        acoustic = []
        pose = []
        lhand = []
        rhand = []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word2id[word[0]]) # SDK stores strings as bytes, decode into strings here
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])
                pose.append(_pose[i, :])
                lhand.append(_lhand[i, :])
                rhand.append(_rhand[i, :])

        words = np.asarray(words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)
        pose = np.asarray(pose)
        lhand = np.asarray(lhand)
        rhand = np.asarray(rhand)

        # z-normalization per instance and remove nan/infs
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
        pose = np.nan_to_num((pose - pose.mean(0, keepdims=True)) / (EPS + np.std(pose, axis=0, keepdims=True)))
        lhand = np.nan_to_num((lhand - lhand.mean(0, keepdims=True)) / (EPS + np.std(lhand, axis=0, keepdims=True)))
        rhand = np.nan_to_num((rhand - rhand.mean(0, keepdims=True)) / (EPS + np.std(rhand, axis=0, keepdims=True)))
        
        train.append(((words, visual, acoustic, pose, lhand, rhand), label, segment))
        #if is_male[vid]:
        #    male.append(((words, visual, acoustic, pose, lhand, rhand), label, segment))
        #else:
        #    female.append(((words, visual, acoustic, pose, lhand, rhand), label, segment))


    print(f"Total number of {num_drop} datapoints have been dropped.")

    # turn off the word2id - define a named function here to allow for pickling
    def return_unk():
        return UNK
    word2id.default_factory = return_unk

    # 
    return train, dataset, features, label_field