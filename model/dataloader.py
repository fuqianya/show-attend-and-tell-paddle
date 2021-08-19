# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
dataloader.py
~~~~~~~~~~~~~

A class responsibe for load batch of data for training and eval.
"""
import os
import h5py
import numpy as np

# paddle
import paddle
from paddle.io import Dataset, DataLoader

class CaptionLoader(Dataset):
    def __init__(self, feat_dir, captions):
        self.feat_dir = feat_dir
        self.captions = list(captions.items())

    def __getitem__(self, index):
        img_name, caps = self.captions[index]
        feats = np.load(os.path.join(self.feat_dir, img_name[:-4] + '.npz'))
        fc_feat = feats['fc_feat']
        att_feat = feats['att_feat']

        return img_name, caps, fc_feat, att_feat

    def __len__(self):
        return len(self.captions)

def create_collate_fn(pad_index, max_seq_len):
    """Process captions like clip and pad."""
    def collate_fn(dataset):
        groundtruth = {}
        tmp = []
        for img_id, caps, fc_feat, att_feat in dataset:
            groundtruth[img_id] = [c[:max_seq_len] for c in caps]
            for cap in caps:
                tmp.append([img_id, cap, fc_feat, att_feat])

        dataset = tmp
        dataset.sort(key=lambda p: len(p[1]), reverse=True)
        img_ids, caps, fc_feats, att_feats = zip(*dataset)
        fc_feats = np.array(fc_feats, dtype='float32')
        att_feats = np.array(att_feats, dtype='float32')

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_array = np.zeros((len(caps), max(lengths)), dtype='int64')
        caps_array.fill(pad_index)
        mask_array = np.zeros((len(caps), max(lengths) - 1))
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            caps_array[i, :end_cap] = np.array(c[:end_cap])
            mask_array[i, :end_cap - 1] = 1
        lengths = [l - 1 for l in lengths]
        return img_ids, fc_feats, att_feats, (caps_array, lengths), mask_array, groundtruth

    return collate_fn

def get_dataloader(feat_dir, captions, pad_index, max_seq_len, batch_size, num_workers, shuffle=True, collate_fn=True):
    dataset = CaptionLoader(feat_dir, captions)

    # collate_fn is set True for training and eval
    # and set False for test
    if collate_fn:
        collate_function = create_collate_fn(pad_index, max_seq_len + 1)
    else:
        collate_function = None
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_function)

    return dataloader