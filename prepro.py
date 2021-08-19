"""
prepro.py
~~~~~~~~~

Preprocessing the Flickr8k dataset for training and evluation.
"""
import os
import sys
import json
import skimage.io
import argparse
import pandas as pd
from collections import Counter

import numpy as np
from tqdm import tqdm

# paddle
import paddle
# encoder
from model.encoder import Encoder

def read_imlist(file):
    imname_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            imname_list.append(line.strip())

    return imname_list

def process_captions(params):
    """Prepare caption tokens, groundtruth sentences (test split) and vocab dict."""
    print('Start to process captions ... ')
    train_split_file = params['flickr8k_split_file'].format('train')
    dev_split_file = params['flickr8k_split_file'].format('dev')
    test_split_file = params['flickr8k_split_file'].format('test')

    # read split from file
    train_imname_list = read_imlist(train_split_file)
    dev_imname_list = read_imlist(dev_split_file)
    test_imname_list = read_imlist(test_split_file)
    print('Found %d, %d and %d images for training, val and test, respectively.'
          % (len(train_imname_list), len(dev_imname_list), len(test_imname_list)))

    annotation_path = params['flickr8k_caption_file']
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    captions = annotations['caption'].values
    image_names = annotations['image'].values

    groundtruth_test = {}
    groundtruth_val = {}
    idx2word = Counter()
    cap_obj = {'train': {}, 'val': {}, 'test': {}}
    for image_name, caption in zip(image_names, captions):
        # check if im_name is in splits
        im_name, caption_id = image_name.split('#')
        assert int(caption_id) >= 0 and int(caption_id) <= 4

        if im_name in train_imname_list:
            split = 'train'
        elif im_name in dev_imname_list:
            split = 'val'
        elif im_name in test_imname_list:
            split = 'test'
        else:
            continue

        caption = caption.replace('\n', '').replace('"', '').replace('.', '').strip().lower()
        token = caption.split()

        # update the vocabulary
        idx2word.update(token)

        if im_name not in cap_obj[split]:
            cap_obj[split][im_name] = []
        cap_obj[split][im_name].append(token)

        if split == 'val':
            if im_name not in groundtruth_val:
                groundtruth_val[im_name] = []
            groundtruth_val[im_name].append(caption)

        if split == 'test':
            if im_name not in groundtruth_test:
                groundtruth_test[im_name] = []
            groundtruth_test[im_name].append(caption)

    # dump captions into output_captions
    with open(params['output_captions'], 'w') as f:
        json.dump(cap_obj, f)

    # dump groundtruth into output_groundtruth
    with open(params['output_groundtruth'] + '_val.json', 'w') as f:
        json.dump(groundtruth_val, f)

    with open(params['output_groundtruth'] + '_test.json', 'w') as f:
        json.dump(groundtruth_test, f)

    # filter the words that appear less than five times
    idx2word = idx2word.most_common()
    idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [w[0] for w in idx2word if w[1] > 4]
    print('Found %d words in the dataset' % len(idx2word))

    # dump idx2word into output_idx2word
    with open(params['output_idx2word'], 'w') as f:
        json.dump(idx2word, f)

    print('Process captions done!\n')

def prepare_features(params):
    """Using encoder to extract features."""
    print('Start to process features ... ')

    # set up encoder
    encoder = Encoder()

    splits = ['train', 'dev', 'test']
    img_dir = params['flickr8k_image_dir']
    for split in splits:
        print('Processing split: %s ...' % split)
        f = open(params['flickr8k_split_file'].format(split))
        image_list = [i.strip() for i in f]

        for img_nm in image_list:
            with paddle.no_grad():
                img = encoder.preprocess_image(os.path.join(img_dir, img_nm))
                img_fc, img_att = encoder(img)
                fc_feat = img_fc.numpy().reshape((2048,))
                att_feat = img_att.numpy().reshape((196, 2048))
                save_path = os.path.join(params['output_feature_dir'], img_nm[:-4] + '.npz')
                np.savez(save_path, fc_feat=fc_feat, att_feat=att_feat)

    print('Process features done!')

def main(params):
    output_feat_dir = params['output_feature_dir']
    if not os.path.isdir(output_feat_dir ):
        os.makedirs(output_feat_dir)

    # prepare captions
    process_captions(params)

    # prepare features
    prepare_features(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input files
    parser.add_argument('--flickr8k_image_dir', type=str, default='./images/Flicker8k_Dataset/',
                        help='path to the images of flickr8k')
    parser.add_argument('--flickr8k_caption_file', type=str, default='./data/Flickr8k.token.txt',
                        help='path to the caption file of flickr8k.')
    parser.add_argument('--flickr8k_split_file', type=str, default='./data/Flickr_8k.{}Images.txt',
                        help='path to the split files of flickr8k')

    # output files
    parser.add_argument('--output_feature_dir', type=str, default='./data/feats',
                        help='folder to be store the preprocessed features.')
    parser.add_argument('--output_captions', type=str, default='./data/captions.json',
                        help='path to store the preprocessed captions.')
    parser.add_argument('--output_idx2word', type=str, default='./data/idx2word.json',
                        help='path to store the word-index mapping dictionary.')
    parser.add_argument('--output_groundtruth', type=str, default='./data/groundtruth',
                        help='path to store the captions of test split, which prepares for evaluation.')

    opt = parser.parse_args()
    params = vars(opt)  # convert to dictionary

    # call main()
    main(params)