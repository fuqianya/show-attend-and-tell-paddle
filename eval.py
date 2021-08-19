"""
train.py
~~~~~~~~

A script to eval the captioner.
"""
import os
import json
import random

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# paddle
import paddle
# options
from config.config import parse_opt
# model
from model.decoder import Captioner
# dataloader
from model.dataloader import get_dataloader
# criterion
from model.loss import XECriterion
# utils
from utils.eval_utils import eval_split

def main(params):
    # load dataset from json files
    idx2word = json.load(open(params['idx2word'], 'r'))
    captions = json.load(open(params['captions'], 'r'))

    # set random seed
    paddle.seed(params['seed'])
    random.seed(params['seed'])

    # set up model
    captioner = Captioner(idx2word, params['settings'])

    # load state dict
    checkpoint_file = params['eval_model']
    # checkpoint_file is in the format of ./checkpoint/epoch_{number}.pth
    epoch = checkpoint_file.split('_')[-1].split('.')[0]
    checkpoint = paddle.load(checkpoint_file)
    captioner.set_state_dict(checkpoint)

    # process image captions before set up dataloader
    print('\n====> process image captions begin')
    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    captions_id = {}
    for split, caps in captions.items():
        captions_id[split] = {}
        for img_id, seqs in tqdm(caps.items(), ncols=100):
            tmp = []
            for seq in seqs:
                tmp.append([captioner.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [captioner.eos_id])
            captions_id[split][img_id] = tmp
    captions = captions_id
    print('\n====> process image captions end')

    # set up dataloader
    feat_dir = params['feat_dir']
    pad_id = captioner.pad_id
    batch_size = params['batch_size']
    max_seq_len = params['max_seq_len']
    num_workers = params['num_workers']

    test_captions = {}
    for fn in captions['test']:
        test_captions[fn] = [[]]
    test_loader = get_dataloader(feat_dir, test_captions, pad_id, max_seq_len, batch_size,
                                 num_workers=4, shuffle=False, collate_fn=False)

    # evaluate the model on val split
    print('Start to evaluate test split ...')
    eval_split(captioner, epoch, test_loader, opt, val_or_test='test')

if __name__ == '__main__':
    opt = parse_opt()
    params = vars(opt)  # convert to dict

    # call main()
    main(params)