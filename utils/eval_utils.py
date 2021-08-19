"""
eval_utils.py
~~~~~~~~~~~~~

A module that contains utils for evaluation.
"""
import os
import sys
import json
from tqdm import tqdm

from pyutils.cap_eval import eval

# padddle
import paddle

def eval_split(model, epoch, loader, opt, val_or_test='val'):
    # set mode
    model.eval()
    results = []
    for fns, _, fc_feats, att_feats in loader:
        for i, fn in enumerate(fns):
            fc_feat = fc_feats[i]
            att_feat = att_feats[i]

            with paddle.no_grad():
                rest, _ = model.sample(fc_feat, att_feat, beam_size=opt.beam_size, max_seq_len=opt.max_seq_len)
            results.append({'image_id': fn, 'caption': rest[0]})

    if val_or_test == 'val':
        result_file = os.path.join(opt.result, 'epoch_val' + str(epoch) + '.json')
    elif val_or_test == 'test':
        result_file = os.path.join(opt.result, 'epoch_test' + str(epoch) + '.json')
    else:
        raise ValueError
    json.dump(results, open(result_file, 'w'))

    # evaluate with metrics
    if val_or_test == 'val':
        annotation_file = './data/groundtruth_val.json'
    elif val_or_test == 'test':
        annotation_file = './data/groundtruth_test.json'
    else:
        raise ValueError
    eval.evaluate(annotation_file, result_file)
