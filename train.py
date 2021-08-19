"""
train.py
~~~~~~~~

A script to train the captioner.
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
    checkpoint_dir = params['checkpoint']
    result_dir = params['result']
    if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.isdir(result_dir): os.makedirs(result_dir)

    idx2word = json.load(open(params['idx2word'], 'r'))
    captions = json.load(open(params['captions'], 'r'))

    # set random seed
    paddle.seed(params['seed'])
    random.seed(params['seed'])

    # set up model
    captioner = Captioner(idx2word, params['settings'])

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

    train_loader = get_dataloader(feat_dir, captions['train'], pad_id, max_seq_len, batch_size, num_workers=num_workers)
    val_loss_loader = get_dataloader(feat_dir, captions['val'], pad_id, max_seq_len, batch_size, num_workers=4, shuffle=False)

    val_captions = {}
    for fn in captions['val']:
        val_captions[fn] = [[]]
    val_cap_loader = get_dataloader(feat_dir, val_captions, pad_id, max_seq_len, batch_size,
                                    num_workers=4, shuffle=False, collate_fn=False)

    # set up criterion
    xe_criterion = XECriterion()

    # set up optimizer
    learning_rate = params['learning_rate']
    optimizer = paddle.optimizer.Adam(learning_rate=params['learning_rate'],
                                      parameters=captioner.parameters(),
                                      grad_clip=paddle.fluid.clip.ClipGradByValue(params['grad_clip']))

    def lossFun(dataloader, training=True, ss_prob=0.0):
        """Train or val for per epoch."""
        # set mode
        if training: captioner.train()
        else: captioner.eval()

        loss_val = 0.0
        for img_ids, fc_feats, att_feats, (caps_tensor, _), mask_tensor, _  in dataloader:
            pred = captioner(fc_feats, att_feats, caps_tensor, ss_prob=ss_prob)
            loss = xe_criterion(pred, caps_tensor[:, 1:], mask_tensor)

            loss_val += float(loss)
            if training:
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

        return loss_val / len(dataloader)

    previous_loss = None
    patiance = 0
    for epoch in range(params['max_epochs'] + 1):
        print('====> start epoch: %d' % epoch)

        # set up scheduled sampling probability
        ss_prob = 0.0
        if epoch > params['scheduled_sampling_start'] >= 0:
            frac = (epoch - params['scheduled_sampling_start']) // params['scheduled_sampling_increase_every']
            ss_prob = min(params['scheduled_sampling_increase_prob'] * frac, params['scheduled_sampling_max_prob'])

        # train the model for one epoch
        train_loss = lossFun(train_loader, training=True, ss_prob=ss_prob)

        # eval the model
        with paddle.no_grad():
            val_loss = lossFun(val_loss_loader, training=False)

        # decay the learning rates
        if previous_loss is not None and val_loss > previous_loss:
            learning_rate = learning_rate * 0.5
            print('Set learning rate to {}.'.format(learning_rate))
            optimizer.set_lr(learning_rate)


        previous_loss = val_loss

        # save the model
        if epoch % params['save_checkpoint_every'] == 0:
            # evaluate the model on val split
            print('Start to evaluate val split ...')
            eval_split(captioner, epoch, val_cap_loader, opt, val_or_test='val')

            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': params['settings'],
                'idx2word': idx2word
            }
            chkpoint_path = os.path.join(checkpoint_dir, 'epoch_' + str(epoch) + '.pth')
            paddle.save(chkpoint, chkpoint_path)

        print('epoch: %d, train_loss: %.4f, val_loss: %.4f'
              % (epoch, train_loss, val_loss))

if __name__ == '__main__':
    opt = parse_opt()
    params = vars(opt)  # convert to dict

    # call main()
    main(params)