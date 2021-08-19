"""
config.py
~~~~~~~~~

Defines configuration of our model and training procedure.
"""
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument('--learning_rate', type=float, default=4e-4)  # 4e-4 for xe, 4e-5 for rl
    parser.add_argument('--resume', type=str, default='',
                        help='continue training from saved model at this path.')  # required for rl
    parser.add_argument('--max_epochs', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)

    # scheduled sampling settings
    parser.add_argument('--scheduled_sampling_start', type=int, default=0)
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5)
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    parser.add_argument('--idx2word', type=str, default='./data/idx2word.json',
                        help='path to store the word-index mapping dictionary.')
    parser.add_argument('--captions', type=str, default='./data/captions.json',
                        help='path to store the preprocessed captions.')
    parser.add_argument('--feat_dir', type=str, default='./data/feats/',
                        help='folder contains the pre-computed feature for each image.')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                        help='folder to save the models.')
    parser.add_argument('--result', type=str, default='./result/',
                        help='folder to save the training results')
    parser.add_argument('--max_seq_len', type=int, default=16,
                        help='clip the captions on this value.')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                        help='clip the gradients on this value.')
    parser.add_argument('--seed', type=int, default=123,
                        help='random number generator seed to use')
    parser.add_argument('--save_checkpoint_every', type=int, default=3,
                        help='how often (epoch) to save a model checkpoint?')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--gpuid', type=int, default=0)

    # eval settings
    parser.add_argument('-e', '--eval_model', type=str, default='./checkpoint/epoch_27.pth',
                        help='which model do you want to evaluate?')

    args = parser.parse_args()

    # decoder settings
    settings = dict()
    settings['word_emb_dim'] = 256
    settings['fc_feat_dim'] = 2048
    settings['att_feat_dim'] = 2048
    settings['feat_emb_dim'] = 512
    settings['dropout_p'] = 0.5
    settings['rnn_hid_dim'] = 512
    settings['att_hid_dim'] = 512
    args.settings = settings

    return args
