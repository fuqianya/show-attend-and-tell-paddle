"""
loss.py
~~~~~~~

Loss functions used to compute loss.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class XECriterion(nn.Layer):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, mask):
        """Inputs:
         - pred: [batch_size, seq_len, vocab_size].
         - target: [batch_size, seq_len].
         - mask: [batch_size, seq_len].
        """
        # truncate to the same size
        target = target[:, :pred.shape[1]]
        mask = mask[:, :pred.shape[1]]

        loss_ = F.cross_entropy(pred, target, reduction='none')
        loss_ *= mask

        return paddle.sum(loss_) / paddle.sum(mask)