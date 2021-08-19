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
encoder.py
~~~~~~~~~~

Most common image captioner models are encoder-decoder frameworks, where encoder
is used to extract feature and decoder is to generate words.
"""
import cv2
import numpy as np

# paddle
import paddle
import paddle.nn as nn

from paddle.vision.models import resnet101

class Encoder(nn.Layer):
    def __init__(self, att_size=14):
        super(Encoder, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.eval()
        self.adp_avg_pool = nn.AdaptiveAvgPool2D(output_size=[att_size, att_size])

    def preprocess_image(self, img_path):
        """ preprocess_image """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        img = img.astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        img = np.expand_dims(img, axis=0).copy()

        return paddle.to_tensor(img)

    def forward(self, img, att_size=14):

        x = img

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # [batch_Size, 2048, h, w] --> [batch_size, 2048]
        fc = x.mean(3).mean(2).squeeze()
        att = self.adp_avg_pool(x).squeeze().transpose((1, 2, 0))

        return fc, att