# MIT License
#
# Deep Joint Demosaicking and Denoising
# Siggraph Asia 2016
# Michael Gharbi, Gaurav Chaurasia, Sylvain Paris, Fredo Durand
# 
# Copyright (c) 2016 Michael Gharbi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Python implementation of layers used by the denoising/demosaicking network."""

import caffe
import json
import numpy as np


class BayerMosaickLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        sz = list(bottom[0].data.shape)
        if sz[1] != 3:
            raise Exception("Input should have 3 channels.")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        """Bayer Mosaick.
            
           G R G R G
           B G B G B
           G R G R G
        """
        top[0].data[:, :, :, :] = 0
        top[0].data[:, 1, ::2, ::2] = bottom[0].data[:, 1, ::2, ::2]      # G
        top[0].data[:, 0, ::2, 1::2] = bottom[0].data[:, 0, ::2, 1::2]    # R
        top[0].data[:, 2, 1::2, ::2] = bottom[0].data[:, 2, 1::2, ::2]    # B
        top[0].data[:, 1, 1::2, 1::2] = bottom[0].data[:, 1, 1::2, 1::2]  # G

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class XTransMosaickLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        sz = list(bottom[0].data.shape)
        if sz[1] != 3:
            raise Exception("Input should have 3 channels.")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        """XTrans Mosaick.

           G b G G r G
           r G r b G b
           G b G G r G
           G r G G b G
           b G b r G r
           G r G G b G
        """

        g_mask = np.zeros((6,6))
        g_mask[0,0] = 1
        g_mask[0,2] = 1
        g_mask[0,3] = 1
        g_mask[0,5] = 1

        g_mask[1,1] = 1
        g_mask[1,4] = 1

        g_mask[2,0] = 1
        g_mask[2,2] = 1
        g_mask[2,3] = 1
        g_mask[2,5] = 1

        g_mask[3,0] = 1
        g_mask[3,2] = 1
        g_mask[3,3] = 1
        g_mask[3,5] = 1

        g_mask[4,1] = 1
        g_mask[4,4] = 1

        g_mask[5,0] = 1
        g_mask[5,2] = 1
        g_mask[5,3] = 1
        g_mask[5,5] = 1

        r_mask = np.zeros((6,6))
        r_mask[0,4] = 1
        r_mask[1,0] = 1
        r_mask[1,2] = 1
        r_mask[2,4] = 1
        r_mask[3,1] = 1
        r_mask[4,3] = 1
        r_mask[4,5] = 1
        r_mask[5,1] = 1

        b_mask = np.zeros((6,6))
        b_mask[0,1] = 1
        b_mask[1,3] = 1
        b_mask[1,5] = 1
        b_mask[2,1] = 1
        b_mask[3,4] = 1
        b_mask[4,0] = 1
        b_mask[4,2] = 1
        b_mask[5,4] = 1

        mask = np.dstack((r_mask,g_mask,b_mask))
        mask = mask.transpose([2, 0, 1])

        sz = list(bottom[0].data.shape)

        mask = mask[None, :, :, :]

        h = sz[2]
        w = sz[3]
        nh = np.ceil(h*1.0/6)
        nw = np.ceil(w*1.0/6)

        mask = np.tile(mask,(sz[0], 1, nh, nw))
        mask = mask[:, :, :h,:w]

        top[0].data[...] = bottom[0].data[...]*mask

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class PackBayerMosaickLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        sz = list(bottom[0].data.shape)
        if sz[2] % 2 != 0 or sz[3] % 2 != 0:
            raise Exception("Input should have a spatial extent divisible by 2.")

        if sz[1] != 3:
            raise Exception("Input should be a trichromatic Bayer array.")

    def reshape(self, bottom, top):
        sz = list(bottom[0].data.shape)
        sz[1] = 4
        sz[2] /= 2
        sz[3] /= 2
        top[0].reshape(*sz)

    def forward(self, bottom, top):
        top[0].data[:, 0, :, :] = bottom[0].data[:, 1, ::2, ::2]  # G
        top[0].data[:, 1, :, :] = bottom[0].data[:, 0, ::2, 1::2]  # R
        top[0].data[:, 2, :, :] = bottom[0].data[:, 2, 1::2, ::2]  # B
        top[0].data[:, 3, :, :] = bottom[0].data[:, 1, 1::2, 1::2]  # G

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[:, 1, ::2, ::2]   = top[0].diff[:, 0, :, :] # G
            bottom[0].diff[:, 0, ::2, 1::2]  = top[0].diff[:, 1, :, :]  # R
            bottom[0].diff[:, 2, 1::2, ::2]  = top[0].diff[:, 2, :, :]  # B
            bottom[0].diff[:, 1, 1::2, 1::2] = top[0].diff[:, 3, :, :]   # G


class UnpackBayerMosaickLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        sz = list(bottom[0].data.shape)
        if sz[1] != 12:
            raise Exception("Input should have 12 channels.")

    def reshape(self, bottom, top):
        sz = list(bottom[0].data.shape)
        sz[1] = 3
        sz[2] *= 2
        sz[3] *= 2
        top[0].reshape(*sz)

    def forward(self, bottom, top):
        for c in range(3):
            top[0].data[:, c, ::2, ::2] = bottom[0].data[:, 4*c, :, :]
            top[0].data[:, c, ::2, 1::2] = bottom[0].data[:, 4*c+1, :, :]
            top[0].data[:, c, 1::2, ::2] = bottom[0].data[:, 4*c+2, :, :]
            top[0].data[:, c, 1::2, 1::2] = bottom[0].data[:, 4*c+3, :, :]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            for c in range(3):
                bottom[0].diff[:, 4*c, :, :]   = top[0].diff[:, c, ::2, ::2]
                bottom[0].diff[:, 4*c+1, :, :] = top[0].diff[:, c, ::2, 1::2] 
                bottom[0].diff[:, 4*c+2, :, :] = top[0].diff[:, c, 1::2, ::2] 
                bottom[0].diff[:, 4*c+3, :, :] = top[0].diff[:, c, 1::2, 1::2]


class AddGaussianNoiseLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 2:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        try:
            params = json.loads(self.param_str)
            if 'min_noise' in params:
                self.min_noise = params['min_noise']
            else:
                raise ValueError("Min noise not provided")
            if 'max_noise' in params:
                self.max_noise = params['max_noise']
            else:
                raise ValueError("Max noise not provided")
            if self.min_noise > self.max_noise:
                raise ValueError("Min noise is greater than max noise")
        except:
            raise ValueError("Could not parse param string.")
        

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(bottom[0].data.shape[0])

    def forward(self, bottom, top):
        sz = bottom[0].data.shape
        noise_levels = np.random.rand(sz[0])
        noise_levels *= self.max_noise-self.min_noise
        noise_levels += self.min_noise

        top[1].data[...] = noise_levels[...]
        for n in range(sz[0]):
            noise_level = noise_levels[n]
            noise = noise_level*np.random.randn(1, sz[1], sz[2], sz[3])
            top[0].data[n, :, :, :] = bottom[0].data[n, :, :, :]+noise

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class CropLikeLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Needs two inputs.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        if len(bottom[1].data.shape) != 4:
            raise Exception("Needs 4D input.")

    def reshape(self, bottom, top):
        n, _, h, w = bottom[1].data.shape
        c = bottom[0].data.shape[1]
        top[0].reshape(n, c, h, w)
        src_sz = bottom[0].data.shape
        dst_sz = bottom[1].data.shape
        self.offset = [(s-d)/2 for d,s in zip(dst_sz, src_sz)]

    def forward(self, bottom, top):
        src_sz = bottom[0].data.shape
        dst_sz = bottom[1].data.shape

        for n in range(src_sz[0]):
            for c in range(src_sz[1]):
                top[0].data[n, c, :, :] = bottom[0].data[n, c,
                    self.offset[2]:self.offset[2]+dst_sz[2],
                    self.offset[3]:self.offset[3]+dst_sz[3]]

    def backward(self, top, propagate_down, bottom):
        src_sz = bottom[0].data.shape
        dst_sz = bottom[1].data.shape
        bottom[0].diff[...] = 0
        for n in range(src_sz[0]):
            for c in range(src_sz[1]):
                bottom[0].diff[n, c, 
                    self.offset[2]:self.offset[2]+dst_sz[2],
                    self.offset[3]:self.offset[3]+dst_sz[3]] = top[0].diff[n, c, :, :]



class RandomFlipLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        sz = bottom[0].data.shape
        flips = np.random.randint(0, 2, sz[0])
        for n in range(sz[0]):
            if flips[n] == 1:
                for c in range(sz[1]):
                    top[0].data[n, c, :, :] = np.fliplr(bottom[0].data[n, c, :, :])
            else:
                top[0].data[n, :, :, :] = bottom[0].data[n, :, :, :]

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class RandomRotLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        sz = bottom[0].data.shape
        rot = np.random.randint(0, 4, sz[0])
        for n in range(sz[0]):
            if rot[n] == 0:
                top[0].data[n, :, :, :] = bottom[0].data[n, :, :, :]
            else:
                for c in range(sz[1]):
                    top[0].data[n, c, :, :] = np.rot90(bottom[0].data[n, c, :, :], rot[n])

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class RandomOffsetLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Needs one input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        try:
            params = json.loads(self.param_str)
            if 'offset_x' in params:
                self.offset_x = params['offset_x']
            else:
                raise ValueError("Offset x not provided")
            if 'offset_y' in params:
                self.offset_y = params['offset_y']
            else:
                raise ValueError("Offset y not provided")
        except:
            raise ValueError("Could not parse param string.")
        

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        sz = bottom[0].data.shape
        for n in range(sz[0]):
            offset_y = np.random.randint(0, self.offset_y)
            offset_x = np.random.randint(0, self.offset_x)

            if offset_y > 0:
                if offset_x > 0:
                    top[0].data[n, :, offset_y:, offset_x:] = bottom[0].data[n, :, :-offset_y, :-offset_x]
                    for y in range(offset_y):
                        top[0].data[n, :, y, offset_x:] = bottom[0].data[n, :, 0, :-offset_x]
                    for x in range(offset_x):
                        top[0].data[n, :, offset_y:, x] = bottom[0].data[n, :, :-offset_y, 0]
                        for y in range(offset_y):
                            top[0].data[n, :, y, x] = bottom[0].data[n, :, 0, 0]
                else:
                    top[0].data[n, :, offset_y:, :] = bottom[0].data[n, :, :-offset_y, :]
                    for y in range(offset_y):
                        top[0].data[n, :, y, :] = bottom[0].data[n, :, 0, :]
            else:
                if offset_x > 0:
                    top[0].data[n, :, :, offset_x:] = bottom[0].data[n, :, :, :-offset_x]
                    for x in range(offset_x):
                        top[0].data[n, :, :, x] = bottom[0].data[n, :, :, 0]
                else:
                    top[0].data[n, :, :, :] = bottom[0].data[n, :, :, :]

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class ReplicateLikeLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Needs two inputs.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 1:
            raise Exception("Needs 1D input.")

    def reshape(self, bottom, top):
        sz = list(bottom[1].data.shape)
        sz[1] = 1
        top[0].reshape(*sz)

        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("Inputs batch size do not match.")

    def forward(self, bottom, top):
        sz = bottom[1].data.shape
        for n in range(sz[0]):
            top[0].data[n, :, :, :] = bottom[0].data[n]

    def backward(self, top, propagate_down, bottom):
        raise Exception('gradient is invalid')


class NormalizedEuclideanLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Needs two input.")

        if len(top) != 1:
            raise Exception("Needs one output.")

        if len(bottom[0].data.shape) != 4:
            raise Exception("Needs 4D input.")

        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("Inputs shapes should match")

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff = bottom[0].data-bottom[1].data
        top[0].data[...] = np.sum(np.square(self.diff))/bottom[0].count

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff/bottom[0].count
        if propagate_down[1]:
            bottom[1].diff[...] = -self.diff/bottom[0].count
