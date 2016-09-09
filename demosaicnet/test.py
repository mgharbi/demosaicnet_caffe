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
"""Tests for the addtional Python layers"""

import os
import tempfile
import unittest

import caffe
import numpy as np
import skimage.io


class TestPythonLayer(unittest.TestCase):
    def python_net_file(self, bsize, c, h, w, layer):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("""name: 'pythonnet' force_backward: true
            input: 'data' input_shape { dim:%d dim: %d dim: %d dim: %d }
            layer { type: 'Python' name: 'output' bottom: 'data' top: 'output'
              python_param { module: 'demosaicnet.layers' layer: '%s' } }""" % (bsize, c, h, w, layer))
            return f.name


class TestBayerMosaickLayer(TestPythonLayer):
    def setUp(self):
        net_file = self.python_net_file(16, 3, 4, 4, 'BayerMosaickLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        self.net.blobs['data'].data[:, 0, :, :] = 1
        self.net.blobs['data'].data[:, 1, :, :] = 2
        self.net.blobs['data'].data[:, 2, :, :] = 3
        self.net.forward()
        out = self.net.blobs['output'].data

        # Check red channel
        assert (out[:,0,::2,::2]==0).all()
        assert (out[:,0,1::2,1::2]==0).all()
        assert (out[:,0,1::2,::2]==0).all()
        assert (out[:,0,::2,1::2]==1).all()

        # Check green channel
        assert (out[:,1,::2,::2]==2).all()
        assert (out[:,1,1::2,1::2]==2).all()
        assert (out[:,1,1::2,::2]==0).all()
        assert (out[:,1,::2,1::2]==0).all()

        # Check blue channel
        assert (out[:,2,::2,::2]==0).all()
        assert (out[:,2,1::2,1::2]==0).all()
        assert (out[:,2,1::2,::2]==3).all()
        assert (out[:,2,::2,1::2]==0).all()

    def test_output(self):
        net_file = self.python_net_file(1, 3, 128, 128, 'BayerMosaickLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        for i in range(128):
            self.net.blobs['data'].data[:, :, :, i] = i

        self.net.forward()
        out = self.net.blobs['output'].data

        im = np.squeeze(out[0,:,:,:]).transpose([1,2,0])
        im /= 128
        skimage.io.imsave('output/test_bayer.png', im)

class TestXTransMosaicLayer(TestPythonLayer):
    def setUp(self):
        net_file = self.python_net_file(16, 3, 6, 6, 'XTransMosaickLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        self.net.blobs['data'].data[:, 0, :, :] = 1
        self.net.blobs['data'].data[:, 1, :, :] = 2
        self.net.blobs['data'].data[:, 2, :, :] = 3
        self.net.forward()
        out = self.net.blobs['output'].data

        # Check red channel
        assert (out[:,0,0,4]==1).all()
        assert (out[:,0,1,0]==1).all()
        assert (out[:,0,1,2]==1).all()
        assert (out[:,0,2,4]==1).all()
        assert (out[:,0,3,1]==1).all()
        assert (out[:,0,4,3]==1).all()
        assert (out[:,0,4,5]==1).all()
        assert (out[:,0,5,1]==1).all()

        # Check blue channel
        assert (out[:,2,0,1]==3).all()
        assert (out[:,2,1,3]==3).all()
        assert (out[:,2,1,5]==3).all()
        assert (out[:,2,2,1]==3).all()
        assert (out[:,2,3,4]==3).all()
        assert (out[:,2,4,0]==3).all()
        assert (out[:,2,4,2]==3).all()
        assert (out[:,2,5,4]==3).all()

    def test_output(self):
        net_file = self.python_net_file(1, 3, 128, 128, 'XTransMosaickLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        for i in range(128):
            self.net.blobs['data'].data[:, :, :, i] = i

        self.net.forward()
        out = self.net.blobs['output'].data

        im = np.squeeze(out[0,:,:,:]).transpose([1,2,0])
        im /= 128
        skimage.io.imsave('output/test_xtrans.png', im)


class TestPackBayerMosaicLayer(TestPythonLayer):
    def setUp(self):
        net_file = self.python_net_file(16, 3, 2, 2, 'PackBayerMosaickLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        self.net.blobs['data'].data[:, 1, 0, 0] = 1  # G
        self.net.blobs['data'].data[:, 0, 0, 1] = 2  # R
        self.net.blobs['data'].data[:, 2, 1, 0] = 3  # B
        self.net.blobs['data'].data[:, 1, 1, 1] = 4  # G
        self.net.forward()

        out = self.net.blobs['output'].data

        assert out.shape[1] == 4
        assert out.shape[2] == 1
        assert out.shape[3] == 1
        assert (out[:, 0, 0, 0] == 1).all()
        assert (out[:, 1, 0, 0] == 2).all()
        assert (out[:, 2, 0, 0] == 3).all()
        assert (out[:, 3, 0, 0] == 4).all()

    def test_backward(self):
        self.net.blobs['output'].diff[:, 0, 0, 0] = 1
        self.net.blobs['output'].diff[:, 1, 0, 0] = 2
        self.net.blobs['output'].diff[:, 2, 0, 0] = 3
        self.net.blobs['output'].diff[:, 3, 0, 0] = 4
        self.net.backward()

        diff = self.net.blobs['data'].diff

        assert (diff[:, 1, 0, 0] == 1).all()
        assert (diff[:, 0, 0, 1] == 2).all()
        assert (diff[:, 2, 1, 0] == 3).all()
        assert (diff[:, 1, 1, 1] == 4).all()


class TestUnpackBayerMosaicLayer(TestPythonLayer):
    def setUp(self):
        net_file = self.python_net_file(16, 12, 1, 1, 'UnpackBayerMosaickLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        for i in range(12):
            self.net.blobs['data'].data[:, i, 0, 0] = i+1
        self.net.forward()

        out = self.net.blobs['output'].data

        assert out.shape[1] == 3
        assert out.shape[2] == 2
        assert out.shape[3] == 2
        assert (out[:, 0, ::2, ::2] == 1).all()
        assert (out[:, 0, ::2, 1::2] == 2).all()
        assert (out[:, 0, 1::2, ::2] == 3).all()
        assert (out[:, 0, 1::2, 1::2] == 4).all()

        assert (out[:, 1, ::2, ::2] == 5).all()
        assert (out[:, 1, ::2, 1::2] == 6).all()
        assert (out[:, 1, 1::2, ::2] == 7).all()
        assert (out[:, 1, 1::2, 1::2] == 8).all()

        assert (out[:, 2, ::2, ::2] == 9).all()
        assert (out[:, 2, ::2, 1::2] == 10).all()
        assert (out[:, 2, 1::2, ::2] == 11).all()
        assert (out[:, 2, 1::2, 1::2] == 12).all()

    def test_backward(self):
        for c in range(3):
            self.net.blobs['output'].diff[:, c, 0, 0] = 4*c
            self.net.blobs['output'].diff[:, c, 0, 1] = 4*c+1
            self.net.blobs['output'].diff[:, c, 1, 0] = 4*c+2
            self.net.blobs['output'].diff[:, c, 1, 1] = 4*c+3
        self.net.backward()

        diff = self.net.blobs['data'].diff

        for i in range(12):
            assert (diff[:, i, 0, 0] == i).all()


class TestRandomFlipLayer(TestPythonLayer):
    def setUp(self):
        net_file = self.python_net_file(16, 3, 128, 128, 'RandomFlipLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        for i in range(128):
            self.net.blobs['data'].data[:, :, :, i] = i
        self.net.forward()

        out = self.net.blobs['output'].data
        for i in range(16):
            im = np.squeeze(out[i,:,:,:]).transpose([1,2,0])
            im /= 128
            skimage.io.imsave('output/test_flip{}.png'.format(i), im)


class TestRandomRotLayer(TestPythonLayer):
    def setUp(self):
        net_file = self.python_net_file(16, 3, 128, 128, 'RandomRotLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        for i in range(128):
            self.net.blobs['data'].data[:, :, :, i] = i
        self.net.forward()

        out = self.net.blobs['output'].data
        for i in range(16):
            im = np.squeeze(out[i,:,:,:]).transpose([1,2,0])
            im /= 128
            skimage.io.imsave('output/test_rot{}.png'.format(i), im)


class TestRandomOffsetLayer(TestPythonLayer):
    def python_net_file(self, bsize, c, h, w, layer):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("""name: 'pythonnet' force_backward: true
            input: 'data' input_shape { dim:%d dim: %d dim: %d dim: %d }
            layer { type: 'Python' name: 'output' bottom: 'data' top: 'output'
              python_param { module: 'demosaicnet.layers' layer: '%s' 
              param_str:'{"offset_x":2, "offset_y":2}' } }""" % (bsize, c, h, w, layer))
            return f.name

    def setUp(self):
        net_file = self.python_net_file(16, 3, 8, 8, 'RandomOffsetLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        self.net.blobs['data'].data[:, :, ::2, ::2] = 1
        self.net.blobs['data'].data[:, :, 1::2, 1::2] = 1
        self.net.forward()

        out = self.net.blobs['output'].data
        for i in range(16):
            im = np.squeeze(out[i,:,:,:]).transpose([1,2,0])
            skimage.io.imsave('output/test_offset{}.png'.format(i), im)


class TestCropLikeLayer(TestPythonLayer):
    def python_net_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("""name: 'pythonnet' force_backward: true
            input: 'data' input_shape { dim:16 dim:3 dim: 128 dim: 128 }
            input: 'target' input_shape { dim:16 dim: 3 dim: 64 dim: 64 }
            layer { type: 'Python' name: 'output' bottom: 'data' bottom: 'target'
              top: 'output'
              python_param { module: 'demosaicnet.layers' layer: 'CropLikeLayer'}}""")
            return f.name

    def setUp(self):
        net_file = self.python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        for i in range(64):
            for j in range(64):
                self.net.blobs['data'].data[:, :, 32+i, 32+j] = i*j
        self.net.forward()

        out = self.net.blobs['output'].data
        assert out.shape == self.net.blobs['target'].data.shape

        for i in range(64):
            for j in range(64):
                assert (out[:, :, i, j] == i*j).all()

    def test_backward(self):
        for i in range(64):
            for j in range(64):
                self.net.blobs['output'].diff[:, :, i, j] = i*j
        self.net.backward()

        diff = self.net.blobs['data'].diff

        for i in range(64):
            for j in range(64):
                assert (diff[:, :, 32+i, 32+j] == i*j).all()


class TestAddGaussianNoiseLayer(TestPythonLayer):
    def python_net_file(self, bsize, c, h, w, layer):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("""name: 'pythonnet' force_backward: true
            input: 'data' input_shape { dim:%d dim: %d dim: %d dim: %d }
            layer { type: 'Python' name: 'output' bottom: 'data' top: 'output'
              top: 'noise_level'
              python_param { module: 'demosaicnet.layers' layer: '%s' 
              param_str:'{"min_noise":0, "max_noise":0.1}' } }""" % (bsize, c, h, w, layer))
            return f.name

    def setUp(self):
        net_file = self.python_net_file(16, 3, 128, 128, 'AddGaussianNoiseLayer')
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        for i in range(128):
            self.net.blobs['data'].data[:, :, :, i] = i
        self.net.blobs['data'].data[...] /= 128.0
        self.net.forward()

        out = self.net.blobs['output'].data
        noise_level = self.net.blobs['noise_level'].data
        assert noise_level.size == 16
        assert np.amax(noise_level) <= 0.1
        out[out>1] = 1
        out[out<0] = 0
        for i in range(16):
            im = np.squeeze(out[i,:,:,:]).transpose([1,2,0])
            skimage.io.imsave('output/test_add_gaussian_noise{}.png'.format(i), im)


class TestReplicateLikeLayer(TestPythonLayer):
    def python_net_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("""name: 'pythonnet' force_backward: true
            input: 'data' input_shape { dim:16 }
            input: 'target' input_shape { dim:16 dim: 3 dim: 64 dim: 64 }
            layer { type: 'Python' name: 'output' bottom: 'data' bottom: 'target'
              top: 'output'
              python_param { module: 'demosaicnet.layers' layer: 'ReplicateLikeLayer'}}""")
            return f.name

    def setUp(self):
        net_file = self.python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        for i in range(16):
            self.net.blobs['data'].data[i] = i
        self.net.forward()

        out = self.net.blobs['output'].data
        sz = list(self.net.blobs['target'].data.shape)
        sz[1] = 1
        assert list(out.shape) == sz
        for i in range(16):
            assert (out[i, 0, :, :] == self.net.blobs['data'].data[i]).all()

class TestNormalizedEuclideaanLayer(TestPythonLayer):
    def python_net_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("""name: 'pythonnet' force_backward: true
            input: 'data' input_shape { dim:16 dim: 3 dim: 64 dim: 64 }
            input: 'target' input_shape { dim:16 dim: 3 dim: 64 dim: 64 }
            layer { type: 'Python' name: 'output' bottom: 'data' bottom: 'target'
              top: 'output'
              python_param { module: 'demosaicnet.layers' layer: 'NormalizedEuclideanLayer'}}""")
            return f.name

    def setUp(self):
        net_file = self.python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward_backward(self):
        a = np.random.rand(16, 3, 64, 64)
        b = np.random.rand(16, 3, 64, 64)
        self.net.blobs['data'].data[...] = a
        self.net.blobs['target'].data[...] = b

        self.net.forward()
        self.net.backward()

        out = self.net.blobs['output'].data
        outdiff = self.net.blobs['data'].diff
        outdiff2 = self.net.blobs['target'].diff

        diff = a-b
        count = 16*3*64*64

        assert len(out) == 1
        assert (out[0] - np.sum(np.square(diff))/count) < 1e-6
        assert (out[0] - np.sum(np.square(diff))/count) < 1e-6
        assert np.amax(outdiff - diff/count) < 1e-6
        assert np.amax(outdiff2 - (-diff/count)) < 1e-6
