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
"""Models for 'Deep Joint demosaicking and Denoising'."""

from caffe import layers as L, params as P
import caffe

__all__ = ['demosaic']


def _convolution(bottom, width, ksize, pad=True, bias=True):
    """Parametrized convolution layer."""
    if pad:
        padding = (ksize-1)/2
    else:
        padding = 0

    if bias:
        return L.Convolution(
            bottom=bottom,
            param=[{'lr_mult': 1, 'decay_mult': 1},
                   {'lr_mult': 2, 'decay_mult': 0}],
            convolution_param={
                'num_output': width,
                'kernel_size': ksize,
                'pad': padding,
                'weight_filler': {
                    'type': 'msra',
                    'variance_norm': P.Filler.AVERAGE,
                },
                'bias_filler': {
                    'type': 'constant',
                    'value': 0,
                }
            })
    else:
        return L.Convolution(
            bottom=bottom,
            param=[{'lr_mult': 1, 'decay_mult': 1}],
            convolution_param={
                'num_output': width,
                'kernel_size': ksize,
                'pad': padding,
                'weight_filler': {
                    'type': 'msra',
                    'variance_norm': P.Filler.AVERAGE,
                },
            })

#pylint: disable=too-many-arguments
def demosaic(depth, width, ksize, batch_size,
             non_linearity='relu',
             mosaic_type='bayer', trainset=None,
             train_mode=True,
             min_noise=0, max_noise=0, pad=True,
             batch_norm=False):
    """Network to denoise/demosaic Bayer arrays."""

    if non_linearity == 'relu':
      NL = L.ReLU
    else:
      NL = L.TanH

    if mosaic_type not in ['bayer', 'xtrans']:
        raise ValueError('Unknown mosaic type "{}".'.format(mosaic_type))

    offset_x = 1
    offset_y = 1
    if mosaic_type == 'xtrans':
        offset_x = 5
        offset_y = 5

    net = caffe.NetSpec()

    add_noise = (min_noise > 0 or max_noise > 0)

    if add_noise and min_noise > max_noise:
        raise ValueError('min noise is greater than max_noise')

    if trainset is not None:  # Build the network with database connection
        # Read from an LMDB database for train and validation sets
        net.demosaicked = L.Data(
            data_param={'source': trainset,
                        'backend': P.Data.LMDB,
                        'batch_size': batch_size},
            transform_param={'scale': 0.00390625})

        # Extend the data
        net.offset = L.Python(bottom='demosaicked',
                              python_param={'module':'demosaicnet.layers',
                                            'layer': 'RandomOffsetLayer',
                                            'param_str': '{"offset_x": %s, "offset_y":%s}' % (offset_x, offset_y)})
        net.rot = L.Python(bottom='offset',
                            python_param={'module':'demosaicnet.layers',
                                          'layer': 'RandomRotLayer'})
        net.groundtruth = L.Python(bottom='rot',
                              python_param={'module':'demosaicnet.layers',
                                            'layer': 'RandomFlipLayer'})

        # Add noise
        if add_noise:
            net.noisy, net.noise_level = L.Python(
                    ntop=2,
                    bottom='groundtruth',
                    python_param={'module':'demosaicnet.layers',
                                  'layer': 'AddGaussianNoiseLayer',
                                  'param_str': '{"min_noise": %f, "max_noise":%f}' % (min_noise, max_noise)})
            layer_to_mosaick = 'noisy'
        else:
            layer_to_mosaick = 'groundtruth'

        # ---------------------------------------------------------------------
        if mosaic_type == 'bayer':
            net.mosaick = L.Python(bottom=layer_to_mosaick,
                                   python_param={'module':'demosaicnet.layers',
                                                 'layer': 'BayerMosaickLayer'})
        else:
            net.mosaick = L.Python(bottom=layer_to_mosaick,
                                   python_param={'module':'demosaicnet.layers',
                                                 'layer': 'XTransMosaickLayer'})
        # ---------------------------------------------------------------------


    else:  # Build the test network
        net.mosaick = L.Input(shape=dict(dim=[batch_size, 3, 128, 128]))
        if add_noise:
            net.noise_level = L.Input(shape=dict(dim=[batch_size]))

    # -------------------------------------------------------------------------
    if mosaic_type == 'bayer':
        # Pack mosaick (2x2 downsampling)
        net.pack = L.Python(bottom='mosaick',
                            python_param={'module':'demosaicnet.layers',
                                          'layer': 'PackBayerMosaickLayer'})
        pre_noise_layer = 'pack'
    else:
        pre_noise_layer = 'mosaick'
    # -------------------------------------------------------------------------

    # Add noise input
    if add_noise:
        net.replicated_noise_level = L.Python(
                bottom=['noise_level', pre_noise_layer],
                python_param={'module':'demosaicnet.layers',
                              'layer': 'ReplicateLikeLayer'})
        net.pack_and_noise = L.Concat(bottom=[pre_noise_layer, 'replicated_noise_level'])

    # Process
    for layer_id in range(depth):
        name = 'conv{}'.format(layer_id+1)
        if layer_id == 0:
            if add_noise:
                bottom = 'pack_and_noise'
            else:
                bottom = pre_noise_layer
        else:
            if batch_norm:
                bottom = 'conv{}'.format(layer_id)
            else:
                bottom = 'conv{}'.format(layer_id)

        if mosaic_type=='bayer' and layer_id == depth-1:
            nfilters = 12
        else:
            nfilters = width

        if batch_norm:
            bn = 'norm{}'.format(layer_id+1)
            relu = 'relu{}'.format(layer_id+1)
            net[name] = _convolution(bottom, nfilters, ksize, pad=pad)
            net[relu] = NL(net[name], in_place=True)
            net[bn] = L.BatchNorm(net[relu],
                    batch_norm_param=dict(use_global_stats=not train_mode),
                    param=[dict(lr_mult=0, decay_mult=0)]*3, in_place=True)
        else:
            net[name] = _convolution(bottom, nfilters, ksize, pad=pad)
            net['relu{}'.format(layer_id+1)] = NL(net[name], in_place=True)

    # -------------------------------------------------------------------------
    if mosaic_type == 'bayer':
        bottom = 'conv{}'.format(depth)
        # Unpack result
        net.unpack = L.Python(bottom=bottom,
                              python_param={'module':'demosaicnet.layers',
                                            'layer': 'UnpackBayerMosaickLayer'})
        unpack_layer = 'unpack'
    else:
        unpack_layer = 'conv{}'.format(depth)
    # -------------------------------------------------------------------------

    # Fast-forward input mosaick
    if not pad:
        net.cropped_mosaick = L.Python(bottom=['mosaick', unpack_layer],
                                       python_param={'module':'demosaicnet.layers',
                                                     'layer': 'CropLikeLayer'})
        mosaick_layer = 'cropped_mosaick'
    else:
        mosaick_layer = 'mosaick'
    net.residual_and_mosaick = L.Concat(bottom=[unpack_layer, mosaick_layer])

    # Full-res convolution
    net['fullres_conv'] = _convolution('residual_and_mosaick', width, ksize, pad=pad)
    net['fullres_relu'] = NL(net['fullres_conv'], in_place=True)
    net['output'] = _convolution('fullres_conv', 3, 1)

    if trainset is not None:  # Add a loss for the train network
        # Output
        if not pad:
            net.cropped_groundtruth = L.Python(bottom=['groundtruth', 'output'],
                                          python_param={'module':'demosaicnet.layers',
                                                         'layer': 'CropLikeLayer'})
            gt_layer = 'cropped_groundtruth'
        else:
            gt_layer = 'groundtruth'

        net['loss'] = L.Python(bottom=['output', gt_layer],
                                      loss_weight=1.0,
                                      python_param={'module':'demosaicnet.layers',
                                                    'layer':'NormalizedEuclideanLayer'})

    return net
#pylint: enable=too-many-arguments
