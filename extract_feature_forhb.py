__author__ = 'wangyufei'

import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/mnt/ilcompf2d0/project/yuwang/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import scipy.io as sio
import cPickle

def extract_feature(in_file, save_name, model_name, weight_name , blob_name, img_size):
    imgs = []
    #if name == 'test':
    #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
    with open(in_file, 'r') as data:
        for line in data:
            imgs.append(line.split()[0])
    mean_file = np.load('/mnt/ilcompf2d0/project/yuwang/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe.set_mode_gpu()
    net = caffe.Net(model_name,
                weight_name,
                caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_file.mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(1,3,img_size,img_size)
    features = []
    count = 0
    for img in imgs:
        print img, count
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
        out = net.forward()
        a = net.blobs[blob_name].data.copy()
        features.append(a)
        count += 1
    sio.savemat(save_name,{'features': features})

    #f = open(save_name+'.cPickle','wb')
    #cPickle.dump(features, f)
    #f.close()
if __name__ == "__main__":
    args = sys.argv
    assert len(args) > 2
    in_file = args[1]
    save_name = args[2]
    if len(args) > 3:
        model_name = args[3]
    else:
        model_name = '/mnt/ilcompf2d0/project/yuwang/caffe/models/bvlc_alexnet/deploy.prototxt'
    if len(args) > 4:
        weight_name = args[4]
    else:
        weight_name = '/mnt/ilcompf2d0/project/yuwang/event_curation/wedding_CNN_net/pre_train_model/bvlc_alexnet.caffemodel'

    if len(args) > 5:
        blob_name = args[5]
    else:
        blob_name = 'fc8'
    if len(args) > 6:
        img_size = int(args[6])
    else:
        img_size = 227

    extract_feature(in_file, save_name, model_name, weight_name, blob_name, img_size)