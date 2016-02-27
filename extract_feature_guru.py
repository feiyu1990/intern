__author__ = 'wangyufei'

import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/home/feiyu1990/local/caffe-mine/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import os

import cPickle

def extract_feature(net_name, model_name, name, blob_names, img_size = 227):
    imgs = []
    img_file = '/home/feiyu1990/local/event_curation/wedding_'+name+'_path.txt'
    #if name == 'test':
    #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
    with open(img_file, 'r') as data:
        for line in data:
            imgs.append(line.split(' ')[0])
    model_name = '/home/feiyu1990/local/event_curation/wedding_CNN_net/net/' + model_name
    weight_name = '/home/feiyu1990/local/event_curation/wedding_CNN_net/snapshot/caffenet_wedding' + net_name + '.caffemodel'
    mean_file = np.load('/home/feiyu1990/local/caffe-mine/python/caffe/imagenet/ilsvrc_2012_mean.npy')
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
    features = [[] for i in blob_names]
    count = 0
    for img in imgs:
        print img, count
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
        out = net.forward()
        for i in xrange(len(blob_names)):
            a = net.blobs[blob_names[i]].data.copy()
            features[i].append(a)
        count += 1
    for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/wedding_CNN_net/features/'+name+'_'+blob_names[i]+ net_name + '.cPickle','wb')
        cPickle.dump(features[i], f)
        f.close()
if __name__ == "__main__":

    extract_feature('_ranking_euclidean_iter_10000', 'python_deploy_siamese.prototxt', 'test', ['fc8_value'])
   # extract_feature('_sigmoid_3round_iter_250000', 'python_deploy.prototxt', 'test', ['fc8_value'])
   # extract_feature('_learnlast_2round_iter_160000', 'python_deploy.prototxt', 'test', ['fc8_value'])
   # extract_feature('_largelr_iter_110000', 'python_deploy.prototxt', 'test', ['fc8_value'])
   # extract_feature('_iter_120000', 'python_deploy.prototxt', 'test', ['fc8_value'])
   # extract_feature('_classification0.55_0.8_iter_70000', 'deploy_classification.prototxt', 'test', ['fc8_class'])
   # extract_feature('_classification_learnlast0.55_0.8_iter_40000', 'deploy_classification.prototxt', 'test', ['fc8_class'])
   # extract_feature('_classification_learnlast_multilayer0.55_0.8_iter_70000', 'deploy_classification_multilayer.prototxt', 'test', ['fc8_class'])
   # extract_feature('_ranking_sigmoid_iter_40000', 'python_deploy_siamese.prototxt', 'test', ['sigmoid9'])
   # extract_feature('_vgg_sigmoid_iter_450000', 'deploy_python_vgg.prototxt', 'test', ['fc8_value'], 224)
   # extract_feature('_vgg_euclidean_iter_390000', 'deploy_python_vgg.prototxt', 'test', ['fc8_value'], 224)