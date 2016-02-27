__author__ = 'wangyufei'

import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/home/feiyu1990/local/caffe-mine-test/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import os

import cPickle

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}


def extract_feature(event_name, net_name, model_name, name, blob_names, img_size = 227):
    imgs = []
    img_file = '/home/feiyu1990/local/event_curation/baseline_all/'+event_name+'/guru_'+name+'_path.txt'
    #if name == 'test':
    #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
    with open(img_file, 'r') as data:
        for line in data:
            imgs.append(line.split(' ')[0])
    model_name = '/home/feiyu1990/local/event_curation/CNN_all_event/' + model_name
    weight_name = '/home/feiyu1990/local/event_curation/CNN_all_event/snapshot/' + net_name + '.caffemodel'
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
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
        out = net.forward()
        for i in xrange(len(blob_names)):
            a = net.blobs[blob_names[i]].data.copy()
            features[i].append(a)
            print img, count
        count += 1
    for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event/features/'+event_name + '_' +name+'_'+blob_names[i]+ net_name + '.cPickle','wb')
        cPickle.dump(features[i], f)
        f.close()
if __name__ == "__main__":

    for event_name in dict_name2:
       extract_feature(event_name, 'segment_iter_70000','python_deploy_siamese_old.prototxt','test',['fc8_multevent'])
