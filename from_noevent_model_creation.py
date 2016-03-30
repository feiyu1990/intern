__author__ = 'wangyufei'

import numpy as np

# root = '/home/feiyu1990/local/event_curation/CNN_all_event_google/'
# root = '/home/feiyu1990/local/event_curation/CNN_all_event_1009/'
# root = '/home/feiyu1990/local/event_curation/CNN_all_event_corrected/'
root = '/home/feiyu1990/local/event_curation/CNN_all_event_vgg/'

caffe_root = '/home/feiyu1990/local/caffe-mine-test/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#this is from affinity clustering (cluster #6)
dict_subcategory = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 3, 7: 4, 8: 4, 9: 2, 10: 4, 11: 3, 12: 4, 13: 4, 14: 4,
                    15: 5, 16: 3, 17: 4, 18: 4, 19: 2, 20: 4, 21: 4, 22: 4}

#this is from spectral clustering (cluster #3)
dict_subcategory2 = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 2, 10: 0, 11: 2, 12: 0,
                     13: 0, 14: 0, 15: 2, 16: 2, 17: 0, 18: 0, 19: 2, 20: 1, 21: 2, 22: 0}

def model_creation(model_name, out_name):
        caffe.set_device(1)
        caffe.set_mode_gpu()
        in_model_name = root + 'VGG_deploy_noevent.prototxt'
        in_weight_name = root + 'snapshot/'+model_name
        out_weight_name = root + 'snapshot/' + out_name
        in_net = caffe.Net(in_model_name,
                    in_weight_name,
                    caffe.TEST)
        out_model_name = root + 'VGG_train_val_alex_siamese_allevent_fromnoevent.prototxt'
        out_net = caffe.Net(out_model_name,
                    in_weight_name,
                    caffe.TEST)

        a = in_net.params['fc8_multevent'][0].data
        print a.shape
        out_net.params['fc8_multevent1'][0].data[...] = np.tile(a,(23,1))
        out_net.params['fc8_multevent1_p'][0].data[...] = np.tile(a,(23,1))
        a = in_net.params['fc8_multevent'][1].data
        out_net.params['fc8_multevent1'][1].data[...] = np.tile(a,(23))
        out_net.params['fc8_multevent1_p'][1].data[...] = np.tile(a,(23))
        print out_net.params['fc8_multevent1_p'][0].data
        print out_net.params['fc8_multevent1_p'][1].data
        out_net.save(out_weight_name)
def model_creation_vgg(model_name, out_name):
        caffe.set_device(1)
        caffe.set_mode_gpu()
        in_model_name = root + 'VGG_deploy_noevent.prototxt'
        in_weight_name = root + 'snapshot/'+model_name
        out_weight_name = root + 'snapshot/' + out_name
        in_net = caffe.Net(in_model_name,
                    in_weight_name,
                    caffe.TEST)
        out_model_name = root + 'training/VGG_train_val_alex_siamese_allevent_fromnoevent.prototxt'
        out_net = caffe.Net(out_model_name,
                    in_weight_name,
                    caffe.TEST)

        a = in_net.params['fc8_multevent'][0].data
        print a.shape
        out_net.params['fc8_multevent1'][0].data[...] = np.tile(a,(23,1))
        out_net.params['fc8_multevent1_p'][0].data[...] = np.tile(a,(23,1))
        a = in_net.params['fc8_multevent'][1].data
        out_net.params['fc8_multevent1'][1].data[...] = np.tile(a,(23))
        out_net.params['fc8_multevent1_p'][1].data[...] = np.tile(a,(23))
        # print out_net.params['fc8_multevent1_p'][0].data
        # print out_net.params['fc8_multevent1_p'][1].data
        out_net.save(out_weight_name)
def model_creation_superclass(model_name):
        caffe.set_device(1)
        caffe.set_mode_gpu()
        in_model_name = root + 'train_val_noevent.prototxt'
        in_weight_name = root + 'snapshot/'+model_name
        out_weight_name = root + 'snapshot/segment_fromnoevent_superclass_iter_start.caffemodel'
        print out_weight_name
        in_net = caffe.Net(in_model_name,
                    in_weight_name,
                    caffe.TEST)
        out_model_name = root + 'training/train_val_superevent.prototxt'
        out_net = caffe.Net(out_model_name,
                    in_weight_name,
                    caffe.TEST)

        a = in_net.params['fc8_multevent'][0].data
        out_net.params['fc8_multevent1'][0].data[...] = np.tile(a,(6,1))
        out_net.params['fc8_multevent1_p'][0].data[...] = np.tile(a,(6,1))
        a = in_net.params['fc8_multevent'][1].data
        out_net.params['fc8_multevent1'][1].data[...] = np.tile(a,(6))
        out_net.params['fc8_multevent1_p'][1].data[...] = np.tile(a,(6))
        # print out_net.params['fc8_multevent1_p'][0].data
        # print out_net.params['fc8_multevent1_p'][1].data
        out_net.save(out_weight_name)
def model_creation_fromsuperclass(model_name):
        caffe.set_device(1)
        caffe.set_mode_gpu()
        in_model_name = root + 'training/train_val_superevent.prototxt'
        in_weight_name = root + 'snapshot/'+model_name
        out_weight_name = root + 'snapshot/segment_fromsuperclass_frombegin_iter_100000_start.caffemodel'
        print out_weight_name
        in_net = caffe.Net(in_model_name,
                    in_weight_name,
                    caffe.TEST)
        out_model_name = root + 'train_val_fromsuper.prototxt'
        out_net = caffe.Net(out_model_name,
                    in_weight_name,
                    caffe.TEST)

        # a = in_net.params['conv1'][0].data
        # b = in_net.params['conv1'][1].data
        # print a.shape, b.shape
        a = in_net.params['fc8_multevent1'][0].data
        b = np.zeros((23,a.shape[1]))
        for i in xrange(23):
            b[i,:] = a[dict_subcategory2[i],:]
        out_net.params['fc8_multevent2'][0].data[...] = b
        out_net.params['fc8_multevent2_p'][0].data[...] = b


        a = in_net.params['fc8_multevent1'][1].data
        b = np.zeros((23,))
        for i in xrange(23):
            b[i] = a[dict_subcategory2[i]]
        out_net.params['fc8_multevent2'][1].data[...] = b
        out_net.params['fc8_multevent2_p'][1].data[...] = b

        out_net.save(out_weight_name)
def model_creation_twoloss(model_name, out_name):
        caffe.set_device(1)
        caffe.set_mode_gpu()
        in_model_name = root + 'python_deploy_siamese_noevent.prototxt'
        in_weight_name = root + 'snapshot/'+model_name
        out_weight_name = root + 'snapshot/' + out_name
        in_net = caffe.Net(in_model_name,
                    in_weight_name,
                    caffe.TEST)
        out_model_name = root + 'training/train_val_fromnoevent_twoloss.prototxt'
        out_net = caffe.Net(out_model_name,
                    in_weight_name,
                    caffe.TEST)

        a = in_net.params['fc8_multevent'][0].data
        print a.shape
        out_net.params['fc8_loss1'][0].data[...] = np.tile(a,(23,1))
        out_net.params['fc8_loss1_p'][0].data[...] = np.tile(a,(23,1))
        a = in_net.params['fc8_multevent'][1].data
        out_net.params['fc8_loss1'][1].data[...] = np.tile(a,(23))
        out_net.params['fc8_loss1_p'][1].data[...] = np.tile(a,(23))
        print out_net.params['fc8_loss1_p'][0].data
        print out_net.params['fc8_loss1_p'][1].data
        out_net.save(out_weight_name)

if __name__ == '__main__':
    model_creation('VGG_noevent_0.5_iter_100000.caffemodel','VGG_fromnoevent_iter_start_0.5_10w.caffemodel')
    # model_creation_twoloss('segment_noevent_iter_100000.caffemodel','segment_twoloss_fromnoevent_iter_start.caffemodel')
    # model_creation('segment_noevent_iter_100000.caffemodel','segment_fromnoevent_iter_start.caffemodel')

    # model_creation_superclass('segment_noevent_iter_100000.caffemodel')
    # model_creation_fromsuperclass('segment_superevent_iter_20000.caffemodel')
    # model_creation_fromsuperclass('segment_superevent_frombegin_iter_100000.caffemodel')
