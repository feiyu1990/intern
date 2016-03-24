

import shutil
import os
import cPickle
import random
import csv
import numpy as np
import h5py
import copy
from sklearn.cluster import KMeans
from itertools import izip

import re
import operator
import scipy.stats
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, chi2
from collections import Counter, defaultdict
# root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
root = '/home/feiyu1990/local/event_curation/'
#root = '/home/ubuntu/event_curation/'

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}



'''
def hdf5_creation():
    weight_name = root + "wedding_CNN_net/pre_train_model/VGG_ILSVRC_16_layers.caffemodel"
    model_name = root + 'CNN_all_event_vgg/VGG_deploy.prototxt'
    mean_file = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe.set_device(1)
    caffe.set_mode_gpu()
    img_dims = (224,224)
    raw_scale = 255
    channel_swap = (2,1,0)
    net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)
    features = []
    count = 0
    write_folder = (root + 'CNN_all_event_vgg/data/training_images_')
    file_count = 0
    features = np.zeros((10000,512,7,7))
    labels = np.zeros((10000,))
    i = 0
    img_count = 0
    batch_size = 50
    with open(root + 'CNN_all_event_old/data/guru_ranking_reallabel_training_nomargin.txt','r') as data:
        for line in data:
            img_path, img_label = line.split(' ')
            img_label = int(img_label[:-1])
            temp = caffe.io.load_image(img_path)
            input_2 = caffe.io.resize_image(temp, (256,256))
            # print input_2.shape
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            x,y = [random.sample(range(29),1)[0],random.sample(range(29),1)[0]]
            input_2 = input_2[x:x+224,y:y+224,:]
            # print x,y,input_2.shape
            # break
            inputs = [input_2]
            out = net.predict(inputs,oversample = False)
            features[i,:,:,:] = out
            labels[i] = img_label
            i += 1; count += 1
            # if i == 1000:
            print out.shape, count

            if i == 10000:
                i = 0
                f = h5py.File(write_folder + str(file_count) + '.h5','w')
                f.create_dataset('data',data=features)
                f.create_dataset('label',data=labels)
                f.close()
                features = np.zeros((10000,512,7,7))
                labels = np.zeros((10000,))

    features = features[:i,:,:,:]
    labels = labels[:i,:,:,:]
    f = h5py.File(write_folder + str(file_count) + '.h5','w')
    f.create_dataset('data',data=features)
    f.create_dataset('label',data=labels)
    f.close()


    ///////////
    inputs = []; img_label = []
    with open(root + 'CNN_all_event_old/data/guru_ranking_reallabel_training_nomargin.txt','r') as data:
        for line in data:
            img_path, temp = line.split(' ')
            img_label.append(int(temp[:-1]))
            temp = caffe.io.load_image(img_path)
            input_2 = caffe.io.resize_image(temp, (256,256))
            # print input_2.shape
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            x,y = [random.sample(range(29),1)[0],random.sample(range(29),1)[0]]
            input_2 = input_2[x:x+224,y:y+224,:]
            inputs.append(input_2)
            if img_count == batch_size:
                out = net.predict(inputs,oversample = False)
                features[i:i+batch_size,:,:,:] = np.array(out)
                labels[i:i+batch_size] = np.array(img_label)
                inputs = [];img_label = []
                i += batch_size; count += batch_size
                #if i == 1000:
                print out.shape, count

                if i == 10000:
                    i = 0
                    f = h5py.File(write_folder + str(file_count) + '.h5','w')
                    f.create_dataset('data',data=features)
                    f.create_dataset('label',data=labels)
                    f.close()
                    features = np.zeros((10000,512,7,7))
                    labels = np.zeros((10000,))

    out = net.predict(inputs,oversample = False)
    features[i:i+batch_size,:,:,:] = np.array(out)
    labels[i:i+batch_size] = np.array(img_label)
    inputs = [];img_label = []
    i += batch_size; count += batch_size
    features = features[:i,:,:,:]
    labels = labels[:i,:,:,:]
    f = h5py.File(write_folder + str(file_count) + '.h5','w')
    f.create_dataset('data',data=features)
    f.create_dataset('label',data=labels)
    f.close()
    '''

def kmeans(event_name, num_center = 10):
    feature_path = root + 'baseline_all_0509/'+event_name+'/vgg_test_features.cPickle'
    # feature_path = '/Users/wangyufei/Documents/Study/intern_adobe/rebuttal/snapshot/vgg_test_features.cPickle'
    f = open(feature_path, 'r')
    feature = cPickle.load(f)
    f.close()
    # feature = np.array(feature)
    # print feature.shape
    # img_list_path = root + 'baseline_all_0509/'+event_name+'guru_test_path.txt'
    # img_list_path = '/Users/wangyufei/Documents/Study/intern_adobe/rebuttal/snapshot/guru_test_path.txt'
    # img_list = []
    # with open(img_list_path, 'r') as data:
    #     for line in data:
    #         img_list.append(line.split(' ')[0])
    # print len(img_list)

    event_info_path = root + 'baseline_all_0509/'+event_name+'/test_image_ids.cPickle'
    # event_info_path = '/Users/wangyufei/Documents/Study/intern_adobe/rebuttal/snapshot/test_image_ids.cPickle'
    f = open(event_info_path, 'r')
    event_info = cPickle.load(f)
    f.close()

    feature_this = []
    # img_this = []
    prediction = []
    event_previous = ''
    for img_id, temp_feature in zip(event_info, feature):
        event_this = img_id.split('/')[0]
        if event_this != event_previous:
            if event_previous != '':
                start_i = len(prediction)
                prediction.extend([0]*len(feature_this))
                feature_this = np.array(feature_this)
                kmeans = KMeans(init='k-means++', n_clusters=num_center, n_init=10)
                kmeans.fit(feature_this)
                feature_label = kmeans.predict(feature_this)
                label_count = list(Counter(feature_label).items())
                label_count_sorted = sorted(label_count, key=lambda x: x[1], reverse=True)
                print label_count_sorted
                center_this = kmeans.cluster_centers_
                for i in xrange(num_center):
                    center_n = label_count_sorted[i][0]
                    center_feature = center_this[center_n,:]
                    temp = feature_this - center_feature
                    # print temp.shape
                    l2_norm = np.linalg.norm(temp, axis=1)
                    # print l2_norm.shape
                    this_img_id = np.argmin(l2_norm)
                    prediction[start_i + this_img_id] = num_center - i

                feature_this = []
            event_previous = event_this
        feature_this.append(temp_feature)

    start_i = len(prediction)
    prediction.extend([0]*len(feature_this))
    feature_this = np.array(feature_this)
    kmeans = KMeans(init='k-means++', n_clusters=num_center, n_init=10)
    kmeans.fit(feature_this)
    feature_label = kmeans.predict(feature_this)
    label_count = list(Counter(feature_label).items())
    label_count_sorted = sorted(label_count, key=lambda x: x[1], reverse=True)
    center_this = kmeans.cluster_centers_
    for i in xrange(num_center):
        center_n = label_count_sorted[i][0]
        center_feature = center_this[center_n,:]
        temp = feature_this - center_feature
        l2_norm = np.linalg.norm(temp, axis=1)
        this_img_id = np.argmin(l2_norm)
        prediction[start_i + this_img_id] = num_center - i
    print len(prediction)
    print len(event_info)
    f = open('/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name + '/test_kmeans_prediction_'+str(num_center)+'.cPickle','wb')
    cPickle.dump(prediction, f)
    f.close()

def kmeans_percent(event_name):
    feature_path = root + 'baseline_all_0509/'+event_name+'/vgg_test_features.cPickle'
    # feature_path = '/Users/wangyufei/Documents/Study/intern_adobe/rebuttal/snapshot/vgg_test_features.cPickle'
    f = open(feature_path, 'r')
    feature = cPickle.load(f)
    f.close()
    event_info_path = root + 'baseline_all_0509/'+event_name+'/test_image_ids.cPickle'
    f = open(event_info_path, 'r')
    event_info = cPickle.load(f)
    f.close()

    feature_this = []
    # img_this = []
    prediction = []
    event_previous = ''
    for img_id, temp_feature in zip(event_info, feature):
        event_this = img_id.split('/')[0]
        if event_this != event_previous:
            if event_previous != '':
                start_i = len(prediction)
                len_album = max(int(float(len(feature_this)) / 5), 10)
                prediction.extend([0]*len(feature_this))
                feature_this = np.array(feature_this)
                kmeans = KMeans(init='k-means++', n_clusters=len_album, n_init=10)
                kmeans.fit(feature_this)
                feature_label = kmeans.predict(feature_this)
                label_count = list(Counter(feature_label).items())
                label_count_sorted = sorted(label_count, key=lambda x: x[1], reverse=True)
                center_this = kmeans.cluster_centers_
                for i in xrange(len_album):
                    center_n = label_count_sorted[i][0]
                    center_feature = center_this[center_n,:]
                    temp = feature_this - center_feature
                    # print temp.shape
                    l2_norm = np.linalg.norm(temp, axis=1)
                    # print l2_norm.shape
                    this_img_id = np.argmin(l2_norm)
                    prediction[start_i + this_img_id] = len_album - i

                feature_this = []
            event_previous = event_this
        feature_this.append(temp_feature)

    start_i = len(prediction)
    len_album = max(int(float(len(feature_this)) / 5), 10)
    prediction.extend([0]*len(feature_this))
    feature_this = np.array(feature_this)
    kmeans = KMeans(init='k-means++', n_clusters=len_album, n_init=10)
    kmeans.fit(feature_this)
    feature_label = kmeans.predict(feature_this)
    label_count = list(Counter(feature_label).items())
    label_count_sorted = sorted(label_count, key=lambda x: x[1], reverse=True)
    center_this = kmeans.cluster_centers_
    for i in xrange(len_album):
        center_n = label_count_sorted[i][0]
        center_feature = center_this[center_n,:]
        temp = feature_this - center_feature
        l2_norm = np.linalg.norm(temp, axis=1)
        this_img_id = np.argmin(l2_norm)
        prediction[start_i + this_img_id] = len_album - i
    print len(prediction)
    print len(event_info)
    f = open('/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name + '/test_kmeans_prediction_20percent.cPickle','wb')
    cPickle.dump(prediction, f)
    f.close()

def check_distribution():
    bin = [0]*51
    with open("/home/feiyu1990/local/event_curation/CNN_all_event_old/data/guru_ranking_reallabel_training_nomargin.txt") as textfile1, open("/home/feiyu1990/local/event_curation/CNN_all_event_old/data/guru_ranking_reallabel_training_nomargin_p.txt") as textfile2:
        for x, y in izip(textfile1, textfile2):
            x = int(x.split(' ')[1].split()[0])
            y = int(y.split(' ')[1].split()[0])
            bin[int(float(abs(x-y))/80*50)] += 1
    print bin
    sum = [(0, bin[0])]
    for iter in range(1,len(bin)):
        sum.append((sum[iter-1][0]+1, sum[iter-1][1] + bin[iter]))
    print sum
    mean_sum = [(float(i[0])/50, float(i[1])/1980973) for i in sum]
    print mean_sum


dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

def training_test_coincide():
    root_this = '/Volumes/WD_backup/from_guru2/event_curation/baseline_all_0509/'
    training_test_pair = defaultdict(list)
    test_event_abandon = []
    for event_name in dict_name2:
        if event_name in ['Wedding','Birthday','Graduation']:
            continue
        f = open(root_this + event_name + '/test_event_id.cPickle','r')
        test = cPickle.load(f)
        f.close()
        test_event_set = set([i.split('_')[1].split('/')[0] for i in test])
        f = open(root_this + event_name + '/training_event_id.cPickle','r')
        training = cPickle.load(f)
        f.close()
        training_event_set = set([i.split('_')[1].split('/')[0] for i in training])
        for i in test_event_set:
            if i in training_event_set:
                worker_this = i
                test_event = []
                training_event = []
                for ii in test:
                    if ii.split('_')[1].split('/')[0] == worker_this:
                        test_event.append(ii)
                for ii in training:
                    if ii.split('_')[1].split('/')[0] == worker_this:
                        training_event.append(ii)
                training_test_pair[event_name].append([test_event, training_event])
                test_event_abandon.extend(test_event)
    print training_test_pair
    f = open('test_event_abandoned_noILE.pkl','w')
    cPickle.dump(test_event_abandon, f)
    f.close()
if __name__ == '__main__':
    for event_name in dict_name2:
        print event_name
        kmeans(event_name,5)
    #     kmeans(event_name,10)
    #     kmeans_percent(event_name)
    # check_distribution()
    # training_test_coincide()
