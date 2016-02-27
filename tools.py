__author__ = 'wangyufei'

import cPickle
from sklearn.preprocessing import normalize
import numpy as np
import os

def from_txt_to_pickle(in_path, out_path):
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            feature = [float(i) for i in meta]
            features.append(feature)
    print len(feature)
    print len(features)
    f = open(out_path,'w')
    cPickle.dump(features, f)
    f.close()

def create_l2_distance_mtx(feature):
    similarity_matrix = np.zeros((len(feature), len(feature)))
    feature_normalize = normalize(feature, axis=1)
    for i in xrange(len(feature)):
        for j in xrange(i, len(feature)):
            temp = np.dot(feature_normalize[i,:], feature_normalize[j,:])
            similarity_matrix[i][j] = temp
            similarity_matrix[j][i] = temp

def modify_img_list():
    root = '/mnt/ilcompf2d0/project/yuwang/event_curation/face_recognition/face_features_wedding/'
    lists = [o for o in os.listdir(root) if os.path.isdir(root+o)]
    for file in lists:
        file = root + file + '/all-scores-faces-list'
        out_file = file + '-new'
        f = open(out_file, 'w')
        with open(file, 'r') as data:
            for line in data:
                meta = line.split('/')
                f.write('/mnt/ilcompf2d0/project/yuwang/event_curation/face_recognition/face_features_wedding/' + '/'.join(meta[2:]))
        f.close()



if __name__ == '__main__':
    #from_txt_to_pickle('/Users/wangyufei/Documents/Study/intern_adobe/check_correctness/all-scores-faces-list-feat.txt','/Users/wangyufei/Documents/Study/intern_adobe/check_correctness/all-scores-faces-list-feat.cPickle')
    modify_img_list()