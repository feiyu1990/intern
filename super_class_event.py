__author__ = 'wangyufei'
root = '/home/feiyu1990/local/event_curation/'

# caffe_root = '/home/feiyu1990/local/caffe-mine-test/'  # this file is expected to be in {caffe_root}/examples
# import sys
# sys.path.insert(0, caffe_root + 'python')
# import caffe
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, chi2
from scipy.cluster import hierarchy
import cPickle
import os
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

input_feature_name = '_training_sigmoid9_23_segment_iter_70000.cPickle'
def correlation_from_network_output(n_cluster):
    if os.path.exists(root + 'correlation_23_events.cPickle'):
        f = open(root + 'correlation_23_events.cPickle','r')
        correlation_events = cPickle.load(f)
        f.close()
    else:
        correlation_events = np.zeros((23,23))
        # np.fill_diagonal(correlation_events, 1)
        dict_event_all = {}
        for event_name in dict_name2:
            path = root+'baseline_all_noblock/' + event_name+ '/training_image_ids.cPickle'
            f = open(path, 'r')
            all_event_ids = cPickle.load(f)
            f.close()
            try:
                f = open(root + 'CNN_all_event_old/features/'+event_name + input_feature_name, 'r')
                predict_score = cPickle.load(f)
                f.close()
            except:
                break
            for (name_, score) in zip(all_event_ids, predict_score):
                event_id = name_.split('/')[0]
                if event_id in dict_event_all:
                        dict_event_all[event_id].append(score)
                else:
                        dict_event_all[event_id] = [score]


        print len(dict_event_all)
        for album in dict_event_all:
            dict_event_all[album] = np.asarray(dict_event_all[album])
        for event_name_1 in dict_name2:
            for event_name_2 in dict_name2:
                if event_name_1 == event_name_2:
                    continue
                for album in dict_event_all:
                #     print dict_event_all[album].shape
                #     print dict_event_all[album][:,dict_name2[event_name_1]-1]
                    rho, p = spearmanr(dict_event_all[album][:, dict_name2[event_name_1]-1],dict_event_all[album][:,dict_name2[event_name_2]-1])
                    # w = kendalltau(dict_event_all[album][:, dict_name2[event_name_1]-1],dict_event_all[album][:,dict_name2[event_name_2]-1])
                    correlation_events[dict_name2[event_name_1]-1,dict_name2[event_name_2]-1] += rho
        correlation_events /= len(dict_event_all)
        np.fill_diagonal(correlation_events, 1)

        f = open(root + 'correlation_23_events.cPickle','w')
        cPickle.dump(correlation_events, f)
        f.close()
    print correlation_events

    correlation_events = (correlation_events - np.min(correlation_events)) / ( np.max(correlation_events) -  np.min(correlation_events))
    spectral = cluster.SpectralClustering(n_clusters=n_cluster,
                                          eigen_solver='arpack',
                                          affinity="precomputed")

    y = spectral.fit_predict(correlation_events)
    # agglo = cluster.AgglomerativeClustering(n_clusters=6, affinity='euclidean', connectivity=None, n_components=None, compute_full_tree='auto', linkage='ward')
    # y = agglo.fit_predict(correlation_events)
    print y

    dict_transfer = {}
    for i in xrange(len(y)):
        dict_transfer[i] = y[i]
    print dict_transfer
    dict_class = {}
    for i in xrange(n_cluster):
        dict_class[i] = []
    for event_name in dict_name2:
        dict_class[y[dict_name2[event_name]-1]].append(event_name)
    print dict_class


if __name__ == '__main__':
    correlation_from_network_output(int(sys.argv[1]))