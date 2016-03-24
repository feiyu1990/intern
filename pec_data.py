import shutil
import os
from os.path import isdir, isfile
import math
import cPickle
import random
import csv
import numpy as np
import h5py
import copy
from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc

caffe_root = '/home/feiyu1990/local/caffe-mine-test/'  # this file is expected to be in {caffe_root}/examples
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe
import ujson
import re
import operator
import scipy.stats
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, chi2
from collections import Counter, defaultdict

# import Bio.Cluster
# combine_face_model = '_combined_10_fromnoevent.cPickle'
# combine_face_model = '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle'
global_permutation_time = 10
from sklearn.decomposition import PCA

dict_name2 = {'ThemePark': 1, 'UrbanTrip': 2, 'BeachTrip': 3, 'NatureTrip': 4,
              'Zoo': 5, 'Cruise': 6, 'Show': 7,
              'Sports': 8, 'PersonalSports': 9, 'PersonalArtActivity': 10,
              'PersonalMusicActivity': 11, 'ReligiousActivity': 12,
              'GroupActivity': 13, 'CasualFamilyGather': 14,
              'BusinessActivity': 15, 'Architecture': 16, 'Wedding': 17, 'Birthday': 18, 'Graduation': 19, 'Museum': 20,
              'Christmas': 21,
              'Halloween': 22, 'Protest': 23}

dict_reverse = dict([(dict_name2[i], i) for i in dict_name2])
map_to_new = {'birthday': 'Birthday',
              'children_birthday': 'Birthday',
              'christmas': 'Christmas',
              'cruise': 'Cruise',
              # 'exhibition': 'BusinessActivity',
              'exhibition': 'Museum',
              'halloween': 'Halloween',
              'road_trip': 'NatureTrip',
              # 'skiing': 'PersonalSports',
              'concert': 'Show',
              'graduation': 'Graduation',
              'hiking': 'NatureTrip',
              'wedding': 'Wedding'}
new_ind = {4: 0, 6: 1, 7: 2, 17: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 8}  # , 15:10}

#roadtrip&hiking 0
# cruise 1  *****
# show 2
# personalsports 3
# wedding 4
# birthday 5 ******
# graduation 6
# museum 7  ******
# christmas 8
# halloween 9 *****


# #urban trip not defined very well
# 1 roadtrip: urbantrip         roadtrip->naturetrip
# 2 hiking: naturetrip          hiking->cruise
# 3 cruise: cruise
# 4 concert: show
# 5 skiing: personalsports      skiiing->hiking
# 6 wedding
# 7 birthday                    birthday->wedding
# 8 graduation
# 9 exhibition: museum         exhibition->concert
# 10 christmas                 christmas->birthday
# 11 halloween




def create_comparison_path(name='train'):
    path = '/home/feiyu1990/local/event_curation/pec/images/'
    with open(path + '../meta/' + name + '.json') as f:
        test_folders = ujson.load(f)
    album_list = []
    for event in test_folders:
        event_this = test_folders[event]
        for album in event_this:
            album_list.append(path + event + '/' + album)

    img_path = []
    for album in album_list:
        with open(album + '/ordering.json') as f:
            img_list = ujson.load(f)
            for i in img_list:
                img_path.append(album + '/' + str(i[0]) + '.jpg 0\n')
    with open(path + '../' + name + '_path.txt', 'w') as f:
        for line in img_path:
            f.write(line)


def create_test_and_train_result(path, name):
    f = open('/home/feiyu1990/local/event_curation/pec/meta/' + name + '.json')
    test_ = ujson.load(f)

    with open(root_feature + path + '_event_list.pkl') as f:
        event_list = cPickle.load(f)
    feature = np.load(root_feature + path + '.npy')
    print feature.shape, len(event_list)

    test_feature_dict = dict()
    count = 0
    for event_ in event_list:
        event_type = event_.split('/')[0]
        album = event_.split('/')[1]
        order_ = test_[event_type][album]
        len_ = len(order_)
        feature_this = feature[count:count + len_, :]
        count += len_
        test_feature_dict[event_] = feature_this

    print count

    with open(root_feature + path + '_result_dict.pkl', 'w') as f:
        cPickle.dump(test_feature_dict, f)


def prediction_cnn(path):
    with open(root_feature + path + '_result_dict.pkl') as f:
        test_feature_dict = cPickle.load(f)
    confusion_matrix = np.zeros((14, 14))
    for event in test_feature_dict:
        type_old = event.split('/')[0]
        if type_old not in map_to_new:
            continue
        type_new = dict_name2[map_to_new[type_old]]
        # print type_old, type_new
        feature_this = test_feature_dict[event]
        prediction_ = np.argsort(np.mean(feature_this, axis=0))
        # print prediction_
        for i in xrange(len(prediction_)):
            if prediction_[-i - 1] + 1 not in new_ind:
                continue
            confusion_matrix[new_ind[type_new], new_ind[prediction_[-i - 1] + 1]] += 1
            break
            # confusion_matrix[type_new, prediction_] += 1
    print confusion_matrix
    print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix)


def prediction_cnn_event(test_feature_dict, print_=False):
    wrong_event = []
    confusion_matrix = np.zeros((14, 14), dtype=int)
    for event_old in test_feature_dict:
        if '/' in event_old:
            event = '_'.join(event_old.split('/'))
        else:
            event = event_old
        type_old = '_'.join(event.split('_')[:-1])
        if type_old not in map_to_new:
            # print type_old
            continue
        type_new = dict_name2[map_to_new[type_old]]
        # print type_old, type_new
        feature_this = test_feature_dict[event_old]
        prediction_ = np.argsort(feature_this)
        # print prediction_
        for i in xrange(len(prediction_)):
            if prediction_[-i-1] + 1 not in new_ind:
                # break
                continue
            confusion_matrix[new_ind[type_new], new_ind[prediction_[-i-1]+1]] += 1
            if new_ind[type_new] !=  new_ind[prediction_[-i-1]+1]:
                wrong_event.append((event, dict_reverse[prediction_[-i-1]+1]))
            break

        # confusion_matrix[type_new, prediction_] += 1
    if print_:
        for i in range(14):
            print list(confusion_matrix[i,:])
        print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix)
    print wrong_event
    return float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), confusion_matrix


def prediction_cnn_event_only_top1(test_feature_dict, print_=False):
    wrong_event = []
    confusion_matrix = np.zeros((14, 14), dtype=int)
    for event_old in test_feature_dict:
        if '/' in event_old:
            event = '_'.join(event_old.split('/'))
        else:
            event = event_old
        type_old = '_'.join(event.split('_')[:-1])
        if type_old not in map_to_new:
            # print type_old
            continue
        type_new = dict_name2[map_to_new[type_old]]
        # print type_old, type_new
        feature_this = test_feature_dict[event_old]
        prediction_ = np.argsort(feature_this)
        # print prediction_
        for i in xrange(len(prediction_)):
            if prediction_[-i-1] + 1 not in new_ind:
                # print event_old, dict_reverse[prediction_[-i-1] + 1]
                break
                # continue
            confusion_matrix[new_ind[type_new], new_ind[prediction_[-i-1]+1]] += 1
            if new_ind[type_new] !=  new_ind[prediction_[-i-1]+1]:
                wrong_event.append((event, dict_reverse[prediction_[-i-1]+1]))
            break

        # confusion_matrix[type_new, prediction_] += 1
    if print_:
        for i in range(14):
            print list(confusion_matrix[i,:])
        print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix)
    # print wrong_event
    if np.sum(confusion_matrix)==0:
        return 0, None
    return float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), confusion_matrix



def prediction_cnn_event_top2(path):
    wrong_event = []
    with open(root_feature + path + '.pkl') as f:
        test_feature_dict = cPickle.load(f)
    confusion_matrix = np.zeros((14, 14))
    for event in test_feature_dict:
        type_old = '_'.join(event.split('_')[:-1])
        if type_old not in map_to_new:
            # print type_old
            continue
        type_new = dict_name2[map_to_new[type_old]]
        # print type_old, type_new
        feature_this = test_feature_dict[event]
        prediction_ = np.argsort(feature_this)
        # print prediction_
        # for i in xrange(len(prediction_)):
        if prediction_[-1] + 1 not in new_ind:
            wrong_event.append((event, dict_reverse[prediction_[-1] + 1]))
            continue
        confusion_matrix[new_ind[type_new], new_ind[prediction_[-1] + 1]] += 1
        if new_ind[type_new] != new_ind[prediction_[-1] + 1]:
            # if prediction_[-2]+1 in new_ind and new_ind[type_new] != new_ind[prediction_[-2]+1]:
            #     continue
            wrong_event.append((event, dict_reverse[prediction_[-1] + 1]))

            # confusion_matrix[type_new, prediction_] += 1
    print confusion_matrix
    print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix)
    print wrong_event


def prediction_cnn_event_2(path1, path2):
    notmatching = []
    with open(root_feature + path1 + '.pkl') as f:
        test_feature_dict = cPickle.load(f)
    with open(root_feature + path2 + '.pkl') as f:
        test_feature_dict_2 = cPickle.load(f)
    # confusion_matrix = np.zeros((10, 10))
    for event in test_feature_dict:
        type_old = '_'.join(event.split('_')[:-1])
        if type_old not in map_to_new:
            continue
        # type_new = dict_name2[map_to_new[type_old]]
        feature_this = test_feature_dict[event]
        feature_this2 = test_feature_dict_2[event]
        prediction_ = np.argsort(feature_this)
        prediction_2 = np.argsort(feature_this2)
        # print prediction_, prediction_2
        if prediction_[-1] != prediction_2[-1]:
            notmatching.append((event, dict_reverse[prediction_[-1] + 1] + dict_reverse[prediction_[-2] + 1],
                                dict_reverse[prediction_2[-1] + 1] + dict_reverse[prediction_2[-2] + 1]))

    # print confusion_matrix
    # print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix)
    print notmatching


def prediction_cnn_event_cufed(test_feature_dict, top1=False, print_=False):
    wrong_event = []
    confusion_matrix = np.zeros((23, 23), dtype=int)
    cross_entropy = 0
    for event_id in test_feature_dict:
        # print ground_truth_event[event_id]
        feature_this = test_feature_dict[event_id]
        feature_this = feature_this / np.sum(feature_this)
        prediction_ = np.argmax(feature_this)
        ground_ = np.zeros((23,))
        for i in ground_truth_event[event_id]:
            ground_[dict_name2[i[0]] - 1] = i[1]
        cross_entropy += np.sum(ground_ * np.log(feature_this))
        if top1:
            event_type = ground_truth_event[event_id][0][0]
            confusion_matrix[dict_name2[event_type]-1, prediction_] += 1
        else:
            ground_truth_this_list = [dict_name2[i[0]]-1 for i in  ground_truth_event[event_id]]
            if prediction_ not in ground_truth_this_list:
                confusion_matrix[ground_truth_this_list[0], prediction_] += 1
                wrong_event.append((event_id, ground_truth_event[event_id], dict_reverse[prediction_ + 1]))
            else:
                confusion_matrix[prediction_, prediction_] += 1
                # wrong_event.append((event_id, ground_truth_event[event_id], dict_reverse[prediction_ + 1]))
    if print_:
        for i in range(23):
            print list(confusion_matrix[i,:])
        print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix), cross_entropy
    # print wrong_event
    return float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), confusion_matrix, cross_entropy



def prediction_cnn_event_cufed_type1(test_feature_dict):
    confusion_matrix = np.zeros((23, 23), dtype=int)
    cross_entropy = 0
    for event_id in test_feature_dict:
        # print ground_truth_event[event_id]
        feature_this = test_feature_dict[event_id]
        feature_this = feature_this / np.sum(feature_this)
        prediction_ = np.argmax(feature_this)
        ground_ = np.zeros((23,))
        for i in ground_truth_event[event_id]:
            ground_[dict_name2[i[0]] - 1] = i[1]
        cross_entropy += np.sum(ground_ * np.log(feature_this))
        ground_truth_this_list = [dict_name2[i[0]]-1 for i in  ground_truth_event[event_id]]
        for i in ground_truth_this_list:
            confusion_matrix[i, prediction_] += 1
    for i in range(23):
        print list(confusion_matrix[i,:])
    print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix), cross_entropy
    return float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), confusion_matrix, cross_entropy

def prediction_cnn_event_cufed_type2(test_feature_dict):
    confusion_matrix = np.zeros((23, 23), dtype=float)
    cross_entropy = 0
    for event_id in test_feature_dict:
        # print ground_truth_event[event_id]
        feature_this = test_feature_dict[event_id]
        feature_this = feature_this / np.sum(feature_this)
        prediction_ = np.argmax(feature_this)
        ground_ = np.zeros((23,))
        for i in ground_truth_event[event_id]:
            ground_[dict_name2[i[0]] - 1] = i[1]
        cross_entropy += np.sum(ground_ * np.log(feature_this))
        ground_truth_this_list = [dict_name2[i[0]]-1 for i in  ground_truth_event[event_id]]
        for i in ground_truth_this_list:
            confusion_matrix[i, prediction_] += float(1) / len(ground_truth_this_list)
    for i in range(23):
        print list(confusion_matrix[i,:])
    print float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), np.sum(confusion_matrix), cross_entropy
    return float(np.trace(confusion_matrix)) / np.sum(confusion_matrix), confusion_matrix, cross_entropy


def prediction_roc(test_feature_dict):
    confusion_matrix = np.zeros((23, 23), dtype=float)
    cross_entropy = 0
    predict_all = []; ground_all = []
    for event_id in test_feature_dict:
        # print ground_truth_event[event_id]
        feature_this = test_feature_dict[event_id]
        feature_this = feature_this / np.sum(feature_this)
        prediction_ = np.argmax(feature_this)
        ground_ = np.zeros((23,))
        for i in ground_truth_event[event_id]:
            ground_[dict_name2[i[0]] - 1] = 1
        predict_ = np.zeros((23,))
        predict_[prediction_] = 1
        ground_all.append(ground_)
        predict_all.append(predict_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    predict_all = np.array(predict_all); print predict_all.shape
    ground_all = np.array(ground_all); print predict_all.ground_all
    for i in range(23):
        fpr[i], tpr[i], _ = roc_curve(ground_all[:, i], predict_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_all.ravel(), predict_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(23)]))




# def subclass_creation(in_path):



def e_step(importance_feature, img_recognition, all_img_ids, threshold=0, poly=1, poly2=1):
    event_recognition = dict()
    last_event_id = ''

    for i, j, k in zip(importance_feature, img_recognition, all_img_ids):
        event_id = k.split('/')[0]
        if last_event_id != event_id:
            last_event_id = event_id
            event_recognition[event_id] = np.zeros((23,))
        if i < threshold:
            continue
        # event_recognition[event_id] += i **poly * j
        event_recognition[event_id] += (i ** poly2) * (j ** poly)
        # event_recognition[event_id] += sigmoid(i)  * j
    for event_id in event_recognition:
        event_recognition[event_id] /= np.sum(event_recognition[event_id])

    return event_recognition


def m_step(importance_feature, event_recognition, all_img_ids, threshold_1=0.8):
    importance_out_all = []
    importance_out = []
    last_event_id = ''
    for i, j in zip(importance_feature, all_img_ids):
        event_id = j.split('/')[0]
        if event_id != last_event_id:
            last_event_id = event_id
            if len(importance_out) > 0:
                # if np.max(importance_out) < 0:
                #     importance_out = list(np.ones((len(importance_out), )))
                importance_out = [ii - np.sort(importance_out)[0] for ii in importance_out]
                # importance_out = [ii - np.sort(importance_out)[len(importance_out) / 10 ] for ii in importance_out]
                importance_out_all.extend(importance_out / np.max(importance_out))
                importance_out = []
        event_rec_this = event_recognition[event_id]
        if np.max(event_rec_this) - np.min(event_rec_this) != 0:
            event_rec_this = (event_rec_this - np.min(event_rec_this)) / (
            np.max(event_rec_this) - np.min(event_rec_this))
        if threshold_1 == 1:
            low_values_indices = event_rec_this < np.max(event_rec_this)
        else:
            low_values_indices = event_rec_this < threshold_1 * np.max(event_rec_this)
        # low_values_indices = event_rec_this < sorted(event_rec_this)[-3]  # Where values are low
        event_rec_this[low_values_indices] = 0
        # event_rec_this = np.power(event_rec_this, 2)
        # event_rec_this = event_rec_this / np.sum(event_rec_this)
        # print i, event_rec_this, np.sum(event_rec_this * i)
        importance_out.append(np.sum(event_rec_this * i))
    # if np.max(importance_out) < 0:
    #     importance_out = list(np.ones((len(importance_out), )))
    # importance_out = [ii - np.sort(importance_out)[len(importance_out) / 10 ] for ii in importance_out]
    importance_out = [ii - np.sort(importance_out)[0] for ii in importance_out]
    importance_out_all.extend(importance_out / np.max(importance_out))
    # print np.max(importance_out_all), np.min(importance_out_all)
    return importance_out_all


def em_combine_event_recognition_curation_corrected(img_ids, test_list,
                                                    threshold, threshold_m, poly=1, poly2=1, img_importance=None,
                                                    img_recognition=None,
                                                    # importance_path='importance_test',
                                                    # event_path = 'recognition_test',
                                                    stop_criterion=0.01, max_iter=101,

                                                    ):
    importance_scores = []

    importance_ini = np.ones((len(img_recognition),))
    event_recognition = e_step(importance_ini, img_recognition, img_ids, threshold, poly, poly2)
    # print event_recognition
    # event_lengths[event_type] = len(event_recognition)
    importance_score = m_step(img_importance, event_recognition, img_ids, threshold_m)
    iter = 0
    diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
    while diff > stop_criterion:
        iter += 1
        if iter >= max_iter:
            if max_iter > 1:
                event_recognition = e_step([(i + j + k + q) / 4 for i, j, k, q in
                                            zip(importance_scores[-1], importance_scores[-2], importance_scores[-3],
                                                importance_scores[-4])], img_recognition, img_ids, threshold, poly,
                                           poly2)
            break
        event_recognition = e_step(importance_score, img_recognition, img_ids, threshold, poly, poly2)
        importance_score_new = m_step(img_importance, event_recognition, img_ids, threshold_m)
        importance_scores.append(importance_score_new)
        diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
        importance_score = importance_score_new
    # if max_iter > 1:
    #     importance_score = m_step(importance_score, event_recognition, img_ids, 0)
    #
    img_importance_all_event = defaultdict(list)
    for img_id, importance in zip(img_ids, importance_score):
        event_id = img_id.split('/')[0]
        img_importance_all_event[event_id].append(importance)

    return event_recognition, img_importance_all_event


def em_combine_event_recognition_curation_corrected_new(img_ids, test_list,
                                                    threshold, threshold_m, poly, poly2, lstm_event_recognition,
                                                    img_importance=None,
                                                    img_recognition=None,
                                                    # importance_path='importance_test',
                                                    # event_path = 'recognition_test',
                                                    stop_criterion=0.01, max_iter=101, average=True,
                                                    combine_lstm=True,

                                                    ):
    importance_scores = []
    event_recognitions = []
    img_importance_all_event = defaultdict(list)
    importance_ini = np.ones((len(img_recognition),))
    event_recognition = e_step(importance_ini, img_recognition, img_ids, threshold, poly, poly2)
    if max_iter == 0:
        event_recognition1 = dict()
        for event_type in event_recognition:
            event_recognition1[event_type] = np.ones((23,), dtype=float) / 23
        importance_score = np.array(m_step(img_importance, event_recognition1, img_ids, threshold_m))
        for img_id, importance in zip(img_ids, importance_score):
            event_id = img_id.split('/')[0]
            img_importance_all_event[event_id].append(importance)
        return event_recognition, img_importance_all_event
    # if max_iter == 1:
    #     return event_recognition, None
    if combine_lstm:
        event_recognition = combine_lstm_cnn_result(event_recognition, lstm_event_recognition, poly1=1)
    event_recognitions.append(event_recognition)
    # print event_recognition
    # event_lengths[event_type] = len(event_recognition)
    importance_score = np.array(m_step(img_importance, event_recognition, img_ids, threshold_m))
    importance_scores.append(importance_score)
    iter = 0
    diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
    while diff > stop_criterion:
        iter += 1
        if iter >= max_iter:
            if max_iter > 1 and average:
                temp = np.stack(importance_scores[-4:])
                importance_score = np.mean(temp, axis=0)
                event_recognition = combine_result_multiple_sources(event_recognitions[-4:])

            break
        event_recognition = e_step(importance_score, img_recognition, img_ids, threshold, poly, poly2)
        if combine_lstm:
            event_recognition = combine_lstm_cnn_result(event_recognition, lstm_event_recognition, poly1=1)
        event_recognitions.append(event_recognition)
        importance_score_new = np.array(m_step(img_importance, event_recognition, img_ids, threshold_m))
        importance_scores.append(importance_score_new)
        diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
        importance_score = importance_score_new
    for img_id, importance in zip(img_ids, importance_score):
        event_id = img_id.split('/')[0]
        img_importance_all_event[event_id].append(importance)

    return event_recognition, img_importance_all_event



def em_combine_only_e(img_ids, img_importance=None,img_recognition=None

                                                    ):
    importance_scores = []

    importance_ini = np.ones((len(img_recognition),))
    event_recognition = e_step(importance_ini, img_recognition, img_ids, 0, 1, 1)
    for event_ in event_recognition:
        event_recognition[event_] = np.ones((23, ), dtype=float) / 1
    # print event_recognition
    # event_lengths[event_type] = len(event_recognition)
    importance_score = m_step(img_importance, event_recognition, img_ids, 0)
    img_importance_all_event = defaultdict(list)
    for img_id, importance in zip(img_ids, importance_score):
        event_id = img_id.split('/')[0]
        img_importance_all_event[event_id].append(importance)


    return  img_importance_all_event

def combine_result_multiple_sources(cnn_result_dict_list):
    # print poly1, poly2
    cnn_result_dict_new = dict()
    for dict_i in cnn_result_dict_list:
        for event_id in dict_i:
            if event_id in cnn_result_dict_new:
                temp = np.array(dict_i[event_id])
                temp = temp/np.sum(temp)
                cnn_result_dict_new[event_id] += temp
            else:
                temp = np.array(dict_i[event_id])
                temp = temp/np.sum(temp)
                cnn_result_dict_new[event_id] = temp
    for event in cnn_result_dict_new:
        cnn_result_dict_new[event] /= len(cnn_result_dict_list)
    # print cnn_result_dict_new
    return cnn_result_dict_new


def combine_lstm_cnn_result(cnn_result_dict, lstm_result_dict, poly1, poly2=None):
    # print poly1, poly2
    cnn_result_dict_new = dict()
    for event_id in lstm_result_dict:
        temp = np.array(cnn_result_dict[event_id])
        temp = temp/np.sum(temp)
        temp1 = temp ** poly1
        temp = np.array(lstm_result_dict[event_id])
        temp = temp/np.sum(temp)
        if poly2:
            temp2 = temp ** poly2
        else:
            temp2 = temp ** poly1
        if poly1 == 0:
            cnn_result_dict_new[event_id] = temp2
        elif poly2 != None and poly2 == 0:
            # print 'hi!'
            cnn_result_dict_new[event_id] = temp1
        else:
            cnn_result_dict_new[event_id] = temp1  + temp2
    return cnn_result_dict_new


def cross_validation_process_pec(fold=5):
    # poly = 10; poly2 = 10
    # threshold_m = 10
    event_path = 'pec_all_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    importance_path = 'pec_all_sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000'
    # lstm_path = 'pec_vote_multilabel_oversample_20_0.5_crossvalidation_prediction_dict'
    # combined_path = 'RECOGNITION_'+importance_path + '_' + event_path + '_em_9'


    with open(root_feature + '../meta/train_test.json') as f:
        event_dict = ujson.load(f)
    with open(root_feature + event_path + '_event_list.pkl') as f:
        test_list = cPickle.load(f)
    img_importance = np.load(root_feature + importance_path + '.npy')
    print img_importance.shape
    img_recognition = np.load(root_feature + event_path + '.npy')
    print img_recognition.shape
    # cross_fold_list = range(len(test_list))
    # cross_fold_list = [i%5 for i in cross_fold_list]
    # random.shuffle(cross_fold_list)
    # # print cross_fold_list
    # np.save(root_feature + 'cross_validation_list.npy', cross_fold_list)
    cross_fold_list = np.load(root_feature + 'cross_validation_list.npy')

    test_confusion_all = np.zeros((14, 14))
    test_result_dict_all = dict()
    for fold_i in range(fold):
        print '+++++++++++FOLD', fold_i, '++++++++++++++'
        count = 0
        test_list_fold = []
        train_list_fold = []
        img_test_indicator = []
        img_train_indicator = []
        for ind, j in enumerate(cross_fold_list):
            event_this = test_list[ind]
            len_event_this = len(event_dict[event_this.split('/')[0]][event_this.split('/')[1]])
            if j == fold_i:
                test_list_fold.append(test_list[ind])
                img_test_indicator.extend(range(count, len_event_this + count))
            else:
                train_list_fold.append(test_list[ind])
                img_train_indicator.extend(range(count, len_event_this + count))
            count += len_event_this
        print len(test_list_fold), len(train_list_fold)
        print len(img_test_indicator), len(img_train_indicator), count

        test_img_ids_fold = []
        train_img_ids_fold = []
        for event in test_list_fold:
            for img in event_dict[event.split('/')[0]][event.split('/')[1]]:
                test_img_ids_fold.append(event.split('/')[0] + '_' + event.split('/')[1] + '/' + str(img[0]))
        for event in train_list_fold:
            for img in event_dict[event.split('/')[0]][event.split('/')[1]]:
                train_img_ids_fold.append(event.split('/')[0] + '_' + event.split('/')[1] + '/' + str(img[0]))

        test_img_importance_fold = img_importance[img_test_indicator, :]
        train_img_importance_fold = img_importance[img_train_indicator, :]
        test_img_recognition_fold = img_recognition[img_test_indicator, :]
        train_img_recognition_fold = img_recognition[img_train_indicator, :]
        print train_img_recognition_fold.shape, test_img_recognition_fold.shape
        threshold_m = 10
        poly1 = 10
        poly2 = 10
        scores = np.zeros((6, 5))
        threshold_list = range(0, 11, 2)
        poly2_list = range(2, 19, 4)
        # threshold_list = [0]; poly2_list = [0]
        for ind1, threshold_m in enumerate(threshold_list):
            for ind2, poly2 in enumerate(poly2_list):
                print 'T:', threshold_m, 'P:', poly2
                recog_result, _ = em_combine_event_recognition_curation_corrected(train_img_ids_fold, train_list_fold,
                                                                               0, float(threshold_m) / 10,
                                                                               float(poly1) / 10, float(poly2) / 10,
                                                                               img_importance=train_img_importance_fold,
                                                                               img_recognition=train_img_recognition_fold,
                                                                               max_iter=9)
                score_this, confusion_this = prediction_cnn_event(recog_result)
                scores[ind1, ind2] = score_this
        print scores
        temp = np.where(scores == np.max(scores))
        print temp
        temp = [(temp[0][i], temp[1][i]) for i in xrange(len(temp[0]))]
        temp = sorted(temp, key=lambda x: x[0], reverse=True)
        threshold_best_ind, poly2_best_ind = temp[0]
        threshold_best = threshold_list[threshold_best_ind]
        poly2_best = poly2_list[poly2_best_ind]
        #
        # threshold_best = threshold_m; poly2_best = poly2
        # # for ii in range(len(scores)- 1, -1, -1):
        # for ii in range(len(scores)):
        #     if scores[ii] == temp:
        #         # threshold_best = threshold_list[ii]
        #         poly2_best = poly2_list[ii]
        recog_result_test, _ = em_combine_event_recognition_curation_corrected(test_img_ids_fold, test_list_fold,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10,
                                                                               float(poly2_best) / 10,
                                                                               img_importance=test_img_importance_fold,
                                                                               img_recognition=test_img_recognition_fold,
                                                                               max_iter=1)
        score_this_test_iter1, confusion_test_iter1 = prediction_cnn_event(recog_result_test)
        recog_result_test, _ = em_combine_event_recognition_curation_corrected(test_img_ids_fold, test_list_fold,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10,
                                                                               float(poly2_best) / 10,
                                                                               img_importance=test_img_importance_fold,
                                                                               img_recognition=test_img_recognition_fold,
                                                                               max_iter=9)
        score_this_test, confusion_test = prediction_cnn_event(recog_result_test)
        test_result_dict_all.update(recog_result_test)
        print 'FOLD:', fold_i, 'BEST THRESHOLD:', threshold_best, 'BEST POLY2:', poly2_best, 'BEST ACCU:', score_this_test, '/', score_this_test_iter1, 'length:', len(
            recog_result_test)
        test_confusion_all += confusion_test
    print '------OVERALL-------'
    print test_confusion_all
    print float(np.trace(test_confusion_all)) / np.sum(test_confusion_all), np.sum(test_confusion_all)
    f = open(root_feature + 'THRESHOLM_POLY2_cross_validation_combine_best.pkl', 'w')
    cPickle.dump(test_result_dict_all, f)
    f.close()

def cross_validation_lstm_pec(fold=5):
    # combined_path = 'EM1_cross_validation_combine_best.pkl'

    combined_path = 'THRESHOLDM_cross_validation_combine_best.pkl'
    event_path = 'pec_all_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    # lstm_path = 'pec_vote_multilabel_oversample_20_0.5_crossvalidation_prediction_dict.pkl'
    # lstm_path = 'pec_vote_multilabel_crossvalidation_prediction_dict.pkl'
    lstm_path = 'pec_vote_softall_multilabel_crossvalidation_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET

    with open(root_feature + event_path + '_event_list.pkl') as f:
        test_list = cPickle.load(f)

    cross_fold_list = np.load(root_feature + 'cross_validation_list.npy')
    with open(root_feature + lstm_path) as f:
        lstm_result_dict_all = cPickle.load(f)
    for key in lstm_result_dict_all:
        lstm_result_dict_all['_'.join(key.split('/'))] = lstm_result_dict_all.pop(key)
    with open(root_feature + combined_path) as f:
        combine_result_dict_all = cPickle.load(f)
    for key in combine_result_dict_all:
        combine_result_dict_all['_'.join(key.split('/'))] = combine_result_dict_all.pop(key)
    # print combine_result_dict_all.keys()
    test_confusion_all = np.zeros((14, 14))
    test_result_dict_all = dict()

    for fold_i in range(fold):
        print '+++++++++++FOLD', fold_i, '++++++++++++++'
        combine_dict_test = defaultdict(dict); lstm_dict_test = defaultdict(dict)
        combine_dict_training = defaultdict(dict); lstm_dict_training = defaultdict(dict)
        for ind, j in enumerate(cross_fold_list):
            event_this = '_'.join(test_list[ind].split('/'))
            if j == fold_i:
                combine_dict_test[event_this] = combine_result_dict_all[event_this]
                lstm_dict_test[event_this] = lstm_result_dict_all[event_this]
            else:
                combine_dict_training[event_this] = combine_result_dict_all[event_this]
                lstm_dict_training[event_this] = lstm_result_dict_all[event_this]

        poly_list = range(0, 42, 2)
        poly_list = [float(i) / 20 for i in poly_list]
        scores = []
        for poly in poly_list:
            combine_lstm_result = combine_lstm_cnn_result(combine_dict_training, lstm_dict_training, poly)
            score_this_training, confusion_training = prediction_cnn_event(combine_lstm_result)
            scores.append(score_this_training)
        print scores
        poly_best = poly_list[np.argmax(scores)]
        combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, poly_best)
        score_this_test, confusion_test = prediction_cnn_event(combine_lstm_result_test)
        test_result_dict_all.update(combine_lstm_result_test)
        test_confusion_all += confusion_test

        print 'FOLD:', fold_i, 'BEST POLY:', poly_best, 'BEST ACCU:', score_this_test, 'length:', len(combine_lstm_result_test)

    print '------OVERALL-------'
    print test_confusion_all
    print float(np.trace(test_confusion_all)) / np.sum(test_confusion_all), np.sum(test_confusion_all)
    # f = open(root_feature + 'LSTM_SOFT_COMBINE_POLY_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(test_result_dict_all, f)
    # f.close()


def cross_validation_process_cufed(fold=5):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_new/'
    root_feature = root + folder_name
    # event_path = 'test_predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000'
    # importance_path = 'test_sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000'

    event_path = 'test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    importance_path = 'test_sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000'

    event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_img_dict[img.split('/')[0]].append(img)
        with open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_img_dict[img.split('/')[0]].append(img)

    with open(root + folder_name + event_path + '_event_list.pkl') as f:
        event_recognition_all_list = cPickle.load(f)

    print 'length of events', len(event_recognition_all_list)
    count = 0
    event_recognition_all_event_dict = defaultdict(list)
    for event in event_recognition_all_list:
        for img in event_img_dict[event]:
            event_recognition_all_event_dict[event].append(count)
            count += 1
    img_recognition_old = np.load(root + folder_name + event_path + '.npy')

    with open(root + folder_name + importance_path + '_event_list.pkl') as f:
        importance_feature_all_list = cPickle.load(f)
    count = 0
    importance_feature_all_event_dict = defaultdict(list)
    for event in importance_feature_all_list:
        for img in event_img_dict[event]:
            importance_feature_all_event_dict[event].append(count)
            count += 1
    img_importance_old = np.load(root + folder_name + importance_path + '.npy')
    # print img_importance_old.shape, img_recognition_old.shape

    img_importance = np.zeros((0, 23))
    img_recognition = np.zeros((0, 23))
    test_list = importance_feature_all_list
    for event in test_list:
        img_importance = np.concatenate(
            (img_importance, img_importance_old[importance_feature_all_event_dict[event], :]), axis=0)
        img_recognition = np.concatenate(
            (img_recognition, img_recognition_old[event_recognition_all_event_dict[event], :]), axis=0)
    print img_importance.shape, img_recognition.shape

    # cross_fold_list = range(len(test_list))
    # cross_fold_list = [i%5 for i in cross_fold_list]
    # random.shuffle(cross_fold_list)
    # np.save(root_feature + 'cross_validation_list.npy', cross_fold_list)
    cross_fold_list = np.load(root_feature + 'cross_validation_list.npy')

    ###need: img_importance, img_recognition, test_list, event_img_dict(dict event->images)###
    test_confusion_all = np.zeros((23, 23))
    test_result_dict_all = dict()
    test_importance_result_dict_all = dict()
    for fold_i in range(fold):
        print '+++++++++++FOLD', fold_i, '++++++++++++++'
        count = 0
        test_list_fold = []
        train_list_fold = []
        img_test_indicator = []
        img_train_indicator = []
        for ind, j in enumerate(cross_fold_list):
            event_this = test_list[ind]
            len_event_this = len(event_img_dict[event_this])
            if j == fold_i:
                test_list_fold.append(test_list[ind])
                img_test_indicator.extend(range(count, len_event_this + count))
            else:
                train_list_fold.append(test_list[ind])
                img_train_indicator.extend(range(count, len_event_this + count))
            count += len_event_this
        # print len(test_list_fold), len(train_list_fold)
        # print len(img_test_indicator), len(img_train_indicator), count

        test_img_ids_fold = []
        train_img_ids_fold = []
        for event in test_list_fold:
            for img in event_img_dict[event]:
                test_img_ids_fold.append(img)
        for event in train_list_fold:
            for img in event_img_dict[event]:
                train_img_ids_fold.append(img)

        test_img_importance_fold = img_importance[img_test_indicator, :]
        train_img_importance_fold = img_importance[img_train_indicator, :]
        test_img_recognition_fold = img_recognition[img_test_indicator, :]
        train_img_recognition_fold = img_recognition[img_train_indicator, :]
        # print train_img_recognition_fold.shape, test_img_recognition_fold.shape
        threshold_m = 10
        poly1 = 10
        poly2 = 10
        scores = np.zeros((6, 5))
        # threshold_list = range(0, 11, 2)
        # poly2_list = range(2, 11, 2)
        threshold_list = [0]
        poly2_list = [10]
        for ind1, threshold_m in enumerate(threshold_list):
            for ind2, poly2 in enumerate(poly2_list):
                # print 'T:', threshold_m, 'P:', poly2
                recog_result, importance_result = em_combine_event_recognition_curation_corrected(train_img_ids_fold, train_list_fold,
                                                                               0, float(threshold_m) / 10,
                                                                               float(poly1) / 10, float(poly2) / 10,
                                                                               img_importance=train_img_importance_fold,
                                                                               img_recognition=train_img_recognition_fold,
                                                                               max_iter=9)
                score_this, confusion_this = prediction_cnn_event_cufed(recog_result, True)
                scores[ind1, ind2] = score_this
        print scores
        temp = np.where(scores == np.max(scores))
        temp = [(temp[0][i], temp[1][i]) for i in xrange(len(temp[0]))]
        temp = sorted(temp, key=lambda x: x[0])

        threshold_best_ind, poly2_best_ind = temp[0]
        threshold_best = threshold_list[threshold_best_ind]
        poly2_best = poly2_list[poly2_best_ind]

        recog_result_test, importance_result = em_combine_event_recognition_curation_corrected(test_img_ids_fold, test_list_fold,
                                                                            0, float(threshold_best) / 10,
                                                                            float(poly1) / 10, float(poly2_best) / 10,
                                                                            img_importance=test_img_importance_fold,
                                                                            img_recognition=test_img_recognition_fold,
                                                                            max_iter=1)
        score_this_test_iter1, confusion_test_iter1 = prediction_cnn_event_cufed(recog_result_test, True)
        recog_result_test, importance_result = em_combine_event_recognition_curation_corrected(test_img_ids_fold, test_list_fold,
                                                                            0, float(threshold_best) / 10,
                                                                            float(poly1) / 10, float(poly2_best) / 10,
                                                                            img_importance=test_img_importance_fold,
                                                                            img_recognition=test_img_recognition_fold,
                                                                            max_iter=9)
        score_this_test, confusion_test = prediction_cnn_event_cufed(recog_result_test, True)
        test_result_dict_all.update(recog_result_test)
        test_importance_result_dict_all.update(importance_result)
        print 'FOLD:', fold_i, 'BEST THRESHOLD:', threshold_best, 'BEST POLY2:', poly2_best, 'BEST ACCU:', score_this_test, '/', score_this_test_iter1, 'length:', len(
            recog_result_test)
        test_confusion_all += confusion_test
    print '------OVERALL-------'
    print test_confusion_all
    print float(np.trace(test_confusion_all)) / np.sum(test_confusion_all), np.sum(test_confusion_all)
    f = open(root_feature + 'THRESHOLDM_POLY2_SOFT_recognition_cross_validation_combine_best_v1.pkl', 'w')
    cPickle.dump(test_result_dict_all, f)
    f.close()
    f = open(root_feature + 'THRESHOLDM_POLY2_SOFT_importance_cross_validation_combine_best_v1.pkl', 'w')
    cPickle.dump(test_importance_result_dict_all, f)
    f.close()
def cross_validation_lstm_cufed(fold=5):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_new/'
    root_feature = root + folder_name
    # event_path = 'test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    # combined_path = 'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best_v2.pkl'
    # lstm_path = 'vote_softall_multilabel_crossvalidation_prediction_dict.pkl' #73.9->76.8

    lstm_path = 'vote_multilabel_prediction_dict.pkl' #THIS IS GOOD 75.3 -> 77.6
    event_path = 'test_predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000'
    combined_path = 'THRESHOLDM_POLY2_recognition_cross_validation_combine_best_v2.pkl'

    lstm_path = 'vote_softall_multilabel_crossvalidation_prediction_dict.pkl'


    with open(root_feature + event_path + '_event_list.pkl') as f:
        test_list = cPickle.load(f)

    cross_fold_list = np.load(root_feature + 'cross_validation_list.npy')
    with open(root_feature + lstm_path) as f:
        lstm_result_dict_all = cPickle.load(f)
    with open(root_feature + combined_path) as f:
        combine_result_dict_all = cPickle.load(f)
    test_confusion_all = np.zeros((23, 23))
    test_result_dict_all = dict()

    for fold_i in range(fold):
        print '+++++++++++FOLD', fold_i, '++++++++++++++'
        combine_dict_test = defaultdict(dict); lstm_dict_test = defaultdict(dict)
        combine_dict_training = defaultdict(dict); lstm_dict_training = defaultdict(dict)
        for ind, j in enumerate(cross_fold_list):
            event_this = '_'.join(test_list[ind].split('/'))
            if j == fold_i:
                combine_dict_test[event_this] = combine_result_dict_all[event_this]
                lstm_dict_test[event_this] = lstm_result_dict_all[event_this]
            else:
                combine_dict_training[event_this] = combine_result_dict_all[event_this]
                lstm_dict_training[event_this] = lstm_result_dict_all[event_this]

        poly_list = range(0, 67, 2)
        poly_list = [float(i) / 20 for i in poly_list]
        # poly_list=[1]
        scores = []
        for poly in poly_list:
            combine_lstm_result = combine_lstm_cnn_result(combine_dict_training, lstm_dict_training, poly)
            score_this_training, confusion_training = prediction_cnn_event_cufed(combine_lstm_result)
            scores.append(score_this_training)
        print scores
        poly_best = poly_list[np.argmax(scores)]
        combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, poly_best)
        score_this_test, confusion_test = prediction_cnn_event_cufed(combine_lstm_result_test)
        test_result_dict_all.update(combine_lstm_result_test)
        test_confusion_all += confusion_test

        print 'FOLD:', fold_i, 'BEST POLY:', poly_best, 'BEST ACCU:', score_this_test, 'length:', len(combine_lstm_result_test)

    print '------OVERALL-------'
    print test_confusion_all
    print float(np.trace(test_confusion_all)) / np.sum(test_confusion_all), np.sum(test_confusion_all)
    # f = open(root_feature + 'LSTM_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(test_result_dict_all, f)
    # f.close()


def create_validation_list(fold=4):
    f = open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_new/'
             'test_predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl')
    event_list = cPickle.load(f)
    f.close()
    with open('/home/feiyu1990/local/event_curation/0208_correction/all_input_and_result/new_multiple_result_2round_removedup_vote.pkl') as f:
        event_dict = cPickle.load(f)

    test_event_dict = defaultdict(list)
    random.shuffle(event_list)
    for event in event_list:
        test_event_dict[event_dict[event][0][0]].append(event)

    test_list = []; validation_list = []
    for event_type in test_event_dict:
        this_ = test_event_dict[event_type]
        validation_list.extend(this_[:len(this_)/4])
        test_list.extend(this_[len(this_)/4:])


    with open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/test_list.pkl', 'w') as f:
        cPickle.dump(test_list, f)
    with open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/validation_list.pkl', 'w') as f:
        cPickle.dump(validation_list, f)
    print validation_list
    print test_list

def create_img_list():
    event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_img_dict[img.split('/')[0]].append(img)
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_img_dict[img.split('/')[0]].append(img)
    with open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/test_list.pkl') as f:
        test_list = cPickle.load(f)
    with open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/validation_list.pkl') as f:
        validation_list = cPickle.load(f)
    validation_img_list = []
    for event in validation_list:
        for img in event_img_dict[event]:
            validation_img_list.append(img)

    test_img_list = []
    for event in test_list:
        for img in event_img_dict[event]:
            test_img_list.append(img)

    with open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/test_img_list.pkl', 'w') as f:
        cPickle.dump(test_img_list, f)
    print len(test_img_list), len(validation_img_list)
    with open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/validation_img_list.pkl', 'w') as f:
        cPickle.dump(validation_img_list, f)


def split_valid_test():
    input_path = '/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features/'
    out_path = '/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/'
    file_name = 'test_sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000'
    out_file_name = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000'
    feature = np.load(input_path + file_name + '.npy')
    order_list = cPickle.load(open(input_path + file_name+'_event_list.pkl'))
    event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_img_dict[img.split('/')[0]].append(img)
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_img_dict[img.split('/')[0]].append(img)
    count_dict = defaultdict(list)
    count = 0
    for event in order_list:
        for img in event_img_dict[event]:
            count_dict[event].append(count)
            count += 1
    print count
    validation_feature = np.zeros((0, feature.shape[1]))
    test_feature = np.zeros((0, feature.shape[1]))

    test_event_list = cPickle.load(open(out_path+'test_list.pkl'))
    valid_event_list = cPickle.load(open(out_path+'validation_list.pkl'))

    for event in test_event_list:
        indx_this = count_dict[event]
        test_feature = np.concatenate((test_feature, feature[indx_this]), axis=0)

    for event in valid_event_list:
        indx_this = count_dict[event]
        validation_feature = np.concatenate((validation_feature, feature[indx_this]), axis=0)
    print test_feature.shape, validation_feature.shape

    np.save(out_path+'test_'+out_file_name+'.npy', test_feature)
    np.save(out_path+'validation_'+out_file_name+'.npy', validation_feature)

def validation_process_cufed(max_iter, threshold_best,poly2_best, soft=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_validation/'
    root_feature = root + folder_name

    '''soft'''
    if soft:
        event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'soft_vote_multilabel_test_all_prediction_dict.pkl'
        training_lstm_path = 'soft_vote_multilabel_validation_prediction_dict.pkl'
    else:
        '''non-soft'''
        event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'vote_multilabel_test_all_v2_prediction_dict.pkl'
        training_lstm_path = 'vote_multilabel_validation_v2_prediction_dict.pkl'

    img_id_list = cPickle.load(open(root+folder_name+'validation_img_list.pkl'))
    img_recognition = np.load(root + folder_name + 'validation_' + event_path)
    img_importance = np.load(root + folder_name + 'validation_' + importance_path)
    event_lstm = cPickle.load(open(root + folder_name + training_lstm_path))

    #
    # img_id_list = cPickle.load(open(root+folder_name+'test_img_list.pkl'))
    # img_recognition = np.load(root + folder_name + 'test_' + event_path)
    # img_importance = np.load(root + folder_name + 'test_' + importance_path)
    # event_lstm = cPickle.load(open(root +folder_name + test_lstm_path))

    ###need: img_importance, img_recognition, test_list, event_img_dict(dict event->images)###
    poly1 = 10
    scores = np.zeros((12, 30))
    cross_entropies = np.zeros((12, 30))
    f_scores = np.zeros((12, 30))
    threshold_list = range(10, 11)
    poly2_list = range(1, 31)
    for ind1, threshold_m in enumerate(threshold_list):
        for ind2, poly2 in enumerate(poly2_list):
                recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_id_list, None,
                                                                               0, float(threshold_m) / 10,
                                                                               float(poly1) / 10, float(poly2) / 10,
                                                                               lstm_event_recognition=event_lstm,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=9)
                score_this, confusion_this, cross_entropy_this = prediction_cnn_event_cufed(recog_result)
                print 'THRESHOLD:', threshold_m, 'POLY:', poly2, 'SCORE:',score_this, 'CE:', cross_entropy_this
                score_this, f_core_this , macro_f_score= recall_topk_cufed(recog_result, 2)
                print 'THRESHOLD:', threshold_m, 'POLY:', poly2, 'SCORE:',score_this, 'f-score:', f_core_this, 'macro-f:',macro_f_score
                scores[ind1, ind2] = score_this
                f_scores[ind1, ind2] = f_core_this
                cross_entropies[ind1, ind2] = cross_entropy_this

    print scores
    print f_scores
    print cross_entropies
    # temp = np.where(scores > 0.74)
    temp = np.where(scores == np.max(scores))
    temp = [(threshold_list[temp[0][i]], poly2_list[temp[1][i]]) for i in xrange(len(temp[0]))]
    print temp
    temp = sorted(temp, key=lambda x: x[0])
    temp_both = []
    for i,j in temp:
        temp_both.extend((i,j,cross_entropies[i][j]))
    print temp_both
    temp = sorted(temp_both, key=lambda x: x[1], reverse=True)

    # poly1 = 10
    # img_id_list = cPickle.load(open(root+folder_name+'test_img_list.pkl'))
    # img_recognition = np.load(root + folder_name + 'test_' + event_path)
    # img_importance = np.load(root + folder_name + 'test_' + importance_path)
    # event_lstm = cPickle.load(open(root +folder_name + test_lstm_path))
    # # threshold_best = 4; poly2_best = 14 ### this is for soft
    # # threshold_best = 0; poly2_best = 14 ### this is for soft from recall at top@3
    # # threshold_best = 6; poly2_best = 14 ### this is for non-soft new
    # # threshold_best = 9; poly2_best = 4 ### this is for non-soft new
    # # recog_result, importance_result = em_combine_event_recognition_curation_corrected(img_id_list, None,
    # #                                                                            0, float(threshold_best) / 10,
    # #                                                                            float(poly1) / 10, float(poly2_best) / 10,
    # #                                                                            img_importance=img_importance,
    # #                                                                            img_recognition=img_recognition,
    # #                                                                            max_iter=9)
    # recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_id_list, None,
    #                                                                            0, float(threshold_best) / 10,
    #                                                                            float(poly1) / 10, float(poly2_best) / 10,
    #                                                                            lstm_event_recognition=event_lstm,
    #                                                                            img_importance=img_importance,
    #                                                                            img_recognition=img_recognition,
    #                                                                            max_iter=max_iter)
    # score_this, confusion_this, cross_entropy_this = prediction_cnn_event_cufed(recog_result, print_=True)
    # _, fscore, f_macro = recall_topk_cufed(recog_result,top_n=2)
    # print fscore, f_macro
    # # score_this, f_core_this  = recall_topk_cufed(recog_result, 3)
    # #
    # # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_iter'+str(max_iter) + '_recognition_cross_validation_combine_best.pkl', 'w')
    # # cPickle.dump(recog_result, f)
    # # f.close()
    # # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_iter'+str(max_iter) + '_importance_cross_validation_combine_best.pkl', 'w')
    # # cPickle.dump(importance_result, f)
    # # f.close()
    #
    # # importance_result = em_combine_only_e(img_id_list,img_importance=img_importance, img_recognition=img_recognition)
    # # f = open(root_feature + 'SOFT_EM1_importance_cross_validation_combine_best.pkl', 'w')
    # # cPickle.dump(importance_result, f)
    # # f.close()

def validation_process_cufed_test(bests, soft=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_validation/'
    #
    if soft:
        event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'soft_vote_multilabel_test_all_prediction_dict.pkl'
    else:
        '''non-soft'''
        event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'vote_multilabel_test_all_v2_prediction_dict.pkl'


    img_id_list = cPickle.load(open(root+folder_name+'test_img_list.pkl'))
    img_recognition = np.load(root + folder_name + 'test_' + event_path)
    img_importance = np.load(root + folder_name + 'test_' + importance_path)
    event_lstm = cPickle.load(open(root +folder_name + test_lstm_path))

    ###need: img_importance, img_recognition, test_list, event_img_dict(dict event->images)###

    poly1 = 10
    results = np.zeros((len(bests), 9))
    i = 0
    for threshold_best, poly2_best in bests:
        for max_iter in range(1, 10):
            recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_id_list, None,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10, float(poly2_best) / 10,
                                                                               lstm_event_recognition=event_lstm,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=max_iter,
                                                                               average=False)
            score_this, confusion_this, cross_entropy_this = prediction_cnn_event_cufed(recog_result)
            results[i, max_iter - 1] = score_this
        print '(', threshold_best, ',', poly2_best, '):', results[i, :]
        i += 1
    print results
    return results

def validation_process_cufed_validation(bests, soft=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_validation/'
    root_feature = root + folder_name
    if soft:
        event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'soft_vote_multilabel_test_all_prediction_dict.pkl'
        training_lstm_path = 'soft_vote_multilabel_validation_prediction_dict.pkl'
    else:
        '''non-soft'''
        event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'vote_multilabel_test_all_v2_prediction_dict.pkl'
        training_lstm_path = 'vote_multilabel_validation_v2_prediction_dict.pkl'

    img_id_list = cPickle.load(open(root+folder_name+'validation_img_list.pkl'))
    img_recognition = np.load(root + folder_name + 'validation_' + event_path)
    img_importance = np.load(root + folder_name + 'validation_' + importance_path)
    event_lstm = cPickle.load(open(root + folder_name + training_lstm_path))

    ###need: img_importance, img_recognition, test_list, event_img_dict(dict event->images)###

    poly1 = 10
    results = np.zeros((len(bests), 9))
    i = 0
    for threshold_best, poly2_best in bests:
        print threshold_best, poly2_best
        for max_iter in range(9, 10):
            recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_id_list, None,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10, float(poly2_best) / 10,
                                                                               lstm_event_recognition=event_lstm,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=max_iter)
            score_this, confusion_this, cross_entropy_this = prediction_cnn_event_cufed(recog_result, print_=True)
            results[i, max_iter - 1] = score_this
            f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_VALIDATION_poly_'+str(poly2_best) +'_threshold_' +str(threshold_best)+ '_importance_cross_validation_combine_best.pkl', 'w')
            cPickle.dump(importance_result, f)
            f.close()
        i += 1

    return results

def validation_process_cufed_test_importance(threshold_best, poly2_best, soft=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_validation/'
    root_feature = root + folder_name

    if soft:
        event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'soft_vote_multilabel_test_all_prediction_dict.pkl'
        training_lstm_path = 'soft_vote_multilabel_validation_prediction_dict.pkl'
    else:
        '''non-soft'''
        event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
        importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
        test_lstm_path = 'vote_multilabel_test_all_v2_prediction_dict.pkl'
        training_lstm_path = 'vote_multilabel_validation_v2_prediction_dict.pkl'

    img_id_list = cPickle.load(open(root+folder_name+'test_img_list.pkl'))
    img_recognition = np.load(root + folder_name + 'test_' + event_path)
    img_importance = np.load(root + folder_name + 'test_' + importance_path)
    event_lstm = cPickle.load(open(root +folder_name + test_lstm_path))

    ###need: img_importance, img_recognition, test_list, event_img_dict(dict event->images)###

    poly1 = 10
    for max_iter in range(0, 10):
        recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_id_list, None,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10, float(poly2_best) / 10,
                                                                               lstm_event_recognition=event_lstm,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=max_iter,
                                                                               average=False)
        score_this, confusion_this, cross_entropy_this = prediction_cnn_event_cufed(recog_result,print_=True)

        f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_iter'+str(max_iter) + '_importance_cross_validation_combine_best.pkl', 'w')
        cPickle.dump(importance_result, f)
        f.close()


def validation_process_pec(max_iter,combine_lstm=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'pec/features_validation/'
    root_feature = root + folder_name
    # event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
    # importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
    event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
    importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
    training_lstm_path = 'pec_vote_multilabel_validation_prediction_dict.pkl'
    test_lstm_path = 'pec_vote_multilabel_test_prediction_dict.pkl'



    with open(root_feature + '../meta/train_test.json') as f:
        event_dict = ujson.load(f)


    img_recognition = np.load(root + folder_name + 'pec_training_' + event_path)
    img_importance = np.load(root + folder_name + 'pec_training_' + importance_path)
    with open(root_feature + 'pec_training_' + event_path.split('.')[0] + '_event_list.pkl') as f:
        test_list = cPickle.load(f)
    lstm_recognition = cPickle.load(open(root + folder_name + training_lstm_path))

    # img_recognition = np.load(root + folder_name + 'pec_test_' + event_path)
    # img_importance = np.load(root + folder_name + 'pec_test_' + importance_path)
    # with open(root_feature + 'pec_test_' + event_path.split('.')[0] + '_event_list.pkl') as f:
    #     test_list = cPickle.load(f)
    # lstm_recognition = cPickle.load(open(root + folder_name + test_lstm_path))

    for key in lstm_recognition:
        lstm_recognition['_'.join(key.split('/'))] = lstm_recognition.pop(key)
    img_ids = []
    for event in test_list:
        for img in event_dict[event.split('/')[0]][event.split('/')[1]]:
            img_ids.append(event.split('/')[0] + '_' + event.split('/')[1] + '/' + str(img[0]))



    ###need: img_importance, img_recognition, test_list, event_img_dict(dict event->images)###
    poly1 = 10

    scores = np.zeros((12, 30))
    scores1 = np.zeros((12, 30))
    cross_entropies = np.zeros((12, 30))
    f1_scores = np.zeros((12, 30))
    threshold_list = range(0, 11)[::-1]
    poly2_list = range(1, 31)

    for ind1, threshold_m in enumerate(threshold_list):
        for ind2, poly2 in enumerate(poly2_list):
                recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_ids, None,
                                                                               0, float(threshold_m) / 10,
                                                                               float(poly1) / 10, float(poly2) / 10,
                                                                                lstm_recognition,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=9,average=True,combine_lstm=combine_lstm)

                score_this, confusion_this = prediction_cnn_event(recog_result)
                score_this2, f1_score = recall_topk(recog_result, top_n=2)
                scores[ind1, ind2] = score_this
                scores1[ind1, ind2] = score_this2
                f1_scores[ind1, ind2] = f1_score
                print 'THRESHOLD:', threshold_m, 'POLY:', poly2, 'SCORE:',score_this, 'f1_score:', f1_score, 'score_2:',score_this2

    print scores
    print f1_scores
    print scores1
    temp = np.where(scores == np.max(scores))
    temp = [(threshold_list[temp[0][i]], poly2_list[temp[1][i]]) for i in xrange(len(temp[0]))]
    print temp
    # temp = sorted(temp, key=lambda x: x[0])
    # temp_both = []
    # for i,j in temp:
    #     temp_both.extend[(i,j,f1_scores[i][j])]
    # print temp_both
    # temp = sorted(temp_both, key=lambda x: x[1], reverse=True)

    # poly1 = 10
    # # threshold_best = 10; poly_best = 10
    # # threshold_best = 10; poly_best = 6
    # # threshold_best = 5; poly_best = 2
    # threshold_best = 6; poly_best = 8
    # # threshold_best = 6; poly_best = 8
    # recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_ids, None,
    #                                                                            0, float(threshold_best) / 10,
    #                                                                            float(poly1) / 10, float(poly_best) / 10,
    #                                                                                       lstm_recognition,
    #                                                                            img_importance=img_importance,
    #                                                                            img_recognition=img_recognition,
    #                                                                            max_iter=max_iter)
    # # recog_result, importance_result = em_combine_event_recognition_curation_corrected(img_ids, None,
    # #                                                                            0, float(threshold_best) / 10,
    # #                                                                            float(poly1) / 10, float(poly_best) / 10,
    # #                                                                            img_importance=img_importance,
    # #                                                                            img_recognition=img_recognition,
    # #                                                                            max_iter=9)
    #
    # score_this, confusion_this = prediction_cnn_event(recog_result,True)
    # # f = open(root_feature + 'RECALL_PARAM_recognition_cross_validation_combine_best.pkl', 'w')
    # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_recognition_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(recog_result, f)
    # f.close()
    # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_importance_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(importance_result, f)
    # f.close()


def validation_process_pec_test(bests,combine_lstm=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'pec/features_validation/'
    root_feature = root + folder_name
    # event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
    # importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
    # test_lstm_path = 'pec_vote_multilabel_soft_test_prediction_dict.pkl'
    event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
    importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
    test_lstm_path = 'pec_vote_multilabel_test_prediction_dict.pkl'


    with open(root_feature + '../meta/train_test.json') as f:
        event_dict = ujson.load(f)


    img_recognition = np.load(root + folder_name + 'pec_test_' + event_path)
    img_importance = np.load(root + folder_name + 'pec_test_' + importance_path)
    with open(root_feature + 'pec_test_' + event_path.split('.')[0] + '_event_list.pkl') as f:
        test_list = cPickle.load(f)
    lstm_recognition = cPickle.load(open(root + folder_name + test_lstm_path))

    for key in lstm_recognition:
        lstm_recognition['_'.join(key.split('/'))] = lstm_recognition.pop(key)
    img_ids = []
    for event in test_list:
        for img in event_dict[event.split('/')[0]][event.split('/')[1]]:
            img_ids.append(event.split('/')[0] + '_' + event.split('/')[1] + '/' + str(img[0]))


    poly1 = 10
    results = np.zeros((len(bests), 9))
    # threshold_best = 10; poly_best = 10
    # threshold_best = 10; poly_best = 6
    # threshold_best = 5; poly_best = 2
    # threshold_best = 6; poly_best = 8
    i = 0
    for threshold_best, poly2_best in bests:
        for max_iter in range(10):
            recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_ids, None,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10, float(poly2_best) / 10,
                                                                                          lstm_recognition,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=max_iter, average=False,
                                                                               combine_lstm=combine_lstm)

            score_this, confusion_this = prediction_cnn_event_only_top1(recog_result)
            results[i, max_iter - 1] = score_this
        print '(', threshold_best, ',', poly2_best, '):', results[i, :]
        i += 1
    print results
    # f = open(root_feature + 'RECALL_PARAM_recognition_cross_validation_combine_best.pkl', 'w')
    # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_recognition_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(recog_result, f)
    # f.close()
    # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_importance_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(importance_result, f)
    # f.close()
    return results

def validation_process_pec_validation(bests, combine_lstm=True):
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'pec/features_validation/'
    root_feature = root + folder_name
    # event_path = 'predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy'
    # importance_path = 'sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000_event_list.pkl.npy'
    event_path = 'predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy'
    importance_path = 'sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy'
    training_lstm_path = 'pec_vote_multilabel_validation_prediction_dict.pkl'

    with open(root_feature + '../meta/train_test.json') as f:
        event_dict = ujson.load(f)

    img_recognition = np.load(root + folder_name + 'pec_training_' + event_path)
    img_importance = np.load(root + folder_name + 'pec_training_' + importance_path)
    with open(root_feature + 'pec_training_' + event_path.split('.')[0] + '_event_list.pkl') as f:
        test_list = cPickle.load(f)
    lstm_recognition = cPickle.load(open(root + folder_name + training_lstm_path))

    for key in lstm_recognition:
        lstm_recognition['_'.join(key.split('/'))] = lstm_recognition.pop(key)
    img_ids = []
    for event in test_list:
        for img in event_dict[event.split('/')[0]][event.split('/')[1]]:
            img_ids.append(event.split('/')[0] + '_' + event.split('/')[1] + '/' + str(img[0]))


    poly1 = 10
    results = np.zeros((len(bests), 9))
    # threshold_best = 10; poly_best = 10
    # threshold_best = 10; poly_best = 6
    # threshold_best = 5; poly_best = 2
    # threshold_best = 6; poly_best = 8
    i = 0
    for threshold_best, poly2_best in bests:
        for max_iter in range(1, 10):
            recog_result, importance_result = em_combine_event_recognition_curation_corrected_new(img_ids, None,
                                                                               0, float(threshold_best) / 10,
                                                                               float(poly1) / 10, float(poly2_best) / 10,
                                                                                          lstm_recognition,
                                                                               img_importance=img_importance,
                                                                               img_recognition=img_recognition,
                                                                               max_iter=max_iter, average=False,
                                                                               combine_lstm=combine_lstm)

            score_this, confusion_this = prediction_cnn_event(recog_result)
            results[i, max_iter - 1] = score_this
        print '(', threshold_best, ',', poly2_best, '):', results[i, :]
        i += 1
    print results
    return results



def validation_lstm_cufed():
    root = '/home/feiyu1990/local/event_curation/'
    folder_name = 'CNN_all_event_corrected_multi/features_validation/'
    root_feature = root + folder_name
    # event_path = 'test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    # combined_path = 'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best_v2.pkl'
    # lstm_path = 'vote_softall_multilabel_crossvalidation_prediction_dict.pkl' #73.9->76.8

    test_lstm_path = 'vote_multilabel_test_all_v2_prediction_dict.pkl'
    test_combined_path = 'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl'
    training_lstm_path = 'vote_multilabel_validation_v2_prediction_dict.pkl'
    training_combined_path = 'THRESHOLDM_POLY2_validation_recognition_cross_validation_combine_best.pkl'
    #
    #
    # test_lstm_path = 'soft_vote_multilabel_test_all_prediction_dict.pkl'
    # test_combined_path = 'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl'
    # training_lstm_path = 'soft_vote_multilabel_validation_prediction_dict.pkl'
    # training_combined_path = 'SOFT_2THRESHOLDM_POLY2_validation_recognition_cross_validation_combine_best.pkl'
    #


    combine_dict_training = cPickle.load(open(root_feature + training_combined_path))
    lstm_dict_training = cPickle.load(open(root_feature + training_lstm_path))
    lstm_dict_test = cPickle.load(open(root_feature + test_lstm_path))
    combine_dict_test = cPickle.load(open(root_feature + test_combined_path))


    poly_list = range(0, 67, 2)
    poly_list = [float(i) / 20 for i in poly_list]
    # poly_list=[1]
    scores = []
    for poly in poly_list:
            combine_lstm_result = combine_lstm_cnn_result(combine_dict_training, lstm_dict_training, poly)
            score_this_training, confusion_training,cross_entropy = prediction_cnn_event_cufed(combine_lstm_result)
            scores.append(score_this_training)
    print scores
    poly_best = 1
    # poly_best = poly_list[np.argmax(scores)]
    print 'BEST POLY:',poly_best

    poly_best = 1

    #poly_best soft: 1.3
    #poly_best non soft: 0.1
    combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, poly_best)

    score_this_test, confusion_test,cross_entropy = prediction_cnn_event_cufed(combine_lstm_result_test, print_=True)

    f = open(root_feature + 'LSTM_COMBINE_POLY1_recognition_cross_validation_combine_best.pkl', 'w')
    cPickle.dump(combine_lstm_result_test, f)
    f.close()

    print 'LSTM'
    combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, 0,1)
    score_this_test, confusion_test,cross_entropy = prediction_cnn_event_cufed(combine_lstm_result_test, print_=True)

    # f = open(root_feature + 'LSTM_1_recognition_cross_validation_combine_best.pkl', 'w')
    # cPickle.dump(combine_lstm_result_test, f)
    # f.close()
    #
    # print 'CNN'
    # combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, 1,0)
    score_this_test, confusion_test,cross_entropy = prediction_cnn_event_cufed(combine_lstm_result_test, print_=True)

def validation_lstm_pec():

    # combined_path = 'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl'
    # validation_combined_path = 'SOFT_THRESHOLDM_POLY2_VALIDATION_recognition_cross_validation_combine_best.pkl'
    # lstm_path = 'pec_vote_multilabel_soft_validation_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET
    # validation_lstm_path = 'pec_vote_multilabel_soft_test_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET
    #
    # combined_path = 'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl'
    # validation_combined_path = 'THRESHOLDM_POLY2_VALIDATION_recognition_cross_validation_combine_best.pkl'
    # lstm_path = 'pec_vote_multilabel_validation_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET
    # validation_lstm_path = 'pec_vote_multilabel_test_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET

    ###REVERSE!!!!!!!!!!!!!!!!
    validation_combined_path = 'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl'
    combined_path = 'SOFT_THRESHOLDM_POLY2_VALIDATION_recognition_cross_validation_combine_best.pkl'
    validation_lstm_path = 'pec_vote_multilabel_soft_validation_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET
    lstm_path = 'pec_vote_multilabel_soft_test_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET


    validation_combined_path = 'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best_REMOVE_LABEL.pkl'
    combined_path = 'SOFT_THRESHOLDM_POLY2_VALIDATION_recognition_cross_validation_combine_best_REMOVE_LABEL.pkl'
    validation_lstm_path = 'pec_vote_multilabel_soft_validation_prediction_dict_REMOVE_LABEL.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET
    lstm_path = 'pec_vote_multilabel_soft_test_prediction_dict_REMOVE_LABEL.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET



    # validation_lstm_path = 'pec_vote_multilabel_validation_prediction_dict.pkl'
    # lstm_path = 'pec_vote_multilabel_test_prediction_dict.pkl'  #SHOULD USE THIS BECAUSE IT IS TRAINED ON SOFT TARGET

    combine_dict_training = cPickle.load(open(root_feature + validation_combined_path))
    lstm_dict_training = cPickle.load(open(root_feature + validation_lstm_path))
    lstm_dict_test = cPickle.load(open(root_feature + lstm_path))
    combine_dict_test = cPickle.load(open(root_feature + combined_path))


    for key in combine_dict_training:
        combine_dict_training['_'.join(key.split('/'))] = combine_dict_training.pop(key)
    for key in lstm_dict_training:
        lstm_dict_training['_'.join(key.split('/'))] = lstm_dict_training.pop(key)
    for key in lstm_dict_test:
        lstm_dict_test['_'.join(key.split('/'))] = lstm_dict_test.pop(key)
    for key in combine_dict_test:
        combine_dict_test['_'.join(key.split('/'))] = combine_dict_test.pop(key)

    # print combine_dict_test.keys()
    # print lstm_dict_test.keys()

    poly_list = range(2, 42, 2)
    poly_list = [float(i) / 20 for i in poly_list]
    print poly_list
    scores = []; scores_top2 = []; f1_scores = []
    for poly in poly_list:
            combine_lstm_result = combine_lstm_cnn_result(combine_dict_training, lstm_dict_training, poly)
            score_this_training, confusion_training = prediction_cnn_event(combine_lstm_result)
            score_this_top2, f1_score = recall_topk(combine_lstm_result, top_n=2)
            scores.append(score_this_training)
            scores_top2.append(score_this_top2)
            f1_scores.append(f1_score)
    print scores
    print scores_top2
    print f1_scores
    poly_best = poly_list[np.argmax(scores)]
    # print poly_list[np.argmax(scores)], poly_list[np.argmax(scores_top2)], poly_list[np.argmax(f1_scores)]
    # poly_best = poly_list[4]
    print 'COMBINE'
    print poly_best
    combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test,  poly_best)
    score_this_test, confusion_test = prediction_cnn_event(combine_lstm_result_test, print_=True)

    f = open(root_feature + 'SOFT_LSTM_COMBINE_POLY_REVERSEVALIDATION_cross_validation_combine_best_REMOVE_LABEL.pkl', 'w')
    cPickle.dump(combine_lstm_result_test, f)
    f.close()

    print 'LSTM'
    combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, 0, 1)
    score_this_test, confusion_test = prediction_cnn_event(combine_lstm_result_test, print_=True)

    f = open(root_feature + 'SOFT_LSTM_REVERSEVALIDATION_cross_validation_combine_best_REMOVE_LABEL.pkl', 'w')
    cPickle.dump(combine_lstm_result_test, f)
    f.close()

    print 'CNN'
    combine_lstm_result_test = combine_lstm_cnn_result(combine_dict_test, lstm_dict_test, 1, 0)
    score_this_test, confusion_test = prediction_cnn_event(combine_lstm_result_test, print_=True)


def recall(path):
    with open(path) as f:
        result_dict = cPickle.load(f)
    recall_curve = []

    for n in xrange(1, 10):
        recall_this = []
        for event_old in result_dict:
            if '/' in event_old:
                event = '_'.join(event_old.split('/'))
            else:
                event = event_old
            prediction = result_dict[event_old]
            type_old = '_'.join(event.split('_')[:-1])
            if type_old not in map_to_new:
                continue
            ground_ = new_ind[dict_name2[map_to_new[type_old]]]
            ind_mapping = [3, 5, 6, 16, 17, 18, 19, 20, 21]
            prediction = np.array(prediction[ind_mapping])
            predict_ = list(np.argsort(prediction)[::-1])
            if ground_ in predict_[:n]:
                recall_this.append(1)
            else:
                recall_this.append(0)
        recall_curve.append(np.sum(recall_this))

    print recall_curve

def recall_real(path):
    with open(path) as f:
        result_dict = cPickle.load(f)
    recall_curve = []

    for n in xrange(1, 24):
        recall_this = []
        for event_old in result_dict:
            if '/' in event_old:
                event = '_'.join(event_old.split('/'))
            else:
                event = event_old
            prediction = result_dict[event_old]
            type_old = '_'.join(event.split('_')[:-1])
            if type_old not in map_to_new:
                continue
            ground_ = new_ind[dict_name2[map_to_new[type_old]]]
            # ind_mapping = [3, 5, 6, 16, 17, 18, 19, 20, 21]
            # prediction = np.array(prediction[ind_mapping])
            predict_ = list(np.argsort(prediction)[::-1])
            predict_new = []
            for ii in predict_:
                if ii+1 in new_ind:
                    predict_new.append(new_ind[ii+1])
                else:
                    predict_new.append(-1)
            if ground_ in predict_new[:n]:
                recall_this.append(1)
            else:
                recall_this.append(0)
        recall_curve.append(np.sum(recall_this))

    print recall_curve


def recall_topk(result_dict, top_n=1):
        wrong_dict_name = []
        ground_list = []; predict_list = []
        recall_this = []
        for event_old in result_dict:
            if '/' in event_old:
                event = '_'.join(event_old.split('/'))
            else:
                event = event_old
            prediction = result_dict[event_old]
            type_old = '_'.join(event.split('_')[:-1])
            if type_old not in map_to_new:
                continue
            ground_ = new_ind[dict_name2[map_to_new[type_old]]]
            ground_list.append(ground_)
            ind_mapping = [3, 5, 6, 16, 17, 18, 19, 20, 21]
            prediction = np.array(prediction[ind_mapping])
            predict_ = list(np.argsort(prediction)[::-1])
            predict_list.append(predict_[0])
            if ground_ in predict_[:top_n]:
                recall_this.append(1)
            else:
                recall_this.append(0)
                wrong_dict_name.append((event_old, [dict_reverse[(np.argsort(result_dict[event_old]))[::-1][ii] + 1] for ii in range(3)]))
        # print ground_list, predict_list
        f1 = f1_score(ground_list, predict_list)
        # print wrong_dict_name
        return float(np.sum(recall_this)) / len(recall_this), f1

def recall_cufed(path):
    with open(path) as f:
        test_feature_dict = cPickle.load(f)
    recall_curve = []
    for n in xrange(1, 24):
        recall_this = []
        for event_id in test_feature_dict:
            # print ground_truth_event[event_id]
            feature_this = test_feature_dict[event_id]
            feature_this = feature_this / np.sum(feature_this)
            prediction_ = np.argsort(feature_this)[::-1]
            ground_ = np.zeros((23,))
            for i in ground_truth_event[event_id]:
                ground_[dict_name2[i[0]] - 1] = i[1]
            predict_n = prediction_[:n]
            ground_n = np.where(ground_)[0]
            found = False
            for i in ground_n:
                if i in predict_n:
                    recall_this.append(1)
                    found = True
                    break
            if not found:
                recall_this.append(0)
        recall_curve.append(np.sum(recall_this))
    print recall_curve


def recall_topk_cufed(test_feature_dict, top_n=2):
        macro_f1 = defaultdict(list)
        ground_list = []; predict_list = []
        recall_this = [] ;f1_list = []
        for event_id in test_feature_dict:
            # print ground_truth_event[event_id]
            feature_this = test_feature_dict[event_id]
            feature_this = feature_this / np.sum(feature_this)
            prediction_ = np.argsort(feature_this)[::-1]
            ground_ = np.zeros((23,))
            for i in ground_truth_event[event_id]:
                ground_[dict_name2[i[0]] - 1] = i[1]
            predict_n = prediction_[:top_n]
            ground_n = np.where(ground_)[0]
            # ground_list.append([int(i>0) for i in ground_])
            # predict_list.append([int(i==prediction_[0]) for i in range(23)])
            # print [int(i == prediction_[0]) for i in feature_this]
            # ground_list.append(dict_name2[ground_truth_event[event_id][0][0]] - 1)
            # predict_list.append(prediction_[0])
            found = False
            for i in ground_n:
                if i in predict_n:
                    recall_this.append(1)
                    found = True
                    break
            if not found:
                recall_this.append(0)
            predict_n = np.where(feature_this == np.max(feature_this))[0]
            inter_ = len(set(predict_n).intersection(ground_n))
            p_ =float(inter_) / len(predict_n); r_ = float(inter_) / len(ground_n)
            if p_+r_ == 0:
                f1_list.append(0)
            else:
                f1_list.append(2*p_*r_/(p_+r_))
            for i in ground_n:
                macro_f1[i].append((p_, r_))

        pr_all = []; len_all = 0
        for i in macro_f1:
            temp = macro_f1[i]
            len_all += len(temp)
            pr_all.append((sum([i[0] for i in temp]), sum(i[1] for i in temp)))
        # print pr_all
        macro_f1_list = [2*i*j/max((i+j), 0.001) for i,j in pr_all]
        macro_f1_value = float(np.sum(macro_f1_list)) / len_all
        # print len(ground_list)
        # print len(predict_list)
        labels = []
        # for j in xrange(23):
        #     labels.append([int(i==j) for i in range(23)])
        # print labels
        # labels = range(23)
        # f1 = f1_score(ground_list, predict_list,average='macro', labels=labels)
        # f1 = f1_score(predict_list, ground_list,average='weighted')
        # print Counter(ground_list)
        return  float(np.sum(recall_this))/len(recall_this), np.average(f1_list), macro_f1_value


def create_pec_remove_label(path):
    ind_mapping = [3, 5, 6, 16, 17, 18, 19, 20, 21]
    with open(path) as f:
        dict_ = cPickle.load(f)
    for event in dict_:
        temp = dict_[event]
        for i in range(len(temp)):
            if i not in ind_mapping:
                temp[i] = 0
        temp = np.array(temp)
        temp = temp / np.sum(temp)
        dict_[event] = temp
    with open(path.split('.')[0] + '_REMOVE_LABEL.pkl', 'w') as f:
        cPickle.dump(dict_, f)


def create_importance(event_id):
    f = open(root_feature + 'SOFT_THRESHOLDM_POLY2_VALIDATION_importance_cross_validation_combine_best.pkl')
    importance = cPickle.load(f)
    f.close()
    f = open(root_feature + '../meta/test.json')
    test_event_list = ujson.load(f)
    f.close()
    importance_this = importance[event_id]
    list_this = test_event_list['_'.join(event_id.split('_')[:-1])][event_id.split('_')[-1]]
    importance_img_list = [(i[0],j) for i, j in zip(list_this, importance_this)]
    importance_sorted_ = sorted(importance_img_list, key=lambda x: x[1], reverse=True)
    print importance_img_list
    print '*****'
    print importance_sorted_
    f = open(root_feature + 'SOFT_LSTM_COMBINE_POLY_REVERSEVALIDATION_cross_validation_combine_best.pkl')
    # f = open(root_feature + 'SOFT_THRESHOLDM_POLY2_VALIDATION_recognition_cross_validation_combine_best.pkl')
    result_ = cPickle.load(f)
    f.close()
    # print result_[event_id]
    # print np.sum(result_[event_id])
    print result_[event_id] / np.sum(result_[event_id])
    f = open(root_feature + 'EM1_VALIDATION_recognition_cross_validation_combine_best.pkl')
    result_ = cPickle.load(f)
    f.close()
    print result_[event_id]

    f = open(root_feature + 'pec_test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_event_list.pkl')
    event_list = cPickle.load(f)
    f.close()
    feature_ = np.load(root_feature + 'pec_test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    with open(root_feature + '../meta/train_test.json') as f:
        event_dict = ujson.load(f)
    count = 0
    event_img_dict = defaultdict(list)
    for event in event_list:
        event = '_'.join(event.split('/'))
        this_event = event_dict['_'.join(event.split('_')[:-1])][event.split('_')[-1]]
        for img in this_event:
            event_img_dict[event].append((img, [dict_reverse[i + 1] for i in np.argsort(feature_[count, :])[::-1][:2]]))
            count += 1

    temp = event_img_dict[event_id]

    dict_ = dict()
    for img in temp:
        dict_[img[0][0]] = [img[1]]
    list_sorted = []
    for img in importance_sorted_:
        list_sorted.append((img[0], dict_[img[0]], img[1]))
    print list_sorted

def create_importance_cufed(event_id):
    root_feature = '/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/'
    f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_iter7_importance_cross_validation_combine_best_dict.pkl')
    importance = cPickle.load(f)
    f.close()
    f = open(root_feature + 'test_list.pkl')
    test_event_list = cPickle.load(f)
    f.close()
    f = open(root_feature + 'test_img_list.pkl')
    test_img_list = cPickle.load(f)
    f.close()
    test_event_dict = defaultdict(list)
    for img in test_img_list:
        event = img.split('/')[0]
        test_event_dict[event].append(img)
    list_this = test_event_dict[event_id]
    print list_this
    importance_this = importance[event_id]
    importance_img_list = [(i.split('/')[1],j[2]) for i, j in zip(list_this, importance_this)]
    importance_sorted_ = sorted(importance_img_list, key=lambda x: x[1], reverse=True)
    print importance_img_list
    print '*****'
    print importance_sorted_
    # f = open(root_feature + 'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')
    # result_ = cPickle.load(f)
    # f.close()
    f = open(root_feature + 'test_sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_event_list.pkl')
    event_list = cPickle.load(f)
    f.close()
    feature_ = np.load(root_feature + 'test_sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000.npy')

    img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            img_dict[img.split('/')[0]].append(img)
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            img_dict[img.split('/')[0]].append(img)


    event_img_dict = defaultdict(list)
    count = 0
    for event in event_list:
        for img in img_dict[event]:
            event_img_dict[event].append((img.split('/')[1], [dict_reverse[i + 1] for i in np.argsort(feature_[count, :])[::-1][:2]]))
            count += 1
    temp1 = event_img_dict[event_id]
    temp_dict = dict(temp1)
    temp = []
    for img in importance_sorted_:
        temp.append((img[0], img[1], temp_dict[img[0]]))

    print temp

if __name__ == '__main__':
    root_feature = '/home/feiyu1990/local/event_curation/pec/features_validation/'
    root_feature = '/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/'
    with open('/home/feiyu1990/local/event_curation/0208_correction/all_input_and_result/'
              'new_multiple_result_2round_removedup_vote.pkl') as f:
        ground_truth_event = cPickle.load(f)
    # prediction_cnn_event_cufed(cPickle.load(open(root_feature + 'soft_vote_multilabel_test_all_prediction_dict.pkl')),print_=True)
    # prediction_cnn_event_cufed(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_iter1_recognition_cross_validation_combine_best.pkl')))

    # split_valid_test()
    # validation_process_cufed(None)
    # for i in range(1, 9):
    #     validation_process_cufed(i, 3,19)
    # validation_process_cufed(1,0,1)
    # bests = validation_process_cufed(None,None,None)


    # bests = validation_process_cufed(9,10,19)
    # bests = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 22), (0, 23), (0, 24), (0, 25), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (2, 1), (2, 2), (2, 3), (2, 4), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (3, 1), (3, 2), (3, 3), (3, 4), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (4, 1), (4, 2), (4, 3), (4, 4), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 21), (6, 22), (6, 23), (6, 24), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 21), (7, 22), (7, 23), (7, 24), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30)]
    # bests = [(10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26)]
    # bests = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30)]
    # validation_process_cufed_validation(bests)
    # bests = [(10,1)]
    # validation_process_cufed_test(bests)
    # validation_process_cufed_test([(7,16),(7,14),(3,6),(2,9),(1,8),(1,1)])
    # validation_process_cufed_test_importance(10, 20)




    # validation_process_pec(9,combine_lstm=False)
    # bests = [(6,11),(7,7)]
    # bests = [(5, 4), (5, 5), (3, 5),(6,11)]
    bests = [(6, 11)]
    validation_process_pec_test(bests)
    # validation_process_pec_test(bests,combine_lstm=False)
    # validation_process_pec_validation([(3,16)],combine_lstm=False)



    # split_valid_test()
    # for i,j in [(1,4),(1,19),(1,20),(2,18),(2,19),(2,20),
    #             (3,18),(3,19),(3,20),(4,18),(4,19),(4,20),
    #             (5,18),(5,19),(5,20) ,(6,20) ,(7,4), (7,20), (8,4) ,(8,20) ,(9,4), (9,20),
    #             (10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),
    #             (10,12),(10,13),(10,14),(10,15),(10,16),(10,17),(10,18),(10,19),(10,20)]:
    #     validation_process_cufed(i,j)
    # for i in range(1,9):
    #     validation_process_pec(i)

    # create_importance_cufed('101_26582481@N08')
    # cufed softmax DONE!
    # validation_process_pec()
    # create_pec_remove_label(root_feature+'pec_vote_multilabel_soft_validation_prediction_dict.pkl')
    # create_pec_remove_label(root_feature+'pec_vote_multilabel_soft_test_prediction_dict.pkl')
    # create_pec_remove_label(root_feature+'SOFT_THRESHOLDM_POLY2_VALIDATION_recognition_cross_validation_combine_best.pkl')
    # create_pec_remove_label(root_feature+'SOFT_LSTM_REVERSEVALIDATION_cross_validation_combine_best.pkl')

    # root_feature = '/home/feiyu1990/local/event_curation/pec/features_validation/'
    # validation_lstm_pec()


    #
    # recall_cufed(root_feature+'EM1_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'LSTM_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'LSTM_COMBINE_recognition_cross_validation_combine_best.pkl')
    #
    # #
    # recall_cufed(root_feature+'SOFT_EM1_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'SOFT_THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'SOFT_LSTM_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'SOFT_2LSTM_COMBINE_recognition_cross_validation_combine_best.pkl') #this one is correct


    # create_validation_list()
    # split_valid_test()
    # create_img_list()


    # prediction_cnn_event_cufed(cPickle.load(open(root_feature+'EM1_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event_cufed(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event_cufed(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event_cufed(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_LSTM_2nd_recognition_cross_validation_combine_best.pkl')), print_=True)
    # # prediction_cnn_event_cufed(cPickle.load(open(root_feature+'LSTM_recognition_cross_validation_combine_best.pkl')), print_=True) #this one is correct
    # # prediction_cnn_event_cufed(cPickle.load(open(root_feature+'LSTM_COMBINE_recognition_cross_validation_combine_best.pkl')), print_=True) #this one is correct
    # # print '*********'
    # #
    # recall_cufed(root_feature+'EM1_recognition_cross_validation_combine_best.pkl')
    # # recall_cufed(root_feature+'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')
    # recall_cufed(root_feature+'THRESHOLDM_POLY2_LSTM_2nd_recognition_cross_validation_combine_best.pkl')
    # # recall_cufed(root_feature+'LSTM_recognition_cross_validation_combine_best.pkl') #this one is correct
    # # recall_cufed(root_feature+'LSTM_COMBINE_recognition_cross_validation_combine_best.pkl') #this one is correct
    # # print '*********'
    # print recall_topk_cufed(cPickle.load(open(root_feature + 'EM1_recognition_cross_validation_combine_best.pkl')))
    # # print recall_topk_cufed(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl')))
    # print recall_topk_cufed(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')))
    # print recall_topk_cufed(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_2nd_recognition_cross_validation_combine_best.pkl')))
    # # print recall_topk_cufed(cPickle.load(open(root_feature + 'LSTM_recognition_cross_validation_combine_best.pkl')))
    # # print recall_topk_cufed(cPickle.load(open(root_feature + 'LSTM_COMBINE_recognition_cross_validation_combine_best.pkl')))
    #
    #
    # prediction_cnn_event_cufed_type2(cPickle.load(open(root_feature+'EM1_recognition_cross_validation_combine_best.pkl')))
    # # prediction_cnn_event_cufed_type2(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_recognition_cross_validation_combine_best.pkl')))
    # prediction_cnn_event_cufed_type2(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')))
    # prediction_cnn_event_cufed_type2(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_LSTM_2nd_recognition_cross_validation_combine_best.pkl')))
    # # prediction_cnn_event_cufed_type2(cPickle.load(open(root_feature+'LSTM_recognition_cross_validation_combine_best.pkl')))
    # # prediction_cnn_event_cufed_type1(cPickle.load(open(root_feature+'LSTM_COMBINE_recognition_cross_validation_combine_best.pkl')))
    # #
    #
    #
    # prediction_cnn_event_cufed_type1(cPickle.load(open(root_feature+'EM1_recognition_cross_validation_combine_best.pkl')))
    # prediction_cnn_event_cufed_type1(cPickle.load(open(root_feature+'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')))
    #
    #
    #

    # root_feature = '/home/feiyu1990/local/event_curation/pec/features_validation/'
    # prediction_cnn_event(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_NEW_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event(cPickle.load(open(root_feature + 'EM1_new_recognition_cross_validation_combine_best.pkl')), print_=True)
    # prediction_cnn_event(cPickle.load(open(root_feature + 'LSTM_REVERSEVALIDATION_cross_validation_combine_best.pkl')), print_=True)
    # #
    # print '*********'
    # print recall_topk(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')))
    # print recall_topk(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_recognition_cross_validation_combine_best.pkl')))
    # print recall_topk(cPickle.load(open(root_feature + 'THRESHOLDM_POLY2_NEW_recognition_cross_validation_combine_best.pkl')))
    # print recall_topk(cPickle.load(open(root_feature + 'EM1_new_recognition_cross_validation_combine_best.pkl')))
    # print recall_topk(cPickle.load(open(root_feature + 'LSTM_REVERSEVALIDATION_cross_validation_combine_best.pkl')))
    #
    # print '*********'
    # recall(root_feature + 'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')
    # recall(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_recognition_cross_validation_combine_best.pkl')
    # recall(root_feature + 'THRESHOLDM_POLY2_NEW_recognition_cross_validation_combine_best.pkl')
    # recall(root_feature + 'EM1_new_recognition_cross_validation_combine_best.pkl')
    # recall(root_feature + 'LSTM_REVERSEVALIDATION_cross_validation_combine_best.pkl')
    # print '*********'
    # recall_real(root_feature + 'THRESHOLDM_POLY2_LSTM_recognition_cross_validation_combine_best.pkl')
    # recall_real(root_feature + 'THRESHOLDM_POLY2_LSTM_NEW_recognition_cross_validation_combine_best.pkl')
    # recall_real(root_feature + 'THRESHOLDM_POLY2_NEW_recognition_cross_validation_combine_best.pkl')
    # recall_real(root_feature + 'EM1_new_recognition_cross_validation_combine_best.pkl')
    # recall_real(root_feature + 'LSTM_REVERSEVALIDATION_cross_validation_combine_best.pkl')
    #
