

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
#import Bio.Cluster
# combine_face_model = '_combined_10_fromnoevent.cPickle'
# combine_face_model = '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle'
global_permutation_time = 1
from sklearn.decomposition import PCA



correct_list = {'5_19479358@N00':'Museum', '38_59616483@N00':'Museum','136_95413346@N00':'Museum',
                    '0_27302158@N00':'CasualFamilyGather','7_55455788@N00':'Birthday',
                    '144_95413346@N00':'Halloween', '29_13125640@N07':'Christmas', '1_21856707@N00': 'GroupActivity',
                    '0_22928590@N00':'GroupActivity','3_7531619@N05':'Zoo',
                    '16_18108851@N00':'Show', '23_89182227@N00':'Show', '2_27883710@N08':'Sports',
                    '35_8743691@N02':'Wedding', '14_93241698@N00':'Museum', '9_34507951@N07':'BusinessActivity',
                    '32_35578067@N00':'Protest', '20_89138584@N00':'PersonalSports', '18_50938313@N00':'PersonalSports',
                    '376_86383385@N00':'PersonalSports','439_86383385@N00':'PersonalSports','545_86383385@N00':'PersonalSports',
                    '2_43198495@N05':'PersonalSports', '3_60652642@N00':'ReligiousActivity', '9_60053005@N00':'GroupActivity',

                        '56_74814994@N00':'BusinessActivity', '22_32994285@N00':'Sports', '15_66390637@N08':'Sports',
                         '3_54218473@N05':'Zoo', '4_53628484@N00':'Sports', '0_7706183@N06':'GroupActivity',
                         '4_15251430@N03':'Zoo', '63_52304204@N00':'Sports', '2_36319742@N05':'Architecture',
                         '2_12882543@N00':'Sports', '1_75003318@N00':'Sports', '1_88464035@N00':'GroupActivity',
                         '21_49503048699@N01':'CasualFamilyGather', '211_86383385@N00':'Sports',
                         '0_70073383@N00':'PersonalArtActivity'}


block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
root = '/home/feiyu1990/local/event_curation/'
baseline_name = 'baseline_all_correction_removedup_vote/'
# baseline_name = 'baseline_all_0509/'
#root = '/Users/wangyufei/Documents/Study/intern_adobe/'

# global abandoned_test


dict_name = {'Theme park':'ThemePark', 'Urban/City trip':'UrbanTrip', 'Beach trip':'BeachTrip', 'Nature trip':'NatureTrip',
             'Zoo/Aquarium/Botanic garden':'Zoo','Cruise trip':'Cruise','Show (air show/auto show/music show/fashion show/concert/parade etc.)':'Show',
            'Sports game':'Sports','Personal sports':'PersonalSports','Personal art activities':'PersonalArtActivity',
            'Personal music activities':'PersonalMusicActivity','Religious activities':'ReligiousActivity',
            'Group activities (party etc.)':'GroupActivity','Casual family/friends gathering':'CasualFamilyGather',
            'Business activity (conference/meeting/presentation etc.)':'BusinessActivity','Independence Day':'Independence',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture/Art':'Architecture'}

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

dict_name2_reverse = dict([(dict_name2[key], key) for key in dict_name2])
#this is from affinity clustering (cluster #6)
dict_subcategory = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 3, 7: 4, 8: 4, 9: 2, 10: 4, 11: 3, 12: 4, 13: 4, 14: 4,
                    15: 5, 16: 3, 17: 4, 18: 4, 19: 2, 20: 4, 21: 4, 22: 4}

#this is from spectral clustering (cluster #3)
dict_subcategory2 = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 2, 10: 0, 11: 2, 12: 0,
                     13: 0, 14: 0, 15: 2, 16: 2, 17: 0, 18: 0, 19: 2, 20: 1, 21: 2, 22: 0}
class evaluation:
    def __init__(self, net_path, type, validation_name, val_id, face_type = None):
        f = open(root + 'baseline_all_0509/test_event_abandoned.pkl','r')
        self.abandoned_test = cPickle.load(f)
        f.close()
        self.net_path = net_path
        self.validation_name = validation_name
        self.val_id = val_id
        if 'val' in self.validation_name:
            self.validation_path = '/validation_' + str(val_id) + '/'
        else:
            self.validation_path = '/'
        if type == 'none':
            return
        if type == 'worker':
            self.evaluate_present_with_worker_nooverlap(abandon_overlap=False)


    def permute_groundtruth(self):
        f = open(root + baseline_name +'/correction_result_all_v2.pkl','r')
        ground_truth = cPickle.load(f)
        f.close()
        ground_truth_new = {}
        for idx in ground_truth:
            temp = []
            for i in ground_truth[idx]:
                temp.append((i[0], i[1], i[2]+random.uniform(-0.02, 0.02)))
            ground_truth_new[idx] = temp
        f = open(root + baseline_name +'/correction_result_all_v2_permuted.pkl','wb')
        cPickle.dump(ground_truth_new, f)
        f.close()

    def average_precision(self, ground_, predict_, k):
        need_k_index = []
        for i in ground_:
            if ground_[i] <= k:
                need_k_index.append(i)
        retrieved_rank = []
        for i in predict_:
            if i in need_k_index:
                retrieved_rank.append(predict_[i])
        retrieved_rank.sort()
        temp = zip(xrange(1, 1+len(retrieved_rank)), retrieved_rank)
        recall = [min(float(i[0])/i[1],1) for i in temp]
        ap = sum(recall)/len(recall)
        return ap

    def evaluate_MAP_permuted(self, model_names, min_retrieval = 5, permuted = '_permuted',abandon_overlap = False):
        if TIE == False:
            maps = []
            n_ks = []
            f = open(root + baseline_name +'/correction_result_all_v2' +permuted+'.pkl','r')
            ground_truth = cPickle.load(f)
            f.close()
            for model_name in model_names:
                APs = []
                f = open(root + model_name, 'r')
                predict_result = cPickle.load(f)
                f.close()
                for event_id in predict_result:
                    if abandon_overlap and event_id in self.abandoned_test:
                        continue

                    ground_this = ground_truth[event_id]
                    predict_this = predict_result[event_id]
                    predict_ = [i[2] for i in predict_this]
                    ground_ = [i[2] for i in ground_this]

                    predict_ = zip(xrange(len(predict_)), predict_)
                    ground_ = zip(xrange(len(ground_)), ground_)
                    predict_.sort(key = lambda x: x[1], reverse=True)
                    ground_.sort(key = lambda x: x[1], reverse=True)

                    threshold = ground_[max(1, len(ground_)*min_retrieval/100)-1][1]
                    n_k = len([i for i in ground_ if i[1] >= threshold])
                    predict_rank = {}
                    prev = None
                    for i,(k,v) in enumerate(predict_):
                        #if v!=prev:
                        place,prev = i+1,v
                        predict_rank[k] = place


                    ground_rank = {}
                    prev = None
                    for i,(k,v) in enumerate(ground_):
                        if v!=prev:
                            place,prev = i+1,v
                        ground_rank[k] = place
                    AP = self.average_precision(ground_rank, predict_rank, n_k)

                    APs.append([event_id, AP])
                    n_ks.append([n_k, len(ground_)])
                maps.append(sum([i[1] for i in APs])/len(APs))
                percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
            return maps, percent, [i[1] for i in APs]
        else:
            maps = []
            n_ks = []
            f = open(root + baseline_name +'/correction_result_all_v2.pkl','r')

            ground_truth = cPickle.load(f)
            f.close()
            for model_name in model_names:
                APs = []
                f = open(root + model_name, 'r')
                predict_result = cPickle.load(f)
                f.close()
                for event_id in predict_result:
                    if abandon_overlap and event_id in self.abandoned_test:
                        # print event_id
                        continue
                    ground_this = ground_truth[event_id]
                    predict_this = predict_result[event_id]
                    predict_ = [i[2] for i in predict_this]
                    ground_ = [i[2] for i in ground_this]

                    predict_ = zip(xrange(len(predict_)), predict_)
                    ground_ = zip(xrange(len(ground_)), ground_)
                    predict_.sort(key = lambda x: x[1], reverse=True)
                    ground_.sort(key = lambda x: x[1], reverse=True)

                    #threshold = ground_[min_retrieval-1][1]
                    threshold = ground_[max(1, len(ground_)*min_retrieval/100)-1][1]
                    n_k = len([i for i in ground_ if i[1] >= threshold])
                    #if n_k > len(ground_)/2:
                    #    print n_k, len(ground_), event_id
                    predict_rank = {}
                    prev = None
                    for i,(k,v) in enumerate(predict_):
                        #if v!=prev:
                        place,prev = i+1,v
                        predict_rank[k] = place


                    ground_rank = {}
                    prev = None
                    temp_ground = []
                    for i,(k,v) in enumerate(ground_):
                        if v!=prev:
                            place,prev = i+1,v
                        ground_rank[k] = place
                        temp_ground.append(place)
                    temp_ground_count = Counter(temp_ground)
                    ground_map = {}
                    for i in temp_ground_count:
                        temp = temp_ground_count[i]
                        ground_map[i] = i+float(temp-1)/2
                    # print ground_rank
                    for i in ground_rank:
                        ground_rank[i] = ground_map[ground_rank[i]]
                    # print ground_rank
                    AP = self.average_precision(ground_rank, predict_rank, n_k)

                    APs.append([event_id, AP])
                    n_ks.append([n_k, len(ground_)])
                # print model_name, APs
                maps.append(sum([i[1] for i in APs])/len(APs))
                percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
            return maps, percent, [i[1] for i in APs]

    def evaluate_top20_permuted(self, model_names, percent = 20, permuted = '_permuted',abandon_overlap = False):
        retval = []
        f = open(root + baseline_name  + 'correction_result_all_v2'+permuted+'.pkl','r')
        ground_truth = cPickle.load(f)
        f.close()
        for model_name in model_names:
            f = open(root +  model_name, 'r')
            predict_result = cPickle.load(f)
            f.close()
            count_all = 0; n_k_all = 0
            for event_id in predict_result:
                if abandon_overlap and event_id in self.abandoned_test:
                        continue
                ground_this = ground_truth[event_id]
                predict_this = predict_result[event_id]
                predict_ = [i[2] for i in predict_this]
                ground_ = [i[2] for i in ground_this]

                predict_ = zip(xrange(len(predict_)), predict_)
                ground_ = zip(xrange(len(ground_)), ground_)
                predict_.sort(key = lambda x: x[1], reverse=True)
                ground_.sort(key = lambda x: x[1], reverse=True)

                #threshold = ground_[min_retrieval-1][1]
                threshold = ground_[max(1, len(ground_)*percent/100)-1][1]
                n_k = len([i for i in ground_ if i[1] >= threshold])
                dict_wanted = set()
                for i in xrange(n_k):
                    dict_wanted.add(ground_[i][0])
                retrieved_count = 0
                for j in xrange(n_k):
                    if predict_[j][0] in dict_wanted:
                        retrieved_count += 1
                count_all += retrieved_count
                n_k_all += n_k
            retval.append(float(count_all) / n_k_all)
        return retval

    def amt_worker_result_predict_average(self, event_name, min_retrievals = xrange(6,45,5), permuted = '_permuted',abandon_overlap = False):
        f = open(root + baseline_name  + 'correction_result_all_v2'+permuted+'.pkl','r')
        ground_truth = cPickle.load(f)
        f.close()

        input_path = root + baseline_name + event_name+ '/test_image_ids.cPickle'
        f = open(input_path, 'r')
        image_ids = cPickle.load(f)
        f.close()
        event_ids = {}
        for image in image_ids:
            (event_id, img_id) = image.split('/')
            if event_id in event_ids:
                event_ids[event_id].append(image)
            else:
                event_ids[event_id] = [image]
        input_path = root + baseline_name + event_name + '/' +self.validation_name+'.csv'
        line_count = 0
        head_meta = []
        HITs = {}
        all_n_ks = []
        all_aps = []
        all_ps = []
        all_reweighted = []
        with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    head_meta = meta
                elif meta[0] not in HITs:
                    HITs[meta[0]] = [meta]
                else:
                    HITs[meta[0]].append(meta)
                line_count+=1

        image_input_index = {}
        image_output_index = {}
        i = 0
        for field in head_meta:
            if field.startswith('Input.image'):
                image_input_index[int(field[11:])] = i
            if field.startswith('Answer.image'):
                image_output_index[int(field[12:])] = i
            i += 1

        index_worker_id = 15
        index_num_image = 27
        index_event_id = 28
        index_distraction = 31
        for HITId in HITs:
            #print HITId
            this_hit = HITs[HITId]
            num_images = int(this_hit[0][index_num_image])
            distract_image = this_hit[0][index_distraction]
            event_id = this_hit[0][index_event_id]
            [distract1, distract2] = distract_image.split(':')
            distract1 = int(distract1)
            distract2 = int(distract2)
            ground_this = ground_truth[event_id]
            ground_ = [i[2] for i in ground_this]

            ground_ = zip(xrange(len(ground_)), ground_)
            ground_.sort(key = lambda x: x[1], reverse=True)

            for submission in this_hit:
                APs = []
                Ps = []
                n_ks = []
                check_submission = []
                # if submission[index_worker_id] in block_workers:
                #     continue
                #get prediction scores
                ii = 0
                for i in xrange(1, 1+num_images):
                        if i==distract1 or i==distract2:
                            continue
                        score_index = image_output_index[i]
                        score = 0
                        vote = submission[score_index]
                        if vote == 'selected':
                            score += 2
                        elif vote == 'selected_sw':
                            score += 1
                        elif vote == 'selected_irrelevant':
                            score -= 2
                        check_submission.append(score)
                        ii += 1
                predict_rank = {}
                prev = None
                predict_ = zip(xrange(len(check_submission)), check_submission)
                predict_.sort(key = lambda x: x[1], reverse=True)
                for iii,(k,v) in enumerate(predict_):
                        if v!=prev:
                            place,prev = iii+1,v
                        predict_rank[k] = place
                ground_rank = {}
                prev = None
                for iii,(k,v) in enumerate(ground_):
                        if v!=prev:
                            place,prev = iii+1,v
                        ground_rank[k] = place
                temp = [predict_rank[i] for i in predict_rank]
                rank_set =  set(temp)
                new_predict_rank = {}
                for rank in rank_set:
                    this_rank = [i for i in predict_rank if predict_rank[i] == rank]
                    random.shuffle(this_rank)
                    for i in xrange(len(this_rank)):
                        new_predict_rank[this_rank[i]] = rank + i
                for min_retrieval in min_retrievals:
                    threshold = ground_[max(1, len(ground_)*min_retrieval/100)-1][1]
                    #print len(ground_)*min_retrieval/100
                    n_k = len([i for i in ground_ if i[1] >= threshold])
                    APs.append(self.average_precision(ground_rank, new_predict_rank, n_k))
                    n_ks.append([n_k, len(ground_)])

                #print n_ks
                #print APs
                all_aps.append([APs])

                #Precision2. reweighted
                rank_difference = 0
                ground_n = {}
                for i in ground_rank:
                    if ground_rank[i] not in ground_n:
                        ground_n[ground_rank[i]] = 1
                    else:
                        ground_n[ground_rank[i]] += 1
                for i in xrange(len(ground_rank)):
                    p = new_predict_rank[i]
                    g = ground_rank[i]
                    n_same_g =ground_n[g]

                    if p >= g and p < g + n_same_g:
                        continue
                    if p < g:
                        rank_difference += float(g - p) / g
                    else:
                        rank_difference += float(p - g - n_same_g + 1) / g
                all_reweighted.append(rank_difference)

                #Precision3: mean precision

                for min_retrieval in min_retrievals:
                    threshold = ground_[max(1, len(ground_)*min_retrieval/100)-1][1]
                    #print len(ground_)*min_retrieval/100
                    n_k = len([i for i in ground_ if i[1] >= threshold])
                    dict_wanted = set()
                    for i in xrange(n_k):
                        dict_wanted.add(ground_[i][0])
                    retrieved_count = 0
                    for j in xrange(n_k):
                        if predict_[j][0] in dict_wanted:
                            retrieved_count += 1
                    Ps.append([retrieved_count,n_k])

                all_ps.append(Ps)
            all_n_ks.append(n_ks)
            #print '\n'
        return all_n_ks, all_aps, all_reweighted, all_ps

    def baseline_evaluation(self, permute = True,
                            model_names = [], evaluate_worker = True, worker_times = global_permutation_time, abandon_overlap = False):
            if permute:
                self.permute_groundtruth()
            retrievals = [[] for i in model_names]; percent = []; precision = [[] for i in model_names]
            for i in xrange(6, 45, 5):
            # for i in xrange(11, 12, 5):
                temp, percent_temp, AP_temp = self.evaluate_MAP_permuted(model_names, min_retrieval=i, abandon_overlap = abandon_overlap)
                for j in xrange(len(temp)):
                    retrievals[j].append(temp[j])
                percent.append(percent_temp)
            for i in xrange(6, 45, 5):
                temp = self.evaluate_top20_permuted(model_names, percent=i, permuted='_permuted', abandon_overlap = abandon_overlap)
                for j in xrange(len(temp)):
                    precision[j].append(temp[j])
            all_aps=[];all_reweighted=[];all_ps=[]
            # if not evaluate_worker:
            return percent, retrievals , precision, [], []

            #
            # for i in xrange(worker_times):
            #     all_nks, temp2, temp3, temp4 = self.amt_worker_result_predict_average(event_name,  permuted='_permuted', abandon_overlap = abandon_overlap)
            #     all_aps.append([temp2]); all_reweighted.append([temp3]); all_ps.append([temp4])
            # all_aps_average = copy.deepcopy(all_aps[0])
            # for i in xrange(1, worker_times):
            #     for j in xrange(len(all_aps[i])):
            #         for k in xrange(len(all_aps[i][j])):
            #             for l in xrange(len(all_aps[i][j][k])):
            #                 for m in xrange(len(all_aps[i][j][k][l])):
            #                     all_aps_average[j][k][l][m] += all_aps[i][j][k][l][m]
            # for j in xrange(len(all_aps_average)):
            #         for k in xrange(len(all_aps[i][j])):
            #             for l in xrange(len(all_aps[i][j][k])):
            #                 for m in xrange(len(all_aps[i][j][k][l])):
            #                     all_aps_average[j][k][l][m] = float(all_aps_average[j][k][l][m])/worker_times
            #
            # all_ps_average = copy.deepcopy(all_ps[0])
            # for i in xrange(1, worker_times):
            #     for j in xrange(len(all_ps[i])):
            #         for k in xrange(len(all_ps[i][j])):
            #             for l in xrange(len(all_ps[i][j][k])):
            #                 for m in xrange(len(all_ps[i][j][k][l])):
            #                     all_ps_average[j][k][l][m] += all_ps[i][j][k][l][m]
            # for j in xrange(len(all_ps_average)):
            #         for k in xrange(len(all_ps[i][j])):
            #             for l in xrange(len(all_ps[i][j][k])):
            #                 for m in xrange(len(all_ps[i][j][k][l])):
            #                     all_ps_average[j][k][l][m] = float(all_ps_average[j][k][l][m])/worker_times
            #
            # all_reweighted_average = copy.deepcopy(all_reweighted[0])
            # for i in xrange(1, worker_times):
            #     for j in xrange(len(all_reweighted[i])):
            #         for k in xrange(len(all_reweighted[i][j])):
            #                     all_reweighted_average[j][k] += all_reweighted[i][j][k]
            # for j in xrange(len(all_reweighted_average)):
            #         for k in xrange(len(all_reweighted[i][j])):
            #                     all_reweighted_average[j][k] = float(all_reweighted_average[j][k])/worker_times
            #
            # all_aps = all_aps_average[0]
            # all_ps = all_ps_average[0]
            # mean_aps = []
            # for i in all_aps:
            #     for j in i:
            #         mean_aps.append(j)
            # mean_aps = np.mean(mean_aps, axis=0)
            # mean_ps1 = np.zeros(11);mean_ps2 = np.zeros(11)
            # for i in xrange(len(all_ps)):
            #     for j in xrange(len(all_ps[i])):
            #         mean_ps1[j] += all_ps[i][j][0]
            #         mean_ps2[j] += all_ps[i][j][1]
            # mean_ps = [mean_ps1[i]/mean_ps2[i] for i in xrange(len(mean_ps1))]
            # return percent, retrievals , precision, mean_aps, mean_ps

    def create_random_dict(self, validation_name, save_name):
        prediction_dict = {}
        for event_name in dict_name2:
            path = root+'baseline_all_0509/' + event_name+ '/'+validation_name+'_image_ids.cPickle'
            f = open(path, 'r')
            all_event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name+ '/'+validation_name+'_ulr_dict.cPickle', 'r')
            test_url_dict = cPickle.load(f)
            f.close()
            score_random = [random.uniform(0,1) for i in xrange(len(all_event_ids))]
            # f = open(root + baseline_name + event_name+ '/' +validation_name+ '_'+save_name.split('.')[0][:-5]+'.cPickle','wb')
            # cPickle.dump(score_random, f)
            # f.close()
            #

            for score, name_ in zip(score_random, all_event_ids):
                event_name_ = name_.split('/')[0]
                if event_name_ in prediction_dict:
                    prediction_dict[event_name_] += [[name_, test_url_dict[name_], score]]
                else:
                    prediction_dict[event_name_] = [[name_, test_url_dict[name_], score]]

        f = open(root + baseline_name + '/'+validation_name+ '_' + save_name + '.cPickle','wb')
        cPickle.dump(prediction_dict, f)
        f.close()

    def create_predict_dict_from_cpickle_multevent(self, prediction_path):

        feature = np.load(root + ('_').join(prediction_path.split('_')[:-1]) + '.npy')

        event_list_path = root + ('_').join(prediction_path.split('_')[:-1]) + '_event_list.pkl'
        f = open(event_list_path)
        event_list = cPickle.load(f)
        f.close()

        with open(root + '0208_correction/all_input_and_result/new_multiple_result_2round_removedup_vote.pkl') as f:
            event_type_dict = cPickle.load(f)

        with open(root + 'baseline_all_correction_multi_old/event_img_dict.pkl') as f:
            event_img_dict = cPickle.load(f)
        with open(root + 'baseline_all_correction_multi_old/test_ulr_dict.pkl') as f:
            test_url_dict = cPickle.load(f)
        count = 0
        prediction_dict = defaultdict(list)
        for event in event_list:
            event_type = [dict_name2[i[0]] - 1 for i in event_type_dict[event]]
            for img in event_img_dict[event]:
                try:
                    prediction_dict[event] += [[img, test_url_dict[img], np.mean(feature[count, event_type])]]
                except:
                    prediction_dict[event] += [[img, test_url_dict[img], feature[count]]]
                count += 1


        f = open(root + prediction_path,'wb')
        cPickle.dump(prediction_dict, f)
        f.close()


    def create_predict_dict_from_cpickle_multevent_new(self, prediction_path):

        feature_dict = cPickle.load(open(root + prediction_path[:-9] + '.pkl'))
        # with open(root + 'baseline_all_correction_multi_old/new_multiple_result.pkl') as f:
        #     event_type_dict = cPickle.load(f)

        with open(root + 'baseline_all_correction_multi_old/event_img_dict.pkl') as f:
            event_img_dict = cPickle.load(f)
        with open(root + 'baseline_all_correction_multi_old/test_ulr_dict.pkl') as f:
            test_url_dict = cPickle.load(f)
        prediction_dict = defaultdict(list)
        for event in feature_dict:
            count = 0
            for img in event_img_dict[event]:
                # print img
                prediction_dict[event] += [[img, test_url_dict[img], feature_dict[event][count]]]
                count += 1
        f = open(root + prediction_path,'wb')
        cPickle.dump(prediction_dict, f)
        f.close()


    def evaluate_present_with_worker_nooverlap(self, abandon_overlap = True):
        print 'HERE!!!'
        model_names = [
            # 'baseline_all_correction_multi/test_random_dict.cPickle'
            # ,'CNN_all_event_corrected_multi_old/features/test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_dict.pkl'
            # ,'CNN_all_event_corrected_multi/features/test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_dict.pkl'
            #
            # ,'CNN_all_event_corrected_multi_old/features/test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000_dict.pkl'
            # ,'CNN_all_event_corrected_multi_old/features/test_sigmoid9_23_segment_twoloss_fc500_diffweight_2_iter_100000_dict.pkl'
            # ,'CNN_all_event_corrected_multi_old/features/IMPORTANCE_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000_test_predict_multilabel_event_recognition_expand_balanced_3_iter_100000_em_1_dict.pkl'
            # ,'CNN_all_event_corrected_multi_old/features/IMPORTANCE_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000_test_predict_multilabel_event_recognition_expand_balanced_3_iter_100000_em_9_dict.pkl'
            # ,'CNN_all_event_corrected_multi/features/test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_dict.pkl'
            # ,'CNN_all_event_corrected_multi/features/IMPORTANCE_test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_test_predict_multilabel_event_recognition_expand_balanced_4_iter_100000_em_1_dict.pkl'
            # ,'CNN_all_event_corrected_multi/features/IMPORTANCE_test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_test_predict_multilabel_event_recognition_expand_balanced_4_iter_100000_em_9_dict.pkl'
            #  ,'CNN_all_event_corrected_multi/features_validation/EM1_importance_cross_validation_combine_best_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/SOFT_THRESHOLDM_POLY2_importance_cross_validation_combine_best_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/SOFT_2THRESHOLDM_POLY2_2_importance_cross_validation_combine_best_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/SOFT_EM1_importance_cross_validation_combine_best_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_importance_cross_validation_combine_best_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/test_sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_2time_importance_cross_validation_combine_best_dict.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/test_sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000_dict.pkl'
             'CNN_all_event_corrected_multi/features_validation/noevent_importance.pkl'
             # ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter0_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter1_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter2_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter3_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter4_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter5_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter6_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter7_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter8_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter9_importance_cross_validation_combine_best_dict.pkl'

             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter0_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter1_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter2_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter3_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter4_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter5_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter6_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter7_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter8_importance_cross_validation_combine_best_dict.pkl'
             ,'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_SOFT_iter9_importance_cross_validation_combine_best_dict.pkl'

             # ,'CNN_all_event_corrected_multi/features/IMPORTANCE_LSTM_test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000_test_predict_removedup_multilabel_event_recognition_expand_balanced_3_iter_100000_em_29_dict.pkl'


        ]
        # temp = ['CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter1_importance_cross_validation_combine_best_dict',
        #         'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter7_importance_cross_validation_combine_best_dict']
        # model_names = []
        # for event_type in dict_name2:
        #     for i in temp:
        #         model_names.append(i+'_' + event_type+'.pkl')

        # model_names = []
        # # bests = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 22), (0, 23), (0, 24), (0, 25), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (2, 1), (2, 2), (2, 3), (2, 4), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (3, 1), (3, 2), (3, 3), (3, 4), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (4, 1), (4, 2), (4, 3), (4, 4), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 21), (6, 22), (6, 23), (6, 24), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 21), (7, 22), (7, 23), (7, 24), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30)]
        # bests = [(10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26)]
        # bests = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10)]
        # for i,j in bests:
        #     model_names.append('CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_VALIDATION_poly_'+str(j)+'_threshold_'+str(i)+'_importance_cross_validation_combine_best_dict.pkl')

        permutation_times = global_permutation_time
        worker_permutation_times = 1
        retrieval_models = []
        precision_models = []
        for i in model_names:
            retrieval_models.append([])
            precision_models.append([])
        retrieval_worker_all = []
        precision_worker_all = []
        len_all = 0
        self.create_random_dict(self.validation_name, 'random_dict')

        for model_name_this in model_names:
            if model_name_this.endswith('_best_dict.pkl'):
                self.create_predict_dict_from_cpickle_multevent_new(model_name_this)
            elif model_name_this.endswith('_dict.pkl'):
                self.create_predict_dict_from_cpickle_multevent(model_name_this)
        percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
        for i in xrange(permutation_times):
                    percent , retrievals,  precision, _, _ = self.baseline_evaluation(True, model_names, evaluate_worker=False,  worker_times=worker_permutation_times, abandon_overlap=abandon_overlap)
                    percent_all.append(percent)
                    precision_model.append(precision)
                    retrievals_model.append(retrievals)
        percent_average = []; retrievals_model_average = []; precision_model_average = []; #retrievals_worker_average = []; precision_worker_average = []
        for j in xrange(len(retrievals_model[0])):
                retrievals_model_average.append([])
                precision_model_average.append([])
        for i in xrange(len(percent_all[0])):
                percent_average.append(sum(j[i] for j in percent_all)/permutation_times)
                # retrievals_worker_average.append(sum(j[i] for j in retrievals_worker)/permutation_times)
                # precision_worker_average.append(sum(j[i] for j in precision_worker)/permutation_times)
                for j in xrange(len(retrievals_model_average)):
                    retrievals_model_average[j].append(sum(k[j][i] for k in retrievals_model)/permutation_times)
                    precision_model_average[j].append(sum(k[j][i] for k in precision_model)/permutation_times)

        f = open(root + model_names[0],'r')
        temp = cPickle.load(f)
        f.close()
        len_ = len(temp)
        len_all += len_

        # print '>>>>>>>>' + event_name + '<<<<<<<<'
        # print '*ALGORITHM*'
        print 'P:', ', '.join(["%.5f" % v for v in percent_average])
        for i in xrange(len(model_names)):
            # print model_names[i]
            # print ', '.join(["%.3f" % v for v in precision_model_average[i]])
            # print ', '.join(["%.3f" % v for v in retrievals_model_average[i]])
            retrieval_models[i].append([j*len_ for j in retrievals_model_average[i]])
            precision_models[i].append([j*len_ for j in precision_model_average[i]])
        print '*********************************'
        print '*********************************'
        for i in xrange(len(precision_models)):
            print model_names[i]
            #print retrieval_models[i]
            temp = np.array(precision_models[i])
            temp1 = np.sum(temp, axis=0)
            print [j/len_all for j in temp1]
        for i in xrange(len(retrieval_models)):
            print model_names[i]
            #print retrieval_models[i]
            temp = np.array(retrieval_models[i])
            temp1 = np.sum(temp, axis=0)
            print [j/len_all for j in temp1]
            print np.sum([j/len_all for j in temp1])
        #
        # print 'Worker'
        # #print retrieval_worker_all
        # temp = np.array(precision_worker_all)
        # temp1 = np.sum(temp, axis=0)
        # print [i/len_all for i in temp1]
        #
        # temp = np.array(retrieval_worker_all)
        # temp1 = np.sum(temp, axis=0)
        # print [i/len_all for i in temp1]



def extract_feature_10_recognition_traintest_multilabel_pec(img_file, name, net_path, model_name_, net_name, img_size):
    event_prediction_dict = defaultdict(list)
    imgs = []
    with open(img_file, 'r') as data:
        for line in data:
            imgs.append(line.split(' ')[0])
    model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/training/' + model_name_
    weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/snapshot/' + net_name + '.caffemodel'
    mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe.set_device(1)
    caffe.set_mode_gpu()
    img_dims = img_size
    raw_scale = 255
    channel_swap = (2,1,0)
    net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

    # with open('/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+event_name+'/'+name+'_image_ids.cPickle') as f:
    #         event_img_list = cPickle.load(f)
    count = 0
    for img in imgs:
            event_this = '/'.join(img.split('/')[-3:-1])
            # event_this = event_img.split('/')[0]
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)
            event_prediction_dict[event_this].append(out[0])
            if count % 100 == 0:
                print count, out
            count += 1
            # if count == 100:
            #     break
    # f = open('/home/feiyu1990/local/event_curation/'+net_path+'/features/test_predict_'+ net_name + '_dict.pkl','wb')
    # cPickle.dump(event_prediction_dict, f)
    # f.close()

    event_list = []; feature = []
    for i in event_prediction_dict:
        event_list.append(i)
        feature.extend(event_prediction_dict[i])
    feature = np.array(feature)
    f = open('/'.join(file_path.split('/')[:-1]) + '/features/'+name+'_predict_'+ net_name + '_event_list.pkl','wb')
    cPickle.dump(event_list, f)
    f.close()
    np.save('/'.join(file_path.split('/')[:-1]) + '/features/'+name+'_predict_'+ net_name + '.npy', feature)

def extract_feature_10_23_traintest_multilabel_pec(img_file, name, net_path, model_name_, net_name, img_size):
    importance_prediction_dict = defaultdict(list)
    imgs = []
    with open(img_file, 'r') as data:
        for line in data:
            imgs.append(line.split(' ')[0])
    model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/' + model_name_
    weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/snapshot/' + net_name + '.caffemodel'
    mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe.set_device(1)
    caffe.set_mode_gpu()
    img_dims = img_size
    raw_scale = 255
    channel_swap = (2,1,0)
    net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

    count = 0
    for img in imgs:
            event_this = '/'.join(img.split('/')[-3:-1])
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)
            importance_prediction_dict[event_this].append(out[0])
            count += 1
            if count % 100 == 0:
                print out

    event_list = []; feature = []
    for i in importance_prediction_dict:
        event_list.append(i)
        feature.extend(importance_prediction_dict[i])
    feature = np.array(feature)
    f = open('/'.join(img_file.split('/')[:-1]) + '/features/pec_'+name+'_sigmoid9_23_'+ net_name + '_event_list.pkl','wb')
    cPickle.dump(event_list, f)
    f.close()
    np.save('/'.join(img_file.split('/')[:-1]) + '/features/pec_'+name+'_sigmoid9_23_'+ net_name + '.npy', feature)

def extract_feature_10_recognition_traintest_multilabel_pec_fc7(img_file, name, net_path, model_name_, net_name, img_size=227):
    event_prediction_dict = defaultdict(list)
    imgs = []
    with open(img_file, 'r') as data:
        for line in data:
            imgs.append(line.split(' ')[0])
    model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/training/' + model_name_
    weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/snapshot/' + net_name + '.caffemodel'
    mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(model_name, weight_name, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    transformer.set_mean('data', mean_file.mean(1).mean(1))
    net.blobs['data'].reshape(1,3,img_size, img_size)
    count = 0
    for img in imgs:
            event_this = '/'.join(img.split('/')[-3:-1])
            count += 1
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            net.forward()
            a = net.blobs['fc7'].data.copy()
            if count % 100 == 0:
                print count,  a[0]
            event_prediction_dict[event_this].append(a[0])

    event_list = []; feature = []
    for i in event_prediction_dict:
        event_list.append(i)
        feature.extend(event_prediction_dict[i])
    feature = np.array(feature)
    f = open('/'.join(img_file.split('/')[:-1]) + '/features/'+name+'_fc7_'+ net_name + '_event_list.pkl','wb')
    cPickle.dump(event_list, f)
    f.close()
    np.save('/'.join(img_file.split('/')[:-1]) + '/features/'+name+'_fc7_'+ net_name + '.npy', feature)
    f = open('/'.join(img_file.split('/')[:-1]) + '/features/'+name+'_fc7_'+ net_name + '_feature_dict.pkl','wb')
    cPickle.dump(event_prediction_dict, f)
    f.close()


def extract_feature_10_recognition_traintest_multilabel(name, net_path, model_name_, net_name, img_size):
    event_prediction_dict = defaultdict(list)
    for event_name in ['multi_label'] + dict_name2.keys():
        print event_name
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+event_name+'/guru_'+name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/training/' + model_name_
        weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/snapshot/' + net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        with open('/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+event_name+'/'+name+'_image_ids.cPickle') as f:
            event_img_list = cPickle.load(f)
        count = 0
        for img, event_img in zip(imgs, event_img_list):
            event_this = event_img.split('/')[0]
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)
            event_prediction_dict[event_this].append(out[0])
            if count % 100 == 0:
                try:
                    print dict_name2[event_name], np.argmax(out[0])
                except:
                    continue
            count += 1
    # f = open('/home/feiyu1990/local/event_curation/'+net_path+'/features/test_predict_'+ net_name + '_dict.pkl','wb')
    # cPickle.dump(event_prediction_dict, f)
    # f.close()

    event_list = []; feature = []
    for i in event_prediction_dict:
        event_list.append(i)
        feature.extend(event_prediction_dict[i])
    feature = np.array(feature)
    f = open('/home/feiyu1990/local/event_curation/'+net_path+'/features/'+name+'_predict_'+ net_name + '_event_list.pkl','wb')
    cPickle.dump(event_list, f)
    f.close()
    np.save('/home/feiyu1990/local/event_curation/'+net_path+'/features/'+name+'_predict_'+ net_name + '.npy', feature)

def extract_feature_10_recognition_traintest_multilabel_fc7(name, net_path, model_name_, net_name, img_size=227):
    event_prediction_dict = defaultdict(list)
    img_files = []
    for event_name in dict_name2.keys():
            print event_name
            imgs = []
            img_files.append('/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name+'/guru_'+name+'_path.txt')


    for img_file in img_files:
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/training/' + model_name_
        weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/snapshot/' + net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()

        net = caffe.Net(model_name, weight_name, caffe.TEST)
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name+'/'+name+'_image_ids.cPickle') as f:
            event_img_list = cPickle.load(f)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        transformer.set_mean('data', mean_file.mean(1).mean(1))
        net.blobs['data'].reshape(1,3,img_size, img_size)
        count = 0
        for img,event_img in zip(imgs, event_img_list):
            event_this = event_img.split('/')[0]
            count += 1
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            net.forward()
            a = net.blobs['fc7'].data.copy()
            if count % 100 == 0:
                print a[0]
            event_prediction_dict[event_this].append(a[0])

    event_list = []; feature = []
    for i in event_prediction_dict:
        event_list.append(i)
        feature.extend(event_prediction_dict[i])
    feature = np.array(feature)
    f = open('/home/feiyu1990/local/event_curation/'+net_path+'/features/'+name+'_fc7_'+ net_name + '_event_list.pkl','wb')
    cPickle.dump(event_list, f)
    f.close()
    np.save('/home/feiyu1990/local/event_curation/'+net_path+'/features/'+name+'_fc7_'+ net_name + '.npy', feature)

def extract_feature_10_23_traintest_multilabel(name, net_path, model_name_, net_name, img_size):
    importance_prediction_dict = defaultdict(list)
    for event_name in dict_name2.keys():
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name+'/guru_'+name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/' + model_name_
        weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/snapshot/' + net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        img_dims = img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        count = 0
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name+'/'+name+'_image_ids.cPickle') as f:
            event_img_list = cPickle.load(f)
        for img, event_img in zip(imgs, event_img_list):
            event_this = event_img.split('/')[0]
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)
            importance_prediction_dict[event_this].append(out[0])
            count += 1
            if count % 100 == 0:
                print event_name, out

    event_list = []; feature = []
    for i in importance_prediction_dict:
        event_list.append(i)
        feature.extend(importance_prediction_dict[i])
    feature = np.array(feature)
    f = open('/home/feiyu1990/local/event_curation/'+net_path+'/features/'+name+'_sigmoid9_23_'+ net_name + '_event_list.pkl','wb')
    cPickle.dump(event_list, f)
    f.close()
    np.save('/home/feiyu1990/local/event_curation/'+net_path+'/features/'+name+'_sigmoid9_23_'+ net_name + '.npy', feature)

def em_combine_event_recognition_curation_corrected(
        threshold, threshold_m, poly = 1,poly2 = 1,
        importance_path='test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000',
        event_path = 'test_predict_removedup_multilabel_event_recognition_expand_balanced_3_iter_100000',
        folder_name = 'CNN_all_event_corrected_multi',
        baseline_name = 'baseline_all_correction_removedup_vote',
        stop_criterion = 0.01, max_iter = 101,

        # importance_path='test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000',
        # event_path = 'test_predict_multilabel_event_recognition_expand_balanced_3_iter_100000',
        # folder_name = 'CNN_all_event_corrected_multi_old',
        # baseline_name = 'baseline_all_correction_multi_old'

):
    accuracy_events = defaultdict(list)
    event_lengths = dict()

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
    # with open(root + 'baseline_all_correction_multi/event_img_dict.pkl','w') as f:
    #     cPickle.dump(event_img_dict, f)


    with open(root + folder_name+'/features/' + event_path + '_event_list.pkl') as f:
        event_recognition_all_list = cPickle.load(f)
    count = 0
    event_recognition_all_event_dict = defaultdict(list)
    for event in event_recognition_all_list:
        for img in event_img_dict[event]:
            event_recognition_all_event_dict[event].append(count)
            count += 1
    event_recognition_all = np.load(root + folder_name+'/features/' + event_path + '.npy')


    with open(root + folder_name+'/features/' + importance_path + '_event_list.pkl') as f:
        importance_feature_all_list = cPickle.load(f)
    count = 0
    importance_feature_all_event_dict = defaultdict(list)
    for event in importance_feature_all_list:
        for img in event_img_dict[event]:
            importance_feature_all_event_dict[event].append(count)
            count += 1
    importance_feature_all = np.load(root + folder_name+'/features/' + importance_path + '.npy')


    event_recognition_all_event = dict()
    img_importance_all_event = defaultdict(list)
    # threshod_m_start = threshold_m

    for event_type in ['multi_label'] + dict_name2.keys():
        # threshold_m = threshod_m_start
        # print event_type
        with open('/home/feiyu1990/local/event_curation/'+baseline_name+'/' + event_type + '/test_event_id.cPickle') as f:
            event_id_list = cPickle.load(f)
        if len(event_id_list) == 0:
            continue
        recognition = [];importance_feature = []
        if event_type == 'multi_label':
            for id in event_id_list:
                event_name = id[0]
                for i in event_recognition_all_event_dict[event_name]:
                    recognition.append(event_recognition_all[i])
                for i in importance_feature_all_event_dict[event_name]:
                    importance_feature.append(importance_feature_all[i])
        else:
            for id in event_id_list:
                event_name = id
                # print event_name
                for i in event_recognition_all_event_dict[event_name]:
                    recognition.append(event_recognition_all[i])
                for i in importance_feature_all_event_dict[event_name]:
                    importance_feature.append(importance_feature_all[i])

        path = '/home/feiyu1990/local/event_curation/'+baseline_name+'/' + event_type+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()

        importance_scores = []

        # # #initial importance score prediction
        # event_recognition_ini = dict()
        # for i in all_img_ids:
        #     event_id = i.split('/')[0]
        #     if event_id not in event_recognition_ini:
        #         event_recognition_ini[event_id] = np.ones((23,), dtype=float)/23
        # importance_score = m_step(importance_feature, event_recognition_ini, all_img_ids, threshold_m)
        # importance_scores.append(importance_score)
        # diff = np.Inf
        # event_recognition = e_step(importance_score, recognition, all_img_ids, threshold)
        # importance_score = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
        #

        # #initialization of \theta
        importance_ini = np.ones((len(recognition),))
        event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold, poly, poly2)
        event_lengths[event_type] = len(event_recognition)
        accuracy_this = []
        for event_id in event_id_list:
            if event_type == 'multi_label':
                predict_ = np.argmax(event_recognition[event_id[0]])
                ground_ = [dict_name2[i[0]] - 1 for i in event_id[1]]
                accuracy_this.append(predict_ in ground_)
            else:
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_type] - 1)
        accuracy_events[event_type].append(np.sum(accuracy_this))
        importance_score = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
        iter = 0
        # diff = np.Inf
        diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
        while diff > stop_criterion: #* len(importance_score):
            iter += 1
            if iter >= max_iter:
                if max_iter > 1:
                    # print [(i+j)/2 for i,j in zip(importance_scores[-1], importance_scores[-2])]
                    event_recognition = e_step([(i+j+k+q)/4 for i,j,k,q in zip(importance_scores[-1], importance_scores[-2], importance_scores[-3], importance_scores[-4])], recognition, all_img_ids, threshold, poly, poly2)
                    accuracy_this = []
                    for event_id in event_id_list:
                        if event_type == 'multi_label':
                            predict_ = np.argmax(event_recognition[event_id[0]])
                            ground_ = [dict_name2[i[0]] - 1 for i in event_id[1]]
                            accuracy_this.append(predict_ in ground_)
                        else:
                            accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_type] - 1)
                    accuracy_events[event_type].append(np.sum(accuracy_this))
                break
            # threshold_m = threshold_m * (1 -  (float(iter) / max_iter)**2)
            # print threshold_m
            # threshold_m = (float(iter) / max_iter) ** 0.5
            # poly_this = (float(iter) / max_iter) * poly
            event_recognition = e_step(importance_score, recognition, all_img_ids, threshold, poly, poly2)
            # print len(event_recognition), len(event_recognition[event_recognition.keys()[0]])
            accuracy_this = []
            for event_id in event_id_list:
                if event_type == 'multi_label':
                    predict_ = np.argmax(event_recognition[event_id[0]])
                    ground_ = [dict_name2[i[0]] - 1 for i in event_id[1]]
                    accuracy_this.append(predict_ in ground_)
                else:
                    accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_type] - 1)
            accuracy_events[event_type].append(np.sum(accuracy_this))
            importance_score_new = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
            importance_scores.append(importance_score_new)
            diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
            importance_score = importance_score_new


        # # add one time of m-step, to include for all events to predict img_score
        if max_iter > 1:
            importance_score = m_step(importance_feature, event_recognition, all_img_ids, 0)

        for temp in event_recognition:
            event_recognition_all_event[temp] = event_recognition[temp]
        for img_id, importance in zip(all_img_ids, importance_score):
            event_id = img_id.split('/')[0]
            img_importance_all_event[event_id].append(importance)



    f = open(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.pkl', 'w')
    cPickle.dump(img_importance_all_event, f)
    f.close()

    importance_npy = []
    with open(root + folder_name+'/features/'  + importance_path + '_event_list.pkl') as f:
        event_list = cPickle.load(f)
    for event in event_list:
        importance_npy.extend(img_importance_all_event[event])
    np.save(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.npy', importance_npy)
    with open(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'_event_list.pkl','w') as f:
        cPickle.dump(event_list, f)

    f = open(root + folder_name+'/features/RECOGNITION_'  + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.pkl', 'w')
    cPickle.dump(event_recognition_all_event, f)
    f.close()

    print accuracy_events
    accuracy_all = float(np.sum([accuracy_events[event_type][-1] for event_type in accuracy_events])) \
                   / np.sum([event_lengths[event_type] for event_type in event_lengths])
    print accuracy_all


def final_img_importance(
        threshold, threshold_m, poly = 1,
        combine_path = 'CNN_all_event_corrected_multi/features/temp.pkl',
        importance_path='test_sigmoid9_23_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000',
        event_path = 'test_predict_removedup_multilabel_event_recognition_expand_balanced_3_iter_100000',
        folder_name = 'CNN_all_event_corrected_multi',
        baseline_name = 'baseline_all_correction_removedup_vote',max_iter = 101,
):
    img_importance_all_event = defaultdict(list)
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


    with open(root + folder_name+'/features/' + event_path + '_event_list.pkl') as f:
        event_recognition_all_list = cPickle.load(f)
    count = 0
    event_recognition_all_event_dict = defaultdict(list)
    for event in event_recognition_all_list:
        for img in event_img_dict[event]:
            event_recognition_all_event_dict[event].append(count)
            count += 1
    event_recognition_all = np.load(root + folder_name+'/features/' + event_path + '.npy')


    with open(root + folder_name+'/features/' + importance_path + '_event_list.pkl') as f:
        importance_feature_all_list = cPickle.load(f)
    count = 0
    importance_feature_all_event_dict = defaultdict(list)
    for event in importance_feature_all_list:
        for img in event_img_dict[event]:
            importance_feature_all_event_dict[event].append(count)
            count += 1
    importance_feature_all = np.load(root + folder_name+'/features/' + importance_path + '.npy')

    with open(root + combine_path) as f:
        combine_recognition = cPickle.load(f)

    for event_type in ['multi_label'] + dict_name2.keys():
        # threshold_m = threshod_m_start
        # print event_type
        with open('/home/feiyu1990/local/event_curation/'+baseline_name+'/' + event_type + '/test_event_id.cPickle') as f:
            event_id_list = cPickle.load(f)
        if len(event_id_list) == 0:
            continue
        recognition = [];importance_feature = []
        if event_type == 'multi_label':
            for id in event_id_list:
                event_name = id[0]
                for i in event_recognition_all_event_dict[event_name]:
                    recognition.append(event_recognition_all[i])
                for i in importance_feature_all_event_dict[event_name]:
                    importance_feature.append(importance_feature_all[i])
        else:
            for id in event_id_list:
                event_name = id
                # print event_name
                for i in event_recognition_all_event_dict[event_name]:
                    recognition.append(event_recognition_all[i])
                for i in importance_feature_all_event_dict[event_name]:
                    importance_feature.append(importance_feature_all[i])

        path = '/home/feiyu1990/local/event_curation/'+baseline_name+'/' + event_type+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()

        importance_after_lstm = m_step(importance_feature, combine_recognition, all_img_ids, 0)

        for img_id, importance in zip(all_img_ids, importance_after_lstm):
            event_id = img_id.split('/')[0]
            img_importance_all_event[event_id].append(importance)

    f = open(root + folder_name+'/features/IMPORTANCE_LSTM_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.pkl', 'w')
    cPickle.dump(img_importance_all_event, f)
    f.close()

    temp = np.load(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.npy')
    np.save(root + folder_name+'/features/IMPORTANCE_LSTM_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.npy', temp)

    with open(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'_event_list.pkl') as f:
        event_list = cPickle.load(f)
    with open(root + folder_name+'/features/IMPORTANCE_LSTM_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'_event_list.pkl', 'w') as f:
        cPickle.dump(event_list, f)

def em_combine_event_recognition_curation_corrected_involve_lstm(
        threshold, lstm_dict, threshold_m, poly = 1,
        importance_path='test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000',
        event_path = 'test_predict_removedup_multilabel_event_recognition_expand_balanced_3_iter_100000',
        folder_name = 'CNN_all_event_corrected_multi',
        baseline_name = 'baseline_all_correction_removedup_vote',
        stop_criterion = 0.01, max_iter = 101,


        # importance_path='test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000',
        # event_path = 'test_predict_multilabel_event_recognition_expand_balanced_3_iter_100000',
        # folder_name = 'CNN_all_event_corrected_multi_old',
        # baseline_name = 'baseline_all_correction_multi_old'

):
    accuracy_events = defaultdict(list)
    event_lengths = dict()

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
    # with open(root + 'baseline_all_correction_multi/event_img_dict.pkl','w') as f:
    #     cPickle.dump(event_img_dict, f)


    with open(root + folder_name+'/features/' + event_path + '_event_list.pkl') as f:
        event_recognition_all_list = cPickle.load(f)
    count = 0
    event_recognition_all_event_dict = defaultdict(list)
    for event in event_recognition_all_list:
        for img in event_img_dict[event]:
            event_recognition_all_event_dict[event].append(count)
            count += 1
    event_recognition_all = np.load(root + folder_name+'/features/' + event_path + '.npy')


    with open(root + folder_name+'/features/' + importance_path + '_event_list.pkl') as f:
        importance_feature_all_list = cPickle.load(f)
    count = 0
    importance_feature_all_event_dict = defaultdict(list)
    for event in importance_feature_all_list:
        for img in event_img_dict[event]:
            importance_feature_all_event_dict[event].append(count)
            count += 1
    importance_feature_all = np.load(root + folder_name+'/features/' + importance_path + '.npy')


    event_recognition_all_event = dict()
    img_importance_all_event = defaultdict(list)
    # threshod_m_start = threshold_m

    for event_type in ['multi_label'] + dict_name2.keys():
        # threshold_m = threshod_m_start
        # print event_type
        with open('/home/feiyu1990/local/event_curation/'+baseline_name+'/' + event_type + '/test_event_id.cPickle') as f:
            event_id_list = cPickle.load(f)
        if len(event_id_list) == 0:
            continue
        recognition = [];importance_feature = []
        if event_type == 'multi_label':
            for id in event_id_list:
                event_name = id[0]
                for i in event_recognition_all_event_dict[event_name]:
                    recognition.append(event_recognition_all[i])
                for i in importance_feature_all_event_dict[event_name]:
                    importance_feature.append(importance_feature_all[i])
        else:
            for id in event_id_list:
                event_name = id
                # print event_name
                for i in event_recognition_all_event_dict[event_name]:
                    recognition.append(event_recognition_all[i])
                for i in importance_feature_all_event_dict[event_name]:
                    importance_feature.append(importance_feature_all[i])

        path = '/home/feiyu1990/local/event_curation/'+baseline_name+'/' + event_type+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()


        # # # #initial importance score prediction
        # event_recognition_ini = dict()
        # for i in all_img_ids:
        #     event_id = i.split('/')[0]
        #     if event_id not in event_recognition_ini:
        #         event_recognition_ini[event_id] = np.ones((23,), dtype=float)/23
        # importance_score = m_step(importance_feature, event_recognition_ini, all_img_ids, threshold_m)
        # # print importance_score
        # diff = np.Inf
        # event_recognition = e_step(importance_score, recognition, all_img_ids, threshold)
        # importance_score = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
        #

        importance_scores = []
        # #initialization of \theta
        importance_ini = np.ones((len(recognition),))
        # print len(recognition)
        event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold, poly)
        # print len(event_recognition), event_recognition.keys()
        event_lengths[event_type] = len(event_recognition)
        accuracy_this = []
        for event_id in event_id_list:
            if event_type == 'multi_label':
                predict_ = np.argmax(event_recognition[event_id[0]])
                ground_ = [dict_name2[i[0]] - 1 for i in event_id[1]]
                accuracy_this.append(predict_ in ground_)
            else:
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_type] - 1)
        accuracy_events[event_type].append(np.sum(accuracy_this))

        importance_score = m_step_involve_lstm(lstm_dict, importance_feature, event_recognition, all_img_ids, threshold_m)
        importance_scores.append(importance_score)
        iter = 0

        diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
        while diff > stop_criterion: #* len(importance_score):
            iter += 1
            if iter >= max_iter:
                if max_iter > 1:
                    # print [(i+j)/2 for i,j in zip(importance_scores[-1], importance_scores[-2])]
                    event_recognition = e_step([(i+j+k+q)/4 for i,j,k,q in zip(importance_scores[-1], importance_scores[-2], importance_scores[-3], importance_scores[-4])], recognition, all_img_ids, threshold, poly)
                    accuracy_this = []
                    for event_id in event_id_list:
                        if event_type == 'multi_label':
                            predict_ = np.argmax(event_recognition[event_id[0]])
                            ground_ = [dict_name2[i[0]] - 1 for i in event_id[1]]
                            accuracy_this.append(predict_ in ground_)
                        else:
                            accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_type] - 1)
                    accuracy_events[event_type].append(np.sum(accuracy_this))
                break
            # threshold_m = threshold_m * (1 -  (float(iter) / max_iter)**2)
            # print threshold_m
            # threshold_m = (float(iter) / max_iter) ** 0.5
            event_recognition = e_step(importance_score, recognition, all_img_ids, threshold, poly)
            # print len(event_recognition), len(event_recognition[event_recognition.keys()[0]])
            accuracy_this = []
            for event_id in event_id_list:
                if event_type == 'multi_label':
                    predict_ = np.argmax(event_recognition[event_id[0]])
                    ground_ = [dict_name2[i[0]] - 1 for i in event_id[1]]
                    accuracy_this.append(predict_ in ground_)
                else:
                    accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_type] - 1)
            accuracy_events[event_type].append(np.sum(accuracy_this))
            importance_score_new = m_step_involve_lstm(lstm_dict, importance_feature, event_recognition, all_img_ids, threshold_m)
            importance_scores.append(importance_score_new)
            diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
            importance_score = importance_score_new

        # # add one time of m-step, to include for all events to predict img_score
        if max_iter > 1:
            importance_score = m_step_involve_lstm(lstm_dict, importance_feature, event_recognition, all_img_ids, 0)

        for temp in event_recognition:
            event_recognition_all_event[temp] = event_recognition[temp]
        for img_id, importance in zip(all_img_ids, importance_score):
            event_id = img_id.split('/')[0]
            img_importance_all_event[event_id].append(importance)



    f = open(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.pkl', 'w')
    cPickle.dump(img_importance_all_event, f)
    f.close()

    importance_npy = []
    with open(root + folder_name+'/features/'  + importance_path + '_event_list.pkl') as f:
        event_list = cPickle.load(f)
    for event in event_list:
        importance_npy.extend(img_importance_all_event[event])
    np.save(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.npy', importance_npy)
    with open(root + folder_name+'/features/IMPORTANCE_' + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'_event_list.pkl','w') as f:
        cPickle.dump(event_list, f)

    f = open(root + folder_name+'/features/RECOGNITION_'  + importance_path + '_' + event_path  + '_em_'+str(max_iter)+'.pkl', 'w')
    cPickle.dump(event_recognition_all_event, f)
    f.close()

    print accuracy_events
    accuracy_all = float(np.sum([accuracy_events[event_type][-1] for event_type in accuracy_events])) \
                   / np.sum([event_lengths[event_type] for event_type in event_lengths])
    print accuracy_all


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def e_step(importance_feature, img_recognition, all_img_ids, threshold = 0, poly = 1, poly2 = 1):
    event_recognition = dict()
    last_event_id = ''

    for i,j,k in zip(importance_feature, img_recognition, all_img_ids):
        event_id = k.split('/')[0]
        if last_event_id != event_id:
            last_event_id = event_id
            event_recognition[event_id] = np.zeros((23, ))
        if i < threshold:
            continue
        # event_recognition[event_id] += i **poly * j
        event_recognition[event_id] += (i**poly2) * (j ** poly)
        # event_recognition[event_id] += sigmoid(i)  * j
    for event_id in event_recognition:
        event_recognition[event_id] /= np.sum(event_recognition[event_id])

    return event_recognition
def m_step(importance_feature, event_recognition, all_img_ids, threshold_1 = 0.8):
    importance_out_all = []; importance_out = []
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
                importance_out_all.extend(importance_out/np.max(importance_out))
                importance_out = []
        event_rec_this = event_recognition[event_id]
        if np.max(event_rec_this) - np.min(event_rec_this) != 0:
            event_rec_this = (event_rec_this - np.min(event_rec_this)) / (np.max(event_rec_this) - np.min(event_rec_this))
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
    importance_out_all.extend(importance_out/np.max(importance_out))
    # print np.max(importance_out_all), np.min(importance_out_all)
    return importance_out_all

def m_step_involve_lstm(lstm_result, importance_feature, event_recognition, all_img_ids, threshold_1 = 0.8):
    importance_out_all = []; importance_out = []
    last_event_id = ''
    for i, j in zip(importance_feature, all_img_ids):
        event_id = j.split('/')[0]
        if event_id != last_event_id:
            last_event_id = event_id
            if len(importance_out) > 0:
                if np.max(importance_out) < 0:
                    importance_out = list(np.ones((len(importance_out), )))
                importance_out_all.extend(importance_out)#/np.max(importance_out))
            importance_out = []
        event_rec_this = event_recognition[event_id] * lstm_result[event_id]
        if np.max(event_rec_this) - np.min(event_rec_this) != 0:
            event_rec_this = (event_rec_this - np.min(event_rec_this)) / (np.max(event_rec_this) - np.min(event_rec_this))
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
    if np.max(importance_out) < 0:
        importance_out = list(np.ones((len(importance_out), )))
    importance_out_all.extend(importance_out)#/np.max(importance_out))
    # print np.max(importance_out_all), np.min(importance_out_all)
    return importance_out_all


def combine_lstm_cnn_result(cnn_path, lstm_path, poly):
    with open(root + 'CNN_all_event_corrected_multi/features/'+cnn_path+'.pkl') as f:
        cnn_result_dict = cPickle.load(f)
    with open(root + 'CNN_all_event_corrected_multi/features/'+lstm_path+'.pkl') as f:
        lstm_result_dict = cPickle.load(f)

    for event_id in lstm_result_dict:
        cnn_result_dict[event_id] = cnn_result_dict[event_id] / np.sum(cnn_result_dict[event_id])
        cnn_result_dict[event_id] += lstm_result_dict[event_id] ** poly

    with open(root + 'CNN_all_event_corrected_multi/features/temp.pkl', 'w') as f:
        cPickle.dump(cnn_result_dict, f)


def preprocess_lstm():
    root1 = root + 'CNN_all_event_corrected_multi/features/'
    feature_training = np.load(root1 + 'training_fc7_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    print feature_training.shape

    feature_test = np.load(root1 + 'test_fc7_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    print 'doing pca...'
    pca = PCA(n_components=128)
    feature_training_pca = pca.fit_transform(feature_training)
    print 'testing pca...'
    feature_test_pca = pca.transform(feature_test)
    feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)

    np.save(root1 + 'multilabel_iter1w_feature_training_pca.npy', feature_training_pca)
    np.save(root1 + 'multilabel_iter1w_feature_test_pca.npy', feature_test_pca)
    np.save(root1 + 'multilabel_iter1w_feature_all.npy', feature_all)

def create_confusion_matrix_multi(name):
    with open(root + 'CNN_all_event_corrected_multi/features/' + name + '.pkl') as f:
            cnn_result_dict = cPickle.load(f)

    confusion_matrix = np.zeros((23, 23), dtype=float)
    count = 0
    for event_type in dict_name2:
        with open(root + 'baseline_all_correction_removedup_vote/' + event_type + '/test_event_id.cPickle') as f:
            test_event_id = cPickle.load(f)
        for event in test_event_id:
            count += 1
            predict_ = np.argmax(cnn_result_dict[event])
            ground_ = dict_name2[event_type] - 1
            confusion_matrix[ground_, predict_] += 1

    event_type = 'multi_label'
    with open(root + 'baseline_all_correction_removedup_vote/' + event_type + '/test_event_id.cPickle') as f:
        test_event_id = cPickle.load(f)
    for event, event_type in test_event_id:
        count += 1
        predict_ = np.argmax(cnn_result_dict[event])
        event_type_n = [dict_name2[i[0]] - 1 for i in event_type]
        if predict_ in event_type_n:
            confusion_matrix[predict_, predict_] += 1
        else:
            # print event, event_type, dict_name2_reverse[predict_ + 1]
            ground_ = [dict_name2[event_type[0][0]] - 1]
            confusion_matrix[ground_, predict_] += 1

    accuracy = float(np.trace(confusion_matrix)) / count
    for i in xrange(23):
        print([int(j) for j in list(confusion_matrix[i, :])])
    print('Overall accuracy:', accuracy)
    print count

def cross_entropy_loss(path, groundtruth_path):
    print path
    with open(root + 'CNN_all_event_corrected_multi/features/' + path) as f:
            predict_result = cPickle.load(f)

    with open(root + '0208_correction/all_input_and_result/' + groundtruth_path + '.pkl') as f:
        ground_truth = cPickle.load(f)
    loss = 0
    for event in predict_result:
        ground_this = ground_truth[event]
        predict_this = predict_result[event]
        predict_this = np.array(predict_this / np.sum(predict_this))
        ground_this_array = np.zeros((23, ))
        for i in ground_this:
            ground_this_array[dict_name2[i[0]] - 1] = i[1]
        # print np.sum(ground_this_array * np.log(predict_this))
        loss += np.sum(ground_this_array * np.log(predict_this))
    print loss


def split_event_type(in_path):
    with open(in_path) as f:
        in_dict = cPickle.load(f)
    with open('/home/feiyu1990/local/event_curation/0208_correction/all_input_and_result/'
              'new_multiple_result_2round_removedup_vote.pkl') as f:
        ground_truth_event = cPickle.load(f)
    dict_events = defaultdict(dict)
    for event_id in ground_truth_event:
        event_type = ground_truth_event[event_id][0][0]
        if event_id not in in_dict:
            continue
        dict_events[event_type][event_id] = in_dict[event_id]
    for event_type in dict_events:
        print event_type, len(dict_events[event_type])
        with open(in_path.split('.')[0] + '_' + event_type + '.pkl', 'w') as f:
            cPickle.dump(dict_events[event_type], f)

if __name__ == '__main__':
    file_path = '/home/feiyu1990/local/event_curation/pec/train_path.txt'
    # extract_feature_10_recognition_traintest_multilabel_pec(file_path, 'training','CNN_all_event_corrected_multi', 'deploy.prototxt', 'vote_multilabel_event_recognition_expand_balanced_3_iter_100000', (256, 256))
    # extract_feature_10_recognition_traintest_multilabel_pec(file_path, 'training','CNN_all_event_corrected_multi', 'deploy.prototxt', 'vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000', (256, 256))
    # extract_feature_10_23_traintest_multilabel_pec(file_path, 'training', 'CNN_all_event_corrected_multi','python_deploy_siamese_twoloss.prototxt', 'vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000', (256, 256))
    # extract_feature_10_23_traintest_multilabel_pec(file_path, 'training', 'CNN_all_event_corrected_multi','python_deploy_siamese_twoloss.prototxt', 'vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000', (256, 256))
    # extract_feature_10_recognition_traintest_multilabel_pec_fc7(file_path, 'training', 'CNN_all_event_corrected_multi', 'deploy_fc7.prototxt', 'vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000')


    # extract_feature_10_recognition_traintest_multilabel_pec(file_path, 'test','CNN_all_event_corrected_multi', 'deploy.prototxt', 'vote_multilabel_event_recognition_expand_balanced_3_iter_100000', (256, 256))
    # extract_feature_10_recognition_traintest_multilabel_pec(file_path, 'test','CNN_all_event_corrected_multi', 'deploy.prototxt', 'vote_multilabel_event_recognition_expand_balanced_3_iter_100000', (256, 256))
    # extract_feature_10_recognition_traintest_multilabel_fc7('test', 'CNN_all_event_corrected_multi', 'deploy_fc7.prototxt', 'multilabel_event_recognition_expand_balanced_3_iter_100000')
    # extract_feature_10_23_traintest_multilabel('test', 'CNN_all_event_corrected_multi','python_deploy_siamese_twoloss.prototxt', 'vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000', (256, 256))

    TIE=True
    poly = 10; poly2 = 10
    threshold_m = 5
    a = evaluation('CNN_all_event_corrected_multi','worker', 'test', 0)


    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter1_importance_cross_validation_combine_best_dict.pkl')
    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter2_importance_cross_validation_combine_best_dict.pkl')
    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter3_importance_cross_validation_combine_best_dict.pkl')
    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter4_importance_cross_validation_combine_best_dict.pkl')
    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter5_importance_cross_validation_combine_best_dict.pkl')
    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter6_importance_cross_validation_combine_best_dict.pkl')
    # split_event_type(root + 'CNN_all_event_corrected_multi/features_validation/THRESHOLDM_POLY2_LSTM_iter7_importance_cross_validation_combine_best_dict.pkl')
    # baseline_name = 'baseline_all_correction_removedup_vote/'
    # ground_truth_path = 'new_multiple_result_2round_removedup_vote'
    # event_path = 'test_predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000'
    # importance_path = 'test_sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000'

    # importance_path = 'test_sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000'
    # event_path = 'test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    # em_combine_event_recognition_curation_corrected(0,float(threshold_m) / 10, float(poly)/10, max_iter=1,event_path=event_path, importance_path=importance_path, baseline_name=baseline_name)
    # for threshold_m in range(0, 11):
    # for poly2 in range(5, 21, 5):
    # print 'threshold:', threshold_m, 'poly:', poly, 'poly2', poly2
    # em_combine_event_recognition_curation_corrected(0,float(threshold_m) / 10, float(poly)/10,float(poly2)/10, max_iter=1,event_path=event_path, importance_path=importance_path, baseline_name=baseline_name)
    # em_combine_event_recognition_curation_corrected(0,float(threshold_m) / 10, float(poly)/10, float(poly2)/10,max_iter=9,event_path=event_path, importance_path=importance_path, baseline_name=baseline_name)

    # ground_truth_path = 'new_multiple_result_2round_softmaxall_removedup_vote'
    # event_path = 'test_predict_vote_multilabel_event_recognition_expand_balanced_3_iter_100000'
    # importance_path = 'test_sigmoid9_23_vote_multilabel_balance_segment_twoloss_fc300_diffweight_2_iter_100000'
    # lstm_path = 'vote_multilabel_crossvalidation_prediction_dict'

    # ground_truth_path = 'new_multiple_result_2round_softmaxall_removedup_vote'
    # event_path = 'test_predict_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000'
    # importance_path = 'test_sigmoid9_23_vote_soft_segment_twoloss_fc300_diffweight_2_iter_100000'
    # lstm_path = 'vote_softall_multilabel_crossvalidation_prediction_dict'
    #
    #
    # combine_lstm_cnn_result('RECOGNITION_'+importance_path + '_' + event_path + '_em_9', lstm_path, 1)
    # create_confusion_matrix_multi('RECOGNITION_'+importance_path + '_' + event_path + '_em_1')
    # create_confusion_matrix_multi('RECOGNITION_'+importance_path + '_' + event_path + '_em_9')
    # create_confusion_matrix_multi(lstm_path)
    # create_confusion_matrix_multi('temp')
    # cross_entropy_loss('RECOGNITION_'+importance_path + '_' + event_path + '_em_1.pkl', ground_truth_path)
    # cross_entropy_loss('RECOGNITION_'+importance_path + '_' + event_path + '_em_9.pkl', ground_truth_path)
    # cross_entropy_loss(lstm_path + '.pkl', ground_truth_path)
    # cross_entropy_loss('temp.pkl', ground_truth_path)


    # event_list = cPickle.load(open('/home/feiyu1990/local/event_curation/CNN_all_event_corrected_multi/features_validation/test_list.pkl'))
    # no_event_dict = dict()
    # for event_type in dict_name2:
    #     f = open(root + 'CNN_all_event_1009/features/'+event_type+'_test_sigmoid9_10_segment_noevent_twoloss_fc300_iter_100000_dict.cPickle')
    #     temp = cPickle.load(f)
    #     for event in temp:
    #         if event not in event_list:
    #             continue
    #         no_event_dict[event] = temp[event]
    # f = open(root + 'CNN_all_event_corrected_multi/features_validation/noevent_importance.pkl', 'w')
    # cPickle.dump(no_event_dict, f)
    # f.close()

    # # final_img_importance(0, float(threshold_m)/10, max_iter=29)
    # # preprocess_lstm()
    # # extract_feature_10_recognition_traintest_multilabel('test','CNN_all_event_corrected_multi', 'deploy.prototxt', 'event_recognition_expand_balanced_3_iter_100000', (256, 256))
    # # cross_entropy_loss()
    # # combine_lstm_cnn_result()
    #
    # # with open(root + 'CNN_all_event_corrected_multi_old/features/test_multilabel_iter1w_alltraining_prediction_dict.pkl') as f:
    # #     lstm_result_dict = cPickle.load(f)
    # # em_combine_event_recognition_curation_corrected_involve_lstm(0,lstm_result_dict, float(threshold_m) / 10, float(poly)/10, max_iter=29)
    #
    #
