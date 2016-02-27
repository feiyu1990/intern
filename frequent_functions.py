__author__ = 'wangyufei'

import cPickle
import random
import scipy.io as sio
import os
import numpy as np
import operator
import csv
from collections import Counter

root = '/Users/wangyufei/Documents/Study/intern_adobe/'
#root = 'C:/Users/yuwang/Documents/'


dict_name = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}



def create_predict_dict_from_cpickle_multevent(mat_name, event_name, event_index = 16):
    path = root+'baseline_all/'+event_name+'/test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(mat_name + '.cPickle', 'r')
    predict_score = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all'+event_name+'/test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, predict_score):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], score[0][event_index]]]
        else:
            prediction_dict[event_name] = [[name_, test_url_dict[name_], score[0][event_index]]]

    f = open(mat_name + '_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()

def evaluate_rank_reweighted_(model_names, event_name, ):
    for model_name in model_names:
        f = open(root + 'baseline_all/'+event_name+'/vgg_wedding_test_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        for event_id in ground_truth:
            rank_difference[event_id] = 0
            ground_this = ground_truth[event_id]
            predict_this = predict_result[event_id]
            predict_ = [i[2] for i in predict_this]
            ground_ = [i[2] for i in ground_this]
            predict_ = zip(xrange(len(predict_)), predict_)
            ground_ = zip(xrange(len(ground_)), ground_)
            predict_.sort(key = lambda x: x[1])
            ground_.sort(key = lambda x: x[1])

            predict_rank = {}
            prev = None
            for i,(k,v) in enumerate(predict_):
                if v!=prev:
                    place,prev = i+1,v
                predict_rank[k] = place


            ground_rank = {}
            prev = None
            for i,(k,v) in enumerate(ground_):
                if v!=prev:
                    place,prev = i+1,v
                ground_rank[k] = place
            #print ground_rank
            ground_n = {}
            for i in ground_rank:
                if ground_rank[i] not in ground_n:
                    ground_n[ground_rank[i]] = 1
                else:
                    ground_n[ground_rank[i]] += 1
            for i in xrange(len(ground_rank)):
                p = predict_rank[i]
                g = ground_rank[i]
                n_same_g =ground_n[g]

                if p >= g and p < g + n_same_g:
                    continue
                if p < g:
                    rank_difference[event_id] += float(g - p) / g
                else:
                    rank_difference[event_id] += float(p - g - n_same_g + 1) / g

        #print rank_difference
        print model_name, float(sum([rank_difference[i] for i in rank_difference]))/len(rank_difference)
def average_precision(ground_, predict_, k):
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
def evaluate_MAP_(model_names, event_name, min_retrieval = 5):
    maps = []
    n_ks = []
    for model_name in model_names:
        APs = []
        f = open(root + 'baseline_all/'+event_name+'/vgg_wedding_test_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        count_all = 0; n_k_all = 0
        for event_id in ground_truth:
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
            for i,(k,v) in enumerate(ground_):
                if v!=prev:
                    place,prev = i+1,v
                ground_rank[k] = place
            AP = average_precision(ground_rank, predict_rank, n_k)


            APs.append([event_id, AP])
            n_ks.append([n_k, len(ground_)])

            sorted_predict = sorted(predict_rank.items(), key=operator.itemgetter(1))
            first_n_k = sorted_predict[:n_k]
            count = 0
            for i in first_n_k:
                if ground_rank[i[0]] <= n_k:
                    count += 1
            #print count, n_k
            count_all += count; n_k_all += n_k
        #print APs
        maps.append(sum([i[1] for i in APs])/len(APs))
        percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
        #print count_all, n_k_all
    return maps, percent
def evaluate_top20_(model_names, event_name, percent = 20):
    n_ks = []
    for model_name in model_names:
        f = open(root + 'baseline_all/'+event_name+'/vgg_wedding_test_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        count_all = 0; n_k_all = 0
        for event_id in ground_truth:
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
        print model_name,  float(count_all) / n_k_all



def evaluate_rank_reweighted_permuted(model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10.cPickle', 'wedding_CNN_net/wedding_predict_result.cPickle']):
    for model_name in model_names:
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2_permute.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        for event_id in ground_truth:
            rank_difference[event_id] = 0
            ground_this = ground_truth[event_id]
            predict_this = predict_result[event_id]
            predict_ = [i[2] for i in predict_this]
            ground_ = [i[2] for i in ground_this]
            predict_ = zip(xrange(len(predict_)), predict_)
            ground_ = zip(xrange(len(ground_)), ground_)
            predict_.sort(key = lambda x: x[1])
            ground_.sort(key = lambda x: x[1])

            predict_rank = {}
            prev = None
            for i,(k,v) in enumerate(predict_):
                if v!=prev:
                    place,prev = i+1,v
                predict_rank[k] = place


            ground_rank = {}
            prev = None
            for i,(k,v) in enumerate(ground_):
                if v!=prev:
                    place,prev = i+1,v
                ground_rank[k] = place
            #print ground_rank
            ground_n = {}
            for i in ground_rank:
                if ground_rank[i] not in ground_n:
                    ground_n[ground_rank[i]] = 1
                else:
                    ground_n[ground_rank[i]] += 1
            for i in xrange(len(ground_rank)):
                p = predict_rank[i]
                g = ground_rank[i]
                n_same_g =ground_n[g]

                if p >= g and p < g + n_same_g:
                    continue
                    #print 0
                if p < g:
                    rank_difference[event_id] += float(g - p) / g
                    #print float(g - p) / g
                else:
                    rank_difference[event_id] += float(p - g - n_same_g + 1) / g
                    #print float(p - g - n_same_g + 1) / g

        #print rank_difference
        print model_name, float(sum([rank_difference[i] for i in rank_difference]))/len(rank_difference)
def evaluate_MAP_permuted(model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10.cPickle', 'wedding_CNN_net/wedding_predict_result.cPickle'], min_retrieval = 5):
    maps = []
    n_ks = []
    for model_name in model_names:
        APs = []
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2_permute.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        count_all = 0; n_k_all = 0
        for event_id in ground_truth:
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
            for i,(k,v) in enumerate(ground_):
                if v!=prev:
                    place,prev = i+1,v
                ground_rank[k] = place
            AP = average_precision(ground_rank, predict_rank, n_k)


            APs.append([event_id, AP])
            n_ks.append([n_k, len(ground_)])

            sorted_predict = sorted(predict_rank.items(), key=operator.itemgetter(1))
            first_n_k = sorted_predict[:n_k]
            count = 0
            for i in first_n_k:
                if ground_rank[i[0]] <= n_k:
                    count += 1
            #print count, n_k
            count_all += count; n_k_all += n_k
        #print APs
        maps.append(sum([i[1] for i in APs])/len(APs))
        percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
        #print count_all, n_k_all
    return maps, percent
def evaluate_top20_permuted(model_names, percent = 20):
    retval = []
    for model_name in model_names:
        f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2_permute.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        count_all = 0; n_k_all = 0
        for event_id in ground_truth:
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




block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
def amt_worker_result_predict(check_worker = 'A13UC43JQSPAFL', min_retrievals = xrange(5,36,3), type = 'test'):
    input_path = root + 'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle'
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
    input_path = root + 'baseline_wedding_test/wedding_'+type+'.csv'
    line_count = 0
    head_meta = []
    HITs = {}
    all_n_ks = []
    all_aps = []
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
    index_distraction = 31
    for HITId in HITs:

        this_hit = HITs[HITId]
        check_this = False
        for i in this_hit:
            if i[index_worker_id] == check_worker:
                check_this = True
                continue
        if not check_this:
            continue

        APs = []
        n_ks = []
        for min_retrieval in min_retrievals:

            num_images = int(this_hit[0][index_num_image])
            distract_image = this_hit[0][index_distraction]
            [distract1, distract2] = distract_image.split(':')
            distract1 = int(distract1)
            distract2 = int(distract2)
            this_hit_new = []; this_hit_check = []
            check_submission = []
            rest_result = []
            for submission in this_hit:
                if submission[index_worker_id] in block_workers:
                    continue
                if submission[index_worker_id] == check_worker:
                    this_hit_check = submission
                else:
                    this_hit_new.append(submission)
            num_valid_submission = len(this_hit_new)
            ii = 0
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    continue
                score = 0
                score_index = image_output_index[i]
                for submission in this_hit_new:
                    vote = submission[score_index]
                    if vote == 'selected':
                        score += 2
                    elif vote == 'selected_sw':
                        score += 1
                    elif vote == 'selected_irrelevant':
                        score -= 2
                score = float(score)/float(num_valid_submission)
                rest_result.append(score)
                score = 0
                vote = this_hit_check[score_index]
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
            ground_ = zip(xrange(len(rest_result)), rest_result)
            predict_.sort(key = lambda x: x[1], reverse=True)
            ground_.sort(key = lambda x: x[1], reverse=True)

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

            threshold = ground_[max(1, len(ground_)*min_retrieval/100)-1][1]
            #print len(ground_)*min_retrieval/100
            n_k = len([i for i in ground_ if i[1] >= threshold])
            APs.append(average_precision_worker(ground_rank, predict_rank, n_k))
            n_ks.append(float(n_k)/len(ground_))
        print n_ks
        print APs
        print '\n'
        all_n_ks.append(n_ks)
        all_aps.append(APs)
    return all_n_ks, all_aps
def amt_worker_result_predict_average(min_retrievals = xrange(5,36,3), type = 'test'):

    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2_permute.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()

    input_path = root + 'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle'
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
    input_path = root + 'baseline_wedding_test/wedding_'+type+'.csv'
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
        print HITId
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
            if submission[index_worker_id] in block_workers:
                continue
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
                APs.append(average_precision(ground_rank, new_predict_rank, n_k))
                n_ks.append([n_k, len(ground_)])

            #print n_ks
            print APs
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
                p = predict_rank[i]
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
        print '\n'
    return all_n_ks, all_aps, all_reweighted, all_ps

def amt_worker_result_predict_average_2(min_retrievals = xrange(5,36,3), type = 'test'):

    input_path = root + 'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle'
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
    input_path = root + 'baseline_wedding_test/wedding_'+type+'.csv'
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
        print HITId
        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        event_id = this_hit[0][index_event_id]
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)



        for submission in this_hit:

            APs = []
            Ps = []
            n_ks = []
            check_submission = []
            if submission[index_worker_id] in block_workers:
                continue

            #get groundtruth scores
            ground_items = []
            ground_ = []
            for submission_ in this_hit:
                if submission_[index_worker_id] in block_workers:# or submission[index_worker_id] == submission_[index_worker_id]:
                    continue
                ground_items.append(submission_)

            for i in xrange(1, 1+num_images):
                    if i==distract1 or i==distract2:
                        continue
                    score_index = image_output_index[i]
                    score = 0
                    for temp in ground_items:
                        vote = temp[score_index]
                        if vote == 'selected':
                            score += 2
                        elif vote == 'selected_sw':
                            score += 1
                        elif vote == 'selected_irrelevant':
                            score -= 2
                    ground_.append(float(score)/len(ground_items) + random.uniform(-0.02, 0.02))

            ground_ = zip(xrange(len(ground_)), ground_)
            ground_.sort(key = lambda x: x[1], reverse=True)


            #get prediction scores
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
                APs.append(average_precision(ground_rank, new_predict_rank, n_k))
                n_ks.append([n_k, len(ground_)])

            #print n_ks
            print APs
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
                p = predict_rank[i]
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
        print '\n'

    return all_n_ks, all_aps, all_reweighted, all_ps

def average_precision_worker(ground_, predict_, k):
    temp = [predict_[i] for i in predict_]
    rank_set =  set(temp)
    new_predict_ = {}
    for rank in rank_set:
        this_rank = [i for i in predict_ if predict_[i] == rank]
        random.shuffle(this_rank)
        for i in xrange(len(this_rank)):
            new_predict_[this_rank[i]] = rank + i


    need_k_index = []
    for i in ground_:
        if ground_[i] <= k:
            need_k_index.append(i)
    retrieved_rank = []
    for i in new_predict_:
        if i in need_k_index:
            retrieved_rank.append(new_predict_[i])
    retrieved_rank.sort()
    temp = zip(xrange(1, 1+len(retrieved_rank)), retrieved_rank)
    recall = [min(float(i[0])/i[1],1) for i in temp]
    ap = sum(recall)/len(recall)
    return ap
def average_precision_worker_old(ground_, predict_, k):
    temp = [predict_[i] for i in predict_]
    count_ =  Counter(temp)
    new_predict_dict = {}
    for i in count_:
        new_predict_dict[i] = i + float(count_[i] - 1)/2
    new_predict_  = {}
    for i in predict_:
        new_predict_[i] = new_predict_dict[predict_[i]]
    need_k_index = []
    for i in ground_:
        if ground_[i] <= k:
            need_k_index.append(i)
    retrieved_rank = []
    for i in new_predict_:
        if i in need_k_index:
            retrieved_rank.append(new_predict_[i])
    retrieved_rank.sort()
    temp = zip(xrange(1, 1+len(retrieved_rank)), retrieved_rank)
    recall = [min(float(i[0])/i[1],1) for i in temp]
    ap = sum(recall)/len(recall)
    return ap


if __name__ == '__main__':

    all_nks, all_aps, all_reweighted, all_ps = amt_worker_result_predict_average_2()
    mean_aps = []
    for i in all_aps:
        for j in i:
            mean_aps.append(j)
    mean_aps = np.mean(mean_aps, axis=0)


    mean_ps1 = np.zeros(11);mean_ps2 = np.zeros(11)
    for i in xrange(len(all_ps)):
        for j in xrange(len(all_ps[i])):
            mean_ps1[j] += all_ps[i][j][0]
            mean_ps2[j] += all_ps[i][j][1]

    mean_ps = [mean_ps1[i]/mean_ps2[i] for i in xrange(len(mean_ps1))]


    mean_nks1 = np.zeros(11);mean_nks2 = np.zeros(11)
    for i in xrange(len(all_nks)):
        for j in xrange(len(all_nks[i])):
            mean_nks1[j] += all_nks[i][j][0]
            mean_nks2[j] += all_nks[i][j][1]

    mean_nks = mean_nks1/mean_nks2[0]

    print mean_nks
    print mean_aps

    print sum(all_reweighted)/len(all_reweighted)
    print mean_ps
    '''
    type = 'test'
    input_path = root + 'baseline_wedding_test/wedding_'+type+'.csv'
    line_count = 0
    head_meta = []
    HITs = {}
    all_n_ks = []
    all_aps = []
    workers = set()
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                if meta[15] in workers or meta[15] in block_workers:
                    continue
                workers.add(meta[15])
            line_count+=1
    for worker in workers:
        #print worker
        temp1, temp2 = amt_worker_result_predict(worker)
        all_n_ks.append(temp1)
        all_aps.append(temp2)

    mean_aps = []
    for i in all_aps:
        for j in i:
            mean_aps.append(j)
    mean_aps = np.mean(mean_aps, axis=0)

    mean_nks = []
    for i in all_n_ks:
        for j in i:
            mean_nks.append(j)
    mean_nks = np.mean(mean_nks, axis=0)

    print mean_nks
    print mean_aps
    '''

    '''
    event_name = 'BusinessActivity'
    event_id = dict_name[event_name] - 1

    model_names = ['baseline_all/'+event_name+'/vgg_predict_result_10_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multeventvgg_iter_40000_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multevent_iter_10000_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_10000_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_70000_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_20000_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multevent_iter_150000_dict.cPickle']

    for i in model_names:
        if not os.path.exists(root + i):
            create_predict_dict_from_cpickle_multevent(root + ('_').join(i.split('_')[:-1]), event_name)
    evaluate_rank_reweighted_(model_names, event_name)
    evaluate_top20_(model_names, 20)
    retrievals = []; percent = []
    for i in model_names:
        retrievals.append([])
    #for i in [20]:
    for i in xrange(5, 36, 3):
        temp, percent_temp = evaluate_MAP_(model_names,event_name,  min_retrieval=i)
        for j in xrange(len(temp)):
            retrievals[j].append(temp[j])
        percent.append(percent_temp)
    for i in retrievals:
        print i
    print percent
    '''