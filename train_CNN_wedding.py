__author__ = 'wangyufei'

import cPickle
import random
import scipy.io as sio
import os
import numpy as np
import operator

root = '/Users/wangyufei/Documents/Study/intern_adobe/'
#root = 'C:/Users/yuwang/Documents/'
def create_path():
    load_path = root + 'baseline_wedding_test/vgg_wedding_test_result_dict_v2.cPickle'
    f = open(load_path,'r')
    dict_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_image_ids.cPickle','r')
    image_ids_all = cPickle.load(f)
    f.close()
    all_scores = []
    for i in image_ids_all:
        all_scores.append(dict_training[i])

    sio.savemat(root + 'wedding_CNN_net/vgg_wedding_test_result_v2',{'scores': all_scores})
    out_path = root + 'wedding_CNN_net/train.txt'
    f = open(out_path,'w')
    training_ = []
    for i in dict_training:
        img_name = i.split('_')[1]
        img_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/' + img_name + '.jpg'
        training_.append([img_path,dict_training[i]])
    random.shuffle(training_)
    for i in training_:
        f.write(i[0] + ' ' + str(i[1]) + '\n')
    f.close()


    load_path = root + 'baseline_wedding_test/vgg_wedding_test_result_dict_v2.cPickle'
    f = open(load_path,'r')
    dict_training = cPickle.load(f)
    f.close()
    out_path = root + 'wedding_CNN/test.txt'
    f = open(out_path,'w')
    training_ = []
    for i in dict_training:
        img_name = i.split('_')[1]
        img_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/' + img_name + '.jpg'
        training_.append([img_path,dict_training[i]])

    random.shuffle(training_)
    for i in training_:
        f.write(i[0] + ' ' + str(i[1]) + '\n')
    f.close()
def create_predict_dict_from_cpickle_multevent(mat_name, event_index = 16):
    path = root+'baseline_wedding_test/wedding_test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(mat_name + '.cPickle', 'r')
    predict_score = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
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
def create_predict_dict_from_cpickle(mat_name):
    path = root+'baseline_wedding_test/wedding_test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(mat_name + '.cPickle', 'r')
    predict_score = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, predict_score):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], score]]
        else:
            prediction_dict[event_name] = [[name_, test_url_dict[name_], score]]

    f = open(mat_name + '_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()
def create_predict_dict_from_cpickle_classify(cpikcle_name):
    path = root+'baseline_wedding_test/wedding_test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(cpikcle_name + '.cPickle', 'r')
    predict_score = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, predict_score):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], score[0][1]]]
        else:
            prediction_dict[event_name] = [[name_, test_url_dict[name_], score[0][1]]]

    f = open(cpikcle_name + '_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()

def create_predict_dict(mat_name):


    path = root+'baseline_wedding_test/wedding_test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()

    temp = sio.loadmat(mat_name + '.mat')
    predict_score = temp['temp']

    f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, predict_score):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], score[0]]]
        else:
            prediction_dict[event_name] = [[name_, test_url_dict[name_], score[0]]]

    f = open(mat_name + '_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()
def evaluate_rank_reweighted_(model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10.cPickle', 'wedding_CNN_net/wedding_predict_result.cPickle']):
    for model_name in model_names:
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2.cPickle','r')
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
            predict_.sort(key = lambda x: x[1], reverse=True)
            ground_.sort(key = lambda x: x[1], reverse=True)

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
def evaluate_MAP_(model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10.cPickle', 'wedding_CNN_net/wedding_predict_result.cPickle'], min_retrieval = 5):
    maps = []
    n_ks = []
    for model_name in model_names:
        APs = []
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2.cPickle','r')
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
def evaluate_top20_(model_names, percent = 20):
    retval = []
    for model_name in model_names:
        f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2.cPickle','r')
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


def permute_groundtruth():
    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()
    ground_truth_new = {}
    for idx in ground_truth:
        temp = []
        for i in ground_truth[idx]:
            temp.append((i[0], i[1], i[2]+random.uniform(-0.02, 0.02)))
        ground_truth_new[idx] = temp
    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2_permute.cPickle','w')
    cPickle.dump(ground_truth_new, f)
    f.close()

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
            predict_.sort(key = lambda x: x[1], reverse=True)
            ground_.sort(key = lambda x: x[1], reverse=True)

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
def from_txt_to_pickle(name):
    in_path = root + name + '.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)

    f = open(root + name + '.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
def create_knn_from_finetune():
    name = 'alexnet6k_wedding'
    #if not os.path.exists(root + 'baseline_wedding_test/alexnet6k_wedding_training_features.cPickle'):
    #    from_txt_to_pickle('to_guru/wedding_CNN_net/features/training_' + name)
    #    from_txt_to_pickle('to_guru/wedding_CNN_net/features/test_' + name)
    f = open(root + 'baseline_wedding_test/alexnet6k_wedding_training_features.cPickle','r')
    train_feature = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/alexnet6k_wedding_test_features.cPickle','r')
    test_feature = cPickle.load(f)
    f.close()

    #train_feature = np.array(train_feature)
    #test_feature = np.array(test_feature)

    retrieval_scores = []

    f = open(root + 'baseline_wedding_test/wedding_training_ulr_dict.cPickle', 'r')
    training_url_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/wedding_test_image_ids.cPickle','r')
    test_images = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/wedding_training_image_ids.cPickle','r')
    training_images = cPickle.load(f)
    f.close()
    knn_dict = {}

    for i in xrange(len(test_feature)):
        print i
        this_test = np.array(test_feature[i])
        this_test_id = test_images[i]
        knn_dict[i] = [(this_test_id, test_url_dict[this_test_id])]
        this_norm = np.dot(this_test, np.transpose(this_test))
        similarity = []
        for j in xrange(len(train_feature)):
            this_train = np.array(train_feature[j])
            diff = this_test - this_train
            temp = np.exp(-(np.dot(diff, np.transpose(diff))/this_norm)**2)
            similarity.append(temp)
        rank = np.argsort(similarity)[::-1]
        sorted_similarity = [similarity[k] for k in rank]
        count = 0
        for j,k in zip(rank,sorted_similarity):
            if count >= 50:
                break
            count += 1
            knn_dict[i].append((k, training_images[j],training_url_dict[training_images[j]]))


    f = open(root + 'baseline_wedding_test/' + name + '_knn_noapprox.cPickle','wb')
    cPickle.dump(knn_dict,f)
    f.close()
def baseline_predict(name, n_vote = 10):
    f = open(root + 'baseline_wedding_test/vgg_wedding_training_result_dict_v2.cPickle','r')
    training_scores_dict = cPickle.load(f)
    f.close()
    f = open(root + 'wedding_CNN_net/'+name+'_knn.cPickle', 'r')
    knn = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/vgg_wedding_knn.cPickle', 'r')
    knn1 = cPickle.load(f)
    print knn[4][0]
    print [i[2] for i in knn[4][1:]]
    print [i[2] for i in knn1[4][1:]]
    f.close()
    for i in knn:
        this_test_img = knn[i]
        this_test_id = this_test_img[0][0]
        if this_test_id in training_scores_dict:
            print 'ERROR!'

    test_prediction = []
    for i in knn:
        this_test_img = knn[i]
        this_test_id = this_test_img[0][0]
        this_test_url = this_test_img[0][1]
        votes = []
        weight = []
        events = {}
        valid_vote = 0
        for j in xrange(1, 11):
            weight.append(this_test_img[j][0])
            votes.append(training_scores_dict[this_test_img[j][1]])
        votes_ = zip(votes, weight)
        votes_.sort(key=lambda x: x[0])
        votes_ = votes_[1:-1]
        votes = [i[0] for i in votes_]
        weight = [i[1] for i in votes_]
        weight = [i/sum(weight) for i in weight]
        score_predict = sum([votes[i]*weight[i] for i in xrange(len(votes))])
        test_prediction.append([this_test_id, this_test_url, score_predict])

    test_prediction_event = {}
    for i in test_prediction:
        img_id = i
        event_id = img_id[0].split('/')[0]
        if event_id in test_prediction_event:
            test_prediction_event[event_id].append(i)
        else:
            test_prediction_event[event_id] = [i]

    f = open(root + 'wedding_CNN_net/'+name+'_wedding_predict_result_'+str(n_vote)+'.cPickle','wb')
    cPickle.dump(test_prediction_event, f)
    f.close()
def create_retrieval_image(max_display = 10):
    #name = 'fc7_ranking_sigmoid_2round_iter_20000'
    name = 'alexnet6k_wedding'
    f = open(root + 'baseline_wedding_test/vgg_wedding_training_result_dict_v2.cPickle','r')
    training_result = cPickle.load(f)
    f.close()

    #f = open(root + 'to_guru/wedding_CNN_net/features/'+name + '_knn.cPickle','r')
    f = open(root + 'baseline_wedding_test/'+name + '_knn_noapprox.cPickle','r')
    knn_dict = cPickle.load(f)
    f.close()
    html_count = 0
    event_knn = {}
    for i in knn_dict:
        event_id = knn_dict[i][0][0].split('/')[0]
        if event_id in event_knn:
            event_knn[event_id].append(knn_dict[i])
        else:
            event_knn[event_id] = [knn_dict[i]]
    if not os.path.exists(root + 'present_htmls_test/'):
        os.mkdir(root + 'present_htmls_test/')
    for event_id in event_knn:
        if not os.path.exists(root + 'present_htmls_test/' + event_id):
            os.mkdir(root + 'present_htmls_test/' + event_id)
        f = open(root + 'present_htmls_test/' + event_id + '/'+name+'noapprox_retrieval_top10.html','wb')
        f.write('<head>'+name+' Retrieval Result #'+str(html_count)+'</head> <title>'+name+' Retrieval Result '+str(html_count)+'</title>\n' )
        f.write('<center>')
        f.write('<table border="1" style="width:100%">\n')
        this_knn = event_knn[event_id]
        for i in xrange(len(this_knn)):
            test_index = i
            this_test = this_knn[test_index]
            test_id = this_test[0][0]; test_url = this_test[0][1]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "200" /><br /><b>'+test_id+'</b><br /></td>\n')
            for j in xrange(1, max_display+1):
                score = this_test[j][0]; id = this_test[j][1]; url = this_test[j][2]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+url+'\" alt=Loading... width = "200" /><br /><b>('+str(score)+', '+str(training_result[id])+')</b><br /></td>\n')
            f.write('</tr>\n')

        f.write('</table>\n')
        f.write('<style type=\"text/css\">img { height:auto;width:\"200px\";}\n')
        f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
        f.close()
def create_retrieval_image_ranked_score(max_display = 10):
    #name = 'fc7_ranking_sigmoid_2round_iter_20000'
    name = 'fc7_ranking_sigmoid_2round_iter_20000'
    f = open(root + 'baseline_wedding_test/vgg_wedding_training_result_dict_v2.cPickle','r')
    training_result = cPickle.load(f)
    f.close()

    f = open(root + 'to_guru/wedding_CNN_net/features/'+name + '_knn.cPickle','r')
    #f = open(root + 'baseline_wedding_test/'+name + '_knn_noapprox.cPickle','r')
    knn_dict = cPickle.load(f)
    f.close()
    html_count = 0

    f = open(root + 'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_sigmoid_2round_iter_20000_dict.cPickle','r')
    predict_result = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()


    event_knn = {}
    for i in knn_dict:
        event_id = knn_dict[i][0][0].split('/')[0]
        if event_id in event_knn:
            event_knn[event_id].append(knn_dict[i])
        else:
            event_knn[event_id] = [knn_dict[i]]


    if not os.path.exists(root + 'present_htmls_test/'):
        os.mkdir(root + 'present_htmls_test/')
    for event_id in event_knn:
        ground_ = ground_truth[event_id]
        predict_ = predict_result[event_id]
        imgs_ = []
        for i,j in zip(predict_, ground_):
            if (i[0] != j[1]):
                print "ERROR!"
                return
            imgs_.append([i[0], i[1], abs(float(1)/(np.exp(-i[2][0][0]) + 1) - float((2+j[2]))/4), float(1)/(np.exp(-i[2][0][0]) + 1), float((2+j[2]))/4])
            #print float(1)/(np.exp(-i[2][0][0]) + 1), float((2+j[2]))/4
        imgs_.sort(key=lambda x: x[2], reverse=True)
        sorted_index = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(imgs_))]
        sorted_index = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(imgs_))]
        dict_sorted = {}
        for i in xrange(len(sorted_index)):
            dict_sorted[sorted_index[i]] = i

        if not os.path.exists(root + 'present_htmls_test/' + event_id):
            os.mkdir(root + 'present_htmls_test/' + event_id)
        f = open(root + 'present_htmls_test/' + event_id + '/'+name+'_retrieval_top10_ranked_score.html','wb')
        f.write('<head>'+name+' Retrieval Result #'+str(html_count)+'</head> <title>'+name+' Retrieval Result '+str(html_count)+'</title>\n' )
        f.write('<center>')
        f.write('<table border="1" style="width:100%">\n')
        this_knn = event_knn[event_id]
        for i in xrange(len(this_knn)):
            this_test = this_knn[dict_sorted[i]]
            test_id = this_test[0][0]; test_url = this_test[0][1]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "200" /><br /><b>'+test_id+', ('+str(imgs_[i][3])+', ' + str(imgs_[i][4])+')</b><br /></td>\n')
            for j in xrange(1, max_display+1):
                score = this_test[j][0]; id = this_test[j][1]; url = this_test[j][2]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+url+'\" alt=Loading... width = "200" /><br /><b>('+str(score)+', '+str(training_result[id])+')</b><br /></td>\n')
            f.write('</tr>\n')

        f.write('</table>\n')
        f.write('<style type=\"text/css\">img { height:auto;width:\"200px\";}\n')
        f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
        f.close()
def create_retrieval_image_ranked(max_display = 10):
    #name = 'fc7_ranking_sigmoid_2round_iter_20000'
    name = 'fc7_ranking_sigmoid_2round_iter_20000'
    f = open(root + 'baseline_wedding_test/vgg_wedding_training_result_dict_v2.cPickle','r')
    training_result = cPickle.load(f)
    f.close()

    f = open(root + 'to_guru/wedding_CNN_net/features/'+name + '_knn.cPickle','r')
    #f = open(root + 'baseline_wedding_test/'+name + '_knn_noapprox.cPickle','r')
    knn_dict = cPickle.load(f)
    f.close()
    html_count = 0

    f = open(root + 'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_sigmoid_2round_iter_20000_dict.cPickle','r')
    predict_result = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_v2.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()


    event_knn = {}
    for i in knn_dict:
        event_id = knn_dict[i][0][0].split('/')[0]
        if event_id in event_knn:
            event_knn[event_id].append(knn_dict[i])
        else:
            event_knn[event_id] = [knn_dict[i]]


    if not os.path.exists(root + 'present_htmls_test/'):
        os.mkdir(root + 'present_htmls_test/')
    for event_id in event_knn:
        ground_ = ground_truth[event_id]
        predict_ = predict_result[event_id]
        if event_id == '0_22634442@N00':
            print event_id

        imgs_ = rank_difference(predict_, ground_)
        imgs_.sort(key=lambda x: x[2], reverse=True)
        sorted_index = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(imgs_))]
        dict_sorted = {}
        for i in xrange(len(sorted_index)):
            dict_sorted[sorted_index[i]] = i

        if not os.path.exists(root + 'present_htmls_test/' + event_id):
            os.mkdir(root + 'present_htmls_test/' + event_id)
        f = open(root + 'present_htmls_test/' + event_id + '/'+name+'_retrieval_top10_ranked.html','wb')
        f.write('<head>'+name+' Retrieval Result #'+str(html_count)+'</head> <title>'+name+' Retrieval Result '+str(html_count)+'</title>\n' )
        f.write('<center>')
        f.write('<table border="1" style="width:100%">\n')
        this_knn = event_knn[event_id]
        for i in xrange(len(this_knn)):
            this_test = this_knn[dict_sorted[i]]
            test_id = this_test[0][0]; test_url = this_test[0][1]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "200" /><br /><b>'+test_id+', ('+str(imgs_[i][2]) + ', ' + str(imgs_[i][3])+', ' + str(imgs_[i][4])+', ' + str(imgs_[i][5])+', ' + str(imgs_[i][6])+')</b><br /></td>\n')
            for j in xrange(1, max_display+1):
                score = this_test[j][0]; id = this_test[j][1]; url = this_test[j][2]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+url+'\" alt=Loading... width = "200" /><br /><b>('+str(score)+', '+str(training_result[id])+')</b><br /></td>\n')
            f.write('</tr>\n')

        f.write('</table>\n')
        f.write('<style type=\"text/css\">img { height:auto;width:\"200px\";}\n')
        f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
        f.close()

def rank_difference(predict_this, ground_this):

            predict_ = [i[2][0][0] for i in predict_this]
            ground_ = [i[2] for i in ground_this]

            predict_ = zip(xrange(len(predict_)), predict_)
            ground_ = zip(xrange(len(ground_)), ground_)
            predict_.sort(key = lambda x: x[1], reverse=True)
            ground_.sort(key = lambda x: x[1], reverse=True)

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

            imgs_ = []
            for i in xrange(len(predict_this)):
                ind = predict_this[i]
                imgs_.append([ind[0], ind[1], abs(ground_rank[i] - predict_rank[i]), ground_rank[i], predict_rank[i],float((2+ground_this[i][2]))/4,  float(1)/(np.exp(-ind[2][0][0]) + 1)])
            return imgs_

def create_face():
    f = open(root + 'to_guru/face_heatmap/features/test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle','r')
    #f = open(root + 'to_guru/face_heatmap/features/Weddingtest_sigmoid_iter_100000.cPickle','r')
    face = cPickle.load(f)
    f.close()

    f = open(root + 'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_70000.cPickle','r')
    feature_ = cPickle.load(f)
    f.close()
    feature = [i[0][16] for i in feature_]
    #print [float(j) for (i,j) in zip(feature, face)]
    feature_new = [max(0.7,float(j))**1.4/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)] #---> predict
    #feature_new = [max(0.8,float(j))**1.4/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)] #---> groundtruth
    #feature_new = [float(j)**0.1*float(i) for (i,j) in zip(feature, face)]
    #feature_new = [float(j>0.5)*(float(j)**0.05) + i for (i,j) in zip(feature, face)]
    #feature_new = [(max(0.5, float(j))*0.1) + i for (i,j) in zip(feature, face)]


    #feature_new = [float(j)/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)]
    #feature_new = [float(j)*float(i) for (i,j) in zip(feature, face)]
    f = open(root + 'to_guru/CNN_all_event/features/face_combine_test.cPickle','w')
    cPickle.dump(feature_new,f)
    f.close()

def combine_cues(name1 = 'cnn_features/test_fc8_class_ranking_learnlast_iter_220000.cPickle', name2 = 'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_70000.cPickle', event_index = 16):
    path = root+'baseline_wedding_test/wedding_test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(root + name1, 'r')
    predict_score_1 = cPickle.load(f)
    f.close()

    f = open(root + name2, 'r')
    predict_score_2 = cPickle.load(f)
    f.close()

    predict_score_1 = np.array(predict_score_1)
    predict_score_2 = np.array(predict_score_2)

    scores = list(predict_score_1 + predict_score_2)

    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, scores):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], score[0][event_index]]]
        else:
            prediction_dict[event_name] = [[name_, test_url_dict[name_], score[0][event_index]]]

    f = open(root + 'to_guru/CNN_all_event/features/combined_test.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()

if __name__ == '__main__':
    #create_retrieval_image()
    #baseline_predict('features_60000')
    #create_knn_from_finetune()
    #from_txt_to_pickle()
    #create_path()


    #create_predict_dict_from_cpickle('fc8_test_features_sigmoid_iter_40000')
    #create_knn_from_finetune()

    create_face()
    #combine_cues()
    #permute_groundtruth()
    model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10.cPickle',
                   #'cnn_features/test_fc8_class_ranking_learnlast_iter_120000_dict.cPickle',
                   #'cnn_features/test_fc8_class_ranking_learnlast_iter_220000_dict.cPickle',
                   'baseline_wedding_test/alexnet6k_wedding_predict_result_10.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_90000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_180000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_260000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_310000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_370000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_0.2_0.5_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_0.2_0.5_iter_40000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_0.2_0.5_iter_60000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_70000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_0.2_0.5_iter_60000_dict.cPickle',
                   'to_guru/CNN_all_event/features/face_combine_test_dict.cPickle',
                   'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_70000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_2round_iter_270000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_2round_iter_750000_dict.cPickle']

                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_60000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_70000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_130000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_260000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_segment_iter_460000_dict.cPickle']


                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_50000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_130000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_160000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_190000_dict.cPickle',

                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_10000_dict.cPickle',

                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_20000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_30000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_40000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_50000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_60000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_70000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_80000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_90000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_100000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_110000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_iter_120000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_2round_iter_50000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_2round_iter_150000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_2round_iter_200000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventsegment_2round_iter_270000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_nomargin_largerbatch_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_nomargin_largerbatch_iter_70000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_nomargin_iter_20000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_nomargin_iter_40000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_nomargin_iter_60000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_nomargin_iter_160000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_lastlayer_iter_30000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_lastlayer_iter_50000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_lastlayer_iter_70000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_lastlayer_iter_180000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_iter_30000_2_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_iter_40000_2_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_iter_50000_2_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_iter_70000_2_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_round2_iter_20000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_round2_iter_90000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_round2_iter_160000_dict.cPickle', #overfitting
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_30000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_50000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_130000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_160000_dict.cPickle']
                   #'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_sigmoid_2round_iter_20000_dict.cPickle',
                   #'to_guru/wedding_CNN_net/features/fc7_ranking_sigmoid_2round_iter_20000_dict.cPickle',
                   #'baseline_wedding_test/alexnet6k_wedding_predict_result_10_noapprox_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_150000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_70000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventmindist_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventnomargin_round2_iter_20000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventvgg_iter_60000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_10000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_70000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multeventl2_iter_20000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_40000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_100000_dict.cPickle',
                   #'to_guru/CNN_all_event/features/test_fc8_multevent_iter_150000_dict.cPickle']
                   #'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_sigmoid_l2_iter_60000_dict.cPickle',
                   #'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_euclidean_sigmoid_iter_30000_dict.cPickle',
                   #'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_euclidean_lossweight_iter_50000_dict.cPickle',
                   #'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_sigmoid_2round_iter_90000_dict.cPickle']
                   #'to_guru/CNN_all_event/features/combined_test.cPickle']


    #model_names = ['cnn_features/test_fc8_value_iter_120000_dict.cPickle',
    #               'cnn_features/test_fc8_value_largelr_iter_110000_dict.cPickle',
    #               'cnn_features/test_sigmoid9_ranking_sigmoid_iter_40000_dict.cPickle',
    #               'cnn_features/test_fc8_value_sigmoid_3round_iter_250000_dict.cPickle',
    #               'cnn_features/test_fc8_value_learnlast_2round_iter_160000_dict.cPickle',
    #               'cnn_features/test_fc8_value_vgg_euclidean_iter_390000_dict.cPickle',
    #               'cnn_features/test_fc8_value_vgg_sigmoid_iter_450000_dict.cPickle']

    #model_names = ['cnn_features/test_fc8_class_classification0.55_0.8_iter_70000_dict.cPickle',
    #               'cnn_features/test_fc8_class_classification_learnlast0.55_0.8_iter_40000_dict.cPickle',
    #               'cnn_features/test_fc8_class_classification_learnlast_multilayer0.55_0.8_iter_70000_dict.cPickle']

    #model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10_facereweighted_old.cPickle']
    #model_names = ['baseline_wedding_test/alexnet6k_wedding_predict_result_10.cPickle','wedding_CNN_net/fc8_test_features_larglr_20000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_sigmoid_iter_40000_dict.cPickle', 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict_dict.cPickle','ranking_loss_CNN/test_sigmoid9_ranking_sigmoid_iter_20000_dict.cPickle','ranking_loss_CNN/test_sigmoid9_ranking_sigmoid_iter_30000_dict.cPickle']
        #, 'wedding_CNN_net/fc8_test_features_learnlast_iter_50000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_sigmoid_iter_40000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_vgg_sigmoid_iter_80000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_vgg_euclidean_iter_290000_dict.cPickle', 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict_dict.cPickle','ranking_loss_CNN/test_sigmoid9_ranking_sigmoid_iter_20000_dict.cPickle']
    #model_names = ['baseline_wedding_test/vgg_wedding_predict_result_10.cPickle','baseline_wedding_test/vgg_wedding_predict_result_10_facereweighted_dict.cPickle','baseline_wedding_test/vgg_wedding_predict_result_10_facereweighted_old_dict.cPickle', 'baseline_wedding_test/face_vgg_wedding_predict_result_10.cPickle',  'baseline_wedding_test/face_vgg_wedding_predict_result_10_facereweighted.cPickle', 'baseline_wedding_test/face_vgg_wedding_predict_result_10_facereweighted_old.cPickle', 'wedding_CNN_net/fc8_test_features_learnlast_iter_50000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_sigmoid_iter_40000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_vgg_sigmoid_iter_80000_dict.cPickle', 'wedding_CNN_net/test_fc8_value_vgg_euclidean_iter_290000_dict.cPickle', 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict_dict.cPickle','ranking_loss_CNN/test_sigmoid9_ranking_sigmoid_iter_20000_dict.cPickle']
    #model_names = ['baseline_wedding_test/alexnet6k_wedding_predict_result_10.cPickle','ranking_loss_CNN/test_sigmoid9_ranking_sigmoid_iter_20000_dict.cPickle','classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict_dict.cPickle']    #model_names = ['classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict_dict.cPickle']

    #create_retrieval_image_ranked_score()
    #create_retrieval_image_ranked()


    for i in model_names:
        if not os.path.exists(root + i):
            try:
                create_predict_dict_from_cpickle_multevent(root + ('_').join(i.split('_')[:-1]))
            except:
                try:
                    create_predict_dict_from_cpickle(root + ('_').join(i.split('_')[:-1]))
                except:
                    create_predict_dict(root + ('_').join(i.split('_')[:-1]))
    evaluate_rank_reweighted_permuted(model_names)

    precisions = []
    for i in xrange(5,36,3):
        precisions.append(evaluate_top20_permuted(model_names, i))
    for i in xrange(len(model_names)):
        temp = []
        for j in precisions:
            temp.append(j[i])
        print model_names[i], temp

    retrievals = []; percent = []
    for i in model_names:
        retrievals.append([])
    #for i in [20]:
    for i in xrange(5, 36, 3):
        temp, percent_temp = evaluate_MAP_permuted(model_names, min_retrieval=i)
        for j in xrange(len(temp)):
            retrievals[j].append(temp[j])
        percent.append(percent_temp)
    for i in retrievals:
        print i
    print percent

    #create_knn_from_finetune()