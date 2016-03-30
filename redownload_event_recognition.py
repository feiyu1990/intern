import csv
import cPickle
import operator
import random
import os
import numpy as np
from collections import Counter
import sys
import urllib
from collections import defaultdict
import h5py
import shutil

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/'

def find_consistent_result(name):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/'

    input_path = root +'results/'+name+'.csv'
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []

    output_index = []
    input_index = []
    tag_index = []
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            output_index.append(i)
        if 'event_id' in meta:
            input_index.append(i)
        if 'tag' in meta:
            tag_index.append(i)
    #feed_back_index = output_index[0]
    #output_index = output_index[1:]
    output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
    output_index = output_index_new
    HIT_result = []
    for meta in metas:
        HITids.append(meta[0])
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k]] for (j,i,k) in zip(input_index, output_index,tag_index)]])

    HIT_result_dic = {}
    for meta in HIT_result:
        HIT = meta[0]
        result = meta[1]
        if HIT in HIT_result_dic:
            HIT_result_dic[HIT].append(result)
        else:
            HIT_result_dic[HIT] = [result]
    count = 0
    agreement_event_id = []
    agreement_event_id_new = []
    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        if len(results) > 1:
            count+=1
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            #if max_result[1] >= 2 and max_result[1] > len_vote/2:
            # if max_result[1] == len(result_this) and max_result[1] > len_vote/2:
            #     agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
            if max_result[1] == len(results) or max_result[1] >= 4:
                agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
    print "number of events with more than 1 vote:", count*10
    print "number of events with more than 3 same votes:",len(agreement_event_id)
    # print "number of events with 3 same votes:",len(agreement_event_id_new)
    return agreement_event_id


def find_consistent_result_new(name):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/'
    img_to_download_list = dict()
    input_path = root +'results/'+name+'.csv'
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []

    output_index = []
    input_index = []
    tag_index = []
    img_index = [[],[],[],[],[],[],[],[],[],[]]
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            output_index.append(i)
        if 'event_id' in meta:
            input_index.append(i)
        if 'tag' in meta:
            tag_index.append(i)
        for ii in xrange(1,11):
            if 'Input.image'+str(ii)+'_' in meta:
                img_index[ii - 1].append(i)

    #feed_back_index = output_index[0]
    #output_index = output_index[1:]
    output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
    output_index = output_index_new
    HIT_result = []
    for meta in metas:
        HITids.append(meta[0])
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k]] for (j,i,k) in zip(input_index, output_index,tag_index)]])

    last_HIT = ''
    for meta in metas:
        HITid = meta[0]
        if HITid == last_HIT:
            continue
        last_HIT = HITid
        for i in xrange(10):
            this_event = meta[input_index[i]]
            img_to_download_list[this_event] = []
            for j in img_index[i]:
                if meta[j] == 'NA':
                    continue
                img_to_download_list[this_event].append(meta[j])

    filtered_img_to_download_list = dict()

    HIT_result_dic = {}
    for meta in HIT_result:
        HIT = meta[0]
        result = meta[1]
        if HIT in HIT_result_dic:
            HIT_result_dic[HIT].append(result)
        else:
            HIT_result_dic[HIT] = [result]
    count = 0
    # agreement_event_id = []
    # agreement_event_id_new = []
    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        if len(results) > 1:
            count+=1
        # len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            #if max_result[1] >= 2 and max_result[1] > len_vote/2:
            # if max_result[1] == len(result_this) and max_result[1] > len_vote/2:
            #     agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
            if max_result[1] == len(results) or max_result[1] >= 4:
                # agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
                filtered_img_to_download_list[results[0][i][1]] = img_to_download_list[results[0][i][1]]

    print len(filtered_img_to_download_list)
    return filtered_img_to_download_list
    # print "number of events with more than 1 vote:", count*10
    # print "number of events with more than 3 same votes:",len(agreement_event_id)
    # print "number of events with 3 same votes:",len(agreement_event_id_new)
    # return agreement_event_id


def create_save_list():
    root = '/home/feiyu1990/local/event_curation/'
    f = open(root+ 'all_event_img_list.pkl','r')
    all_events = cPickle.load(f)
    f.close()
    events_already_have = []
    for event_name in dict_name2:
        f = open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/test_event_id.cPickle','r')
        events_already_have.extend(cPickle.load(f))
        f.close()
        f = open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/training_event_id.cPickle','r')
        events_already_have.extend(cPickle.load(f))
        f.close()
        print len(events_already_have)
    all_events_not_have = dict()
    for event in all_events:
        if event in events_already_have:
            print event
            continue
        all_events_not_have[event] = all_events[event]
    print len(all_events_not_have)

    f = open(root+'needdownload_event_img_list.pkl','w')
    cPickle.dump(all_events_not_have, f)
    f.close()

    # f = open(root + 'download_img_list.txt','w')
    # for event in all_events_not_have:
    #     # os.mkdir('/mnt/ilcompf3d1/user/yuwang/event_recognition/'+ event)
    #     for img in all_events_not_have[event]:
    #         f.write(img + '\t' + '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + '\n')
    # f.close()


def download_img():
    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    f = open(root+'codes/needdownload_event_img_list_2.pkl','r')
    all_events = cPickle.load(f)
    f.close()
    failed_img_list = []
    i = 0
    for event in all_events:
        i += 1
        if not os.path.exists('/mnt/ilcompf3d1/user/yuwang/event_recognition/'+ event):
            os.mkdir('/mnt/ilcompf3d1/user/yuwang/event_recognition/'+ event)
        for img in all_events[event]:
            try:
                urllib.urlretrieve(img,'/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg')
            except:
                failed_img_list.append(img)
        print i, event

    f = open(root+'failed_img_2.pkl','w')
    cPickle.dump(failed_img_list, f)
    f.close()


def create_training_img():
    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    img_list = []
    for event in new_events:
        event_type = event_type_dict[event]
        try:
            dict_name2[event_type]
        except:
            print event_type
            continue
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            img_list.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
    print len(img_list)
    path=  '/mnt/ilcompf3d1/user/yuwang/event_curation/CNN_all_event_1205/data/training_list.txt'
    with open(path, 'r') as data:
        for line in data:
            img_list.append(line)
    print len(img_list)
    random.shuffle(img_list)

    f = open(root + 'CNN_all_event_1205/data/training_img_list_expand.txt','w')
    for img in img_list:
        f.write(img)
    f.close()

def create_training_img_balanced():
    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    img_list_dict = defaultdict(list)
    for event in new_events:
        event_type = event_type_dict[event]
        try:
            dict_name2[event_type]
        except:
            print event_type
            continue
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            img_list_dict[event_type].append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
            # img_list.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
    img_list = []
    threshold = 10000
    for event_type in img_list_dict:
        print event_type, len(img_list_dict[event_type])
        if len(img_list_dict[event_type]) > threshold:
            temp = random.sample(img_list_dict[event_type], threshold)
            img_list.extend(temp)
        else:
            img_list.extend(img_list_dict[event_type])
    print len(img_list)

    # print len(img_list_dict)
    path=  '/mnt/ilcompf3d1/user/yuwang/event_curation/CNN_all_event_1205/data/training_list.txt'
    with open(path, 'r') as data:
        for line in data:
            img_list.append(line)
    print len(img_list)
    random.shuffle(img_list)

    f = open(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced.txt','w')
    for img in img_list:
        f.write(img)
    f.close()

def create_training_img_balanced_fully(expand_ratio = 2):
    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    img_list_dict = defaultdict(list)
    for event in new_events:
        event_type = event_type_dict[event]
        try:
            dict_name2[event_type]
        except:
            print event_type
            continue
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            img_list_dict[event_type].append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
            # img_list.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')

    for event_type in img_list_dict:
        print event_type, len(img_list_dict[event_type])

    ori_img_list_dict = defaultdict(list)
    path=  '/mnt/ilcompf3d1/user/yuwang/event_curation/CNN_all_event_1205/data/training_list.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\n')[0]
            i,j = meta.split(' ')
            ori_img_list_dict[int(j)].append(line)
    for event_type in dict_name2:
        print event_type, len(ori_img_list_dict[dict_name2[event_type] - 1])

    new_img_list_dict = defaultdict(list)
    for event_type in dict_name2:
        len_ = len(ori_img_list_dict[dict_name2[event_type] - 1]) * (expand_ratio - 1)
        if len_ < len(img_list_dict[event_type]):
            temp = random.sample(img_list_dict[event_type], len_)
        else:
            len_needed = len_ - len(img_list_dict[event_type])
            temp = img_list_dict[event_type] + ori_img_list_dict[dict_name2[event_type] - 1]
            temp_ = [random.choice(temp) for i in xrange(len_needed)]
            temp = temp_ + img_list_dict[event_type]
        new_img_list_dict[event_type] = temp + ori_img_list_dict[dict_name2[event_type] - 1]
    for event_type in dict_name2:
        print event_type, len(new_img_list_dict[event_type])


    img_list = []
    for event_type in new_img_list_dict:
        img_list.extend(new_img_list_dict[event_type])
    random.shuffle(img_list)
    print len(img_list)
    f = open(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_'+str(expand_ratio)+'.txt','w')
    for img in img_list:
        f.write(img)
    f.close()

def create_training_img_balanced_equally(expand_ratio = 2):
    ori_img_list_dict = defaultdict(list)
    path=  '/mnt/ilcompf3d1/user/yuwang/event_curation/CNN_all_event_1205/data/training_list.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\n')[0]
            i,j = meta.split(' ')
            ori_img_list_dict[int(j)].append(line)
    for event_type in dict_name2:
        print event_type, len(ori_img_list_dict[dict_name2[event_type] - 1])

    max_ori_len = max([len(ori_img_list_dict[dict_name2[event_type] - 1]) for event_type in dict_name2])
    expand_image_len = max_ori_len * (expand_ratio + 1)

    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    img_list_dict = defaultdict(list)
    for event in new_events:
        event_type = event_type_dict[event]
        try:
            dict_name2[event_type]
        except:
            print event_type
            continue
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            img_list_dict[event_type].append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
            # img_list.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
    for event_type in img_list_dict:
        print event_type, len(img_list_dict[event_type])


    img_list_combined = dict()
    for event_type in img_list_dict:
        need_length = expand_image_len - len(ori_img_list_dict[dict_name2[event_type] - 1])
        if len(img_list_dict[event_type]) > need_length:
            temp = random.sample(img_list_dict[event_type], need_length)
            temp = temp + ori_img_list_dict[dict_name2[event_type] - 1]
        else:
            temp = img_list_dict[event_type] + ori_img_list_dict[dict_name2[event_type] - 1]
            len_needed = expand_image_len - len(temp)
            temp_ = [random.choice(temp) for i in xrange(len_needed)]
            temp = temp + temp_
        img_list_combined[event_type] = temp
        print event_type, len(img_list_combined[event_type])

    img_list = []
    for event_type in img_list_combined:
        img_list.extend(img_list_combined[event_type])
    random.shuffle(img_list)
    # print img_list[:100]

    f = open(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_3_equal.txt','w')
    for img in img_list:
        f.write(img)
    f.close()

def create_training_img_balanced_fully_importance_scaled(expand_ratio = 3, important_threshold = 0):
    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    root = '/home/feiyu1990/local/event_curation/'
    training_score_all = dict()
    for event_name in dict_name2:
        # f = open(root + 'baseline_all_0509/' + event_name +'/training_image_ids.cPickle','r')
        # training_img_id = cPickle.load(f)
        # f.close()
        f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_dict_v2.cPickle','r')
        training_score_dict = cPickle.load(f)
        f.close()
        training_score_all.update(training_score_dict)
    # print training_score_all
    training_score_all_new = dict()
    for img in training_score_all:
        training_score_all_new[img.split('_')[1]] = training_score_all[img]

    importance_score = []
    new_img_list = []
    scores = []
    with open(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_'+str(expand_ratio)+'.txt','r') as data:
        for line in data:
            if 'recognition' in line:
                importance_score.append(0.814)
            else:
                im_id = line.split('/')[-2]+'/' + line.split('/')[-1].split('.')[0]
                score = training_score_all_new[im_id]
                # print score
                if score <= important_threshold:
                    continue
                scores.append(score)
                importance_score.append(score ** 3)
            new_img_list.append(line)
    print len(new_img_list), len(importance_score)
    print np.mean(scores)
    f = h5py.File(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_'+str(expand_ratio)+'_importance_label_scaled_3.h5','w')
    f.create_dataset("importance", data=importance_score)
    f.close()
    f = open(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_'+str(expand_ratio)+'_importance_label_scaled_3.txt','w')
    f.write(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_'+str(expand_ratio)+'_importance_label_scaled_3.h5')
    f.close()

    f = open(root + 'CNN_all_event_1205/data/training_img_list_expand_balanced_'+str(expand_ratio)+'_importance.txt', 'w')
    for line in new_img_list:
        f.write(line)
    f.close()

def consistant_result_record():
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/'
    f = open(root+'results/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8_pre']
    dict_event_type = dict()
    for name in names:
        input_path = root +'results/'+name+'.csv'
        line_count = 0
        head_meta = []
        metas = []
        with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    head_meta = meta
                else:
                    metas.append(meta)
                line_count += 1
        HITids = []

        output_index = []
        input_index = []
        tag_index = []
        img_index = [[],[],[],[],[],[],[],[],[],[]]
        for i in xrange(len(head_meta)):
            meta = head_meta[i]
            if 'Answer.type' in meta:
                output_index.append(i)
            if 'event_id' in meta:
                input_index.append(i)
            if 'tag' in meta:
                tag_index.append(i)
            for ii in xrange(1,11):
                if 'Input.image'+str(ii)+'_' in meta:
                    img_index[ii - 1].append(i)

        #feed_back_index = output_index[0]
        #output_index = output_index[1:]
        output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
        output_index = output_index_new
        HIT_result = []
        for meta in metas:
            HITids.append(meta[0])
            HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k]] for (j,i,k) in zip(input_index, output_index,tag_index)]])


        HIT_result_dic = {}
        for meta in HIT_result:
            HIT = meta[0]
            result = meta[1]
            if HIT in HIT_result_dic:
                HIT_result_dic[HIT].append(result)
            else:
                HIT_result_dic[HIT] = [result]
        count = 0
        # agreement_event_id = []
        # agreement_event_id_new = []
        for HIT in HIT_result_dic:
            results = HIT_result_dic[HIT]

            if len(results) > 1:
                count+=1
            # len_vote = len(results)
            for i in xrange(len(results[0])):
                if results[0][i][1] not in new_events:
                        continue
                result_this = {}
                for result in results:
                    if result[i][2] in result_this:
                        result_this[result[i][2]] += 1
                    else:
                        result_this[result[i][2]] = 1
                max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
                #if max_result[1] >= 2 and max_result[1] > len_vote/2:
                # if max_result[1] == len(result_this) and max_result[1] > len_vote/2:
                #     agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
                if max_result[1] == len(results) or max_result[1] >= 4:
                    # agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
                    # filtered_img_to_download_list[results[0][i][1]] = img_to_download_list[results[0][i][1]]

                    dict_event_type[results[0][i][1]] = max_result[0]
    print len(dict_event_type)
    f = open(root+'new_consistent_img_and_type.pkl','w')
    cPickle.dump(dict_event_type, f)
    f.close()

def create_training_img_balanced_fully_lstm(expand_ratio = 2):
    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'
    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    new_event_dict = defaultdict(dict)
    for event in new_events:
        temp = []
        event_type = event_type_dict[event]
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                # print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            else:
                temp.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg')
        if len(temp) != len(new_events[event]):
            continue
        else:
            new_event_dict[event_type_dict[event]][event] = temp
    for event_type in new_event_dict:
        print len(new_event_dict[event_type])


    ori_event_dict = defaultdict(dict)
    for event_name in dict_name2:
        f = open('/mnt/ilcompf3d1/user/yuwang/event_curation/baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
        temp = cPickle.load(f)
        f.close()
        for img in temp:
            event_ = img.split('/')[0]
            if event_ in ori_event_dict[event_name]:
                ori_event_dict[event_name][event_].append('/mnt/ilcompf3d1/user/yuwang/event_curation/curation_images/' + event_ + '/' + img + '.jpg')
            else:
                ori_event_dict[event_name][event_] = ['/mnt/ilcompf3d1/user/yuwang/event_curation/curation_images/' + event_ + '/' + img + '.jpg']

    img_list_to_copy = []
    combine_event_list = defaultdict(dict)
    for event_name in dict_name2:
        combine_event_list[event_name].update(ori_event_dict[event_name])
        event_to_add = new_event_dict[event_name]
        event_old = ori_event_dict[event_name]
        len_to_add = expand_ratio * len(event_old)
        print event_name, len_to_add, len(event_to_add)
        if len_to_add < len(event_to_add):

            temp = random.sample(range(len(event_to_add)), len_to_add)
            count = 0
            for event_ in event_to_add:
                if count in temp:
                    combine_event_list[event_name][event_] = event_to_add[event_]
                    img_list_to_copy.extend(event_to_add[event_])
                    # print len(img_list_to_copy)
                count += 1
        else:
            combine_event_list[event_name].update(event_to_add)
            for i in event_to_add:
                img_list_to_copy.extend(event_to_add[i])
                # print len(img_list_to_copy)
    for event_name in combine_event_list:
        print event_name, len(combine_event_list[event_name])

    f = open('/mnt/ilcompf3d1/user/yuwang/lstm/new_img_to_copy.pkl', 'w')
    cPickle.dump(img_list_to_copy, f)
    f.close()

    f = open('/mnt/ilcompf3d1/user/yuwang/lstm/new_combine_event_list.pkl', 'w')
    cPickle.dump(combine_event_list, f)
    f.close()


def create_needdownload_img_list():

    with open('/home/feiyu1990/local/event_curation/lstm/new_img_to_copy.pkl') as f:
        img_list = cPickle.load(f)
    with open('/home/feiyu1990/local/event_curation/lstm/new_combine_event_list.pkl') as f:
        event_combine_list = cPickle.load(f)
    event_list = dict()
    for event_name in event_combine_list:
        events_this = event_combine_list[event_name]
        with open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/training_event_id.cPickle') as f:
            event_already_have = cPickle.load(f)
        for event in events_this:
            if event in event_already_have:
                continue
            event_list[event] = event_name
    f = open('/home/feiyu1990/local/event_curation/lstm/new_training_img_list.txt', 'w')
    for img in img_list:
        f.write('/home/feiyu1990/local/event_curation/lstm/new_imgs/' + '/'.join(img.split('/')[-2:]) + ' ' + str(dict_name2[event_list[img.split('/')[-2]]] - 1) + '\n')
    f.close()



def copy_to_redownload():
    with open('/mnt/ilcompf3d1/user/yuwang/lstm/new_img_to_copy.pkl') as f:
        img_list = cPickle.load(f)
    count = 0
    for img in img_list:
        count += 1
        if count % 1000 == 0:
            print count
        if not os.path.exists('/mnt/ilcompf3d1/user/yuwang/lstm/new_imgs/' + img.split('/')[-2]):
            os.mkdir('/mnt/ilcompf3d1/user/yuwang/lstm/new_imgs/' + img.split('/')[-2])
        if not os.path.exists('/mnt/ilcompf3d1/user/yuwang/lstm/new_imgs/' + '/'.join(img.split('/')[-2:])):
            shutil.copy(img, '/mnt/ilcompf3d1/user/yuwang/lstm/new_imgs/' + '/'.join(img.split('/')[-2:]))


def create_training_img_balanced_fully_multilabel(input_dict_name, path_this, expand_ratio = 3):

    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'

    ori_img_list_dict = defaultdict(list)
    path = '/mnt/ilcompf3d1/user/yuwang/event_curation/multi_label_event/'+ input_dict_name
    with open(path) as f:
        multi_event_dict = cPickle.load(f)
    multi_img_dict = defaultdict(list)
    for event_type in dict_name2:
        with open('/mnt/ilcompf3d1/user/yuwang/event_curation/baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            img_list = cPickle.load(f)
        for img in img_list:
            event_id = img.split('/')[0]
            multi_img_dict['/mnt/ilcompf3d1/user/yuwang/event_curation/curation_images/' + event_type + '/' + img.split('_')[1] + '.jpg'] = [dict_name2[i[0]] - 1 for i in multi_event_dict[event_id]]
    # with open('/mnt/ilcompf3d1/user/yuwang/event_curation/multi_label_event/img_event_type_dict.pkl','w') as f:
    #     cPickle.dump(multi_img_dict, f)


    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    img_list_dict = defaultdict(list)
    for event in new_events:
        event_type = event_type_dict[event]
        try:
            dict_name2[event_type]
        except:
            print event_type
            continue
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            img_list_dict[event_type].append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')
            # img_list.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')

    for event_type in img_list_dict:
        print event_type, len(img_list_dict[event_type])

    path=  '/mnt/ilcompf3d1/user/yuwang/event_curation/CNN_all_event_1205/data/training_list.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\n')[0]
            i,j = meta.split(' ')
            event_type_list = multi_img_dict[i]
            if len(event_type_list) >= 2:
                event_type_select = event_type_list
                if len(event_type_list) > 2:
                    print i, event_type_select
            else:
                event_type_select = [event_type_list[0], event_type_list[0]]
            for j in event_type_select:
                ori_img_list_dict[j].append(i + ' ' + str(j) + '\n')
    for event_type in dict_name2:
        print event_type, len(ori_img_list_dict[dict_name2[event_type] - 1])
    with open('/mnt/ilcompf3d1/user/yuwang/event_curation/multi_label_event/ori_img_list.pkl','w') as f:
        cPickle.dump(ori_img_list_dict, f)

    new_img_list_dict = defaultdict(list)
    for event_type in dict_name2:
        len_ = len(ori_img_list_dict[dict_name2[event_type] - 1]) / 2 * (expand_ratio - 1)
        if len_ < len(img_list_dict[event_type]):
            temp = random.sample(img_list_dict[event_type], len_)
        else:
            len_needed = len_ - len(img_list_dict[event_type])
            temp = img_list_dict[event_type] + ori_img_list_dict[dict_name2[event_type] - 1]
            temp_ = [random.choice(temp) for i in xrange(len_needed)]
            temp = temp_ + img_list_dict[event_type]
        new_img_list_dict[event_type] = temp + ori_img_list_dict[dict_name2[event_type] - 1]
    for event_type in dict_name2:
        print event_type, len(new_img_list_dict[event_type])


    img_list = []
    for event_type in new_img_list_dict:
        img_list.extend(new_img_list_dict[event_type])
    random.shuffle(img_list)
    print len(img_list)
    f = open(root + path_this + '/data/multilabel_training_img_list_expand_balanced_'+str(expand_ratio)+'.txt','w')
    for img in img_list:
        f.write(img)
    f.close()


def create_training_img_softmax(input_dict_name, path_this, expand_ratio = 3):

    root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'

    ori_img_list_dict = defaultdict(list)
    path = root + 'multi_label_event/'+ input_dict_name
    with open(path) as f:
        multi_event_dict = cPickle.load(f)
    multi_img_dict = defaultdict(list)
    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            img_list = cPickle.load(f)
        for img in img_list:
            event_id = img.split('/')[0]
            softmax_this = np.zeros((23,))
            for event in multi_event_dict[event_id]:
                softmax_this[dict_name2[event[0]] - 1] = event[1]
            multi_img_dict['/mnt/ilcompf3d1/user/yuwang/event_curation/curation_images/' + event_type + '/' + img.split('_')[1] + '.jpg'] = softmax_this
    # with open(root + 'multi_label_event/img_event_type_dict_softmax.pkl','w') as f:
    #     cPickle.dump(multi_img_dict, f)


    f = open(root+'codes/needdownload_event_img_list.pkl','r')
    new_events = cPickle.load(f)
    f.close()
    f = open(root+'codes/new_consistent_img_and_type.pkl','r')
    event_type_dict = cPickle.load(f)
    f.close()

    img_list_dict = defaultdict(list)
    for event in new_events:
        event_type = event_type_dict[event]
        try:
            dict_name2[event_type]
        except:
            print event_type
            continue
        for img in new_events[event]:
            if os.stat('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg').st_size < 5000:
                print '/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg'
                continue
            softmax_this = np.zeros((23,))
            softmax_this[dict_name2[event_type] - 1] = 1
            img_list_dict[event_type].append(('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' 0\n', softmax_this))
            # img_list.append('/mnt/ilcompf3d1/user/yuwang/event_recognition/' + event + '/' + img.split('/')[-1].split('_')[0] + '.jpg' + ' ' + str(dict_name2[event_type] - 1) + '\n')

    for event_type in img_list_dict:
        print event_type, len(img_list_dict[event_type])

    path=  '/mnt/ilcompf3d1/user/yuwang/event_curation/CNN_all_event_1205/data/training_list.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\n')[0]
            i,j = meta.split(' ')
            event_type_list = multi_img_dict[i]
            ori_img_list_dict[int(j)].append((i + ' 0\n', event_type_list))

    for event_type in dict_name2:
        print event_type, len(ori_img_list_dict[dict_name2[event_type] - 1])
    # with open('/mnt/ilcompf3d1/user/yuwang/event_curation/multi_label_event/ori_img_list_softmax.pkl','w') as f:
    #     cPickle.dump(ori_img_list_dict, f)

    new_img_list_dict = defaultdict(list)
    for event_type in dict_name2:
        len_ = len(ori_img_list_dict[dict_name2[event_type] - 1]) * (expand_ratio - 1)
        if len_ < len(img_list_dict[event_type]):
            temp = random.sample(img_list_dict[event_type], len_)
        else:
            len_needed = len_ - len(img_list_dict[event_type])
            temp = img_list_dict[event_type] + ori_img_list_dict[dict_name2[event_type] - 1]
            temp_ = [random.choice(temp) for i in xrange(len_needed)]
            temp = temp_ + img_list_dict[event_type]
        new_img_list_dict[event_type] = temp + ori_img_list_dict[dict_name2[event_type] - 1]
    for event_type in dict_name2:
        print event_type, len(new_img_list_dict[event_type])


    img_list = []
    for event_type in new_img_list_dict:
        img_list.extend(new_img_list_dict[event_type])
    random.shuffle(img_list)
    print len(img_list)
    f = open(root + path_this + '/data/multilabel_training_img_list_expand_balanced_'+str(expand_ratio)+'_softmax.txt','w')
    label_list = []
    for img in img_list:
        f.write(img[0])
        label_list.append(img[1])
    f.close()
    np.save(root + path_this + '/data/multilabel_training_img_list_expand_balanced_'+str(expand_ratio)+'_softmax_label.npy', np.array(label_list))


def create_h5py():
    feature = np.load('multilabel_training_img_list_expand_balanced_4_softmax_label.npy')
    f = h5py.File('multilabel_training_img_list_expand_balanced_4_softmax_label.h5','w')
    print feature.shape
    f.create_dataset("importance", data=feature)
    f.close()

if __name__ == '__main__':
    create_h5py()
    # a = find_consistent_result_new('0')
    # b = find_consistent_result_new('1')
    # c = find_consistent_result_new('2')
    # e = find_consistent_result_new('3')
    # f = find_consistent_result_new('4')
    # g = find_consistent_result_new('5')
    # h = find_consistent_result_new('8_pre')
    # i = find_consistent_result_new('6')
    # j = find_consistent_result_new('7')
    # all_events = dict()
    # all_events.update(a)
    # all_events.update(b)
    # all_events.update(c)
    # all_events.update(h)
    # all_events.update(e)
    # all_events.update(f)
    # all_events.update(g)
    # all_events.update(i)
    # all_events.update(j)
    # print len(all_events)
    # f = open(root+'all_event_img_list.pkl','w')
    # cPickle.dump(all_events, f)
    # f.close()
    # all_agreements = a+b+c+e+f+g+h+i+j
    # f = open(root+'all_event_3agreements.cPickle','w')
    # cPickle.dump(all_agreements, f)
    # f.close()
    # download_img()
    # consistant_result_record()
    # create_save_list()
    # create_training_img_balanced()
    # create_training_img_balanced_fully(4)
    # create_training_img_balanced_fully_importance_scaled()
    # create_training_img_balanced_equally()
    # create_training_img_balanced_fully_lstm()
    # create_training_img_balanced_fully_lstm()
    # create_needdownload_img_list()
    # copy_to_redownload()

    # name = 'new_multiple_result_2round_removedup_vote.pkl'
    # folder_name = 'CNN_all_event_corrected_multi_removedup_vote'
    # create_training_img_balanced_fully_multilabel(name, folder_name)
    # create_training_img_balanced_fully_multilabel(name,folder_name,1)
    # create_training_img_balanced_fully_multilabel(name,folder_name,4)
    #
    # name = 'new_multiple_result_2round_softmaxall_removedup_vote.pkl'
    # folder_name = 'CNN_all_event_corrected_multi_removedup_vote_soft'
    # create_training_img_softmax(name, folder_name, expand_ratio=4)
    # create_training_img_softmax(name, folder_name)