__author__ = 'wangyufei'
import csv
import random
import cPickle
import numpy as np
import operator
from collections import Counter
from sklearn.preprocessing import normalize
from numpy import linalg
import re
import os
from operator import itemgetter
from PIL import Image
import copy
from scipy import io
from scipy.optimize import curve_fit

root = '/Users/wangyufei/Documents/Study/intern_adobe/'
root = '/home/feiyu1990/local/event_curation/'
#root = '/mnt/ilcompf2d0/project/yuwang/'
#root = 'C:/Users/yuwang/Documents/'
model_name = 'alexnet6k'
#model_name = 'vgg'
n_vote = 10
#model_name = 'alexnet3k'
#face_model = '_birthday_iter_30000_sigmoid1.cPickle'
face_model = '_sigmoidcropped_importance_allevent_iter_100000.cPickle'
#combine_face_model = '_sigmoid9_10_segment_2round_iter_750000.cPickle'
face_combined_model = '_combine_face5'
#combine_face_model = '_combined_10.cPickle'
combine_face_model = '_sigmoid9_10_segment_3time_iter_100000.cPickle'

'''some correctness checking / correcting'''

'''not specific to wedding'''


dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
             'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)','Independence':'Independence Day',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

def create_path_all():
    #root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'
    #root1 = '/mnt/ilcompf2d0/project/yuwang/event_curation/clean_input_and_label/3_event_curation/'
    root1 = root
    in_path = root1 + 'all_output/all_output.csv'
    line_count = -1
    event_ids = []
    with open(in_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            line_count += 1
            if line_count == 0:
                continue
            if meta[28] not in event_ids:
                event_ids.append(meta[28])
    '''
    event_ids_this = []
    in_path = root + 'all_images_curation.txt'
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event = meta[1]+'_'+meta[3]
            if event not in event_ids_this:
                event_ids_this.append(event)
    event_lost = []
    for i in event_ids_this:
        if i not in event_ids:
            event_lost.append(i)
    '''
    in_path = root + 'clean_imgs.txt'
    out_path = root1 + 'all_images_curation.txt'
    f = open(out_path, 'wb')
    with open(in_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            if meta[1] + '_' + meta[3] in event_ids:
                f.write(line)
    f.close()

'''all dataset'''
def create_event_id_dict():
    input_path = root + 'amt_data_collection/3_event_curation/all_output/all_output.csv'
    input_path = root + 'amt/clean_input_and_label/3_event_curation/all_output/all_output.csv'
    line_count = 0
    head_meta = []
    events = {}
    last_id = ''
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] != last_id:
                last_id = meta[0]
                if meta[29] not in events:
                    if meta[29] in dict_name:
                        if dict_name[meta[29]] not in events:
                            events[dict_name[meta[29]]] = [meta[28]]
                        else:
                            events[dict_name[meta[29]]] += [meta[28]]
                    else:
                        events[meta[29]] = [meta[28]]
                else:
                    events[meta[29]] += [meta[28]]
            line_count+=1
    return events
def event_separate_trainval(events, event_name):

    check_type = event_name
    print check_type
    events_set = events[check_type]
    events = list(events_set)
    events_training = random.sample(events,len(events)*3/4)
    events_test = [event for event in events if event not in events_training]
    if not os.path.exists(root + 'baseline_all_old/'+check_type):
        os.mkdir(root + 'baseline_all_old/'+check_type)
    f = open(root + 'baseline_all_old/'+check_type+'/training_event_id.cPickle', 'wb')
    cPickle.dump(events_training, f)
    f.close()
    f = open(root + 'baseline_all_old/'+check_type+'/test_event_id.cPickle', 'wb')
    cPickle.dump(events_test, f)
    f.close()
    create_path(check_type)
def create_path(check_type):
    in_path = root + 'baseline_all_old/'+check_type+'/training_event_id.cPickle'
    f = open(in_path, 'r')
    events_training = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_all_old/'+check_type+'/test_event_id.cPickle'
    f = open(in_path, 'r')
    events_test = cPickle.load(f)
    f.close()

    load_path = root+'all_output/all_images_curation.txt'
    save_paths1 = root + 'baseline_all_old/'+check_type+'/training_path.txt'
    f1 = open(save_paths1, 'wb')
    save_paths2 = root + 'baseline_all_old/'+check_type+'/test_path.txt'
    f2 = open(save_paths2, 'wb')
    with open(load_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            if meta[1]+'_'+meta[3] in events_training:
                path = 'C:\Users\yuwang\Documents\download_event_recognition\\'+meta[3]+'\\'+meta[2]+'.jpg\r\n'
                f1.write(path)
            elif meta[1]+'_'+meta[3] in events_test:
                path = 'C:\Users\yuwang\Documents\download_event_recognition\\'+meta[3]+'\\'+meta[2]+'.jpg\r\n'
                f2.write(path)
    f1.close()
    f2.close()
def linux_create_path(check_type):
    in_path = root + 'baseline_all_old/'+check_type+'/training_event_id.cPickle'
    f = open(in_path, 'r')
    events_training = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_all_old/'+check_type+'/test_event_id.cPickle'
    f = open(in_path, 'r')
    events_test = cPickle.load(f)
    f.close()
    load_path = root+'all_output/all_images_curation.txt'
    save_paths1 = root + 'baseline_all_old/'+check_type+'/linux_training_path.txt'
    f1 = open(save_paths1, 'wb')
    save_paths2 = root + 'baseline_all_old/'+check_type+'/linux_test_path.txt'
    f2 = open(save_paths2, 'wb')
    prefix = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/'
    with open(load_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            if meta[1]+'_'+meta[3] in events_training:
                string = prefix + meta[3]+'/'+meta[2] + '.jpg 0\n'
                f1.write(string)
            elif meta[1]+'_'+meta[3] in events_test:
                string = prefix + meta[3]+'/'+meta[2] + '.jpg 0\n'
                f2.write(string)
    f1.close()
    f2.close()
def from_npy_to_dicts(check_type, type = 'training'):
    #f = open(root + 'baseline_wedding_test/wedding_'+type+'_features.cPickle','r')
    #features = cPickle.load(f)
    #f.close()
    image_ids = []
    in_path = root + 'baseline_all_old/'+check_type+'/linux_'+type+'_path.txt'
    with open(in_path,'r') as data:
        for line in data:
            name = line.split('/')[-1]
            name = name.split('.')[0]
            image_ids.append(name)

    event_ids = []
    i = 0
    in_path = root+'all_output/all_images_curation.txt'
    with open(in_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            id = meta[2]
            if id == image_ids[i]:
                i += 1
                event_ids.append(meta[1] + '_' + meta[3] + '/' + meta[2])
                if i == len(image_ids):
                    break
    '''
    last_event = ''
    event_dict = {}
    for event_id, feature in zip(event_ids, features):
        image_name = event_id.split('/')
        event_id = image_name[0]
        image_name = image_name[1]
        if event_id == last_event:
            event_dict[event_id].append((image_name, feature))
        else:
            last_event = event_id
            event_dict[event_id] = [(image_name, feature)]
    f = open(root + 'baseline_wedding_test/wedding_'+type+'_features_and_events.cPickle','wb')
    cPickle.dump(event_dict, f)
    f.close()
    '''
    f = open(root + 'baseline_all_old/' + check_type + '/' + type+'_image_ids.cPickle','wb')
    cPickle.dump(event_ids, f)
    f.close()
def create_dict_url(check_type):
    path = root + 'baseline_all_old/' + check_type + '/training_image_ids.cPickle'
    f = open(path, 'r')
    image_ids = cPickle.load(f)
    f.close()
    dict = {}

    path = root+'all_output/all_images_curation.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            image_id = meta[1] + '_' + meta[3] + '/'+meta[2]
            if image_id in image_ids:
                dict[image_id] = meta[16]

    f = open(root + 'baseline_all_old/'+check_type+'/training_ulr_dict.cPickle', 'wb')
    cPickle.dump(dict, f)
    f.close()
    path = root + 'baseline_all_old/' + check_type + '/test_image_ids.cPickle'
    f = open(path, 'r')
    image_ids = cPickle.load(f)
    f.close()


    dict = {}
    path = root+'all_output/all_images_curation.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            image_id = meta[1] + '_' + meta[3] + '/'+meta[2]
            if image_id in image_ids:
                dict[image_id] = meta[16]

    f = open(root + 'baseline_all_old/'+check_type+'/test_ulr_dict.cPickle', 'wb')
    cPickle.dump(dict, f)
    f.close()

def create_knn_cPickle(name):
    path = root + 'baseline_all_old/' + name + '/'+model_name+'_knn.txt'
    f = open(root + 'baseline_all_old/'+name+'/training_ulr_dict.cPickle', 'r')
    training_url_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/'+name+'/test_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/'+name+'/test_image_ids.cPickle','r')
    test_images = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/'+name+'/training_image_ids.cPickle','r')
    training_images = cPickle.load(f)
    f.close()
    knn_dict = {}

    with open(path, 'r') as data:
        for line in data:
            test_index = int(line.split(':')[0])
            groups = re.findall('\((.*?)\)',line)
            this_img = test_images[test_index]
            knn_dict[test_index] = [(this_img, test_url_dict[this_img])]
            for i in groups:
                index = int(i.split(',')[0])
                score = float(i.split(',')[1])
                knn_dict[test_index].append((score, training_images[index], training_url_dict[training_images[index]]))
    print len(knn_dict[0]), len(knn_dict)
    f = open(path.split('.')[0]+'.cPickle','wb')
    cPickle.dump(knn_dict,f)
    f.close()
def create_csv(name):
    f = open(root + 'baseline_all_old/'+name+'/training_event_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/'+name+'/training.csv','wb')
    writer = csv.writer(f)
    line_count = 0
    input_path = root + 'all_output/all_output.csv'
    with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    writer.writerow(meta)
                else:
                    if meta[28] in event_ids:
                        writer.writerow(meta)
                line_count += 1
    f.close()

    f = open(root + 'baseline_all_old/'+name+'/test_event_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/'+name+'/test.csv','wb')
    writer = csv.writer(f)
    line_count = 0
    input_path = root + 'all_output/all_output.csv'
    with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    writer.writerow(meta)
                else:
                    if meta[28] in event_ids:
                        writer.writerow(meta)
                line_count += 1
    f.close()
def create_basic_files():
    events = {}
    path = root + 'all_output/all_output.csv'
    with open(path, 'rb') as data:
            line_count = -1
            reader = csv.reader(data)
            for meta in reader:
                line_count += 1
                if line_count == 0:
                    continue
                event_id = meta[28]
                event_type = meta[29]
                if event_type in events:
                    if event_id not in events[event_type]:
                        events[event_type].add(event_id)
                else:
                    if event_type in dict_name and dict_name[event_type] != event_type:
                        if event_id not in events[dict_name[event_type]]:
                            events[dict_name[event_type]].add(event_id)
                    else:
                        events[event_type] = {event_id}
    print events.keys()

    count = 0
    events_new = {}
    for i in events:
        for j in dict_name:
            if dict_name[j] == i:
                new_name = j
                break
        events_new[new_name] = events[i]
        count += len(events_new[new_name])

    print count

    for event_this in events_new:
        if event_this == 'Wedding':
            continue
        print event_this
        event_separate_trainval(events_new, event_this)
        create_path(event_this)
        linux_create_path(event_this)
        from_npy_to_dicts(event_this)
        from_npy_to_dicts(event_this,'test')
        create_dict_url(event_this)

block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
def read_amt_result(name, type = 'training'):
    input_path = root + 'baseline_all_old/' + name + '/' + type + '_image_ids.cPickle'
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
    input_path = root + 'baseline_all_old/' + name + '/' + type+'.csv'
    line_count = 0
    head_meta = []
    HITs = {}
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


    input_and_answers = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_distraction = 31
    for HITId in HITs:
        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        event_id = this_hit[0][index_event_id]
        input_and_answers[event_id] = []
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = []
        for submission in this_hit:
            if submission[index_worker_id] not in block_workers:
                this_hit_new.append(submission)
        num_valid_submission = len(this_hit_new)
        ii = 0
        for i in xrange(1, 1+num_images):
            if i==distract1 or i==distract2:
                print this_hit_new[0][image_input_index[i]]
                continue
            score = 0
            image_index = image_input_index[i]
            score_index = image_output_index[i]
            image_url = this_hit_new[0][image_index]
            for submission in this_hit_new:
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
            score = float(score)/float(num_valid_submission)
            input_and_answers[event_id].append((image_url, event_ids[event_id][ii], score))
            ii += 1
            #print (image_url, score)
    f = open(root + 'baseline_all_old/' + name + '/' + type + '_result_v1.cPickle','wb')
    cPickle.dump(input_and_answers, f)
    f.close()
def from_txt_to_pickle(name):
    in_path = root + 'baseline_all_old/'+name+'/'+model_name+'_training_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)

    f = open(root + 'baseline_all_old/'+name+'/'+model_name+'_training_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
    in_path = root + 'baseline_all_old/'+name+'/'+model_name+'_test_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)
    f = open(root + 'baseline_all_old/'+name+'/'+model_name+'_test_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
def find_similar(name, type = 'training', threshold = 0.9):
    image_path = root + 'baseline_all_old/'+name+'/vgg_' +type+'_features.cPickle'
    f = open(image_path, 'r')
    feature = cPickle.load(f)
    f.close()
    id_path = root + 'baseline_all_old/'+name+'/'+type+'_image_ids.cPickle'
    f = open(id_path, 'r')
    image_ids = cPickle.load(f)
    f.close()

    print len(feature)
    feature = np.asarray(feature)
    #feature = np.transpose(feature)
    print feature.shape
    feature_normalize = normalize(feature, axis=1)
    print linalg.norm(feature_normalize[0,:])
    ids = []
    urls = []
    times = []
    mat_path = root + 'all_output/all_images_curation.txt'
    kk = 0
    with open(mat_path, 'r') as data:
        for line in data:
            id = line.split('\t')[2]
            if kk == len(image_ids):
                break
            if id != image_ids[kk].split('/')[1]:
                continue
            kk += 1
            url = line.split('\t')[16]
            date = line.split('\t')[5]
            temp = date.split(' ')
            date_info = temp[0]
            time_info = temp[1]
            temp = date_info.split('-')
            m = temp[1]
            d = temp[2]
            y = temp[0]
            h = time_info.split(':')[0]
            minute = time_info.split(':')[1]
            second = time_info.split(':')[2]
            time_this = float(y+m+d+h+minute+second)
            times.append(time_this)
            urls.append(url)
            ids.append(id)

    for id, image_id in zip(ids, image_ids):
        if id != image_id.split('/')[1]:
            print 'ERROR DETECTED!'
            return

    feature_prev = feature_normalize[0,:]
    feature_start = feature_normalize[0,:]
    start_id = 0
    remove_list = []
    values = []
    for i in xrange(1, len(ids)):
        feature_this = feature_normalize[i,:]
        value = np.dot(feature_start, feature_this)

        #values.append(value)
        if value > threshold: # or (value>threshold*0.8 and times[i] - times[i-1] < 30):
            if i == start_id + 1:
                remove_list.append([image_ids[i-1], image_ids[i]])
            else:
                remove_list[-1].append(image_ids[i])
        else:
            #feature_prev = feature_normalize[i - 1,:]
            #value = np.dot(feature_prev, feature_this)
            #if value > threshold:
            #    remove_list[-1] = remove_list[-1][:-1]
            #    remove_list.append([image_ids[i-1], image_ids[i]])
            #    start_id = i - 1
            #    feature_start = feature_prev
            #else:
                start_id = i
                feature_start = feature_this
    f = open(root + 'baseline_all_old/'+name+'/vgg_'+type+'_similar_list.cPickle','wb')
    cPickle.dump(remove_list, f)
    f.close()
def correct_amt_result(name, type = 'test'):
    f = open(root + 'baseline_all_old/' + name + '/' + type + '_result_v1.cPickle','r')
    input_and_answers = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all_old/'+name+'/vgg_'+type+'_similar_list.cPickle','r')
    remove_list = cPickle.load(f)
    f.close()

    remove_list_split = {}
    for i in remove_list:
        event_id = i[0].split('/')[0]
        if event_id in remove_list_split:
            remove_list_split[event_id].append(i)
        else:
            remove_list_split[event_id] = [i]

    for event in input_and_answers:
        images = input_and_answers[event]
        if event not in remove_list_split:
            continue
        similar_list = remove_list_split[event]
        i = 0

        for index in xrange(len(images)):
            image =images[index]
            img_id = image[1]
            if img_id != similar_list[i][0]:
                continue
            this_group = similar_list[i]
            length = len(this_group)
            this_group_score = []
            for k in xrange(length):
                this_group_score.append(images[index+k][2])
            score = max(this_group_score)
            for k in xrange(length):
                input_and_answers[event][index+k] = (images[index+k][0],images[index+k][1], score)

    f = open(root + 'baseline_all_old/'+name+'/vgg_'+type+'_result_v2.cPickle','wb')
    cPickle.dump(input_and_answers, f)
    f.close()
    training_scores_dict = {}
    for event in input_and_answers:
        for img in input_and_answers[event]:
            training_scores_dict[img[1]] = img[2]
    f = open(root + 'baseline_all_old/'+name+'/vgg_'+type+'_result_dict_v2.cPickle', 'wb')
    cPickle.dump(training_scores_dict,f)
    f.close()
def baseline_predict(name, n_vote = n_vote):
    f = open(root + 'baseline_all_old/'+name+'/vgg_training_result_dict_v2.cPickle','r')
    training_scores_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/'+name+'/'+model_name+'_knn.cPickle', 'r')
    knn = cPickle.load(f)
    f.close()

    for i in knn:
        this_test_img = knn[i]
        this_test_id = this_test_img[0][0]
        if this_test_id in training_scores_dict:
            print 'ERROR!'

    test_prediction = []
    for ii in knn:
        this_test_img = knn[ii]
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

    f = open(root + 'baseline_all_old/'+name+'/'+model_name+'_predict_result_'+str(n_vote)+'_dict.cPickle','wb')
    cPickle.dump(test_prediction_event, f)
    f.close()

#baseline evaluate deprecated
'''
def evaluate_rank_reweighted_(name):
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_all_old/' + name + '/vgg_test_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_all_old/' + name + '/predict_result_10_dict.cPickle', 'r')
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
                    rank_difference[event_id] += (g - p) / g
                else:
                    rank_difference[event_id] += (p - g - n_same_g + 1) / g

        #print rank_difference
        print float(sum([rank_difference[i] for i in rank_difference]))/len(rank_difference)
def evaluate_MAP_ (name, min_retrieval = 5):
    maps = []
    n_ks = []
    APs = []
    #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
    #         #ground_truth = cPickle.load(f)
    #f.close()
    f = open(root + 'baseline_all_old/' + name + '/vgg_test_result_v2.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()
    rank_difference = {}
    f = open(root + 'baseline_all_old/' + name + '/predict_result_10_dict.cPickle', 'r')
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

            #threshold = ground_[min_retrieval-1][1]
            threshold = ground_[max(1, len(ground_)*min_retrieval/100)-1][1]
            n_k = len([i for i in ground_ if i[1] >= threshold])
            #if n_k > len(ground_)/2:
            #    print n_k, len(ground_), event_id
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
            AP = average_precision(ground_rank, predict_rank, n_k)
            APs.append([event_id, AP])
            n_ks.append([n_k, len(ground_)])
    #print APs
    maps.append(sum([i[1] for i in APs])/len(APs))
    percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))

    return maps, percent
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
    recall = [float(i[0])/i[1] for i in temp]
    ap = sum(recall)/len(recall)
    return ap
'''
def permute_groundtruth(event_name, type = 'val_test'):
    f = open(root + 'baseline_all_old/' + event_name + '/vgg_'+type+'_result_v2.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()
    ground_truth_new = {}
    for idx in ground_truth:
        temp = []
        for i in ground_truth[idx]:
            temp.append((i[0], i[1], i[2]+random.uniform(-0.02, 0.02)))
        ground_truth_new[idx] = temp
    f = open(root + 'baseline_all_old/' + event_name + '/vgg_'+type+'_result_v2_permuted.cPickle','wb')
    cPickle.dump(ground_truth_new, f)
    f.close()

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
def evaluate_rank_reweighted_permuted(event_name, model_names, permuted = '_permuted',  type = 'val_test'):
    f = open(root + 'baseline_all_old/' + event_name + '/vgg_'+type+'_result_v2'+permuted+'.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()
    retval = []
    for model_name in model_names:
        rank_difference = {}
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        #print predict_result
        for event_id in ground_truth:

            rank_difference[event_id] = 0
            ground_this = ground_truth[event_id]
            predict_this = predict_result[event_id]
            predict_ = [i[2] for i in predict_this]
            ground_ = [i[2] for i in ground_this]
            predict_ = zip(xrange(len(predict_)), predict_)
            ground_ = zip(xrange(len(ground_)), ground_)
            predict_.sort(key = lambda x: x[1], reverse=True)
            ground_.sort(key = lambda x: x[1],reverse=True)

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
        retval.append(float(sum([rank_difference[i] for i in rank_difference]))/len(rank_difference))
    return retval
def evaluate_MAP_permuted(event_name, model_names, min_retrieval = 5, permuted = '_permuted', type = 'val_test'):
    maps = []
    n_ks = []
    f = open(root + 'baseline_all_old/' + event_name + '/vgg_'+type+'_result_v2'+permuted+'.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()
    for model_name in model_names:
        APs = []
        f = open(root + model_name, 'r')
        predict_result = cPickle.load(f)
        f.close()
        for event_id in ground_truth:
            #if event_id == '2_87806373@N04':
            #    print event_id
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
        maps.append(sum([i[1] for i in APs])/len(APs))
        percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
    return maps, percent, [i[1] for i in APs]
def evaluate_top20_permuted(event_name, model_names, percent = 20, permuted = '_permuted', type = 'val_test'):
    retval = []
    f = open(root + 'baseline_all_old/' + event_name + '/vgg_'+type+'_result_v2'+permuted+'.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()
    for model_name in model_names:
        f = open(root +  model_name, 'r')
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
def amt_worker_result_predict_average(event_name, min_retrievals = xrange(5,36,3), permuted = '_permuted', type = 'val_test'):

    f = open(root + 'baseline_all_old/' + event_name + '/vgg_'+type+'_result_v2'+permuted+'.cPickle','r')
    ground_truth = cPickle.load(f)
    f.close()

    input_path = root + 'baseline_all_old/' + event_name + '/' + type + '_image_ids.cPickle'
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
    input_path = root + 'baseline_all_old/' + event_name + '/' +type+'.csv'
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

def create_predict_dict_from_cpickle_multevent(validation_name, event_name, mat_name, event_index = 17, multi_event = True):
    path = root+'baseline_all_old/' + event_name+ '/'+validation_name+'_image_ids.cPickle'

    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(mat_name + '.cPickle', 'r')
    predict_score = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all_old/' + event_name+ '/'+validation_name+'_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, predict_score):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            if multi_event:
                prediction_dict[event_name] += [[name_, test_url_dict[name_], score[0][event_index-1]]]
            else:
                prediction_dict[event_name] += [[name_, test_url_dict[name_], score]]
        else:
            if multi_event:
                prediction_dict[event_name] = [[name_, test_url_dict[name_], score[0][event_index-1]]]
            else:
                prediction_dict[event_name] = [[name_, test_url_dict[name_], score]]


    f = open(mat_name + '_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()

def baseline_evaluation(event_name, permute = True,
                        model_names = [], evaluate_worker = True, worker_times = 10, validation_name = 'val_test'):

        model_names_this = [event_name.join(i) for i in model_names]
        if permute:
            permute_groundtruth(event_name, validation_name)
        retrievals = [[] for i in model_names_this]; percent = []; precision = [[] for i in model_names_this]
        reweighted = evaluate_rank_reweighted_permuted(event_name, model_names_this, permuted='_permuted', type = validation_name)
        for i in xrange(5, 36, 3):
            temp, percent_temp, AP_temp = evaluate_MAP_permuted(event_name, model_names_this, min_retrieval=i, permuted='_permuted', type = validation_name)
            for j in xrange(len(temp)):
                retrievals[j].append(temp[j])
            percent.append(percent_temp)
        for i in xrange(5, 36, 3):
            temp = evaluate_top20_permuted(event_name, model_names_this, percent=i, permuted='_permuted', type = validation_name)
            for j in xrange(len(temp)):
                precision[j].append(temp[j])
        all_aps=[];all_reweighted=[];all_ps=[]
        if not evaluate_worker:
            return reweighted, percent, retrievals , precision, [], []


        for i in xrange(worker_times):
            all_nks, temp2, temp3, temp4 = amt_worker_result_predict_average(event_name, permuted='_permuted', type = validation_name)
            all_aps.append([temp2]); all_reweighted.append([temp3]); all_ps.append([temp4])
        all_aps_average = copy.deepcopy(all_aps[0])
        for i in xrange(1, worker_times):
            for j in xrange(len(all_aps[i])):
                for k in xrange(len(all_aps[i][j])):
                    for l in xrange(len(all_aps[i][j][k])):
                        for m in xrange(len(all_aps[i][j][k][l])):
                            all_aps_average[j][k][l][m] += all_aps[i][j][k][l][m]
        for j in xrange(len(all_aps_average)):
                for k in xrange(len(all_aps[i][j])):
                    for l in xrange(len(all_aps[i][j][k])):
                        for m in xrange(len(all_aps[i][j][k][l])):
                            all_aps_average[j][k][l][m] = float(all_aps_average[j][k][l][m])/worker_times

        all_ps_average = copy.deepcopy(all_ps[0])
        for i in xrange(1, worker_times):
            for j in xrange(len(all_ps[i])):
                for k in xrange(len(all_ps[i][j])):
                    for l in xrange(len(all_ps[i][j][k])):
                        for m in xrange(len(all_ps[i][j][k][l])):
                            all_ps_average[j][k][l][m] += all_ps[i][j][k][l][m]
        for j in xrange(len(all_ps_average)):
                for k in xrange(len(all_ps[i][j])):
                    for l in xrange(len(all_ps[i][j][k])):
                        for m in xrange(len(all_ps[i][j][k][l])):
                            all_ps_average[j][k][l][m] = float(all_ps_average[j][k][l][m])/worker_times

        all_reweighted_average = copy.deepcopy(all_reweighted[0])
        for i in xrange(1, worker_times):
            for j in xrange(len(all_reweighted[i])):
                for k in xrange(len(all_reweighted[i][j])):
                            all_reweighted_average[j][k] += all_reweighted[i][j][k]
        for j in xrange(len(all_reweighted_average)):
                for k in xrange(len(all_reweighted[i][j])):
                            all_reweighted_average[j][k] = float(all_reweighted_average[j][k])/worker_times

        all_reweighted = all_reweighted_average[0]
        all_aps = all_aps_average[0]
        all_ps = all_ps_average[0]
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
        return reweighted, percent, retrievals , precision, mean_aps, mean_ps
'''
def baseline_evaluation_present(event_name, permute = True,
                        model_names = []):
        model_names_this = [event_name.join(i) for i in model_names]
        #create_knn_cPickle(event_name)
        #create_csv(event_name)
        #read_amt_result(event_name)
        #from_txt_to_pickle(event_name)
        #find_similar(event_name)
        #find_similar(event_name, 'test')
        #correct_amt_result(event_name, 'training')
        #correct_amt_result(event_name, 'test')
        #baseline_predict(event_name)

        if permute:
            permute_groundtruth(event_name)

        for i in model_names_this:
            #if not os.path.exists(root + i):
                create_predict_dict_from_cpickle_multevent(False, event_name, root + ('_').join(i.split('_')[:-1]), dict_name2[event_name], multi_event=False)

        retrievals = [[] for i in model_names_this]; percent = []; precision = [[] for i in model_names_this]
        print '>>>>>>>>' + event_name + '<<<<<<<<'
        print '*ALGORITHM*'
        reweighted = evaluate_rank_reweighted_permuted(event_name, model_names_this, permuted='_permuted')
        for i in xrange(5, 36, 3):
            temp, percent_temp = evaluate_MAP_permuted(event_name, model_names_this, min_retrieval=i, permuted='_permuted')
            for j in xrange(len(temp)):
                retrievals[j].append(temp[j])
            percent.append(percent_temp)
        for i in xrange(5, 36, 3):
            temp = evaluate_top20_permuted(event_name, model_names_this, percent=i, permuted='_permuted')
            for j in xrange(len(temp)):
                precision[j].append(temp[j])
        print 'P:', ', '.join(["%.3f" % v for v in percent])
        for i in xrange(len(model_names_this)):
            print model_names_this[i], reweighted[i]
            print ', '.join(["%.3f" % v for v in precision[i]])
            print ', '.join(["%.3f" % v for v in retrievals[i]])

        print '*WORKER*'
        all_nks = [];all_aps=[];all_reweighted=[];all_ps=[]
        #for i in xrange(10):
        all_nks, all_aps, all_reweighted, all_ps = amt_worker_result_predict_average(event_name, permuted='_permuted')

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
        if sum([i-j for i,j in zip(mean_nks, percent)]) > 0.5:
            print 'P:', ', '.join(["%.3f" % v for v in mean_nks])
        print sum(all_reweighted)/len(all_reweighted)
        print ', '.join(["%.3f" % v for v in mean_aps])
        print ', '.join(["%.3f" % v for v in mean_ps])
        print '\n'
        #return reweighted, percent, retrievals , precision, mean_aps, mean_ps
'''

def create_random_dict(validation_name, event_name, save_name):
    path = root+'baseline_all_old/' + event_name+ '/'+validation_name+'_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all_old/' + event_name+ '/'+validation_name+'_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()
    score_random = [random.uniform(0,1) for i in xrange(len(all_event_ids))]
    f = open(save_name.split('.')[0][:-5]+'.cPickle','wb')
    cPickle.dump(score_random, f)
    f.close()


    prediction_dict = {}
    for score, name_ in zip(score_random, all_event_ids):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], score]]
        else:
            prediction_dict[event_name] = [[name_, test_url_dict[name_], score]]


    f = open(save_name,'wb')
    cPickle.dump(prediction_dict, f)
    f.close()
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def create_face(event_name, validation_name, face_model_name, original_model_name, name, alpha, beta):

    f = open(root + 'face_heatmap/features/'+event_name+ face_model_name,'r')
    face = cPickle.load(f)
    f.close()
    try:
        face = [i[0][dict_name2[event_name]-1] for i in face]
    except:
        pass

    f = open(root + 'CNN_all_event_old/features/'+event_name+original_model_name,'r')
    feature = cPickle.load(f)
    f.close()
    try:
        feature = [i[0][dict_name2[event_name]-1] for i in feature]
    except:
        pass

    f = open(root + 'face_heatmap/features/'+event_name+training_model_name,'r')
    face_training = cPickle.load(f)
    f.close()
    try:
        face_training = [i[0][dict_name2[event_name]-1] for i in face_training]
    except:
        pass


    min_ = np.min(face_training)
    max_ = np.max(face_training)
    alpha = alpha *(max_-min_) + min_
    feature_new = [max(alpha,float(j))**beta*float(i) for (i,j) in zip(feature, face)] #recently used
    '''
    face_std = np.var(face_training)
    face_mean = np.mean(face_training)
    sorted_face = np.sort(face_training)
    count_last = 0
    count_this = 0
    no_face_value = 0
    for i in xrange(1,len(sorted_face)):
        count_this += 1
        if sorted_face[i] != sorted_face[i-1]:
            if count_this > count_last:
                count_last = count_this
                no_face_value = sorted_face[i-1]
            count_this = 0
    max_ = np.max(face_training)
    min_ = np.min(face_training)
    face_hist = np.histogram(face_training, 30)
    x = face_hist[1][:-1]
    y = face_hist[0]
    print x, y
    try:
        popt,pcov = curve_fit(gaus,x,y,p0=[1,face_mean,np.sqrt(face_std)])
        predict_y = gaus(x,*popt)
        residual = predict_y - y
        norm = [max(0.1, i) for i in predict_y]
        error = np.linalg.norm(residual/norm)
        print error
    except:
        error = 10*2

    larger_num = np.sum(face_training > no_face_value)
    smaller_num = np.sum(face_training < no_face_value)

    if larger_num > smaller_num:
        unbalanced = max(float(smaller_num) / larger_num, 0.01)
    else:
        unbalanced = max(float(larger_num) / smaller_num, 0.01)

    beta = face_std*10**3 / np.sqrt(unbalanced) #*(larger_num + smaller_num) / float(len(face))
    range_ = max_ - min_
    if larger_num > smaller_num:
        feature_new = [max(no_face_value + 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value + 0.2*range_, beta
    else:
        feature_new = [min(no_face_value - 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value- 0.2*range_, beta

    '''

    #print [float(j) for (i,j) in zip(feature, face)]

    #feature_new = [float(j)**0.1*float(i) for (i,j) in zip(feature, face)]
    #feature_new = [float(j>0.5)*(float(j)**0.05) + i for (i,j) in zip(feature, face)]
    #feature_new = [(max(0.5, float(j))*0.1) + i for (i,j) in zip(feature, face)]


    #feature_new = [float(j)/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)]
    #feature_new = [float(j)*float(i) for (i,j) in zip(feature, face)]
    f = open(root + 'CNN_all_event_old/features/'+name + '.cPickle','w')
    cPickle.dump(feature_new,f)
    f.close()
    create_predict_dict_from_cpickle_multevent(validation_name, event_name, root + 'CNN_all_event_old/features/'+name, dict_name2[event_name], multi_event=False)
def create_face_2(event_name, validation_name, face_model_name, original_model_name, name, alpha, beta, theta, training_model_name):
    f = open(root + 'face_heatmap/features/'+event_name+ face_model_name,'r')
    face = cPickle.load(f)
    f.close()
    try:
        face = [i[0][dict_name2[event_name]-1] for i in face]
    except:
        pass

    f = open(root + 'CNN_all_event_old/features/'+event_name+original_model_name,'r')
    feature = cPickle.load(f)
    f.close()
    try:
        feature = [i[0][dict_name2[event_name]-1] for i in feature]
    except:
        pass

    f = open(root + 'face_heatmap/features/'+event_name+training_model_name,'r')
    face_training = cPickle.load(f)
    f.close()
    try:
        face_training = [i[0][dict_name2[event_name]-1] for i in face_training]
    except:
        pass


    min_ = np.min(face_training)
    max_ = np.max(face_training)
    alpha = alpha *(max_-min_) + min_
    beta = beta *(max_-min_) + min_
    feature_new = [min(beta, max(alpha, float(j)))*float(i) for (i,j) in zip(feature, face)]
    #feature_new = [max(alpha,float(j))**beta*float(i) for (i,j) in zip(feature, face)] #recently used
    '''
    face_std = np.var(face_training)
    face_mean = np.mean(face_training)
    sorted_face = np.sort(face_training)
    count_last = 0
    count_this = 0
    no_face_value = 0
    for i in xrange(1,len(sorted_face)):
        count_this += 1
        if sorted_face[i] != sorted_face[i-1]:
            if count_this > count_last:
                count_last = count_this
                no_face_value = sorted_face[i-1]
            count_this = 0
    max_ = np.max(face_training)
    min_ = np.min(face_training)
    face_hist = np.histogram(face_training, 30)
    x = face_hist[1][:-1]
    y = face_hist[0]
    print x, y
    try:
        popt,pcov = curve_fit(gaus,x,y,p0=[1,face_mean,np.sqrt(face_std)])
        predict_y = gaus(x,*popt)
        residual = predict_y - y
        norm = [max(0.1, i) for i in predict_y]
        error = np.linalg.norm(residual/norm)
        print error
    except:
        error = 10*2

    larger_num = np.sum(face_training > no_face_value)
    smaller_num = np.sum(face_training < no_face_value)

    if larger_num > smaller_num:
        unbalanced = max(float(smaller_num) / larger_num, 0.01)
    else:
        unbalanced = max(float(larger_num) / smaller_num, 0.01)

    beta = face_std*10**3 / np.sqrt(unbalanced) #*(larger_num + smaller_num) / float(len(face))
    range_ = max_ - min_
    if larger_num > smaller_num:
        feature_new = [max(no_face_value + 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value + 0.2*range_, beta
    else:
        feature_new = [min(no_face_value - 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value- 0.2*range_, beta

    '''

    #print [float(j) for (i,j) in zip(feature, face)]

    #feature_new = [float(j)**0.1*float(i) for (i,j) in zip(feature, face)]
    #feature_new = [float(j>0.5)*(float(j)**0.05) + i for (i,j) in zip(feature, face)]
    #feature_new = [(max(0.5, float(j))*0.1) + i for (i,j) in zip(feature, face)]


    #feature_new = [float(j)/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)]
    #feature_new = [float(j)*float(i) for (i,j) in zip(feature, face)]
    f = open(root + 'CNN_all_event_old/features/'+name + '.cPickle','w')
    cPickle.dump(feature_new,f)
    f.close()
    create_predict_dict_from_cpickle_multevent(validation_name, event_name, root + 'CNN_all_event_old/features/'+name, dict_name2[event_name], multi_event=False)
def create_face_3(event_name, validation_name, face_model_name, original_model_name, name, alpha):
    f = open(root + 'face_heatmap/features/'+event_name+ face_model_name,'r')
    face = cPickle.load(f)
    f.close()
    try:
        face = [i[0][dict_name2[event_name]-1] for i in face]
    except:
        pass

    f = open(root + 'CNN_all_event_old/features/'+event_name+original_model_name,'r')
    feature = cPickle.load(f)
    f.close()
    try:
        feature = [i[0][dict_name2[event_name]-1] for i in feature]
    except:
        pass

    #feature_new = [(float(j)**alpha)*float(i) for (i,j) in zip(feature, face)]
    feature_new = [(float(j)*alpha)+float(i) for (i,j) in zip(feature, face)]
    #feature_new = [max(alpha,float(j))**beta*float(i) for (i,j) in zip(feature, face)] #recently used
    '''
    face_std = np.var(face_training)
    face_mean = np.mean(face_training)
    sorted_face = np.sort(face_training)
    count_last = 0
    count_this = 0
    no_face_value = 0
    for i in xrange(1,len(sorted_face)):
        count_this += 1
        if sorted_face[i] != sorted_face[i-1]:
            if count_this > count_last:
                count_last = count_this
                no_face_value = sorted_face[i-1]
            count_this = 0
    max_ = np.max(face_training)
    min_ = np.min(face_training)
    face_hist = np.histogram(face_training, 30)
    x = face_hist[1][:-1]
    y = face_hist[0]
    print x, y
    try:
        popt,pcov = curve_fit(gaus,x,y,p0=[1,face_mean,np.sqrt(face_std)])
        predict_y = gaus(x,*popt)
        residual = predict_y - y
        norm = [max(0.1, i) for i in predict_y]
        error = np.linalg.norm(residual/norm)
        print error
    except:
        error = 10*2

    larger_num = np.sum(face_training > no_face_value)
    smaller_num = np.sum(face_training < no_face_value)

    if larger_num > smaller_num:
        unbalanced = max(float(smaller_num) / larger_num, 0.01)
    else:
        unbalanced = max(float(larger_num) / smaller_num, 0.01)

    beta = face_std*10**3 / np.sqrt(unbalanced) #*(larger_num + smaller_num) / float(len(face))
    range_ = max_ - min_
    if larger_num > smaller_num:
        feature_new = [max(no_face_value + 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value + 0.2*range_, beta
    else:
        feature_new = [min(no_face_value - 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value- 0.2*range_, beta

    '''

    #print [float(j) for (i,j) in zip(feature, face)]

    #feature_new = [float(j)**0.1*float(i) for (i,j) in zip(feature, face)]
    #feature_new = [float(j>0.5)*(float(j)**0.05) + i for (i,j) in zip(feature, face)]
    #feature_new = [(max(0.5, float(j))*0.1) + i for (i,j) in zip(feature, face)]


    #feature_new = [float(j)/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)]
    #feature_new = [float(j)*float(i) for (i,j) in zip(feature, face)]
    f = open(root + 'CNN_all_event_old/features/'+name + '.cPickle','w')
    cPickle.dump(feature_new,f)
    f.close()
    create_predict_dict_from_cpickle_multevent(validation_name, event_name, root + 'CNN_all_event_old/features/'+name, dict_name2[event_name], multi_event=False)
def create_face_4(event_name, validation_name, face_model_name, original_model_name, name, alpha, beta, theta, training_model_name):
    f = open(root + 'face_heatmap/features/'+event_name+ face_model_name,'r')
    face = cPickle.load(f)
    f.close()
    try:
        face = [i[0][dict_name2[event_name]-1] for i in face]
    except:
        pass

    f = open(root + 'CNN_all_event_old/features/'+event_name+original_model_name,'r')
    feature = cPickle.load(f)
    f.close()
    try:
        feature = [i[0][dict_name2[event_name]-1] for i in feature]
    except:
        pass

    f = open(root + 'face_heatmap/features/'+event_name+training_model_name,'r')
    face_training = cPickle.load(f)
    f.close()
    try:
        face_training = [i[0][dict_name2[event_name]-1] for i in face_training]
    except:
        pass


    min_ = np.min(face_training)
    max_ = np.max(face_training)
    alpha = alpha *(max_-min_) + min_
    beta = beta *(max_-min_) + min_
    feature_new = [min(beta, max(alpha, float(j)))*theta+float(i) for (i,j) in zip(feature, face)]
    #feature_new = [max(alpha,float(j))**beta*float(i) for (i,j) in zip(feature, face)] #recently used
    '''
    face_std = np.var(face_training)
    face_mean = np.mean(face_training)
    sorted_face = np.sort(face_training)
    count_last = 0
    count_this = 0
    no_face_value = 0
    for i in xrange(1,len(sorted_face)):
        count_this += 1
        if sorted_face[i] != sorted_face[i-1]:
            if count_this > count_last:
                count_last = count_this
                no_face_value = sorted_face[i-1]
            count_this = 0
    max_ = np.max(face_training)
    min_ = np.min(face_training)
    face_hist = np.histogram(face_training, 30)
    x = face_hist[1][:-1]
    y = face_hist[0]
    print x, y
    try:
        popt,pcov = curve_fit(gaus,x,y,p0=[1,face_mean,np.sqrt(face_std)])
        predict_y = gaus(x,*popt)
        residual = predict_y - y
        norm = [max(0.1, i) for i in predict_y]
        error = np.linalg.norm(residual/norm)
        print error
    except:
        error = 10*2

    larger_num = np.sum(face_training > no_face_value)
    smaller_num = np.sum(face_training < no_face_value)

    if larger_num > smaller_num:
        unbalanced = max(float(smaller_num) / larger_num, 0.01)
    else:
        unbalanced = max(float(larger_num) / smaller_num, 0.01)

    beta = face_std*10**3 / np.sqrt(unbalanced) #*(larger_num + smaller_num) / float(len(face))
    range_ = max_ - min_
    if larger_num > smaller_num:
        feature_new = [max(no_face_value + 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value + 0.2*range_, beta
    else:
        feature_new = [min(no_face_value - 0.2*range_,float(j))**beta*float(i) for (i,j) in zip(feature, face)]
        print no_face_value- 0.2*range_, beta

    '''

    #print [float(j) for (i,j) in zip(feature, face)]

    #feature_new = [float(j)**0.1*float(i) for (i,j) in zip(feature, face)]
    #feature_new = [float(j>0.5)*(float(j)**0.05) + i for (i,j) in zip(feature, face)]
    #feature_new = [(max(0.5, float(j))*0.1) + i for (i,j) in zip(feature, face)]


    #feature_new = [float(j)/(1+np.exp(-float(i))) for (i,j) in zip(feature, face)]
    #feature_new = [float(j)*float(i) for (i,j) in zip(feature, face)]
    f = open(root + 'CNN_all_event_old/features/'+name + '.cPickle','w')
    cPickle.dump(feature_new,f)
    f.close()
    create_predict_dict_from_cpickle_multevent(validation_name, event_name, root + 'CNN_all_event_old/features/'+name, dict_name2[event_name], multi_event=False)
def combine_cues(event_type, names, event_index, save_name, validation_name):
    path = root+'baseline_all_old/'+event_type+'/'+validation_name+'_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    #n_combine = len(names)
    predict_score = []
    for name in names:
        f = open(root + name, 'r')
        predict_score.append(np.array(cPickle.load(f)))
        f.close()


    f = open(root +'baseline_all_old/'+event_type+'/'+validation_name+'_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    #print len(all_event_ids)
    for i in xrange(len(all_event_ids)):
        name_ = all_event_ids[i]
        event_name = name_.split('/')[0]
        temp_score = 0
        for score in predict_score:
            #print score[i]
            try:
                temp_score += score[i][0][event_index-1]
            except:
                #print score[i]
                temp_score += score[i]
        if event_name in prediction_dict:
            prediction_dict[event_name] += [[name_, test_url_dict[name_], temp_score]]
        else:
            #print event_name
            prediction_dict[event_name] = [[name_, test_url_dict[name_], temp_score]]

    f = open(root + 'CNN_all_event_old/features/'+event_type + save_name +'_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()
    f = open(root + 'CNN_all_event_old/features/'+event_type+ save_name + '.cPickle','wb')
    cPickle.dump(np.mean(predict_score,axis=0), f)
    f.close()
def grid_search_face(event_name, permute, validation_name, times = 3):
    alphas = np.arange(0,1,0.05)
    betas = np.arange(0,1,0.1)
    all_results = {}
    all_results_std = {}

    for j in xrange(times):
        if permute:
            permute_groundtruth(event_name, validation_name)
        for alpha in alphas:
            for beta in betas:
                retrieval = []
                #create_face(event_name, validation_name,'_'+validation_name + '_sigmoidcropped_importance_allevent_iter_100000.cPickle', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle', validation_name + '_combine_face', alpha, beta)
                create_face(event_name, validation_name,'_'+validation_name + face_model, '_'+validation_name+combine_face_model, validation_name + face_combined_model, alpha, beta, '_training' + face_model)
                #create_face(event_name, validation,'_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', 'test_combine_face', alpha, beta)
                std_ = 0
                for i in xrange(5, 20, 3):
                    temp, percent_temp, APs = evaluate_MAP_permuted(event_name, ['CNN_all_event_old/features/'+validation_name+face_combined_model + '_dict.cPickle'], min_retrieval= i, type=validation_name)
                    std_ += np.std(APs)
                    retrieval.append(temp[0])
                #print (alpha, beta), retrieval, std_
                if (alpha, beta) not in all_results:
                    all_results[(alpha, beta)] = sum(retrieval)
                    all_results_std[(alpha, beta)] = std_
                else:
                    all_results[(alpha, beta)] += sum(retrieval)
                    all_results_std[(alpha, beta)] += std_
    #print all_results
    #f = open(root + 'CNN_all_event_old/features/'+validation_name+face_combined_model,'r')
    #temp = cPickle.load(f)
    #f.close()
    #print temp.keys()
    baseline = all_results[(0,0)] / times
    sorted_all_result_std = sorted(all_results_std.items(), key=operator.itemgetter(1), reverse=True)
    for i in xrange(30):
        all_results.pop(sorted_all_result_std[i][0], None)
        #print sorted_all_result_std[i][0]
    sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1))
    improve = sorted_all_result[-1][1] / times
    #print
    print event_name, improve - baseline, sorted_all_result[-1][0]
def grid_search_face_2(event_name, permute, validation_name, times = 10):
    alphas = np.arange(0,1.05,0.05)
    betas = np.arange(0,1.05,0.05)
    thetas = np.arange(1.0, 1.1,0.5)
    all_results = {}
    all_results_std = {}

    for j in xrange(times):
        ap_original = []
        if permute:
            permute_groundtruth(event_name, validation_name)
        retrieval = []
        for i in xrange(5, 20, 3):
            temp, percent_temp, APs = evaluate_MAP_permuted(event_name, ['CNN_all_event_old/features/'+event_name+'_'+validation_name+combine_face_model[:-8]+'_dict.cPickle'], min_retrieval= i, type=validation_name)
            retrieval.append(temp[0])
            ap_original.append(APs)
        #print ap_original
        if (0, 0, 0) not in all_results:
            all_results[(0, 0, 0)] = sum(retrieval)
        else:
            all_results[(0, 0, 0)] += sum(retrieval)
        for alpha in alphas:
            for beta in betas:
                if beta <= alpha:
                    continue
                for theta in thetas:
                    retrieval = []
                    #create_face(event_name, validation_name,'_'+validation_name + '_sigmoidcropped_importance_allevent_iter_100000.cPickle', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle', validation_name + '_combine_face', alpha, beta)
                    create_face_2(event_name, validation_name,'_'+validation_name + face_model, '_'+validation_name+combine_face_model, validation_name + face_combined_model, alpha, beta, theta, '_training' + face_model)
                    #create_face(event_name, validation,'_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', 'test_combine_face', alpha, beta)
                    std_ = 0
                    for i in xrange(5, 20, 3):
                        temp, percent_temp, APs = evaluate_MAP_permuted(event_name, ['CNN_all_event_old/features/'+validation_name+face_combined_model+'_dict.cPickle'], min_retrieval= i, type=validation_name)
                        #print APs, ap_original[(i-5)/3], [p - q for p, q in zip(APs, ap_original[(i-5)/3])]
                        #print [p - q for p, q in zip(APs, ap_original[(i-5)/3])]
                        #std_ += np.std([p - q for p, q in zip(APs, ap_original[(i-5)/3])])
                        std_ += np.std(APs)
                        retrieval.append(temp[0])
                    #print (alpha, beta), retrieval, std_
                    if (alpha, beta, theta) not in all_results:
                        all_results[(alpha, beta, theta)] = sum(retrieval)
                        all_results_std[(alpha, beta, theta)] = std_
                    else:
                        all_results[(alpha, beta, theta)] += sum(retrieval)
                        all_results_std[(alpha, beta, theta)] += std_
    #print all_results
    #f = open(root + 'CNN_all_event_old/features/'+validation_name+face_combined_model,'r')
    #temp = cPickle.load(f)
    #f.close()
    #print temp.keys()
    baseline = all_results[(0,0,0)] / times
    sorted_all_result_std = sorted(all_results_std.items(), key=operator.itemgetter(1), reverse=True)
    abandoned_ = []
    for i in xrange(30):
        abandoned_.append((sorted_all_result_std[i][0], all_results[sorted_all_result_std[i][0]]))
        all_results.pop(sorted_all_result_std[i][0], None)
        #print sorted_all_result_std[i][0]
    #print 'ABANDONED:', abandoned_

    sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
    print event_name, sorted_all_result[0][1] / times - baseline
    print sorted_all_result
    if sorted_all_result[0][1] / times - baseline < 0.01:
        return [(1,1,1), (1,1,1), (1,1,1)]

    temp_ = [(sorted_all_result[0][0],sorted_all_result[0][0][1] - sorted_all_result[0][0][0]) ]
    for i in xrange(1, len(sorted_all_result)):
        if abs(sorted_all_result[i-1][1] - sorted_all_result[i][1]) < 10**(-5):
            temp_.append((sorted_all_result[i][0], sorted_all_result[i][0][1] - sorted_all_result[i][0][0]))
        else:
            break
    sorted_temp = sorted(temp_, key=operator.itemgetter(1))
    if len(sorted_temp) >= 3:
        #print sorted_temp
        return [i[0] for i in sorted_temp]
    else:
        temp = [i[0] for i in sorted_temp]
        if len(sorted_temp) == 1:
            if sorted_all_result[1][1] / times - baseline < 0.01:
                temp.extend([(1,1,1), (1,1,1)])
            elif sorted_all_result[2][1] / times - baseline < 0.01:
                temp.extend([sorted_all_result[1][0], (1,1,1)])
            else:
                temp.extend([sorted_all_result[1][0], sorted_all_result[2][0]])
            return temp
        else:
            if sorted_all_result[2][1] / times - baseline < 0.01:
                temp.extend([(1,1,1)])
            else:
                temp.extend([sorted_all_result[2][0]])
            return temp
def grid_search_face_4(theta, event_name, permute, validation_name, times = 10):
    alphas = np.arange(0,1.05,0.05)
    betas = np.arange(0,1.05,0.05)
    all_results = {}
    all_results_std = {}

    for j in xrange(times):
        ap_original = []
        if permute:
            permute_groundtruth(event_name, validation_name)
        retrieval = []
        for i in xrange(5, 20, 3):
            temp, percent_temp, APs = evaluate_MAP_permuted(event_name, ['CNN_all_event_old/features/'+event_name+'_'+validation_name+combine_face_model[:-8]+'_dict.cPickle'], min_retrieval= i, type=validation_name)
            retrieval.append(temp[0])
            ap_original.append(APs)
        #print ap_original
        if (0, 0, 0) not in all_results:
            all_results[(0, 0, 0)] = sum(retrieval)
        else:
            all_results[(0, 0, 0)] += sum(retrieval)
        for alpha in alphas:
            for beta in betas:
                if beta <= alpha:
                    continue
                retrieval = []
                #create_face(event_name, validation_name,'_'+validation_name + '_sigmoidcropped_importance_allevent_iter_100000.cPickle', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle', validation_name + '_combine_face', alpha, beta)
                create_face_4(event_name, validation_name,'_'+validation_name + face_model, '_'+validation_name+combine_face_model, validation_name + face_combined_model, alpha, beta, theta, '_training' + face_model)
                #create_face(event_name, validation,'_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', 'test_combine_face', alpha, beta)
                std_ = 0
                for i in xrange(5, 20, 3):
                        temp, percent_temp, APs = evaluate_MAP_permuted(event_name, ['CNN_all_event_old/features/'+validation_name+face_combined_model+'_dict.cPickle'], min_retrieval= i, type=validation_name)
                        #print APs, ap_original[(i-5)/3], [p - q for p, q in zip(APs, ap_original[(i-5)/3])]
                        #print [p - q for p, q in zip(APs, ap_original[(i-5)/3])]
                        #std_ += np.std([p - q for p, q in zip(APs, ap_original[(i-5)/3])])
                        std_ += np.std(APs)
                        retrieval.append(temp[0])
                #print (alpha, beta), retrieval, std_
                if (alpha, beta, theta) not in all_results:
                        all_results[(alpha, beta, theta)] = sum(retrieval)
                        all_results_std[(alpha, beta, theta)] = std_
                else:
                        all_results[(alpha, beta, theta)] += sum(retrieval)
                        all_results_std[(alpha, beta, theta)] += std_
    #print all_results
    #f = open(root + 'CNN_all_event_old/features/'+validation_name+face_combined_model,'r')
    #temp = cPickle.load(f)
    #f.close()
    #print temp.keys()
    baseline = all_results[(0,0,0)] / times
    sorted_all_result_std = sorted(all_results_std.items(), key=operator.itemgetter(1), reverse=True)
    abandoned_ = []
    #for i in xrange(30):
    #    abandoned_.append((sorted_all_result_std[i][0], all_results[sorted_all_result_std[i][0]]))
    #    all_results.pop(sorted_all_result_std[i][0], None)
        #print sorted_all_result_std[i][0]
    #print 'ABANDONED:', abandoned_

    sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_all_result
    if sorted_all_result[0][1] / times - baseline > 0.01:
        print sorted_all_result[0][0]
        return sorted_all_result[0][0]
    else:
        print (0, 0, 0)
        return (0, 0, 0)
def grid_search_face_3(event_name, permute, validation_name, times = 10):
    alphas = np.arange(0,1.05,0.05)
    all_results = {}
    for j in xrange(times):
        if permute:
            permute_groundtruth(event_name, validation_name)
        for alpha in alphas:
                    retrieval = []
                    #create_face(event_name, validation_name,'_'+validation_name + '_sigmoidcropped_importance_allevent_iter_100000.cPickle', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle', validation_name + '_combine_face', alpha, beta)
                    create_face_3(event_name, validation_name,'_'+validation_name + face_model, '_'+validation_name+combine_face_model, validation_name + face_combined_model, alpha)
                    #create_face(event_name, validation,'_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', 'test_combine_face', alpha, beta)
                    for i in xrange(5, 20, 3):
                        temp, percent_temp, APs = evaluate_MAP_permuted(event_name, ['CNN_all_event_old/features/'+validation_name+face_combined_model+'_dict.cPickle'], min_retrieval= i, type=validation_name)
                        #print APs, ap_original[(i-5)/3], [p - q for p, q in zip(APs, ap_original[(i-5)/3])]
                        #print [p - q for p, q in zip(APs, ap_original[(i-5)/3])]
                        #std_ += np.std([p - q for p, q in zip(APs, ap_original[(i-5)/3])])

                        retrieval.append(temp[0])
                    if alpha in all_results:
                        all_results[alpha].append((retrieval, sum(retrieval)))
                    else:
                        all_results[alpha] = [(retrieval, sum(retrieval))]
    for alpha in all_results:
        #print all_results[alpha]
        #print np.mean(all_results[alpha][0], axis = 0)
        #print np.sum(all_results[alpha][1])
        #all_results[alpha] = [np.mean([i[0] for i in all_results[alpha]], axis = 0), np.sum([i[1] for i in all_results[alpha]])]
        all_results[alpha] = np.sum([i[1] for i in all_results[alpha]])
    sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_all_result
    if sorted_all_result[0][0] == 0:
        return (0, 0, 0)
    return grid_search_face_4(sorted_all_result[0][0], event_name, permute, validation_name, times = 10)

def to_guru_file(case = 'to_face'):
    if case == 'to_guru':
        in_path = root + 'baseline_wedding_test/training_validation/val_validation_path.txt'
        out_path = root + 'baseline_wedding_test/training_validation/guru_val_validation_path.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
                for line in data:
                    line = line.split('\r')[0]
                    img_path_new = '/home/feiyu1990/local/event_curation/curation_images/Wedding/' + '/'.join(line.split('\\')[-2:])
                    f.write(img_path_new + ' 0\n')
        f.close()
    if case == 'to_face':
        in_path = root + 'face_heatmap/data/all_event/guru_ranking_reallabel_training_nomargin_p.txt'
        out_path = root + 'face_heatmap/data/all_event/face_ranking_reallabel_training_nomargin_p.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
                for line in data:
                    img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' + '/'.join(line.split('/')[-2:])
                    f.write(img_path_new)
        f.close()
        in_path = root + 'face_heatmap/data/all_event/guru_ranking_reallabel_training_nomargin.txt'
        out_path = root + 'face_heatmap/data/all_event/face_ranking_reallabel_training_nomargin.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
                for line in data:
                    img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' + '/'.join(line.split('/')[-2:])
                    f.write(img_path_new)
        f.close()
    if case == 'to_face_allevent':
        for event_name in dict_name2:
            in_path = root + 'baseline_all_old/'+event_name+'/linux_training_path.txt'
            out_path = root + 'face_heatmap/data/all_event/'+event_name+'_training_path.txt'
            f = open(out_path, 'w')
            with open(in_path,'r') as data:
                    for line in data:
                        img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' + '/'.join(line.split('/')[-2:])
                        f.write(img_path_new)
            f.close()
        for event_name in dict_name2:
            in_path = root + 'baseline_all_old/'+event_name+'/linux_test_path.txt'
            out_path = root + 'face_heatmap/data/all_event/'+event_name+'_test_path.txt'
            f = open(out_path, 'w')
            with open(in_path,'r') as data:
                    for line in data:
                        img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' + '/'.join(line.split('/')[-2:])
                        f.write(img_path_new)
            f.close()
def find_valid_examples_reallabel(name = 'training', event_name = 'Birthday'):
    f = open(root + 'baseline_all_old/'+event_name+'/vgg_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all_old/'+event_name+'/vgg_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_all_old/'+event_name+'/'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_all_old/'+event_name+'/guru_'+name+'_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])

    img_path_dict = {}
    for (i,j) in zip(img_ids, img_paths):
        img_path_dict[i] = j

    img_pair = []
    count_all = 0
    for event in ground_truth_training:
        this_event = ground_truth_training[event]
        this_event.sort(key=lambda x: x[2])
        count = 0
        len_ = len(this_event)
        for i in xrange(len_):
            #if this_event[-1][2] - this_event[i][2] < 1.2:
            #    break
            img_1 = this_event[i]
            for j in xrange(i + 1, len_):
                img_2 = this_event[j]
                #if img_2[2] - img_1[2] < 1.2:
                #    continue
                temp = random.sample([0, 1], 1)
                if temp[0] == 0:
                    img_pair.append((img_1[1], img_2[1], 0))
                else:
                    img_pair.append((img_2[1], img_1[1], 1))
                count += 1
        count_all += count
        print event, len(this_event), count
    print count_all


    random.shuffle(img_pair)
    if name == 'test':
        Img_pair = img_pair[:1000]
    out_path1 = root + 'baseline_all_old/'+event_name+'/data/guru_'+name+'.txt'
    out_path2 = root + 'baseline_all_old/'+event_name+'/data/guru_'+name+'_p.txt'
    f1 = open(out_path1,'w')
    f2 = open(out_path2,'w')
    for i in img_pair:
        line = img_path_dict[i[0]] + ' ' + str(int(20*(ground_truth_training_dict[i[0]]))) + '\n'
        f1.write(line)
        line = img_path_dict[i[1]] + ' ' + str(int(20*(ground_truth_training_dict[i[1]]))) + '\n'
        f2.write(line)
    f1.close()
    f2.close()
    pass
def find_valid_examples_reallabel_face(event_name, name = 'training'):
    f = open(root + 'baseline_all_old/'+event_name+'/vgg_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all_old/'+event_name+'/vgg_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_all_old/'+event_name+'/'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_all_old/'+event_name+'/guru_'+name+'_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])

    img_path_dict = {}
    for (i,j) in zip(img_ids, img_paths):
        img_path_dict[i] = j

    img_pair = []
    count_all = 0
    for event in ground_truth_training:
        this_event = ground_truth_training[event]
        this_event.sort(key=lambda x: x[2])
        count = 0
        len_ = len(this_event)
        for i in xrange(len_):
            if this_event[-1][2] - this_event[i][2] < 1.2:
                break
            img_1 = this_event[i]
            for j in xrange(i + 1, len_):
                img_2 = this_event[j]
                if img_2[2] - img_1[2] < 1.2:
                    continue
                temp = random.sample([0, 1], 1)
                if temp[0] == 0:
                    img_pair.append((img_1[1], img_2[1], 0))
                else:
                    img_pair.append((img_2[1], img_1[1], 1))
                count += 1
        count_all += count
        print event, len(this_event), count
    print count_all


    random.shuffle(img_pair)
    if name == 'test':
        img_pair = img_pair[:1000]
    out_path1 = root + 'face_heatmap/'+event_name+'_'+name+'.txt'
    out_path2 = root + 'face_heatmap/'+event_name+'_'+name+'_p.txt'
    f1 = open(out_path1,'w')
    f2 = open(out_path2,'w')
    for i in img_pair:
        line = img_path_dict[i[0]] + ' ' + str(int(20*(ground_truth_training_dict[i[0]]))) + '\n'
        line = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' +'/'.join(line.split('/')[-2:])
        f1.write(line)
        line = img_path_dict[i[1]] + ' ' + str(int(20*(ground_truth_training_dict[i[1]]))) + '\n'
        line = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' +'/'.join(line.split('/')[-2:])
        f2.write(line)
    f1.close()
    f2.close()
    pass

def evaluate_present_with_worker(validation_name = 'val_test'):

    model_names = [#['baseline_all_old/', '/'+validation_name + '_' + model_name + '_predict_result_'+str(n_vote)+'_dict.cPickle']
                   #,['baseline_all_old/', '/vgg_predict_result_'+str(n_vote)+'_dict.cPickle']
                   #,['CNN_all_event_old/features/', '_combined_val_test_dict.cPickle']
                   #,['CNN_all_event_old/features/', 'val_test_face_combined_dict.cPickle']
                    #['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9segment_iter_70000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_2time_iter_70000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_3time_iter_70000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_3time_iter_100000_dict.cPickle']
                    #['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_3time_iter_180000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_50000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_100000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_150000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_200000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_250000_dict.cPickle']
                    #,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9segment_2round_iter_750000_dict.cPickle']
                    #,['CNN_all_event_0.1/features/', '_' + validation_name + '_sigmoid90.1_iter_70000_dict.cPickle']
                    #,['CNN_all_event_0.1/features/', '_' + validation_name + '_sigmoid90.1_iter_200000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_iter_650000_dict.cPickle']
                    ['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_iter_70000_dict.cPickle']
                    ,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_differentsize_iter_70000_dict.cPickle']
                    ,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_differentsize_iter_110000_dict.cPickle']
                    ,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_differentsize_iter_130000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_2time_iter_70000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_3time_iter_70000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_3time_iter_100000_dict.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_3time_iter_180000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_50000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_100000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_150000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_200000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_250000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_test_sigmoid9_10_segment_2time_iter_250000_dict.cPickle']
                    #,['CNN_all_event_old/features/','_test_sigmoid9segment_2time_iter_180000_dict.cPickle']
                    ,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_2round_iter_750000_dict.cPickle']
                    #,['CNN_all_event_0.1/features/', '_' + validation_name + '_sigmoid9_10_0.1_iter_70000_dict.cPickle']
                    #,['CNN_all_event_0.1/features/', '_' + validation_name + '_sigmoid9_10_0.1_iter_200000_dict.cPickle']
                    #,['CNN_all_event_ranking/features/', '_test_sigmoid9ranking_iter_10000_dict.cPickle']
                    #,['CNN_all_event_ranking/features/', '_test_sigmoid9ranking_iter_20000_dict.cPickle']
                    #,['CNN_all_event_ranking/features/', '_test_sigmoid9ranking_iter_130000_dict.cPickle']
                    #,['CNN_all_event_ranking/features/', '_test_sigmoid9ranking_iter_300000_dict.cPickle']
                    #,['to_guru/CNN_all_event_old/features/', '_test_sigmoid9vgg_segment_iter_460000_dict.cPickle']
                    #,['to_guru/CNN_all_event_old/features/', '_combined_test.cPickle']
                    #,['to_guru/CNN_all_event_old/features/', '_face_combined_dict.cPickle']
                    #,['to_guru/CNN_all_event_old/features/', '_face_combined_groundtruth_dict.cPickle']
                   ]

    permutation_times = 50
    worker_permutation_times = 1
    retrieval_models = []
    for i in model_names:
        retrieval_models.append([])
    retrieval_worker_all = []
    len_all = 0
    for event_name in dict_name2:
        model_names_this = [event_name.join(ii) for ii in model_names]
        for model_name_this in model_names_this:
            try:
                create_predict_dict_from_cpickle_multevent(validation_name, event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=True)
            except:
                try:
                    create_predict_dict_from_cpickle_multevent(validation_name, event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=False)
                except:
                    print 'Skipping creation of dict:', model_name_this

        percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
        for i in xrange(permutation_times):
                if i %10 == 0:
                    print i
                reweighted, percent, retrievals , precision, mean_aps, mean_ps = baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times, validation_name = validation_name)
                percent_all.append(percent)
                retrievals_model.append(retrievals)
                precision_model.append(precision)
                retrievals_worker.append(mean_aps)
                precision_worker.append(mean_ps)
        percent_average = []; retrievals_model_average = []; precision_model_average = []; retrievals_worker_average = []; precision_worker_average = []
        for j in xrange(len(retrievals_model[0])):
            retrievals_model_average.append([])
            precision_model_average.append([])
        for i in xrange(len(percent_all[0])):
            percent_average.append(sum(j[i] for j in percent_all)/permutation_times)
            retrievals_worker_average.append(sum(j[i] for j in retrievals_worker)/permutation_times)
            precision_worker_average.append(sum(j[i] for j in precision_worker)/permutation_times)
            for j in xrange(len(retrievals_model_average)):
                retrievals_model_average[j].append(sum(k[j][i] for k in retrievals_model)/permutation_times)
                precision_model_average[j].append(sum(k[j][i] for k in precision_model)/permutation_times)
        model_names_this = [event_name.join(i) for i in model_names]

        f = open(root + 'CNN_all_event_old/features/' + event_name + '_' + validation_name+'_sigmoid9segment_iter_70000_dict.cPickle','r')
        temp = cPickle.load(f)
        f.close()
        len_ = len(temp)
        len_all += len_

        print '>>>>>>>>' + event_name + '<<<<<<<<'
        print '*ALGORITHM*'
        print 'P:', ', '.join(["%.5f" % v for v in percent_average])
        for i in xrange(len(model_names_this)):
            print model_names_this[i]
            print ', '.join(["%.3f" % v for v in precision_model_average[i]])
            print ', '.join(["%.3f" % v for v in retrievals_model_average[i]])
            retrieval_models[i].append([j*len_ for j in retrievals_model_average[i]])
        print '*WORKER*'
        print ', '.join(["%.3f" % v for v in retrievals_worker_average])
        print ', '.join(["%.3f" % v for v in precision_worker_average])
        print '\n'
        retrieval_worker_all.append([i*len_ for i in retrievals_worker_average])
    print '*********************************'
    print '*********************************'
    for i in xrange(len(retrieval_models)):
        print model_names[i]
        #print retrieval_models[i]
        temp = np.array(retrieval_models[i])
        temp1 = np.sum(temp, axis=0)
        print [j/len_all for j in temp1]
    print 'Worker'
    #print retrieval_worker_all
    temp = np.array(retrieval_worker_all)
    temp1 = np.sum(temp, axis=0)
    print [i/len_all for i in temp1]
def evaluate_present_face(dict_from_validation2):
    model_names = [['baseline_all_old/', '/val_test_' + model_name + '_predict_result_'+str(n_vote)+'_dict.cPickle']
                   #,['baseline_all_old/', '/vgg_predict_result_'+str(n_vote)+'_dict.cPickle']
                   ,['CNN_all_event_old/features/', '_val_test'+combine_face_model[:-8]+'_dict.cPickle']
                   ,['CNN_all_event_old/features/', '_val_test_face_combined_3_dict.cPickle']
                   ]
    permutation_times = 50
    worker_permutation_times = 1
    retrieval_models = []
    for i in model_names:
        retrieval_models.append([])
    retrieval_worker_all = []
    len_all = 0
    #dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (1,1,1),
    #                        'Zoo': (1, 1, 1), 'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (1, 1, 1), 'Christmas': (0.350000000000000003, 0.65000000000000007, 1.0),
    #                        'PersonalArtActivity': (0.10000000000000001, 0.45000000000000001, 1.0), 'GroupActivity': (0.15000000000000002, 0.25, 1.0),
    #                        'Wedding': (0.65000000000000002, 0.95000000000000007, 1.0), 'ReligiousActivity': (1, 1, 1), 'Graduation': (0.45000000000000001, 0.5, 1.0),
    #                        'CasualFamilyGather': (0.0, 0.10000000000000001, 1.0), 'Architecture': (1, 1, 1), 'ThemePark': (0.55000000000000004, 0.70000000000000007, 1.0),
    #                        'Sports': (1,1, 1.0), 'Show': (0.65000000000000002, 0.80000000000000004, 1.0), 'Halloween': (1, 1, 1),
    #                        'BusinessActivity': (1, 1, 1), 'Protest': (1, 1, 1.0), 'Cruise': (1, 1, 1),
    #                        'Birthday': (0.75, 0.80000000000000004, 1.0), 'NatureTrip': (1, 1, 1)
    #                        }


    '''
    #for _sigmoid9_10_stochastic_iter_150000
    dict_from_validation2 ={'PersonalSports': (1, 1, 1), 'Museum': (0.050000000000000003, 0.70000000000000007, 1.0),
    'UrbanTrip': (0.20000000000000001, 0.5, 1.0), 'Zoo': (1, 1, 1), 'BeachTrip': (1, 1, 1),
    'PersonalMusicActivity': (0.45000000000000001, 0.55000000000000004, 1.0), 'Christmas': (0.30000000000000004, 0.80000000000000004, 1.0),
    'PersonalArtActivity': (0.65000000000000002, 0.95000000000000007, 1.0), 'GroupActivity': (1, 1, 1),
    'Wedding': (0.60000000000000009, 0.85000000000000009, 1.0), 'ReligiousActivity': (0.55000000000000004, 0.75, 1.0),
    'Graduation': (1, 1, 1), 'CasualFamilyGather': (0.0, 0.10000000000000001, 1.0), 'Architecture': (1, 1, 1),
    'ThemePark': (0.70000000000000007, 0.90000000000000002, 1.0), 'Sports': (0.85000000000000009, 0.90000000000000002, 1.0),
    'Show': (0.60000000000000009, 0.75, 1.0), 'Halloween': (1, 1, 1), 'BusinessActivity': (1, 1, 1),
    'Protest': (0.55000000000000004, 0.65000000000000002, 1.0), 'Cruise': (0.30000000000000004, 0.45000000000000001, 1.0),
    'Birthday': (1, 1, 1), 'NatureTrip': (1, 1, 1)}

    not working for this!
    '''
    '''
    #for segment_70000
    dict_from_validation = {'PersonalSports':(0.85000000000000009, 0.20000000000000001) ,'BeachTrip': (0.95000000000000007, 0.90000000000000002),
                            'PersonalMusicActivity':(0.050000000000000003, 0.30000000000000004), 'Christmas':(0.35000000000000003, 0.90000000000000002),
                            'PersonalArtActivity':(0.15000000000000002, 0.10000000000000001), 'GroupActivity':(0.20000000000000001, 0.30000000000000004),
                            'Wedding':(0.60000000000000009, 0.90000000000000002), 'ReligiousActivity':(0.85000000000000009, 0.90000000000000002),
                            'Graduation':(0.80000000000000004, 0.90000000000000002), 'CasualFamilyGather':(0.70000000000000007, 0.10000000000000001),
                            'ThemePark':(0.55000000000000004, 0.80000000000000004), 'Sports':(0.90000000000000002, 0.30000000000000004),
                            'Show':(0.65000000000000002, 0.70000000000000007), 'BusinessActivity':(0.85000000000000009, 0.90000000000000002),
                            'Cruise':(0.10000000000000001, 0.10000000000000001), 'Birthday':(0.40000000000000002, 0.30000000000000004),
                            'Protest':(0,0),'NatureTrip':(0,0), 'UrbanTrip':(0,0), 'Museum':(0,0), 'Zoo':(0,0), 'Architecture':(0,0), 'Halloween':(0,0)
    }
    dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (1, 1, 1), 'Zoo': (1, 1, 1),
                            'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (0.45000000000000001, 0.60000000000000009, 1.0),
                            'Christmas': (0.35000000000000003, 0.60000000000000009, 1.0), 'PersonalArtActivity': (0.15000000000000002, 0.5, 1.0),
                            'GroupActivity': (0.20000000000000001, 0.30000000000000004, 1.0), 'Wedding': (0.55000000000000004, 0.90000000000000002, 1.0),
                            'ReligiousActivity': (1, 1, 1), 'Graduation': (0.45000000000000001, 0.5, 1.0), 'CasualFamilyGather': (0.0, 0.10000000000000001, 1.0),
                            'Architecture': (1, 1, 1), 'ThemePark': (0.5, 0.70000000000000007, 1.0), 'Sports': (1, 1, 1), 'Show': (0.65000000000000002, 0.70000000000000007, 1.0),
                            'Halloween': (1, 1, 1), 'BusinessActivity': (1, 1, 1), 'Protest': (0.5, 0.60000000000000009, 1.0), 'Cruise': (1, 1, 1),
                            'Birthday': (0.25, 0.40000000000000002, 1.0), 'NatureTrip': (1, 1, 1)}
    ['CNN_all_event_old/features/', '_val_test_sigmoid9segment_iter_70000_dict.cPickle']
    [0.26944296560127928, 0.31351165261054137, 0.34735032193726495, 0.37898486306883578, 0.40882714335136505, 0.44093269765351328, 0.46734090976260756, 0.49729245993449711, 0.52560433670606699, 0.55428548772899788, 0.58453033049885006]
    ['CNN_all_event_old/features/', '_val_test_face_combined_dict.cPickle']
    [0.27694772322530825, 0.3180026125953469, 0.35087208237580852, 0.38175546722251402, 0.41156822601677717, 0.44389402276511414, 0.47008555518893858, 0.49999453546356659, 0.52815343684225735, 0.55590509650706099, 0.58541886874082794]

    #for combine
    dict_from_validation = {'PersonalSports':(0,0) ,'BeachTrip': (0.95000000000000007, 0.20000000000000009),
                            'PersonalMusicActivity':(0.0, 0.10000000000000001), 'Christmas':(0.45000000000000001, 0.90000000000000002),
                            'PersonalArtActivity': (0.10000000000000001, 0.60000000000000004), 'GroupActivity': (0.0, 0.10000000000000001),
                            'Wedding':(0.55000000000000004, 0.90000000000000002), 'ReligiousActivity':(0.85000000000000009, 0.60000000000000009),
                            'Graduation':(0,0), 'CasualFamilyGather':(0,0),
                            'ThemePark':(0.45000000000000002, 0.40000000000000002), 'Sports':(0.90000000000000002, 0.30000000000000004),
                            'Show':(0.65, 0.80000000000000007), 'BusinessActivity':(0.85000000000000009, 0.90000000000000009),
                            'Cruise': (0.40000000000000002, 0.10000000000000001), 'Birthday':(0.15000000000000002, 0.40000000000000002),
                            'Protest':(0.1, 0.80000000000000002),'NatureTrip':(0.350000000000000002, 0.70000000000000009), 'UrbanTrip':(0.25, 0.80000000000000004), 'Museum':(0.0, 0.0),
                            'Zoo':(0.5, 0.90000000000000002), 'Architecture':(0,0), 'Halloween':(0,0)
    }

    #for combine 2
    dict_from_validation2 = {'PersonalSports':(1,1),'Museum':(1,1),'UrbanTrip':(0.3, 0.6),'Zoo':(1,1),
                            'BeachTrip': (1,1),'PersonalMusicActivity':(0.30000000000000004, 0.60000000000000009),
                            'Christmas':(0.2, 0.65),'PersonalArtActivity': (0.25, 0.5), 'GroupActivity': (0.15, 0.25),
                            'Wedding':(0.55, 0.95), 'ReligiousActivity':(0.5, 0.65),'Graduation':(0.45,0.5), 'CasualFamilyGather':(0,0.1),
                            'Architecture':(1,1),'ThemePark':(0.55, 0.7), 'Sports':(0.9, 0.95),'Show':(0.65, 0.75), 'Halloween':(0.65,0.75),
                            'BusinessActivity':(0.90, 0.95),'Protest':(0.5, 0.6),'Cruise': (1,1), 'Birthday':(0.7, 0.85),
                            'NatureTrip':(0.45, 0.85),
    }



    dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (0.30000000000000004, 0.80000000000000004, 1.0),
                            'Zoo': (1, 1, 1), 'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (1, 1, 1), 'Christmas': (0.050000000000000003, 0.70000000000000007, 1.0),
                            'PersonalArtActivity': (0.10000000000000001, 0.45000000000000001, 1.0), 'GroupActivity': (0.15000000000000002, 0.25, 1.0),
                            'Wedding': (0.65000000000000002, 0.95000000000000007, 1.0), 'ReligiousActivity': (1, 1, 1), 'Graduation': (0.45000000000000001, 0.5, 1.0),
                            'CasualFamilyGather': (0.0, 0.10000000000000001, 1.0), 'Architecture': (1, 1, 1), 'ThemePark': (0.55000000000000004, 0.70000000000000007, 1.0),
                            'Sports': (0.90000000000000002, 0.95000000000000007, 1.0), 'Show': (0.65000000000000002, 0.80000000000000004, 1.0), 'Halloween': (1, 1, 1),
                            'BusinessActivity': (1.0, 1.0, 1.0), 'Protest': (0.5, 0.65000000000000002, 1.0), 'Cruise': (1, 1, 1),
                            'Birthday': (0.75, 0.80000000000000004, 1.0), 'NatureTrip': (1, 1, 1)
                            }

    ['CNN_all_event_old/features/', '_val_test_combined_dict.cPickle']
    [0.26678331314037396, 0.30906886448331017, 0.34269261136374091, 0.37583416679762788, 0.40894495852041901, 0.44325403785189349, 0.4709997067515676, 0.50256757341002545, 0.53190908357390243, 0.55999406722606393, 0.5891123806831976]
    ['CNN_all_event_old/features/', '_val_test_face_combined_dict.cPickle']
    [0.2702247876399364, 0.31111337429449787, 0.3448186776891457, 0.37778947323208245, 0.41127444480155567, 0.44543151291785316, 0.47283405976134324, 0.50376723687748992, 0.53255868695798059, 0.55980391562132437, 0.58841795610971137]

    '''
    '''
    #for combine_10_dict:
    dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (1, 1, 1), 'Zoo': (1, 1, 1),
                            'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (0.5, 0.80000000000000004, 1.0), 'Christmas': (0.35000000000000003, 0.65000000000000002, 1.0),
                            'PersonalArtActivity': (0.15000000000000002, 0.5, 1.0), 'GroupActivity': (0.15000000000000002, 0.30000000000000004, 1.0),
                            'Wedding': (0.75, 0.95000000000000007, 1.0), 'ReligiousActivity': (1, 1, 1), 'Graduation': (0.45000000000000001, 0.5, 1.0),
                            'CasualFamilyGather': (0.0, 0.10000000000000001, 1.0), 'Architecture': (1, 1, 1), 'ThemePark': (0.35000000000000003, 0.90000000000000002, 1.0),
                            'Sports': (1, 1, 1), 'Show': (0.60000000000000009, 0.75, 1.0), 'Halloween': (1, 1, 1), 'BusinessActivity': (1, 1, 1),
                            'Protest': (1, 1, 1), 'Cruise': (0.30000000000000004, 0.40000000000000002, 1.0), 'Birthday': (0.70000000000000007, 0.80000000000000004, 1.0),
                            'NatureTrip': (0.90000000000000002, 1.0, 1.0)}

    ['CNN_all_event_old/features/', '_val_test_combined_10_dict.cPickle']
    [0.27776983673849948, 0.31447920659003592, 0.34828171151963294, 0.38283046622797906, 0.41663920089925094, 0.45156746585533647, 0.47956976152920106, 0.51112490646412545, 0.53890571232744833, 0.5667168893652953, 0.59679201342133958]
    ['CNN_all_event_old/features/', '_val_test_face_combined_dict.cPickle']
    [0.28211851643376284, 0.3170641727932525, 0.35211384274822488, 0.3844196627084982, 0.41768660036036953, 0.45253563067614527, 0.48034886451721576, 0.51154860652371836, 0.53880719892864948, 0.56595394397699883, 0.59527198807563475]

    '''

    '''
    #for sigmoid9_10_stochastic_iter_250000

    dict_from_validation = {'PersonalSports': (0.55000000000000004, 0.60000000000000009),'Museum': (0.0, 0.60000000000000009),
                            'UrbanTrip':(0, 0),
                            'BeachTrip': (0.95000000000000007, 0.70000000000000007), 'PersonalMusicActivity':(0.20000000000000001, 0.30000000000000004),
                            'Christmas':(0.10000000000000001, 0.70000000000000007), 'PersonalArtActivity':(0.70000000000000007, 0.90000000000000002),
                            'GroupActivity':(0.70000000000000007, 0.80000000000000004), 'Wedding':(0.60000000000000009, 0.90000000000000002)
    }
    '''

    '''
            'ThemePark':(0.60000000000000009, 0.5), 'UrbanTrip':(0,0), 'BeachTrip':(0.90000000000000002, 1.3), 'NatureTrip': (0,0),
             'Zoo':(0,0),'Cruise':(0,0),'Show':(0,0),'Sports':(0,0),'PersonalArtActivity': (0,0),
            'PersonalMusicActivity':(0.15000000000000002, 0.30000000000000004),'ReligiousActivity':(0,0),
            'GroupActivity':(0.55000000000000004, 0.60000000000000009),'CasualFamilyGather':(0.0, 0.0),
            'BusinessActivity':(0.0, 0.0), 'Architecture':(0,0), 'Wedding':(0.6,1.1), 'Birthday':(0.60000000000000009, 0.40000000000000002), 'Graduation':(0,0),
            'Museum':(0,0),'Christmas':(0,0),'Halloween':(0.15000000000000002, 0.40000000000000002), 'Protest':(0.60000000000000009, 1.1000000000000001)}
    '''

    '''
    for _sigmoidcropped_importance_allevent_iter_100000
    '''
    dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (0.5, 0.65000000000000002, 1.0),
                             'Zoo': (1, 1, 1),  'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (0.50000000000000009, 0.750000000000000002, 1.0),
                             'Christmas': (0.15, 0.7, 1), 'PersonalArtActivity': (0.2,0.5, 1), 'GroupActivity': (0.30000000000000001, 0.35000000000000003, 1.0),
                             'Wedding': (0.55, 0.9, 1.0), 'ReligiousActivity': (1, 1, 1), 'Graduation': (0.45, 0.5, 1.0),
                             'CasualFamilyGather': (0.05, 0.10000000000000003, 1.0), 'Architecture': (1, 1, 1), 'ThemePark': (0.7, 0.95, 1),
                             'Sports': (1, 1, 1), 'Show': (0.65000000000000007, 0.7, 1.0), 'Halloween': (1, 1, 1),
                             'BusinessActivity': (1, 1, 1), 'Protest': (0.55000000000000004, 0.60000000000000009, 1.0),
                             'Cruise': (0.25, 0.3, 1.0),'Birthday': (0.75, 0.8, 1.0),'NatureTrip': (0.65, 0.95, 1)}

    '''
    for face finetuned _iter_30000_sigmoid1.cPickle

    dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (0.85,0.95, 1.0),
                             'Zoo': (1, 1, 1),  'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (0.6,0.9, 1.0),
                             'Christmas': (1,1, 1), 'PersonalArtActivity': (1,1, 1), 'GroupActivity': (0.1, 0.3, 1.0),
                             'Wedding': (0.6, 0.9, 1.0), 'ReligiousActivity': (0.7,0.95, 1), 'Graduation': (0.35, 0.4, 1.0),
                             'CasualFamilyGather': (0.6, 0.65, 1.0), 'Architecture': (1, 1, 1), 'ThemePark': (1,1, 1),
                             'Sports': (1, 1, 1), 'Show': (0.7, 0.85, 1.0), 'Halloween': (1, 1, 1),
                             'BusinessActivity': (1, 1, 1), 'Protest': (0.1,0.4, 1.0),
                             'Cruise': (0.45, 0.65, 1.0),'Birthday': (0.75, 0.8, 1.0),'NatureTrip': (1,1, 1)}
    '''
    '''
    for _combined_10_new.cPickle only


    dict_from_validation2 = {'PersonalSports': (1, 1, 1), 'Museum': (1, 1, 1), 'UrbanTrip': (1,1, 1.0),
                             'Zoo': (1, 1, 1),  'BeachTrip': (1, 1, 1), 'PersonalMusicActivity': (1,1, 1.0),
                             'Christmas': (0.25, 0.7, 1), 'PersonalArtActivity': (0.2,0.45, 1), 'GroupActivity': (0.30000000000000001, 0.35000000000000003, 1.0),
                             'Wedding': (0.55, 0.9, 1.0), 'ReligiousActivity': (1, 1, 1), 'Graduation': (0.45, 0.5, 1.0),
                             'CasualFamilyGather': (0.05, 0.10000000000000003, 1.0), 'Architecture': (1, 1, 1), 'ThemePark': (0.7, 0.9, 1),
                             'Sports': (1, 1, 1), 'Show': (0.65000000000000007, 0.7, 1.0), 'Halloween': (1, 1, 1),
                             'BusinessActivity': (1, 1, 1), 'Protest': (0.55000000000000004, 0.60000000000000009, 1.0),
                             'Cruise': (0.25, 0.3, 1.0),'Birthday': (1,1, 1.0),'NatureTrip': (1,1, 1)}
    '''
    #dict_from_validation1 = {'PersonalSports': [(0, 0, 0)], 'Museum': [(0, 0, 0)], 'UrbanTrip': [(0, 0, 0)], 'Zoo': [(0, 0, 0)], 'BeachTrip': [(0, 0, 0)], 'PersonalMusicActivity': [(0, 0, 0)], 'Christmas': [(0.25, 0.70000000000000007, 0.80000000000000004)], 'PersonalArtActivity': [(0, 0, 0)], 'GroupActivity': [(0.0, 0.40000000000000002, 0.30000000000000004)], 'Wedding': [(0.5, 0.95000000000000007, 0.20000000000000001)], 'ReligiousActivity': [(0, 0, 0)], 'Graduation': [(0, 0, 0)], 'CasualFamilyGather': [(0, 0, 0)], 'Architecture': [(0, 0, 0)], 'ThemePark': [(0.65000000000000002, 0.90000000000000002, 1.0)], 'Sports': [(0, 0, 0)], 'Show': [(0.45000000000000001, 0.55000000000000004, 0.55000000000000004)], 'Halloween': [(0, 0, 0)], 'BusinessActivity': [(0, 0, 0)], 'Protest': [(0, 0, 0)], 'Cruise': [(0.050000000000000003, 0.5, 0.15000000000000002)], 'Birthday': [(0.050000000000000003, 0.75, 0.050000000000000003)], 'NatureTrip': [(0, 0, 0)]}

    for event_name in dict_from_validation2:
        #face_model = '_' + event_name.lower() + '_iter_30000_sigmoid1.cPickle'
        #create_face(event_name, False, '_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', event_name + '_face_combined_groundtruth', 0.8, 1.4)
        this_param = dict_from_validation1[event_name]
        this_param_2 = dict_from_validation2[event_name]
        print event_name
        print this_param, this_param_2
        create_face_4(event_name, 'val_test', '_val_test' + face_model, '_val_test'+combine_face_model, event_name + '_val_test_face_combined_2', this_param[0][0], this_param[0][1], this_param[0][2], '_training' + face_model)
        #create_face_3(event_name, 'val_test', '_val_test' + face_model, '_val_test'+combine_face_model, event_name + '_val_test_face_combined', this_param[0])
        create_face_2(event_name, 'val_test', '_val_test' + face_model, '_val_test'+combine_face_model, event_name + '_val_test_face_combined', this_param_2[0], this_param_2[1], this_param_2[2], '_training' + face_model)
        f = open(root + 'baseline_all_old/' + event_name + '/' + 'val_test_alexnet6k_predict_result_10_dict.cPickle','r')
        temp = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_old/features/'+event_name+'_val_test_face_combined.cPickle','r')
        a_temp = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_old/features/'+event_name+'_val_test_face_combined_2.cPickle','r')
        b_temp = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_old/features/'+event_name+'_val_test_face_combined_3.cPickle','w')
        cPickle.dump([i*j for i, j in zip(a_temp, b_temp)],f)
        f.close()

        len_ = len(temp)
        #print temp.keys()
        len_all += len_
        model_names_this = [event_name.join(ii) for ii in model_names]
        for model_name_this in model_names_this:
            #if not os.path.exists(root + i):
            try:
                create_predict_dict_from_cpickle_multevent('val_test', event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=True)
            except:
                try:
                    create_predict_dict_from_cpickle_multevent('val_test', event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=False)
                except:
                    print 'Skipping creation of dict:', model_name_this
        percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
        for i in xrange(permutation_times):
                if i %10 == 0:
                    print i
                reweighted, percent, retrievals , precision, mean_aps, mean_ps = baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times)
                percent_all.append(percent)
                retrievals_model.append(retrievals)
                precision_model.append(precision)
                retrievals_worker.append(mean_aps)
                precision_worker.append(mean_ps)
        percent_average = []; retrievals_model_average = []; precision_model_average = []; retrievals_worker_average = []; precision_worker_average = []
        for j in xrange(len(retrievals_model[0])):
            retrievals_model_average.append([])
            precision_model_average.append([])
        for i in xrange(len(percent_all[0])):
            percent_average.append(sum(j[i] for j in percent_all)/permutation_times)
            retrievals_worker_average.append(sum(j[i] for j in retrievals_worker)/permutation_times)
            precision_worker_average.append(sum(j[i] for j in precision_worker)/permutation_times)
            for j in xrange(len(retrievals_model_average)):
                retrievals_model_average[j].append(sum(k[j][i] for k in retrievals_model)/permutation_times)
                precision_model_average[j].append(sum(k[j][i] for k in precision_model)/permutation_times)
        #print percent_all
        #print percent_average


        model_names_this = [event_name.join(i) for i in model_names]
        print '>>>>>>>>' + event_name + '<<<<<<<<'
        print '*ALGORITHM*'
        print 'P:', ', '.join(["%.5f" % v for v in percent_average])
        for i in xrange(len(model_names_this)):
            print model_names_this[i]
            print ', '.join(["%.3f" % v for v in precision_model_average[i]])
            print ', '.join(["%.3f" % v for v in retrievals_model_average[i]])
            retrieval_models[i].append([j*len_ for j in retrievals_model_average[i]])
        print '*WORKER*'
        print ', '.join(["%.3f" % v for v in retrievals_worker_average])
        print ', '.join(["%.3f" % v for v in precision_worker_average])
        print '\n'
        retrieval_worker_all.append([i*len_ for i in retrievals_worker_average])
    print '*********************************'
    print '*********************************'
    for i in xrange(len(retrieval_models)):
        print model_names[i]
        #print retrieval_models[i]
        temp = np.array(retrieval_models[i])
        #print temp
        #print len_all
        temp1 = np.sum(temp, axis=0)
    #print temp1
        print [j/len_all for j in temp1]
    print 'Worker'
    #print retrieval_worker_all
    temp = np.array(retrieval_worker_all)
    temp1 = np.sum(temp, axis=0)
    print [i/len_all for i in temp1]
def evaluate_present_combine(validation_name = 'val_test'):
    model_names_to_combine = [
                    ['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_iter_70000.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_2time_iter_70000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_3time_iter_100000.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_3time_iter_180000.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_150000.cPickle']
                    #['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_iter_70000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_segment_2time_iter_70000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_50000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_100000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_150000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_200000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9_10_stochastic_iter_250000.cPickle']
                    ,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9_10_segment_2round_iter_750000.cPickle']
                   ]
    model_names_to_combine2 = [
                    ['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_2time_iter_70000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_3time_iter_100000.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_3time_iter_180000.cPickle']
                    ,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_150000.cPickle']
                    #['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9segment_2time_iter_70000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_50000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_100000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_150000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_200000.cPickle']
                    #,['CNN_all_event_old/features/','_'+validation_name+'_sigmoid9stochastic_iter_250000.cPickle']
                    ,['CNN_all_event_old/features/', '_'+validation_name+'_sigmoid9segment_2round_iter_750000.cPickle']
                   ]
    model_names = [['CNN_all_event_old/features/','_' + validation_name+'_combined_10_new_dict.cPickle']]#, ['CNN_all_event_old/features/','_' + validation_name+'_combined_10_new_dict.cPickle']]
    permutation_times = 100
    worker_permutation_times = 1
    retrieval_models = []
    for i in model_names:
        retrieval_models.append([])
    retrieval_worker_all = []
    len_all = 0
    for event_name in dict_name2:
        #combine_cues(event_name,[event_name.join(i) for i in model_names_to_combine2], dict_name2[event_name], '_' + validation_name+'_combined', validation_name)
        #combine_cues(event_name,[event_name.join(i) for i in model_names_to_combine2], dict_name2[event_name], '_' + validation_name+'_combined_new', validation_name)
        combine_cues(event_name,[event_name.join(i) for i in model_names_to_combine], dict_name2[event_name], '_' + validation_name+'_combined_10_new', validation_name)
        f = open(root + 'baseline_all_old/' + event_name + '/' + validation_name+'_event_id.cPickle','r')
        temp = cPickle.load(f)
        f.close()
        len_ = len(temp)
        len_all += len_
        percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
        for i in xrange(permutation_times):
                if i %10 == 0:
                    print i

                reweighted, percent, retrievals , precision, mean_aps, mean_ps = baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times, validation_name = validation_name)
                percent_all.append(percent)
                retrievals_model.append(retrievals)
                precision_model.append(precision)
                retrievals_worker.append(mean_aps)
                precision_worker.append(mean_ps)
        percent_average = []; retrievals_model_average = []; precision_model_average = []; retrievals_worker_average = []; precision_worker_average = []
        for j in xrange(len(retrievals_model[0])):
            retrievals_model_average.append([])
            precision_model_average.append([])
        for i in xrange(len(percent_all[0])):
            percent_average.append(sum(j[i] for j in percent_all)/permutation_times)
            retrievals_worker_average.append(sum(j[i] for j in retrievals_worker)/permutation_times)
            precision_worker_average.append(sum(j[i] for j in precision_worker)/permutation_times)
            for j in xrange(len(retrievals_model_average)):
                retrievals_model_average[j].append(sum(k[j][i] for k in retrievals_model)/permutation_times)
                precision_model_average[j].append(sum(k[j][i] for k in precision_model)/permutation_times)

        model_names_this = [event_name.join(i) for i in model_names]
        print '>>>>>>>>' + event_name + '<<<<<<<<'
        print '*ALGORITHM*'
        print 'P:', ', '.join(["%.5f" % v for v in percent_average])
        for i in xrange(len(model_names_this)):
            print model_names_this[i]
            print ', '.join(["%.3f" % v for v in precision_model_average[i]])
            print ', '.join(["%.3f" % v for v in retrievals_model_average[i]])
            retrieval_models[i].append([j*len_ for j in retrievals_model_average[i]])
        print '*WORKER*'
        print ', '.join(["%.3f" % v for v in retrievals_worker_average])
        print ', '.join(["%.3f" % v for v in precision_worker_average])
        print '\n'
        retrieval_worker_all.append([i*len_ for i in retrievals_worker_average])
    print '*********************************'
    print '*********************************'
    for i in xrange(len(retrieval_models)):
        print model_names[i]
        #print retrieval_models[i]
        temp = np.array(retrieval_models[i])
        temp1 = np.sum(temp, axis=0)
    #print temp1
        print [j/len_all for j in temp1]
    print 'Worker'
    #print retrieval_worker_all
    temp = np.array(retrieval_worker_all)
    temp1 = np.sum(temp, axis=0)
    print [i/len_all for i in temp1]

'''
def evaluate_present_without_worker():
    model_names = [['CNN_all_event_old/features/', '_val_test_random_dict.cPickle']]
    permutation_times = 100
    for event_name in dict_name2:
        percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
        for i in xrange(permutation_times):
                if i %10 == 0:
                    print i
                create_random_dict(False, event_name, root+'CNN_all_event_old/features/'+event_name+ '_val_test_random_dict.cPickle')
                reweighted, percent, retrievals , precision, mean_aps, mean_ps = baseline_evaluation(event_name, True, model_names, evaluate_worker=False)
                percent_all.append(percent)
                retrievals_model.append(retrievals)
                precision_model.append(precision)
        percent_average = []; retrievals_model_average = []; precision_model_average = []
        for j in xrange(len(retrievals_model[0])):
            retrievals_model_average.append([])
            precision_model_average.append([])
        for i in xrange(len(percent_all[0])):
            percent_average.append(sum(j[i] for j in percent_all)/permutation_times)
            for j in xrange(len(retrievals_model_average)):
                retrievals_model_average[j].append(sum(k[j][i] for k in retrievals_model)/permutation_times)
                precision_model_average[j].append(sum(k[j][i] for k in precision_model)/permutation_times)
        model_names_this = [event_name.join(i) for i in model_names]
        print '>>>>>>>>' + event_name + '<<<<<<<<'
        print '*RANDOM*'
        print 'P:', ', '.join(["%.5f" % v for v in percent_average])
        for i in xrange(len(model_names_this)):
            print model_names_this[i]
            print ', '.join(["%.3f" % v for v in precision_model_average[i]])
            print ', '.join(["%.3f" % v for v in retrievals_model_average[i]])
'''

if __name__ == '__main__':


    '''
    #if need to begin from train/val separation:
    events = create_event_id_dict()
    for event_id in dict_name:
        event_separate_trainval(events, dict_name[event_id])
        linux_create_path(event_id)
        from_npy_to_dicts(event_id)
        from_npy_to_dicts(event_id,'test')
        create_dict_url(event_id)

        create_knn_cPickle(event_id)
        create_csv(event_id)
        read_amt_result(event_id)
        find_similar(event_id)
        correct_amt_result(event_id)
        baseline_predict(event_id)

    #for random predict result creation
    for event_name in dict_name2:
        create_random_dict(False, event_name, root+'CNN_all_event_old/features/'+event_name+ '_random_dict.cPickle')
    '''
    #evaluate methods and worker method
    #evaluate_present_with_worker('training')
    #evaluate_present_combine('training')
    #evaluate_present_with_worker('val_validation')
    #evaluate_present_with_worker('val_test')
    evaluate_present_with_worker('test')
    #evaluate_present_combine('test')

    #evaluate_present_combine('val_validation')
    #evaluate_present_combine('val_test')
    '''
    dict_from_validation1 = {}

    #combine_face_model_list = ['_combined_new.cPickle','_combined_10_new.cPickle', '_combined.cPickle', '_combined_10.cPickle',
    #                           '_sigmoid9_10_segment_3time_iter_100000.cPickle', '_sigmoid9segment_iter_70000.cPickle',
    #                           '_sigmoid9_10_segment_2round_iter_750000.cPickle', '_sigmoid9_10_stochastic_iter_100000.cPickle']
    global combine_face_model
    global face_model
    combine_face_model_list = ['_combined_10_new.cPickle']

    for name in combine_face_model_list:
        combine_face_model = name
        print combine_face_model
        for event_name in dict_name2:
            print event_name
            #face_model = '_'+event_name.lower()+'_iter_30000_sigmoid1.cPickle'
            #print face_model
            if event_name not in dict_from_validation1:
                #print event_name
                dict_from_validation1[event_name] = [grid_search_face_3(event_name, permute=True, validation_name='val_validation')]
            else:
                #print event_name
                dict_from_validation1[event_name].append(grid_search_face_3(event_name, permute=True, validation_name='val_validation'))
        print dict_from_validation1
    print dict_from_validation1

    evaluate_present_face(dict_from_validation1)
    #if need face: eg. create_face(event_name, False, '_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', event_name + '_face_combined_groundtruth', 0.8, 1.4)
    #evaluate_present_combine()
    #evaluate_present_combine('val_validation')
    '''
        #grid_search_face(event_name, permute=True, validation=True)
    #grid_search_face('Protest', permute=True, validation_name='val_validation')
    '''
    mat_event = {}
    for event_name in dict_name2:
        f = open(root + 'face_heatmap/features/'+event_name+ '_training_sigmoidcropped_importance_allevent_iter_100000.cPickle','r')
        face = cPickle.load(f)
        f.close()
        try:
            face = [i[0][dict_name2[event_name]-1] for i in face]
        except:
            pass
        mat_event[event_name] = face
    io.savemat(root + 'face_heatmap/features/all_training_sigmoidcropped_importance_allevent_iter_100000.mat', mat_event)
    '''

