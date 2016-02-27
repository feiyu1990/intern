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

root = '/Users/wangyufei/Documents/Study/intern_adobe/'
#root = 'C:/Users/yuwang/Documents/'
model_name = 'alexnet6k'
#model_name = 'vgg'
#model_name = 'alexnet3k'


'''some correctness checking / correcting'''
def check_correctness():
    #urls = ['http://farm1.staticflickr.com/217/511765114_6b78ea4dfc.jpg', 'http://farm1.staticflickr.com/232/511764728_ed0abff4fa.jpg']
    urls = ['http://farm1.staticflickr.com/232/511764728_ed0abff4fa.jpg', 'http://farm1.staticflickr.com/213/511764604_14192d4257.jpg']
    count = -1
    image_ids = []
    with open(root+'all_images_curation.txt','r') as data:
        for line in data:
            meta = line.split('\t')
            if meta[16] in urls:
                image_ids.append(meta[1]+'_'+meta[3]+'/'+meta[2])
    line_numbers=[]
    f = open(root + 'baseline_wedding_test/wedding_training_image_ids.cPickle','r')
    image_ids_all = cPickle.load(f)
    f.close()
    for i in image_ids_all:
        if i in image_ids:
            line_numbers.append(image_ids_all.index(i))

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_features.cPickle','r')
    features = cPickle.load(f)
    f.close()
    feature1 = np.asarray(features[line_numbers[0]])
    feature2 = np.asarray(features[line_numbers[1]])
    #feature3 = np.asarray(features[line_numbers[2]])
    features_this = [feature1, feature2]
    feature_normalize = normalize(features_this, axis=1)

    value1 = np.dot(feature_normalize[0,:], feature_normalize[1,:])
    #value2 = np.dot(feature_normalize[2,:], feature_normalize[1,:])
    print value1#, value2
    pass
def correct_wedding(type = 'training'):
    need_delete_event = ['1_31102014@N07','1_58003213@N00','2_58003213@N00']
    f = open(root+'baseline_wedding_test/wedding_'+type+'_id.cPickle','r')
    event_id = cPickle.load(f)
    f.close()
    new_even_id = []
    for i in event_id:
        if i not in need_delete_event:
            new_even_id.append(i)

    f = open(root+'baseline_wedding_test/wedding_'+type+'_id.cPickle','wb')
    cPickle.dump(new_even_id, f)
    f.close()
    need_delete_image = []

    with open(root+'all_images_curation.txt','r') as data:
        for line in data:
            meta = line.split('\t')
            event_id = meta[1] + '_' + meta[3]
            if event_id in need_delete_event:
                need_delete_image.append(meta[2])

    f = open(root+'baseline_wedding_test/wedding_'+type+'_path.txt','wb')
    delete_lines = []
    count = -1
    with open(root+'baseline_wedding_test/wedding_'+type+'_path.txt','r') as data:
        for line in data:
            count += 1
            this_image = line.split('\\')[-1]
            this_image = this_image.split('.')[0]
            if this_image not in need_delete_image:
                f.write(line)
            else:
                delete_lines.append(count)
    f.close()

    f = open(root+'baseline_wedding_test/'+model_name+'_wedding_'+type+'_features.txt','wb')
    count = -1
    with open(root+'baseline_wedding_test/wedding_'+type+'_features.txt','r') as data:
        for line in data:
            count += 1
            if count not in delete_lines:
                f.write(line)
    f.close()

    f = open(root+'baseline_wedding_test/linux_wedding_'+type+'_path.txt','wb')
    count = -1
    with open(root+'baseline_wedding_test/linux_wedding_'+type+'_path.txt','r') as data:
        for line in data:
            count += 1
            if count not in delete_lines:
                f.write(line)
    f.close()



    in_path = root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()

    image_ids = []
    in_path = root + 'baseline_wedding_test/linux_wedding_'+type+'_path.txt'
    with open(in_path,'r') as data:
        for line in data:
            name = line.split('/')[-1]
            name = name.split('.')[0]
            image_ids.append(name)

    event_ids = []
    i = 0
    in_path = root+'all_images_curation.txt'
    with open(in_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            id = meta[2]
            if id == image_ids[i]:
                i += 1
                event_ids.append(meta[1] + '_' + meta[3] + '/' + meta[2])
                if i == len(image_ids):
                    break

    f = open(root + 'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle','wb')
    cPickle.dump(event_ids, f)
    f.close()

'''not specific to wedding'''
def merge_all_result():
    need_delete_event = ['1_31102014@N07','1_58003213@N00','2_58003213@N00']
    f = open(root + 'all_output/all_output.csv','wb')
    writer = csv.writer(f)
    line_count = 0
    head_meta_correct = []
    input_path = root + 'all_output/result4.csv'
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta_correct = meta
                writer.writerow(meta)
                break

    for kkk in xrange(7):
        line_count = 0
        correct_order = {}
        input_path = root + 'all_output/result'+str(kkk)+'.csv'
        with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    for i in xrange(len(meta) - 2):
                        field = meta[i]
                        correct_i = head_meta_correct.index(field)
                        correct_order[correct_i] = i
                else:
                    new_meta = []
                    for i in xrange(len(head_meta_correct) - 2):
                        if i in correct_order:
                            new_meta.append(meta[correct_order[i]])
                        else:
                            new_meta.append('NOTVALID')
                    if new_meta[16] == 'Rejected':
                        continue
                    if new_meta[28] in need_delete_event and new_meta[29] == 'Wedding':
                        continue
                    writer.writerow(new_meta)

                line_count += 1


    f.close()
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

'''wedding dataset'''
def event_separate_trainval():
    events = {}
    for i in xrange(7):
        path = root + 'all_output/result'+str(i)+'.csv'
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
                    events[event_type] = {event_id}


    check_type = 'Wedding'
    events_set = events[check_type]
    events = list(events_set)
    events_training = random.sample(events,len(events)*3/4)
    events_test = [event for event in events if event not in events_training]
    f = open(root + 'baseline_wedding_test/wedding_training.cPickle', 'wb')
    cPickle.dump(events_training, f)
    f.close()
    f = open(root + 'baseline_wedding_test/wedding_test.cPickle', 'wb')
    cPickle.dump(events_test, f)
    f.close()
    create_path()
def create_path():
    in_path = root + 'baseline_wedding_test/wedding_training_id.cPickle'
    f = open(in_path, 'r')
    events_training = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_wedding_test/wedding_test_id.cPickle'
    f = open(in_path, 'r')
    events_test = cPickle.load(f)
    f.close()

    load_path = root+'all_images_curation.txt'
    save_paths1 = root + 'baseline_wedding_test/wedding_training_path.txt'
    f1 = open(save_paths1, 'wb')
    save_paths2 = root + 'baseline_wedding_test/wedding_test_path.txt'
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
def linux_create_path():
    in_path = root + 'baseline_wedding_test/wedding_training_id.cPickle'
    f = open(in_path, 'r')
    events_training = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_wedding_test/wedding_test_id.cPickle'
    f = open(in_path, 'r')
    events_test = cPickle.load(f)
    f.close()
    load_path = root+'all_images_curation.txt'
    save_paths1 = root + 'baseline_wedding_test/linux_wedding_training_path.txt'
    f1 = open(save_paths1, 'wb')
    save_paths2 = root + 'baseline_wedding_test/linux_wedding_test_path.txt'
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
def from_npy_to_dicts(type = 'training'):
    #f = open(root + 'baseline_wedding_test/wedding_'+type+'_features.cPickle','r')
    #features = cPickle.load(f)
    #f.close()
    image_ids = []
    in_path = root + 'baseline_wedding_test/linux_wedding_'+type+'_path.txt'
    with open(in_path,'r') as data:
        for line in data:
            name = line.split('/')[-1]
            name = name.split('.')[0]
            image_ids.append(name)

    event_ids = []
    i = 0
    in_path = root+'all_images_curation.txt'
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
    f = open(root + 'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle','wb')
    cPickle.dump(event_ids, f)
    f.close()
def create_dict_url():
    path = root + 'baseline_wedding_test/wedding_training_image_ids.cPickle'
    f = open(path, 'r')
    image_ids = cPickle.load(f)
    f.close()
    dict = {}

    path = root + 'all_images_curation.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            image_id = meta[1] + '_' + meta[3] + '/'+meta[2]
            if image_id in image_ids:
                dict[image_id] = meta[16]

    f = open(root + 'baseline_wedding_test/wedding_training_ulr_dict.cPickle', 'wb')
    cPickle.dump(dict, f)
    f.close()
    path = root + 'baseline_wedding_test/wedding_test_image_ids.cPickle'
    f = open(path, 'r')
    image_ids = cPickle.load(f)
    f.close()


    dict = {}
    path = root + 'all_images_curation.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            image_id = meta[1] + '_' + meta[3] + '/'+meta[2]
            if image_id in image_ids:
                dict[image_id] = meta[16]

    f = open(root + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'wb')
    cPickle.dump(dict, f)
    f.close()
def create_knn_cPickle(path = root + 'baseline_wedding_test/face_'+model_name+'_wedding_knn.txt'):
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

    with open(path, 'r') as data:
        for line in data:
            test_index = int(line.split(':')[0])
            groups = re.findall('\((.*?)\)',line)
            #if len(groups) != 50:
            #    print 'ERROR DETECTED!'
            #    return
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
def create_csv_wedding():
    f = open(root + 'baseline_wedding_test/wedding_training_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/wedding_training.csv','wb')
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

    f = open(root + 'baseline_wedding_test/wedding_test_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/wedding_test.csv','wb')
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
block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
def read_amt_result(type = 'training'):
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
    #event_ids_this = [i for i in event_ids]
    #input_path = root + 'baseline_wedding_test/wedding_'+type+'_id.cPickle'
    #f = open(input_path, 'r')
    #event_ids_true = cPickle.load(f)
    #f.close()
    #event_lost = [i for i in event_ids_true if i not in event_ids_this]
    #print 'lost events:', len(event_lost)

    input_path = root + 'baseline_wedding_test/wedding_'+type+'.csv'
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
    f = open(root + 'baseline_wedding_test/wedding_'+type+'_result_v1.cPickle','wb')
    cPickle.dump(input_and_answers, f)
    f.close()
def correct_amt_result(type = 'training'):
    f = open(root + 'baseline_wedding_test/wedding_'+type+'_result_v1.cPickle','r')
    input_and_answers = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_similar_list.cPickle','r')
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

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_result_v2.cPickle','wb')
    cPickle.dump(input_and_answers, f)
    f.close()
    training_scores_dict = {}
    for event in input_and_answers:
        for img in input_and_answers[event]:
            training_scores_dict[img[1]] = img[2]
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_result_dict_v2.cPickle', 'wb')
    cPickle.dump(training_scores_dict,f)
    f.close()


'''wedding curation training & test'''
def from_txt_to_pickle():
    in_path = root + 'baseline_wedding_test/'+model_name+'_wedding_training_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
    in_path = root + 'baseline_wedding_test/'+model_name+'_wedding_test_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
def find_similar(threshold, type = 'training'):
    image_path = root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_features.cPickle'
    f = open(image_path, 'r')
    feature = cPickle.load(f)
    f.close()
    id_path = root + 'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle'
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
    mat_path = root + 'all_images_curation.txt'
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
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_'+type+'_similar_list.cPickle','wb')
    cPickle.dump(remove_list, f)
    f.close()
def baseline_predict(n_vote = 10):
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    training_scores_dict = cPickle.load(f)
    f.close()
    #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn_noapprox.cPickle', 'r')
    f = open(root + 'to_guru/wedding_CNN_net/features/fc7_ranking_sigmoid_2round_iter_20000_knn.cPickle','r')
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
    f = open(root + 'to_guru/wedding_CNN_net/features/fc7_ranking_sigmoid_2round_iter_20000_dict.cPickle','wb')
    #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_predict_result_'+str(n_vote)+'_noapprox_dict.cPickle','wb')
    cPickle.dump(test_prediction_event, f)
    f.close()
def baseline_predict_nomore2(n_vote = 10):
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    training_scores_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn.cPickle', 'r')
    knn = cPickle.load(f)
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
        for j in xrange(1, len(this_test_img)):
            this_train_event = this_test_img[j][1].split('/')[0]
            if this_train_event not in events:
                events[this_train_event] = 1
            else:
                events[this_train_event] += 1
                if events[this_train_event] > 2:
                    continue
            valid_vote += 1
            weight.append(this_test_img[j][0])
            votes.append(training_scores_dict[this_test_img[j][1]])
            if valid_vote == 10:
                break
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

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_predict_result_'+str(n_vote)+'_v2.cPickle','wb')
    cPickle.dump(test_prediction_event, f)
    f.close()


'''result display'''
def create_result_htmls(type, event_ids, input_path):
    root1 = root + 'baseline_wedding_test/'+type+'_htmls/'
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
                line_count+=1
            if meta[28] not in event_ids:
                continue
            elif meta[0] not in HITs:
                HITs[meta[0]] = [meta]
            else:
                HITs[meta[0]].append(meta)


    for HITId in HITs:

        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        event_id = this_hits[0][28]
        input_field = []
        output_field = []
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_type' in field:
                    name = value1
                    print name
                    name = ''.join(e for e in name if e.isalnum())
                    name = name[:10]
                if 'num_image' in field:
                    print value1
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        if event_id == '84_7273032@N04':
            print event_id
        out_file = root1 +'present_'+ event_id + '.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_predict.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)

        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]
                    len_selection = len(curr_hit)
                    if field == 'feedback':
                        pass
                    else:

                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if key == '' or key == 'NOTVALID':
                            continue
                        if field != 'difficulty':
                            line_stack[i] += '\ndocument.getElementById("'+field+'selected").value="'+str(float(score)/len(curr_hit))+'";\n'
                        if score > 7*len_selection/5:
                            number_1 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >3*len_selection/5 and score <= 7*len_selection/5:
                            number_2 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >= 0*len_selection/5 and score <=3*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score < 0*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
        print number_1, number_2

        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_htmls_rearranged():
    input_path = root + 'baseline_wedding_test/wedding_test.csv'
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

    for HITId in HITs:
        num_images = 0
        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        input_field = []
        output_field = []
        event_id = ''
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_id' in field:
                    event_id = value1
                if 'event_type' in field:
                    name = value1
                    print name
                    name = ''.join(e for e in name if e.isalnum())
                    name = name[:10]
                if 'num_image' in field:
                    print value1
                    num_images = int(value1)
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        #if not os.path.exists(root + 'baseline_wedding_test/test_htmls/'):
        #    os.mkdir(root + 'baseline_wedding_test/test_htmls/')
        out_file = root + 'present_htmls_test/' + event_id + '/rearranged_groundtruth.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)
        scores = {}
        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]

                    if field == 'feedback':
                        pass
                    else:
                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            #line_stack[i] += '\ndocument.getElementById("'+field+key+'").value="'+str(count_type[key])+'";\n'
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if field != 'difficulty' and int(field[5:]) <= num_images:
                            scores[int(field[5:])] = score


        for i in xrange(len(line_stack)):
            if 'var images = ' in line_stack[i]:
                image_list = line_stack[i].split('","')
                image_list[0] = image_list[0].split('"')[1]
                image_list[-1] = image_list[0].split('"')[0]
        image_list = image_list[:num_images]
        scores_ordered = [scores[i+1] for i in xrange(num_images)]
        sorted_image_list = sorted(zip(image_list, scores_ordered),key=lambda x: x[1], reverse=True)

        this_line = 'var images = ['
        for k in sorted_image_list:
            this_line += '"'+k[0]+'",'
        this_line = this_line[:-1] + '];'
        for i in xrange(len(line_stack)):
            if 'var images = ' in line_stack[i]:
                line_stack[i] = this_line
            if '$(document).ready(function()' in line_stack[i]:
                for k in xrange(len(sorted_image_list)):
                    line_stack[i] += '\ndocument.getElementById("image'+str(k+1)+'selected").value="'+str(float(sorted_image_list[k][1])/20 + 0.5)+'";\n'
                    score = sorted_image_list[k][1]
                    len_selection = len(output_field[0]) - 1
                    if score > 7*len_selection/5:
                            number_1 += 1
                            line_stack[i]+='$("label[name=\'image'+str(k+1)+ '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    if score >3*len_selection/5 and score <= 7*len_selection/5:
                            number_2 += 1
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    if score >= 0*len_selection/5 and score <=3*len_selection/5:
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    if score < 0*len_selection/5:
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'


        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def images_for_contrsim(type='training', col_num = 9, write_tags = True):
    list_path = root+'all_images_curation.txt'
    similar_path = root+'baseline_wedding_test/'+model_name+'_wedding_'+type+'_similar_list.cPickle'
    f = open(similar_path, 'r')
    similar_list = cPickle.load(f)
    f.close()

    path = root+'baseline_wedding_test/wedding_'+type+'_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()


    event_ids = []
    for i in similar_list:
        event_ids.extend(i)
    f.close()
    html_count = 0
    out_path = root+'baseline_wedding_test/similar_contrast_htmls/'+model_name+'_'+type+'nosim_contrast_'+str(html_count)+'.html'
    f = open(out_path, 'wb')
    f.write('<head>Training wedding images #'+str(html_count)+'</head> <title> Training wedding images '+str(html_count)+'</title>\n' )
    count = 0
    last_event = ''
    over_all_count = 0
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = 0
    id_count = 0
    count_similar_event = 0
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1] + '_' + meta[3]
            image_id = meta[2]
            id_this = this_event + '/' + image_id
            if id_count == len(all_event_ids) or id_this != all_event_ids[id_count]:
                continue
            id_count += 1
            if id_this in event_ids:
                if id_this not in similar_list[count_similar_event]:
                    count_similar_event += 1
                delete_it = True
            else:
                delete_it = False

            tag_common = meta[0]

            if this_event != last_event:
                event_number += 1
                count = 0
                f.write('</table>\n')
                if event_number % 40 == 0:
                    f.write('<style type=\"text/css\">img { height:auto;width:\"200px\";}\n')
                    f.write('img.delete1{ border: 3px ridge #FF0000;}\n img.delete0{ border: 3px ridge #0000FF;}</style>')
                    f.close()
                    html_count += 1
                    out_path = root+'baseline_wedding_test/similar_contrast_htmls/'+model_name+'_'+type+'nosim_contrast_'+str(html_count)+'.html'
                    f = open(out_path, 'wb')
                    f.write('<head>Training wedding images #'+str(html_count)+'</head> <title> Training wedding images '+str(html_count)+'</title>\n' )
                    f.write('<center>')
                    f.write('<table border="1" style="width:100%">\n')

                f.write('<br><p><b>Event id:' + this_event + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common +'</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                last_event = this_event

            path = meta[16]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            if not delete_it:
                f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            else:
                f.write('\t\t\t<img class=\"delete'+str(count_similar_event%2)+'\" src=\"'+path+'\" alt=Loading... width = "200" />\n')
            date = meta[5]
            f.write('\t\t\t<br /><b>'+str(over_all_count)+' '+date+'</b><br />\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"200px\";}\n')
    f.write('img.delete1{ border: 3px ridge #FF0000;}\n img.delete0{ border: 3px ridge #0000FF;}</style>')
    f.close()
def create_retrieval_image(max_display = 10):
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    training_result = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn_facereweighted_old.cPickle','r')
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

    for event_id in event_knn:
        f = open(root + 'present_htmls_test/' + event_id + '/'+model_name+'_retrieval_top10_facereweighted_old.html','wb')
        f.write('<head>'+model_name+' Retrieval Result #'+str(html_count)+'</head> <title>'+model_name+' Retrieval Result '+str(html_count)+'</title>\n' )
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
def show_test_predict_img():
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_predict_result_10.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()
    root1 = root + 'baseline_wedding_test/predict_htmls/'
    if not os.path.exists(root1):
        os.mkdir(root1)

    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        for img in this_event:
            img_ids.append(img[0])
            img_urls.append(img[1])
            img_scores.append(img[2])

        html_path = root1 + model_name +'_' +event_id + '.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_predict.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'
                    if score > 1:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >0.8:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >= 0:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score < 0:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def show_test_predict_img_rearranged():
    model_name = 'alexnet6k'
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_predict_result_10.cPickle','r')
    #f = open(root + 'to_guru/wedding_CNN_net/features/test_fc8_class_ranking_sigmoid_2round_iter_20000_dict.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()
    #root1 = root + 'baseline_wedding_test/predict_htmls/'
    #if not os.path.exists(root1):
    #    os.mkdir(root1)

    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        for img in this_event:
            img_ids.append(img[0])
            img_urls.append(img[1])
            img_scores.append(img[2])
        temp = zip(img_ids, img_urls, img_scores)
        temp = sorted(temp, key=lambda x: x[2], reverse=True)
        img_ids = [i[0] for i in temp]
        img_urls = [i[1] for i in temp]
        img_scores = [i[2] for i in temp]
        #html_path = root + 'present_htmls_test/' + event_id + '/rearranged_ranking_sigmoid_result.html'
        html_path = root + 'present_htmls_test/' + event_id + '/rearranged_'+model_name+'_result.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]#[0][0]
                    #score = 1/(1 + np.exp(-score))
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'
                    if score > 1:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >0.8:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >= 0:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score < 0:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()


'''evaluation'''

def evaluate_baseline(model_names = ['alexnet6k','alexnet3k','vgg']):
    for model_name in model_names:
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name + '_wedding_predict_result_10.cPickle', 'r')
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
                    rank_difference[event_id] += g - p
                else:
                    rank_difference[event_id] += p - g - n_same_g + 1

        print rank_difference
        print sum([rank_difference[i] for i in rank_difference])
def evaluate_MAP(min_retrieval = 5, model_names = ['alexnet6k','alexnet3k','vgg']):
    maps = []
    n_ks = []
    for model_name in model_names:
        APs = []
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name + '_wedding_predict_result_10.cPickle', 'r')
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
def evaluate_rank_reweighted(model_names = ['alexnet6k','alexnet3k','vgg']):
    for model_name in model_names:
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name + '_wedding_predict_result_10.cPickle', 'r')
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
def evaluate_rank_reweighted2(model_names = ['', 'face_']):
    for model_name in model_names:
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name + 'vgg_wedding_predict_result_10.cPickle', 'r')
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
def evaluate_MAP2(min_retrieval = 5, model_names = ['','face_']):
    maps = []
    n_ks = []
    for model_name in model_names:
        APs = []
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name + 'vgg_wedding_predict_result_10.cPickle', 'r')
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

def evaluate_rank_reweighted_(model_names):
    for model_name in model_names:
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name, 'r')
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
def evaluate_MAP_ (model_names, min_retrieval = 5):
    maps = []
    n_ks = []
    for model_name in model_names:
        APs = []
        #f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_test_result_v2.cPickle','r')
        #ground_truth = cPickle.load(f)
        #f.close()
        f = open(root + 'baseline_wedding_test/wedding_test_result_v1.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        rank_difference = {}
        f = open(root + 'baseline_wedding_test/' + model_name, 'r')
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


'''separate training and validation'''
def separate_validation():
    in_path = root + 'baseline_wedding_test/wedding_training_id.cPickle'
    f = open(in_path, 'r')
    training_id = cPickle.load(f)
    f.close()

    validation_id = random.sample(xrange(len(training_id)), 50)
    validation = [training_id[i] for i in validation_id]
    training = [training_id[i] for i in xrange(len(training_id)) if i not in validation_id]

    f = open(root + 'baseline_wedding_test/training_validation/validation_id.cPickle','wb')
    cPickle.dump(validation, f)
    f.close()
    f = open(root + 'baseline_wedding_test/training_validation/training_id.cPickle','wb')
    cPickle.dump(training, f)
    f.close()
    print training_id
def create_path_validation():
    in_path = root + 'baseline_wedding_test/training_validation/training_id.cPickle'
    f = open(in_path, 'r')
    events_training = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_wedding_test/training_validation/validation_id.cPickle'
    f = open(in_path, 'r')
    events_test = cPickle.load(f)
    f.close()

    v_ = []; t_ = []
    load_path = root+'all_images_curation.txt'
    save_paths1 = root + 'baseline_wedding_test/training_validation/val_training_path.txt'
    f1 = open(save_paths1, 'wb')
    save_paths2 = root + 'baseline_wedding_test/training_validation/val_validation_path.txt'
    f2 = open(save_paths2, 'wb')
    with open(load_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            if meta[1]+'_'+meta[3] in events_training:
                if meta[1]+'_'+meta[3] not in t_:
                    t_.append(meta[1]+'_'+meta[3])
                path = 'C:\Users\yuwang\Documents\download_event_recognition\\'+meta[3]+'\\'+meta[2]+'.jpg\r\n'
                f1.write(path)
            elif meta[1]+'_'+meta[3] in events_test:

                if meta[1]+'_'+meta[3] not in v_:
                    v_.append(meta[1]+'_'+meta[3])
                path = 'C:\Users\yuwang\Documents\download_event_recognition\\'+meta[3]+'\\'+meta[2]+'.jpg\r\n'
                f2.write(path)
    f1.close()
    f2.close()
def create_knn_cPickle_validation():
    image_ids = []
    in_path = root + 'baseline_wedding_test/training_validation/val_training_path.txt'
    with open(in_path,'r') as data:
        for line in data:
            name = line.split('\\')[-1]
            name = name.split('.')[0]
            image_ids.append(name)

    event_ids = []
    i = 0
    in_path = root+'all_images_curation.txt'
    with open(in_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            id = meta[2]
            if id == image_ids[i]:
                i += 1
                event_ids.append(meta[1] + '_' + meta[3] + '/' + meta[2])
                if i == len(image_ids):
                    break

    f = open(root + 'baseline_wedding_test/training_validation/val_training_image_ids.cPickle','wb')
    cPickle.dump(event_ids, f)
    f.close()

    image_ids = []
    in_path = root + 'baseline_wedding_test/training_validation/val_validation_path.txt'
    with open(in_path,'r') as data:
        for line in data:
            name = line.split('\\')[-1]
            name = name.split('.')[0]
            image_ids.append(name)

    event_ids = []
    i = 0
    in_path = root+'all_images_curation.txt'
    with open(in_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            id = meta[2]
            if id == image_ids[i]:
                i += 1
                event_ids.append(meta[1] + '_' + meta[3] + '/' + meta[2])
                if i == len(image_ids):
                    break

    f = open(root + 'baseline_wedding_test/training_validation/val_validation_image_ids.cPickle','wb')
    cPickle.dump(event_ids, f)
    f.close()



    f = open(root + 'baseline_wedding_test/wedding_training_ulr_dict.cPickle', 'r')
    training_url_dict = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/training_validation/val_validation_image_ids.cPickle','r')
    test_images = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/training_validation/val_training_image_ids.cPickle','r')
    training_images = cPickle.load(f)
    f.close()
    path = root + 'baseline_wedding_test/training_validation/val_'+model_name+'_knn.txt'
    knn_dict = {}

    with open(path, 'r') as data:
        for line in data:
            test_index = int(line.split(':')[0])
            groups = re.findall('\((.*?)\)',line)
            if len(groups) != 50:
                print 'ERROR DETECTED!'
                return
            this_img = test_images[test_index]
            knn_dict[test_index] = [(this_img, training_url_dict[this_img])]
            for i in groups:
                index = int(i.split(',')[0])
                score = float(i.split(',')[1])
                knn_dict[test_index].append((score, training_images[index], training_url_dict[training_images[index]]))

    f = open(root + 'baseline_wedding_test/training_validation/val_'+model_name+'_knn.cPickle','wb')
    cPickle.dump(knn_dict,f)
    f.close()
def from_txt_to_pickle_validation():
    in_path = root + 'baseline_wedding_test/training_validation/val_'+model_name+'_training_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)

    f = open(root + 'baseline_wedding_test/training_validation/val_'+model_name+'_training_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
    in_path = root + 'baseline_wedding_test/training_validation/val_'+model_name+'_validation_features.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)
    f = open(root + 'baseline_wedding_test/training_validation/val_'+model_name+'_validation_features.cPickle','wb')
    cPickle.dump(features, f)
    f.close()
def baseline_predict_validation(n_vote = 10):
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    training_scores_dict = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_wedding_test/training_validation/val_'+model_name+'_knn_facereweighted.cPickle', 'r')
    knn = cPickle.load(f)
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

    f = open(root + 'baseline_wedding_test/training_validation/val_'+model_name+'_predict_result_'+str(n_vote)+'_facereweighted.cPickle','wb')
    cPickle.dump(test_prediction_event, f)
    f.close()
def create_retrieval_image_validation(max_display = 10):
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    training_result = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/training_validation/val_'+model_name+'_knn.cPickle','r')
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

    for event_id in event_knn:
        f = open(root + 'baseline_wedding_test/training_validation/knn_htmls/'+model_name+'_retrieval_top10_'+event_id+'.html','wb')
        f.write('<head>'+model_name+' Retrieval Result #'+str(html_count)+'</head> <title>'+model_name+' Retrieval Result '+str(html_count)+'</title>\n' )
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



'''rerank knn result with face positions'''
def find_face_positions():
    #create_knn_cPickle(path = root + 'baseline_wedding_test/' + model_name + '_wedding_knn_all.txt')
    root1 = root + 'face_recognition/'
    lists = [o for o in os.listdir(root1) if os.path.isdir(root1+o)]
    img_face_and_sizes = {}
    download_img_root = root + 'download_event_recognition/'
    for name in lists:
        folder_path = root1 + name
        event_name = name.split('-')[0]
        lines = []
        print folder_path + '/all-scores.txt'
        with open(folder_path + '/all-scores.txt','r') as data:
            for line in data:
                lines.append(line)
        i = 0
        while i < len(lines):
            temp_sizes = []
            line = lines[i]
            if line[0] != '/':
                print 'ERROR!'
            img_name = line.split('/')[-1]
            img_name = img_name.split()[0]
            img_path = download_img_root + event_name.split('_')[1] + '/' + img_name
            im = Image.open(img_path)
            width, height = im.size
            i += 1
            line = lines[i]
            num_img = int(line.split()[0])
            for j in xrange(num_img):
                i += 1
                line = lines[i]
                x,y,size1, size2, not_used= line.split(' ')
                x = int(x); y=int(y); size1=float(size1); size2=float(size2)
                #print size1 * size2 / (width * height)
                if size1 * size2 / (width * height) < 1/40:
                    continue
                temp_sizes.append(((x + size1/2)/width, (y + size2/2)/height, size1/width, size2/height, size1/width*size2/height))
            temp_sizes_sorted = sorted(temp_sizes,key=itemgetter(0))
            if len(temp_sizes) > 3:
                temp_sizes_sorted = [(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)]
            else:
                temp_sizes_sorted = sorted(temp_sizes,key=itemgetter(4))
                temp_sizes_sorted += [(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)]
                temp_sizes_sorted = temp_sizes_sorted[:3]

            img_face_and_sizes[event_name + '/' + img_name.split('.')[0]] = temp_sizes_sorted
            i += 1


    f = open(root + 'knn_face_combine/img_face_and_sizes_dict_v2.cPickle','wb')
    cPickle.dump(img_face_and_sizes, f)
    f.close()
def face_img_match(face_1, face_2):
    similarity_ = 0
    for i,j in zip(face_1, face_2):
        i = (i[0],i[1],i[2]*2,i[3]*2)
        j = (j[0],j[1],j[2]*2,j[3]*2)
        if i[2] == 0 and i[1] == 0:
            similarity_ += 1/2
        else:
            temp = np.dot(i,j)/(np.sum(np.square(i)) + np.sum(np.square(j)))
            similarity_ += temp
    return similarity_ * 2 / 3
def rerank_knn_with_face():
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn.cPickle', 'r')
    knn = cPickle.load(f)
    f.close()

    f = open(root + 'knn_face_combine/img_face_and_sizes_dict.cPickle','r')
    img_face_and_sizes = cPickle.load(f)
    f.close()
    for i in knn:
        this_test_img = knn[i]
        this_test_id = this_test_img[0][0]
        #if this_test_id == '0_52046304@N00/319956609':
        #        print this_test_img
        img_face_size_query = img_face_and_sizes[this_test_id]
        count = 0
        face_similarity_list = []
        for j in xrange(1, len(this_test_img)):
            img_name = this_test_img[j][1]
            img_face_size_this = img_face_and_sizes[img_name]
            face_similarity_list.append(face_img_match_2(img_face_size_query, img_face_size_this))

        temp = [int(ii>0) for ii in face_similarity_list]
        if sum(temp) < 10:
            continue
        for j in xrange(len(face_similarity_list)):
            weight = this_test_img[j+1][0]
            face_similarity = face_similarity_list[j]
            this_test_img[j+1] = (weight * (face_similarity + 1),this_test_img[j+1][1], this_test_img[j+1][2])
            count += 1
        temp = this_test_img[1:]
        temp = sorted(temp,key=itemgetter(0), reverse=True)
        this_test_img = [this_test_img[0]] + temp
        knn[i] = this_test_img
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn_facereweighted.cPickle', 'wb')
    cPickle.dump(knn, f)
    f.close()
def rerank_knn_with_face_old():
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn.cPickle', 'r')
    knn = cPickle.load(f)
    f.close()

    f = open(root + 'knn_face_combine/img_face_and_sizes_dict.cPickle','r')
    img_face_and_sizes = cPickle.load(f)
    f.close()
    for i in knn:
        this_test_img = knn[i]
        this_test_id = this_test_img[0][0]
        img_face_size_query = img_face_and_sizes[this_test_id]
        count = 0
        face_similarity_list = []
        for j in xrange(1, len(this_test_img)):
            img_name = this_test_img[j][1]
            img_face_size_this = img_face_and_sizes[img_name]
            face_similarity = (face_img_match(img_face_size_query, img_face_size_this))
            if count == 0 and face_similarity == 0:
                continue
            weight = this_test_img[j][0]
            this_test_img[j] = (weight * face_similarity,this_test_img[j][1], this_test_img[j][2])
            count += 1
        temp = this_test_img[1:]
        temp = sorted(temp,key=itemgetter(0), reverse=True)
        this_test_img = [this_test_img[0]] + temp
        knn[i] = this_test_img
    f = open(root + 'baseline_wedding_test/'+model_name+'_wedding_knn_facereweighted_old.cPickle', 'wb')
    cPickle.dump(knn, f)
    f.close()

def face_intersection(face_1, face_2):
    if (face_1[2] == 0 and face_1[3] == 0) or (face_2[2] == 0 and face_2[3] == 0):
        return 0
    x1,y1,w1,h1,s1 = face_1
    x2,y2,w2,h2,s2 = face_2
    x_min1 = x1-w1;x_max1 = x1+w1
    x_min2 = x2-w2;x_max2 = x2+w2
    y_min1 = y1-h1;y_max1 = y1+h1
    y_min2 = y2-h2;y_max2 = y2+h2

    if x_min1 > x_max2 or x_max1 < x_min2 or y_min1 > y_max2 or y_max1 < y_min2:
        intersection = 0
    else:
        w_i = min(x_max1, x_max2) - max(x_min1, x_min2)
        h_i = min(y_max1, y_max2) - max(y_min1, y_min2)
        intersection = w_i * h_i
    return intersection
def face_img_match_2(face_1,face_2):
    intersection = [[]]
    for i in face_1:
        for j in face_2:
            intersection[-1].append(face_intersection(i,j))
        intersection += [[]]
    intersection = intersection[:-1]

    idx_2 = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    inter_sum = []
    for i in idx_2:
        inter_sum.append(intersection[0][i[0]] + intersection[1][i[1]] + intersection[2][i[2]])
    #argsort_sum = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(inter_sum))]
    temp1 = 0
    for i in face_1:
        temp1 += i[4]
    temp2 = 0
    for i in face_2:
        temp2 += i[4]

    if temp1 == 0 and temp2 == 0:
        return 1
    return float(max(inter_sum))/(max(temp1,temp2))


if __name__ == '__main__':

    baseline_predict()

    #create_result_htmls_rearranged()
    #show_test_predict_img_rearranged()

    #correct_wedding('test')

    #f = open(root + 'baseline_wedding_test/wedding_training_id.cPickle','r')
    #event_ids = cPickle.load(f)
    #f.close()

    #input_path = root + 'all_output/all_output.csv'
    #create_result_htmls('training', event_ids, input_path)

    #create_path_all()
    #create_path()
    #linux_create_path()
    #create_csv_wedding()
    #read_amt_result('training')
    #read_amt_result('test')


    #from_npy_to_dicts('training')
    #from_npy_to_dicts('test')
    #create_dict_url()


    #create_result_htmls_rearranged()
    #from_txt_to_pickle()
    #create_knn_cPickle()
    #find_similar(threshold = 0.9, type='training')
    #images_for_contrsim('training')
    #find_similar(threshold = 0.9, type='test')
    #images_for_contrsim('test')
    #correct_amt_result()
    #correct_amt_result('test')
    #create_retrieval_image()


    #model_name = 'alexnet6k'
    #baseline_predict()
    #model_name = 'vgg'
    #baseline_predict()
    #model_name = 'alexnet3k'
    #baseline_predict()
    #model_name = 'alexnet6k'
    #baseline_predict_nomore2()
    #model_name = 'vgg'
    #baseline_predict_nomore2()
    #model_name = 'alexnet3k'
    #baseline_predict_nomore2()

    #show_test_predict_img()

    #create_result_htmls_rearranged()
    #show_test_predict_img_rearranged()
    #create_retrieval_image()

    #evaluate_different_model()


    #create_retrieval_image()


    #find_face_positions()
    #rerank_knn_with_face_old()
    #rerank_knn_with_face()
    #baseline_predict()

    #create_retrieval_image()
    '''
    model_names = ['vgg_wedding_predict_result_10.cPickle', 'face_vgg_wedding_predict_result_10.cPickle', 'face_vgg_wedding_predict_result_10_facereweighted.cPickle']
    evaluate_rank_reweighted_(model_names)

    no_face= []; retrieval = []; face_retrieval = []; percent = []
    for i in xrange(5, 36, 3):
        temp, percent_temp = evaluate_MAP_(model_names, min_retrieval=i)
        no_face.append(temp[0])
        retrieval.append(temp[1])
        face_retrieval.append(temp[2])
        percent.append(percent_temp)
    print no_face
    print retrieval
    print face_retrieval
    print percent
    

    #separate_validation()
    #create_path_validation()
    #create_knn_cPickle_validation()
    #from_txt_to_pickle_validation()
    #baseline_predict_validation()
    #create_retrieval_image_validation()
    #show_test_predict_img_rearranged()


    #find_face_positions()
    #rerank_knn_with_face()
    #create_retrieval_image()


    #create_result_htmls_rearranged()
    #baseline_predict()
    #show_test_predict_img_rearranged()

    #rerank_knn_with_face()
    '''