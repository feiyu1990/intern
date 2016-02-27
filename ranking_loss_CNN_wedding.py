__author__ = 'wangyufei'

import cPickle
import random
import scipy.io as sio
import os
import numpy as np
import csv
import h5py
root = '/Users/wangyufei/Documents/Study/intern_adobe/'
#root = 'C:/Users/yuwang/Documents/'
dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
             'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)', 'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}


dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}
def find_valid_examples(name = 'training'):
    f = open(root + 'baseline_wedding_test/vgg_wedding_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_wedding_test/linux_wedding_'+name+'_path.txt'
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
    img_pair = img_pair[:500]
    out_path1 = root + 'ranking_loss_CNN/ranking_part_'+name+'.txt'
    out_path2 = root + 'ranking_loss_CNN/ranking_part_'+name+'_p.txt'
    f1 = open(out_path1,'w')
    f2 = open(out_path2,'w')
    for i in img_pair:
        line = img_path_dict[i[0]] + ' ' + str(i[2]) + '\n'
        f1.write(line)
        line = img_path_dict[i[1]] + ' ' + str(i[2]) + '\n'
        f2.write(line)
    f1.close()
    f2.close()
    pass
def find_valid_examples_multilabel(name = 'training'):
    f = open(root + 'baseline_wedding_test/vgg_wedding_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/vgg_wedding_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_wedding_test/wedding_'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_wedding_test/linux_wedding_'+name+'_path.txt'
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
    img_pair = img_pair[:300]
    out_path1 = root + 'ranking_loss_CNN/ranking_multilabel_part_'+name+'.txt'
    out_path2 = root + 'ranking_loss_CNN/ranking_multilabel_part_'+name+'_p.txt'
    f1 = open(out_path1,'w')
    f2 = open(out_path2,'w')
    for i in img_pair:
        line = img_path_dict[i[0]] + ' ' + str(i[2]) + ' ' + str(int(20*ground_truth_training_dict[i[0]])) + '\n'
        f1.write(line)
        line = img_path_dict[i[1]] + ' ' + str(i[2]) + ' ' + str(int(20*ground_truth_training_dict[i[1]])) + '\n'
        f2.write(line)
    f1.close()
    f2.close()
    pass
def find_valid_examples_reallabel(name = 'training'):
    f = open(root + 'baseline_wedding_test/vgg_wedding_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/vgg_wedding_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_wedding_test/wedding_'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_wedding_test/linux_wedding_'+name+'_path.txt'
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
    #img_pair = img_pair[:300]
    out_path1 = root + 'ranking_loss_CNN/ranking_reallabel_part_'+name+'.txt'
    out_path2 = root + 'ranking_loss_CNN/ranking_reallabel_part_'+name+'_p.txt'
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


def change_to_guru(name = 'training'):
    in_path = root + 'ranking_loss_CNN/ranking_'+name+'.txt'
    out_path = root + 'ranking_loss_CNN/guru_ranking_'+name+'.txt'
    f = open(out_path, 'w')
    with open(in_path,'r') as data:
        for line in data:
            img_path_new = '/home/feiyu1990/local/event_curation/wedding_images/' + '/'.join(line.split('/')[9:])
            f.write(img_path_new)

    in_path = root + 'ranking_loss_CNN/ranking_'+name+'_p.txt'
    out_path = root + 'ranking_loss_CNN/guru_ranking_'+name+'_p.txt'
    f = open(out_path, 'w')
    with open(in_path,'r') as data:
        for line in data:
            img_path_new = '/home/feiyu1990/local/event_curation/wedding_images/' + '/'.join(line.split('/')[9:])
            f.write(img_path_new)

def merge_all_examples(threshold, name):
    imgs = []
    for event_name in dict_name:
        in_file_name = root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/'+str(threshold)+'_'+event_name+'_ranking_reallabel_'+name+'.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs.append([meta + ' ' + event_name])
    count = 0
    for event_name in dict_name:
        in_file_name = root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/'+str(threshold)+'_'+event_name+'_ranking_reallabel_'+name+'_p.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs[count].append(meta + ' ' + event_name)
                count += 1


    random.shuffle(imgs)
    f1 = open(root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/ranking_reallabel_'+name+'_all.txt', 'w')
    f2 = open(root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/ranking_reallabel_'+name+'_all_p.txt', 'w')
    for i,j in imgs:
        f1.write(' '.join(i.split(' ')[:-1]) + '\n')
        f2.write(' '.join(j.split(' ')[:-1]) + '\n')
    f1.close()
    f2.close()

    events = []
    for i,j in imgs:
        events.append(j.split(' ')[-1])
    event_labels = []
    for event_label in events:
            event_label = dict_name2[event_label]
            temp = np.zeros((23,))
            temp[event_label-1] = 1
            event_labels.append(temp)
    event_labels = np.array(event_labels)
    f = h5py.File(root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/'+name+'_label.h5','w')
    f.create_dataset("event_label", data=event_labels)
    f.close()



    '''
    f3 = open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_training_all_events_only.txt', 'w')
    for event_name in events:
        f3.write(event_name + ' ' + str(dict_name2[event_name]) + '\n')
    f3.close()

    imgs = []
    for event_name in dict_name:
        in_file_name = root + 'ranking_loss_CNN/training_CNN_all_events/' + event_name + '_ranking_part_test.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs.append([meta + ' ' + event_name])
    count = 0
    for event_name in dict_name:
        in_file_name = root + 'ranking_loss_CNN/training_CNN_all_events/' + event_name + '_ranking_part_test_p.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs[count].append(meta + ' ' + event_name)
                count += 1


    random.shuffle(imgs)


    events = []
    f1 = open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_part_test_all_event_name.txt', 'w')
    f2 = open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_part_test_all_p_event_name.txt', 'w')
    for i,j in imgs:
        f1.write(i + '\n')
        f2.write(j + '\n')
        events.append(j.split(' ')[-1])
    f1.close()
    f2.close()

    f3 = open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_part_test_all_events_only.txt', 'w')
    for event_name in events:
        f3.write(event_name + ' ' + str(dict_name2[event_name]) + '\n')
    f3.close()

    f1 = open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_part_test_all.txt', 'w')
    f2 = open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_part_test_all_p.txt', 'w')
    for i,j in imgs:
        f1.write(' '.join(i.split(' ')[:-1]) + '\n')
        f2.write(' '.join(j.split(' ')[:-1]) + '\n')
    f1.close()
    f2.close()
    '''
def create_hd5f_file():
    event_labels = []
    with open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_training_all_events_only.txt','r') as data:
        for line in data:
            event_label = line.split(' ')[1]
            event_label = int(event_label.split()[0])
            temp = np.zeros((23,))
            temp[event_label-1] = 1
            event_labels.append(temp)
    event_labels = np.array(event_labels)
    f = h5py.File(root + 'ranking_loss_CNN/training_CNN_all_events/training_label.h5','w')
    f.create_dataset("event_label", data=event_labels)
    f.close()

    event_labels = []
    with open(root + 'ranking_loss_CNN/training_CNN_all_events/ranking_reallabel_part_test_all_events_only.txt','r') as data:
        for line in data:
            event_label = line.split(' ')[1]
            event_label = int(event_label.split()[0])
            temp = np.zeros((23,))
            temp[event_label-1] = 1
            event_labels.append(temp)
    event_labels = np.array(event_labels)
    f = h5py.File(root + 'ranking_loss_CNN/training_CNN_all_events/part_test_label.h5','w')
    f.create_dataset("event_label", data=event_labels)
    f.close()

def guru_find_valid_examples_all_reallabel(threshold, event_name, name = 'training'):
    f = open(root + 'baseline_all/' + event_name + '/vgg_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all/' + event_name + '/vgg_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_all/' + event_name + '/'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_all/' + event_name + '/linux_'+name+'_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            temp = line.split(' ')[0]
            img_path_new = '/home/feiyu1990/local/event_curation/curation_images/'+event_name + '/' + '/'.join(temp.split('/')[9:])
            img_paths.append(img_path_new)

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
            if this_event[-1][2] - this_event[i][2] < threshold:
                break
            img_1 = this_event[i]
            for j in xrange(i + 1, len_):
                img_2 = this_event[j]
                if img_2[2] - img_1[2] < threshold:
                    continue
                temp = random.sample([0, 1], 1)
                if temp[0] == 0:
                    img_pair.append((img_1[1], img_2[1], 0))
                else:
                    img_pair.append((img_2[1], img_1[1], 1))
                count += 1
        count_all += count
    random.shuffle(img_pair)
    if not os.path.exists(root + 'to_guru/CNN_all_event_'+str(threshold)):
        os.mkdir(root + 'to_guru/CNN_all_event_'+str(threshold))
    if not os.path.exists(root + 'to_guru/CNN_all_event_'+str(threshold)+'/data'):
        os.mkdir(root + 'to_guru/CNN_all_event_'+str(threshold)+'/data')
    if name == 'test':
        img_pair = img_pair[:len(img_pair)/10]
    out_path1 = root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/'+str(threshold) + '_'+event_name+'_ranking_reallabel_'+name+'.txt'
    print out_path1
    out_path2 = root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/'+str(threshold) + '_'+event_name+'_ranking_reallabel_'+name+'_p.txt'
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
    return count_all
def find_valid_examples_all_reallabel(event_name, name = 'training'):
    f = open(root + 'baseline_all/' + event_name + '/vgg_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all/' + event_name + '/vgg_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_all/' + event_name + '/'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_all/' + event_name + '/linux_'+name+'_path.txt'
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
    random.shuffle(img_pair)
    if name == 'test':
        img_pair = img_pair[:len(img_pair)/10]
        out_path1 = root + 'ranking_loss_CNN/'+name + '_data/'+event_name+'_ranking_reallabel_part_'+name+'.txt'
        print out_path1
        out_path2 = root + 'ranking_loss_CNN/'+name + '_data/'+event_name+'_ranking_reallabel_part_'+name+'_p.txt'
    else:
        out_path1 = root + 'ranking_loss_CNN/'+name + '_data/'+event_name+'_ranking_reallabel_'+name+'.txt'
        print out_path1
        out_path2 = root + 'ranking_loss_CNN/'+name + '_data/'+event_name+'_ranking_reallabel_'+name+'_p.txt'
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
    return count_all

def find_valid_examples_all_reallabel_nomargin(event_name, name = 'training'):
    f = open(root + 'baseline_all/' + event_name + '/vgg_'+name+'_result_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all/' + event_name + '/vgg_'+name+'_result_dict_v2.cPickle','r')
    ground_truth_training_dict = cPickle.load(f)
    f.close()

    set_ = set()
    for i in ground_truth_training_dict:
        if ground_truth_training_dict[i] not in set_:
            set_.add(ground_truth_training_dict[i])

    f = open(root + 'baseline_all/' + event_name + '/'+name+'_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_all/' + event_name + '/linux_'+name+'_path.txt'
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
    random.shuffle(img_pair)
    img_pair = img_pair[:len(img_pair)/10]
    out_path1 = root + 'ranking_loss_CNN/'+ name + '_data_nomargin/'+event_name+'_ranking_reallabel_part_'+name+'.txt'
    print out_path1
    out_path2 = root + 'ranking_loss_CNN/'+ name + '_data_nomargin/'+event_name+'_ranking_reallabel_part_'+name+'_p.txt'
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
    return count_all
def merge_all_examples_nomargin():
    imgs = []
    for event_name in dict_name:
        in_file_name = root + 'ranking_loss_CNN/training_data_nomargin/' + event_name + '_ranking_reallabel_training.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs.append([meta + ' ' + event_name])
    count = 0
    for event_name in dict_name:
        in_file_name = root + 'ranking_loss_CNN/training_data_nomargin/' + event_name + '_ranking_reallabel_training_p.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs[count].append(meta + ' ' + event_name)
                count += 1


    random.shuffle(imgs)
    f1 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_training_all.txt', 'w')
    f2 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_training_all_p.txt', 'w')
    for i,j in imgs:
        f1.write(' '.join(i.split(' ')[:-1]) + '\n')
        f2.write(' '.join(j.split(' ')[:-1]) + '\n')
    f1.close()
    f2.close()

    events = []
    f1 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_training_all_event_name.txt', 'w')
    f2 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_training_all_p_event_name.txt', 'w')
    for i,j in imgs:
        f1.write(i + '\n')
        f2.write(j + '\n')
        events.append(j.split(' ')[-1])
    f1.close()
    f2.close()

    f3 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_training_all_events_only.txt', 'w')
    for event_name in events:
        f3.write(event_name + ' ' + str(dict_name2[event_name]) + '\n')
    f3.close()

    imgs = []
    for event_name in dict_name:
        in_file_name = root + 'ranking_loss_CNN/test_data_nomargin/' + event_name + '_ranking_reallabel_part_test.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs.append([meta + ' ' + event_name])
    count = 0
    for event_name in dict_name:
        in_file_name = root + 'ranking_loss_CNN/test_data_nomargin/' + event_name + '_ranking_reallabel_part_test_p.txt'
        with open(in_file_name, 'r') as data:
            for line in data:
                meta = line[:-1]
                imgs[count].append(meta + ' ' + event_name)
                count += 1


    random.shuffle(imgs)


    events = []
    f1 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_part_test_all_event_name.txt', 'w')
    f2 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_part_test_all_p_event_name.txt', 'w')
    for i,j in imgs:
        f1.write(i + '\n')
        f2.write(j + '\n')
        events.append(j.split(' ')[-1])
    f1.close()
    f2.close()

    f3 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_part_test_all_events_only.txt', 'w')
    for event_name in events:
        f3.write(event_name + ' ' + str(dict_name2[event_name]) + '\n')
    f3.close()

    f1 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_part_test_all.txt', 'w')
    f2 = open(root + 'ranking_loss_CNN/training_data_nomargin/ranking_reallabel_part_test_all_p.txt', 'w')
    for i,j in imgs:
        f1.write(' '.join(i.split(' ')[:-1]) + '\n')
        f2.write(' '.join(j.split(' ')[:-1]) + '\n')
    f1.close()
    f2.close()

if __name__ == '__main__':
    #change_to_guru('part_test')
    #change_to_guru('training')
    #change_to_guru()
    '''
    for event_name in dict_name2:
        if event_name == 'Wedding':
            continue
        in_path = root + 'baseline_all/'+event_name+'/linux_test_path.txt'
        out_path = root + 'baseline_all/'+event_name+ '/guru_test_path.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
            for line in data:
                img_path_new = '/home/feiyu1990/local/event_curation/curation_images/'+event_name + '/' + '/'.join(line.split('/')[9:])
                f.write(img_path_new)
        in_path = root + 'baseline_all/'+event_name+'/linux_training_path.txt'
        out_path = root + 'baseline_all/'+event_name+ '/guru_training_path.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
            for line in data:
                img_path_new = '/home/feiyu1990/local/event_curation/curation_images/'+event_name + '/' + '/'.join(line.split('/')[9:])
                f.write(img_path_new)



    '''
    #originally, for margin = 0.3 -> threshold = 1.2

    threshold = 0.1
    count_all = 0
    #f1 = h5py.File(root + 'to_guru/CNN_all_event_'+str(threshold)+'/data/'+str(threshold)+'_training_label.h5')
    #event_label = f1['event_label'].value
    #print event_label.shape, len(event_label)

    for event_name in dict_name:
        guru_find_valid_examples_all_reallabel(threshold, event_name, 'test')
        count = guru_find_valid_examples_all_reallabel(threshold, event_name, 'training')
        print event_name, count
        count_all += count
    print count_all

    merge_all_examples(threshold, 'training')
    merge_all_examples(threshold, 'test')
    #create_hd5f_file()