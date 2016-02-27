__author__ = 'wangyufei'


import shutil
import os
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
import re
import operator
import scipy.stats
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, chi2
from collections import Counter, defaultdict
#import Bio.Cluster
# combine_face_model = '_combined_10_fromnoevent.cPickle'
# combine_face_model = '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle'
global_permutation_time = 1



block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
root = '/home/feiyu1990/local/event_curation/'
baseline_name = 'baseline_all_0509/'
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

#this is from affinity clustering (cluster #6)
dict_subcategory = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 3, 7: 4, 8: 4, 9: 2, 10: 4, 11: 3, 12: 4, 13: 4, 14: 4,
                    15: 5, 16: 3, 17: 4, 18: 4, 19: 2, 20: 4, 21: 4, 22: 4}

#this is from spectral clustering (cluster #3)
dict_subcategory2 = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 2, 10: 0, 11: 2, 12: 0,
                     13: 0, 14: 0, 15: 2, 16: 2, 17: 0, 18: 0, 19: 2, 20: 1, 21: 2, 22: 0}


class create_cross_validation:
    def __init__(self):
        #self.copy_old_to_new()
        # self.dict_name = dict_name2
        # for event_name in self.dict_name:
        # #     self.separate_training_to_val(event_name)
        #     for i in xrange(5):
        # #         self.create_path(event_name, i)
        # #         self.from_txt_to_imgid(event_name, 'val_training', i)
        # #         self.from_txt_to_imgid(event_name, 'val_validation', i)
        # #         self.create_csv(event_name, i)
        # #         self.create_dict_url(event_name, i)
        #         self.create_ground_truth(event_name, 'val_validation', i)
        #         self.create_ground_truth(event_name, 'val_training', i)
        # for i in xrange(5):
        #     self.to_guru_file(i, 'training')
        #     self.to_guru_file(i, 'test')



        # for event_name in dict_name2:
        #     if not os.path.exists(root + 'baseline_all_0509/' + event_name + '/alexnet6k_knn.cPickle'):
        #         print 'creating: ', root + 'baseline_all_0509/' + event_name + '/alexnet6k_knn.cPickle'
        #         self.create_knn_cPickle(event_name, 'alexnet6k')
        #     if not os.path.exists(root + 'baseline_all_0509/' + event_name + '/vgg_knn.cPickle'):
        #         print 'creating: ', root + 'baseline_all_0509/' + event_name + '/vgg_knn.cPickle'
        #         self.create_knn_cPickle(event_name, 'vgg')
        #     self.create_csv_traintest(event_name)
        #
        #     self.read_amt_result(event_name)
        #     self.read_amt_result(event_name, 'test')
        #     self.correct_amt_result(event_name, 'test')
        #     self.correct_amt_result(event_name, 'training')
        #     for i in xrange(5):
        #       self.create_ground_truth(event_name, 'val_validation', i)
        #       self.create_ground_truth(event_name, 'val_training', i)
        #       self.create_csv(event_name, i)
        #
        #     self.baseline_predict(event_name, 'alexnet6k')
        #     self.baseline_predict(event_name, 'vgg')


        for i in xrange(5):
            self.to_guru_file(i, 'training')
            self.to_guru_file(i, 'test')
            self.to_guru_file(i, 'val_training')
            self.to_guru_file(i, 'val_validation')
    @staticmethod
    def copy_old_to_new():
        copy_list = ['alexnet6k_knn.cPickle','training.csv','vgg_knn.cPickle','alexnet6k_knn.txt','training_event_id.cPickle',
                     'vgg_knn.txt','alexnet6k_predict_result_10_dict.cPickle','training_image_ids.cPickle','vgg_predict_result_10_dict.cPickle',
                     'alexnet6k_test_tags.txt','training_path.txt','vgg_test_features.cPickle','alexnet6k_train_tags.txt',
                     'training_result_v1.cPickle','vgg_test_features.txt','guru_test_path.txt','training_ulr_dict.cPickle',
                     'vgg_test_result_dict_v2.cPickle','guru_training_path.txt','train_tags.txt','vgg_test_result_v2.cPickle',
                     'vgg_test_result_v2_permuted.cPickle','vgg_test_similar_list.cPickle','linux_test_path.txt','vgg_training_features.cPickle',
                     'linux_training_path.txt','vgg_training_features.txt','predict_result_10_dict.cPickle','test.csv',
                     'vgg_training_result_v2.cPickle','test_event_id.cPickle','vgg_training_result_v2_permuted.cPickle',
                     'test_image_ids.cPickle','vgg_training_similar_list.cPickle','test_path.txt','test_result_v1.cPickle',
                     'test_tags.txt','test_ulr_dict.cPickle','vgg_training_result_dict_v2.cPickle']
        for event_name in dict_name2:
            if not os.path.exists(root + 'baseline_all_0509/' + event_name):
                os.mkdir(root + 'baseline_all_0509/' + event_name)
            for i in copy_list:
                src = root + 'baseline_all_0509/' + event_name + '/' + i
                dst = root + 'baseline_all_0509/' + event_name + '/' + i
                try:
                    shutil.copyfile(src, dst)
                except:
                    pass
    @staticmethod
    def chunkIt(seq, num):
      avg = len(seq) / float(num)
      out = []
      last = 0.0
      while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
      return out

    def separate_training_to_val(self, event_name):
        in_path = root + 'baseline_all_0509/'+event_name+'/training_event_id.cPickle'
        f = open(in_path, 'r')
        event_training = cPickle.load(f)
        f.close()
        print event_name, len(event_training)
        random.shuffle(event_training)
        separated_val = self.chunkIt(event_training, 5)
        for i in xrange(5):
            if not os.path.exists(root + 'baseline_all_0509/' + event_name + '/validation_' + str(i)):
                os.mkdir(root + 'baseline_all_0509/' + event_name + '/validation_' + str(i))
            validation = separated_val[i]
            train = [event_id for event_id in event_training if event_id not in validation]
            print i, '(', len(validation), ',', len(train),')'
            print validation
            print train
            out_path = root + 'baseline_all_0509/' + event_name + '/validation_' + str(i) + '/val_validation_event_id.cPickle'
            f = open(out_path,'w')
            cPickle.dump(validation,f)
            f.close()
            out_path = root + 'baseline_all_0509/' + event_name + '/validation_' + str(i) + '/val_training_event_id.cPickle'
            f = open(out_path,'w')
            cPickle.dump(train,f)
            f.close()
    @staticmethod
    def create_path(check_type, val_id):
        in_path = root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/val_validation_event_id.cPickle'
        f = open(in_path, 'r')
        events_val = cPickle.load(f)
        f.close()
        in_path = root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/val_training_event_id.cPickle'
        f = open(in_path, 'r')
        events_training = cPickle.load(f)
        f.close()
        load_path = root+'all_output/all_images_curation.txt'
        save_paths1 = root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/guru_val_validation_path.txt'
        f1 = open(save_paths1, 'wb')
        save_paths2 = root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/guru_val_training_path.txt'
        f2 = open(save_paths2, 'wb')
        prefix = '/home/feiyu1990/local/event_curation/curation_images/'
        with open(load_path,'r') as data:
            for line in data:
                meta = line.split('\t')
                if meta[1]+'_'+meta[3] in events_val:
                    string = prefix + check_type + '/' + meta[3]+'/'+meta[2] + '.jpg 0\n'
                    f1.write(string)
                elif meta[1]+'_'+meta[3] in events_training:
                    string = prefix + check_type + '/' + meta[3]+'/'+meta[2] + '.jpg 0\n'
                    f2.write(string)
        f1.close()
        f2.close()
    @staticmethod
    def from_txt_to_imgid(check_type, type, val_id):
        image_ids = []
        in_path = root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/guru_'+type+'_path.txt'
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
        f = open(root + 'baseline_all_0509/' + check_type + '/'+'validation_' + str(val_id)+'/' + type+'_image_ids.cPickle','wb')
        cPickle.dump(event_ids, f)
        f.close()
    @staticmethod
    def read_amt_result(name, type = 'training'):
        input_path = root + 'baseline_all_0509/' + name + '/' + type + '_image_ids.cPickle'
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
        input_path = root + 'baseline_all_0509/' + name + '/' + type+'.csv'
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
            this_hit_new = this_hit
            #for submission in this_hit:
                #if submission[index_worker_id] not in block_workers:
                #    this_hit_new.append(submission)
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
        f = open(root + 'baseline_all_0509/' + name + '/' + type + '_result_v1.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
    @staticmethod
    def correct_amt_result(name, type = 'test'):
        f = open(root + 'baseline_all_0509/' + name + '/' + type + '_result_v1.cPickle','r')
        input_and_answers = cPickle.load(f)
        f.close()

        f = open(root + 'baseline_all_0509/'+name+'/vgg_'+type+'_similar_list.cPickle','r')
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

        f = open(root + 'baseline_all_0509/'+name+'/vgg_'+type+'_result_v2.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
        training_scores_dict = {}
        for event in input_and_answers:
            for img in input_and_answers[event]:
                training_scores_dict[img[1]] = img[2]
        f = open(root + 'baseline_all_0509/'+name+'/vgg_'+type+'_result_dict_v2.cPickle', 'wb')
        cPickle.dump(training_scores_dict,f)
        f.close()
    @staticmethod
    def create_knn_cPickle(name, model_name):
        path = root + 'baseline_all_0509/' + name + '/'+model_name+'_knn.txt'
        f = open(root + 'baseline_all_0509/'+name+'/training_ulr_dict.cPickle', 'r')
        training_url_dict = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/test_ulr_dict.cPickle', 'r')
        test_url_dict = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/test_image_ids.cPickle','r')
        test_images = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/training_image_ids.cPickle','r')
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
    @staticmethod
    def baseline_predict(name, model_name, n_vote = 10):
        f = open(root + 'baseline_all_0509/'+name+'/vgg_training_result_dict_v2.cPickle','r')
        training_scores_dict = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/'+model_name+'_knn.cPickle', 'r')
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

        f = open(root + 'baseline_all_0509/'+name+'/'+model_name+'_predict_result_'+str(n_vote)+'_dict.cPickle','wb')
        cPickle.dump(test_prediction_event, f)
        f.close()
    @staticmethod
    def create_csv(name, val_id):
        f = open(root + 'baseline_all_0509/'+name+'/'+'validation_' + str(val_id)+'/val_validation_event_id.cPickle','r')
        event_ids = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/'+'validation_' + str(val_id)+'/val_validation.csv','wb')
        writer = csv.writer(f)
        line_count = 0
        input_path = root + 'all_output/all_output_cleaned.csv'
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

        f = open(root + 'baseline_all_0509/'+name+'/'+'validation_' + str(val_id)+'/val_training_event_id.cPickle','r')
        event_ids = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/'+'validation_' + str(val_id)+'/val_training.csv','wb')
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
    @staticmethod
    def create_csv_traintest(name):
        f = open(root + 'baseline_all_0509/'+name+'/training_event_id.cPickle','r')
        event_ids = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/training.csv','wb')
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

        f = open(root + 'baseline_all_0509/'+name+'/test_event_id.cPickle','r')
        event_ids = cPickle.load(f)
        f.close()
        f = open(root + 'baseline_all_0509/'+name+'/test.csv','wb')
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
    @staticmethod
    def create_dict_url(check_type, val_id):
        path = root + 'baseline_all_0509/' + check_type + '/'+'validation_' + str(val_id)+'/val_validation_image_ids.cPickle'
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

        f = open(root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/val_validation_ulr_dict.cPickle', 'wb')
        cPickle.dump(dict, f)
        f.close()
        path = root + 'baseline_all_0509/' + check_type + '/'+'validation_' + str(val_id)+'/val_training_image_ids.cPickle'
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

        f = open(root + 'baseline_all_0509/'+check_type+'/'+'validation_' + str(val_id)+'/val_training_ulr_dict.cPickle', 'wb')
        cPickle.dump(dict, f)
        f.close()
    @staticmethod
    def to_guru_file(val_id, validation_name):
        if 'val' in validation_name:
            validation_path = '/'+'validation_' + str(val_id)
        else:
            validation_path = ''
        for event_name in dict_name2:
            if not os.path.exists(root + 'face_heatmap/data/'+event_name):
                os.mkdir(root + 'face_heatmap/data/'+event_name)
            if not os.path.exists(root + 'face_heatmap/data/'+event_name+ '/'+'validation_' + str(val_id)):
                os.mkdir(root + 'face_heatmap/data/'+event_name+ '/'+'validation_' + str(val_id))
            # if event_name == 'Wedding':
            #     continue
            in_path = root + 'baseline_all_0509/'+event_name+ validation_path+'/guru_'+validation_name+'_path.txt'
            out_path = root + 'face_heatmap/data/'+event_name+validation_path+'/'+validation_name+'_path.txt'
            f = open(out_path, 'w')
            with open(in_path,'r') as data:
                    for line in data:
                        img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_aesthetic_cropped/' + '/'.join(line.split('/')[-2:])
                        f.write(img_path_new)
            f.close()

        # event_name = 'Wedding'
        # in_path = root + 'baseline_all_0509/'+event_name+ validation_path+'/guru_'+validation_name+'_path.txt'
        # out_path = root + 'face_heatmap/data/'+event_name+ validation_path+'/'+validation_name+'_path.txt'
        # f = open(out_path, 'w')
        # with open(in_path,'r') as data:
        #             for line in data:
        #                 img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped/' + '/'.join(line.split('/')[-2:])
        #                 f.write(img_path_new)
        # f.close()
    @staticmethod
    def create_ground_truth(event_name, type, val_id):
        in_path = root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/'+type+'_event_id.cPickle'
        f = open(in_path,'r')
        test_events = cPickle.load(f)
        f.close()

        in_path = root + 'baseline_all_0509/' +event_name + '/vgg_training_result_v2.cPickle'
        f = open(in_path,'r')
        a = cPickle.load(f)
        f.close()
        dict_ = {}
        out_path = root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/vgg_'+type+'_result_v2.cPickle'
        for event_ in test_events:
            dict_[event_] = a[event_]
        f = open(out_path,'w')
        cPickle.dump(dict_, f)
        f.close()

class create_event_recognition(object):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.guru_find_valid_examples_all_reallabel_traintest_oversample()
    def guru_find_valid_examples_all_reallabel_traintest(self):
        training_img_lists = []
        test_img_lists = []
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name +'/training_image_ids.cPickle','r')
            training_img_id = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name +'/test_image_ids.cPickle','r')
            test_img_id = cPickle.load(f)
            f.close()

            for id in training_img_id:
                path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
                training_img_lists.append(path)
            for id in test_img_id:
                path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
                test_img_lists.append(path)

        random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list.txt'
        out_path2 = root + self.folder_name+'/data/test_list.txt'
        f1 = open(out_path1,'w')
        for i in training_img_lists:
            f1.write(i+'\n')
        f1.close()
        f1 = open(out_path2,'w')
        for i in test_img_lists:
            f1.write(i+'\n')
        f1.close()
    def guru_find_valid_examples_all_reallabel_traintest_highscore(self, important_threshold = 0.6):
        training_img_lists = []
        test_img_lists = []
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name +'/training_image_ids.cPickle','r')
            training_img_id = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_dict_v2.cPickle','r')
            training_score_dict = cPickle.load(f)
            f.close()

            # f = open(root + 'baseline_all_0509/' + event_name +'/test_image_ids.cPickle','r')
            # test_img_id = cPickle.load(f)
            # f.close()

            # f = open(root + 'baseline_all_0509/' + event_name + '/vgg_test_result_dict_v2.cPickle','r')
            # test_score_dict = cPickle.load(f)
            # f.close()

            for id in training_img_id:
                if training_score_dict[id] < important_threshold:
                    continue
                path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
                training_img_lists.append(path)
            # for id in test_img_id:
            #     if training_score_dict[id] < important_threshold:
            #         continue
            #     path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
            #     test_img_lists.append(path)

        random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list_0.6.txt'
        # out_path2 = root + self.folder_name+'/data/test_list.txt'
        f1 = open(out_path1,'w')
        for i in training_img_lists:
            f1.write(i+'\n')
        f1.close()
        # f1 = open(out_path2,'w')
        # for i in test_img_lists:
        #     f1.write(i+'\n')
        # f1.close()
    def guru_find_valid_examples_all_reallabel_traintest_oversample(self, oversample_event = ['PersonalMusicActivity','Protest','ReligiousActivity',
                                                                                              'CasualFamilyGather','GroupActivity','BusinessActivity',
                                                                                              'PersonalArtActivity','Architecture','NatureTrip','Cruise',
                                                                                              'Museum','BeachTrip','Sports']):
        training_img_lists = []
        test_img_lists = []
        for event_name in dict_name2:
            if event_name in oversample_event:
                sample_time = 5
            else:
                sample_time = 1
            f = open(root + 'baseline_all_0509/' + event_name +'/training_image_ids.cPickle','r')
            training_img_id = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_dict_v2.cPickle','r')
            training_score_dict = cPickle.load(f)
            f.close()

            for id in training_img_id:
                path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
                for i in xrange(sample_time):
                    training_img_lists.append(path)

        random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list_oversample.txt'
        f1 = open(out_path1,'w')
        for i in training_img_lists:
            f1.write(i+'\n')
        f1.close()

class create_CNN_training_prototxts(object):
    def __init__(self, threshold, val_name, folder_name, oversample_n = 3, oversample_threshold = 0.3, subsample = False, rate = 0.5, oversample= False):
        self.threshold = threshold
        self.val_name = val_name
        self.folder_name = folder_name
        self.oversample_n = oversample_n
        self.oversample_threshold = oversample_threshold
        if self.threshold == 0:
            self.threshold_prefix = ''
        else:
            self.threshold_prefix = str(self.threshold) + '_'
        if 'val' in val_name:
            for i in xrange(5):
               self.guru_find_valid_examples_all_reallabel(i)
               self.merge_all_examples(i)
               self.create_label_txt(i)
        else:
                if subsample:
                    self.guru_find_valid_examples_all_reallabel_traintest_subsample(rate = rate)
                else:
                    self.guru_find_valid_examples_all_reallabel_traintest(oversample)
                self.merge_all_examples_traintest()
                # self.merge_all_examples_traintest_subcategory()
                self.create_label_txt_traintest()
    def guru_find_valid_examples_all_reallabel(self, val_id):

        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()


            in_path = root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

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
                    if this_event[-1][2] - this_event[i][2] < self.threshold * 4:
                        break
                    img_1 = this_event[i]
                    for j in xrange(i + 1, len_):
                        img_2 = this_event[j]
                        if img_2[2] - img_1[2] < self.threshold * 4:
                            continue
                        temp = random.sample([0, 1], 1)
                        if temp[0] == 0:
                            img_pair.append((img_1[1], img_2[1], 0))
                        else:
                            img_pair.append((img_2[1], img_1[1], 1))
                        count += 1
                count_all += count
            random.shuffle(img_pair)
            if self.val_name == 'val_validation':
                img_pair = img_pair[:len(img_pair)/10]
            if not os.path.exists(root + self.folder_name):
                os.mkdir(root + self.folder_name)
            if not os.path.exists(root + self.folder_name +'validation_' + str(val_id)):
                os.mkdir(root + self.folder_name+'validation_' + str(val_id))
            if not os.path.exists(root + self.folder_name+'validation_' + str(val_id)+'/data'):
                os.mkdir(root + self.folder_name+'validation_' + str(val_id) + '/data/')

            out_path1 = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix + event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            print out_path1
            out_path2 = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
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
            print count_all
    def merge_all_examples(self, val_id):
        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append([meta + ' ' + event_name])
        count = 0
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs[count].append(meta + ' ' + event_name)
                    count += 1


        random.shuffle(imgs)
        f1 = open(root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix  + 'ranking_reallabel_'+self.val_name+'_all.txt', 'w')
        f2 = open(root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix  + 'ranking_reallabel_'+self.val_name+'_all_p.txt', 'w')
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
        f = h5py.File(root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()
    def create_label_txt(self, val_id):
        f = open(root + self.folder_name+'validation_' + str(val_id)+'/data/' + self.val_name[4:] + '_event_label.txt','w')
        f.write(root + self.folder_name+'validation_' + str(val_id)+'/data/' + self.val_name +'_label.h5')
        f.close()

    def guru_find_valid_examples_all_reallabel_traintest(self, oversample):
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name +'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()

            in_path = root + 'baseline_all_0509/' + event_name + '/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

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
                    if this_event[-1][2] - this_event[i][2] < self.threshold * 4:
                        break
                    img_1 = this_event[i]
                    for j in xrange(i + 1, len_):
                        img_2 = this_event[j]
                        if img_2[2] - img_1[2] < self.threshold * 4:
                            continue
                        temp = random.sample([0, 1], 1)
                        if temp[0] == 0:
                            img_pair.append((img_1[1], img_2[1], 0))
                        else:
                            img_pair.append((img_2[1], img_1[1], 1))
                        if oversample:
                            if img_2[2] - img_1[2] < self.oversample_threshold * 4:
                                continue
                            for iter in xrange(self.oversample_n - 1):
                                temp = random.sample([0, 1], 1)
                                if temp[0] == 0:
                                    img_pair.append((img_1[1], img_2[1], 0))
                                else:
                                    img_pair.append((img_2[1], img_1[1], 1))
                        count += 1
                count_all += count
            random.shuffle(img_pair)
            if self.val_name == 'val_validation':
                img_pair = img_pair[:len(img_pair)/10]
            if not os.path.exists(root + self.folder_name):
                os.mkdir(root + self.folder_name)
            if not os.path.exists(root + self.folder_name+'/data'):
                os.mkdir(root + self.folder_name+'/data')
            out_path1 = root + self.folder_name+'/data/'+event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            print out_path1
            out_path2 = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
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
            print count_all
    def guru_find_valid_examples_all_reallabel_traintest_subsample(self, rate = 0.5):
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name +'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            select_event = random.sample(range(len(ground_truth_training)), int(len(ground_truth_training)*rate))

            f = open(root + 'baseline_all_0509/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()

            in_path = root + 'baseline_all_0509/' + event_name + '/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

            img_path_dict = {}
            for (i,j) in zip(img_ids, img_paths):
                img_path_dict[i] = j

            img_pair = []
            count_all = 0
            event_count = -1
            for event in ground_truth_training:
                event_count += 1
                if event_count not in select_event:
                    continue
                this_event = ground_truth_training[event]
                this_event.sort(key=lambda x: x[2])
                count = 0
                len_ = len(this_event)
                for i in xrange(len_):
                    if this_event[-1][2] - this_event[i][2] < self.threshold * 4:
                        break
                    img_1 = this_event[i]
                    for j in xrange(i + 1, len_):
                        img_2 = this_event[j]
                        if img_2[2] - img_1[2] < self.threshold * 4:
                            continue
                        temp = random.sample([0, 1], 1)
                        if temp[0] == 0:
                            img_pair.append((img_1[1], img_2[1], 0))
                        else:
                            img_pair.append((img_2[1], img_1[1], 1))
                        count += 1
                count_all += count
            random.shuffle(img_pair)
            if self.val_name == 'val_validation':
                img_pair = img_pair[:len(img_pair)/10]
            if not os.path.exists(root + self.folder_name):
                os.mkdir(root + self.folder_name)
            if not os.path.exists(root + self.folder_name+'/data'):
                os.mkdir(root + self.folder_name+'/data')
            out_path1 = root + self.folder_name+'/data/'+event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            print out_path1
            out_path2 = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
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
            print count_all
    def merge_all_examples_traintest(self):
        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append([meta + ' ' + event_name])
        count = 0
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs[count].append(meta + ' ' + event_name)
                    count += 1


        random.shuffle(imgs)
        if self.val_name == 'test':
            imgs = imgs[:5000]
        f1 = open(root + self.folder_name+'/data/'+ 'ranking_reallabel_'+self.val_name+'_all.txt', 'w')
        f2 = open(root + self.folder_name+'/data/'+'ranking_reallabel_'+self.val_name+'_all_p.txt', 'w')
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
        f = h5py.File(root + self.folder_name+'/data/' +self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()
    def create_label_txt_traintest(self):
        f = open(root + self.folder_name+'/data/' + self.val_name + '_event_label.txt','w')
        f.write(root + self.folder_name+'/data/' + self.val_name +'_label.h5')
        f.close()

    def merge_all_examples_traintest_subcategory(self):
        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append([meta + ' ' + event_name])
        count = 0
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs[count].append(meta + ' ' + event_name)
                    count += 1


        random.shuffle(imgs)
        f1 = open(root + self.folder_name+'/data/'+ 'ranking_reallabel_'+self.val_name+'_all.txt', 'w')
        f2 = open(root + self.folder_name+'/data/'+'ranking_reallabel_'+self.val_name+'_all_p.txt', 'w')
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
                event_label = dict_subcategory2[dict_name2[event_label]-1]
                temp = np.zeros((6,))
                temp[event_label] = 1
                event_labels.append(temp)
        event_labels = np.array(event_labels)
        f = h5py.File(root + self.folder_name+'/data/' +self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()

class create_CNN_training_prototxts_face(object):
    def __init__(self, threshold, val_name, folder_name):
        self.threshold = threshold
        self.val_name = val_name
        self.folder_name = folder_name
        if self.threshold == 0:
            self.threshold_prefix = ''
        else:
            self.threshold_prefix = str(self.threshold) + '_'
        if 'val' in val_name:
            for i in xrange(5):
               self.guru_find_valid_examples_all_reallabel(i)
               self.merge_all_examples(i)
               self.create_label_txt(i)
        else:
            self.guru_find_valid_examples_all_reallabel_traintest()
            self.merge_all_examples_traintest()
    def guru_find_valid_examples_all_reallabel(self, val_id):

        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()


            in_path = root + 'baseline_all_0509/' + event_name + '/'+'validation_' + str(val_id)+'/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = '/home/feiyu1990/local/event_curation/person_heatmap_images_aesthetic_cropped/'+'/'.join(line.split(' ')[0].split('/')[-2:])
                    img_paths.append(temp)

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
                    if this_event[-1][2] - this_event[i][2] < self.threshold * 4:
                        break
                    img_1 = this_event[i]
                    for j in xrange(i + 1, len_):
                        img_2 = this_event[j]
                        if img_2[2] - img_1[2] < self.threshold * 4:
                            continue
                        temp = random.sample([0, 1], 1)
                        if temp[0] == 0:
                            img_pair.append((img_1[1], img_2[1], 0))
                        else:
                            img_pair.append((img_2[1], img_1[1], 1))
                        count += 1
                count_all += count
            random.shuffle(img_pair)
            if self.val_name == 'val_validation':
                img_pair = img_pair[:len(img_pair)/10]
            if not os.path.exists(root + self.folder_name):
                os.mkdir(root + self.folder_name)
            if not os.path.exists(root + self.folder_name +'validation_' + str(val_id)):
                os.mkdir(root + self.folder_name+'validation_' + str(val_id))
            if not os.path.exists(root + self.folder_name+'validation_' + str(val_id)+'/data'):
                os.mkdir(root + self.folder_name+'validation_' + str(val_id) + '/data/')

            out_path1 = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix + event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            print out_path1
            out_path2 = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
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
            print count_all
    def merge_all_examples(self, val_id):
        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append([meta + ' ' + event_name])
        count = 0
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs[count].append(meta + ' ' + event_name)
                    count += 1


        random.shuffle(imgs)
        f1 = open(root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix  + 'ranking_reallabel_'+self.val_name+'_all.txt', 'w')
        f2 = open(root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix  + 'ranking_reallabel_'+self.val_name+'_all_p.txt', 'w')
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
        f = h5py.File(root + self.folder_name+'validation_' + str(val_id)+'/data/'+self.threshold_prefix +self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()
    def create_label_txt(self, val_id):
        f = open(root + self.folder_name+'validation_' + str(val_id)+'/data/' + self.val_name[4:] + '_label.txt','w')
        f.write(root + self.folder_name+'validation_' + str(val_id)+'/data/' +self.threshold_prefix +self.val_name+'_label.h5')
        f.close()

    def guru_find_valid_examples_all_reallabel_traintest(self):
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name +'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()

            in_path = root + 'baseline_all_0509/' + event_name + '/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = '/home/feiyu1990/local/event_curation/person_heatmap_images_aesthetic_cropped/'+'/'.join(line.split(' ')[0].split('/')[-2:])
                    img_paths.append(temp)

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
                    if this_event[-1][2] - this_event[i][2] < self.threshold * 4:
                        break
                    img_1 = this_event[i]
                    for j in xrange(i + 1, len_):
                        img_2 = this_event[j]
                        if img_2[2] - img_1[2] < self.threshold * 4:
                            continue
                        temp = random.sample([0, 1], 1)
                        if temp[0] == 0:
                            img_pair.append((img_1[1], img_2[1], 0))
                        else:
                            img_pair.append((img_2[1], img_1[1], 1))
                        count += 1
                count_all += count
            random.shuffle(img_pair)
            if self.val_name == 'val_validation':
                img_pair = img_pair[:len(img_pair)/10]
            if not os.path.exists(root + self.folder_name):
                os.mkdir(root + self.folder_name)
            if not os.path.exists(root + self.folder_name+'/data'):
                os.mkdir(root + self.folder_name+'/data')
            out_path1 = root + self.folder_name+'/data/'+event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            print out_path1
            out_path2 = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
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
            print count_all
    def merge_all_examples_traintest(self):
        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append([meta + ' ' + event_name])
        count = 0
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs[count].append(meta + ' ' + event_name)
                    count += 1


        random.shuffle(imgs)
        f1 = open(root + self.folder_name+'/data/'+ 'ranking_reallabel_'+self.val_name+'_all.txt', 'w')
        f2 = open(root + self.folder_name+'/data/'+'ranking_reallabel_'+self.val_name+'_all_p.txt', 'w')
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
        f = h5py.File(root + self.folder_name+'/data/' +self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()
        
class create_CNN_training_prototxts_euclidean(object):
    def __init__(self, val_name, folder_name):
        self.val_name = val_name
        self.folder_name = folder_name
        self.guru_find_valid_examples_all_reallabel()
        self.merge_all_examples()
        self.create_label_txt()
    def guru_find_valid_examples_all_reallabel(self):

        for event_name in dict_name2:
            # f = open(root + 'baseline_all_0509/' + event_name + '/vgg_'+self.val_name+'_result_v2.cPickle','r')
            # ground_truth_training = cPickle.load(f)
            # f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()
            print ground_truth_training_dict

            in_path = root + 'baseline_all_0509/' + event_name +'/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

            img_path_dict = []
            for (i,j) in zip(img_ids, img_paths):
                img_path_dict.append([i,j])
            random.shuffle(img_path_dict)
            # print img_path_dict

            if not os.path.exists(root + self.folder_name):
                os.mkdir(root + self.folder_name)
            if not os.path.exists(root + self.folder_name+'/data'):
                os.mkdir(root + self.folder_name+'/data/')

            out_path1 = root + self.folder_name+'/data/'+ event_name+'_reallabel_'+self.val_name+'.txt'
            print out_path1
            f1 = open(out_path1,'w')
            for i in img_path_dict:
                line = i[1] + ' ' + str(int(20*(ground_truth_training_dict[i[0]]))) + '\n'
                f1.write(line)
            f1.close()
    def merge_all_examples(self):
        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/'+event_name+'_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append(meta + ' ' + event_name)
        random.shuffle(imgs)

        f1 = open(root + self.folder_name+'/data/'+ 'reallabel_'+self.val_name+'_all.txt', 'w')
        for i in imgs:
            f1.write(' '.join(i.split(' ')[:-1]) + '\n')
        f1.close()

        events = []
        for i in imgs:
            events.append(i.split(' ')[-1])
        event_labels = []
        for event_label in events:
                event_label = dict_name2[event_label]
                temp = np.zeros((23,))
                temp[event_label-1] = 1
                event_labels.append(temp)
        event_labels = np.array(event_labels)
        f = h5py.File(root + self.folder_name+'/data/'+self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()
    def create_label_txt(self):
        f = open(root + self.folder_name+'/data/' + self.val_name + '_event_label.txt','w')
        f.write(root + self.folder_name+'/data/' + self.val_name +'_label.h5')
        f.close()

class extract_features:
    def __init__(self, net_path, event_name, net_name, model_name, name, blob_names, img_size = (256,256)):
        self.event_name = event_name
        self.net_name = net_name
        self.model_name = model_name
        self.name = name
        self.img_size = img_size
        self.blob_names = blob_names
        self.net_path = net_path
    def extract_feature_10(self, val_id):
        if 'val' in self.name:
            in_validation_path = '/validation_'+ str(val_id)
        else:
            in_validation_path = '/'
        out_validation_path = '/validation_' + str(val_id)

        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+in_validation_path+ '/guru_'+self.name+'_path.txt'
        #if name == 'test':
        #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/

        # if os.path.exists('/home/feiyu1990/local/event_curation/'+self.net_path + '/' +out_validation_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle'):
        #     return
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/validation_'+ str(val_id)+ '/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            if out.shape[1] == 23:
                features.append(out[0, dict_name2[self.event_name]-1])
            else:
                features.append(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        #for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path + '/' +out_validation_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_10_23_traintest(self):
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        #if name == 'test':
        #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        out = []
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            # if out.shape[1] == 23:
            #     features.append(out[0, dict_name2[self.event_name]-1])
            # else:
            features.append(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        #for i in xrange(len(blob_names)):
        len_event_type = len(out[0])
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_'+str(len_event_type)+'_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_10_superevent_traintest(self):
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        #if name == 'test':
        #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        out = []
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            # if out.shape[1] == 23:
            features.append(out[0, dict_subcategory2[dict_name2[self.event_name]-1]])
            # else:
            #    features.append(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        #for i in xrange(len(blob_names)):
        # len_event_type = len(out[0])
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_super_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_10_traintest(self):
        # if os.path.exists('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle'):
        #     print '/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle already exists!'
        #     return
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        #if name == 'test':
        #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            # print out.shape
            if out.shape[1] == 23:
                features.append(out[0, dict_name2[self.event_name]-1])

            else:
                features.append(out[0])
            # print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
            if count % 100 == 0:
                print event_name, out
        #for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_face(self, val_id):
        if 'val' in self.name:
            in_validation_path = '/validation_'+ str(val_id)
        else:
            in_validation_path = '/'
        out_validation_path = '/validation_' + str(val_id)

        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/'+self.net_path+'/data/'+self.event_name+in_validation_path + '/' +self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+in_validation_path+'/snapshot/' + self.net_name + '.caffemodel'
        caffe.set_mode_gpu()
        net = caffe.Net(model_name,
                    weight_name,
                    caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 2)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        net.blobs['data'].reshape(1,3,self.img_size, self.img_size)
        features = [[] for i in self.blob_names]
        count = 0
        for img in imgs:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            print np.max(net.blobs['data'].data)
            out = net.forward()
            for i in xrange(len(self.blob_names)):
                a = net.blobs[self.blob_names[i]].data.copy()
                features[i].append(a)
                print img, count
            count += 1
        for i in xrange(len(self.blob_names)):
            f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/'+out_validation_path+'/features/'+self.event_name + '_' +self.name+'_'+self.blob_names[i]+ self.net_name + '.cPickle','wb')
            cPickle.dump(features[i], f)
            f.close()
    def extract_feature_face_traintest(self):
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/'+self.net_path+'/data/'+self.event_name + '/' +self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+'/snapshot/' + self.net_name + '.caffemodel'
        caffe.set_mode_gpu()
        net = caffe.Net(model_name,
                    weight_name,
                    caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 2)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        net.blobs['data'].reshape(1,3,self.img_size, self.img_size)
        features = [[] for i in self.blob_names]
        count = 0
        for img in imgs:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            print np.max(net.blobs['data'].data)
            out = net.forward()
            for i in xrange(len(self.blob_names)):
                a = net.blobs[self.blob_names[i]].data.copy()
                features[i].append(a)
                print img, count
            count += 1
        for i in xrange(len(self.blob_names)):
            f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_'+self.blob_names[i]+ self.net_name + '.cPickle','wb')
            cPickle.dump(features[i], f)
            f.close()
    def extract_feature_10_face(self, val_id):
        if 'val' in self.name:
            in_validation_path = '/validation_'+ str(val_id)
        else:
            in_validation_path = '/'
        out_validation_path = '/validation_' + str(val_id)

        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/'+self.net_path+'/data/'+self.event_name+in_validation_path + '/' +self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+in_validation_path+'/snapshot/' + self.net_name + '.caffemodel'
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (55,55))
            caffe_in = input_2
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            if out.shape[1] == 23:
                features.append(out[0, dict_name2[self.event_name]-1])
            else:
                features.append(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/'+out_validation_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_10_traintest_vote(self):
        imgs = defaultdict(list)
        in_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/test_image_ids.cPickle'
        f = open(in_file,'r')
        img_ids = cPickle.load(f)
        f.close()
        
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line, img_id in zip(data, img_ids):
                if line.split('/')[-1].split('.')[0] != img_id.split('/')[1]:
                    print 'WRONG!'
                    return
                event_id = img_id.split('/')[0]
                imgs[event_id].append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = {}
        for event_id in imgs:
            img_event = imgs[event_id]
            len_img = len(img_event)
            print len_img, img_event
            id_indicator = []
            count = 0
            features_this = []
            for i in xrange(len_img):
                id_indicator.append(count)
                img_1 = img_event[i]
                temp = caffe.io.load_image(img_1)
                input_1 = caffe.io.resize_image(temp, (256,256))
                caffe_in_1 = input_1 - mean_file[(2,1,0)]/255
                # print caffe_in_1
                for j in xrange(i + 1, len_img):
                    count += 1
                    img_2 = img_event[j]
                    temp = caffe.io.load_image(img_2)
                    input_2 = caffe.io.resize_image(temp, (256,256))
                    caffe_in_2 = input_2 - mean_file[(2,1,0)]/255
                    inputs = [caffe_in_1, caffe_in_2]
                    # print inputs
                    out = net.predict_vote(inputs)#,oversample = False)
                    # print out
                    # print out[0, dict_name2[self.event_name]-1]
                    features_this.append(out[0, dict_name2[self.event_name]-1])
            for i in xrange(len_img):
                temp = []
                for j in xrange(i):
                    temp.append(-features_this[id_indicator[j] + i - j - 1])
                if i != len_img - 1:
                    for j in xrange(id_indicator[i], id_indicator[i+1]):
                        temp.append(features_this[j])
                print len_img, len(temp)
                img_this = img_event[i]
                features[img_this] = (np.sum(temp))
        features_new = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                features_new.append(features[line.split(' ')[0]])
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'perevent_sigmoid9_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features_new, f)
        f.close()
    def extract_feature_10_traintest_vote_new(self):
        imgs = defaultdict(list)
        in_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/test_image_ids.cPickle'
        f = open(in_file,'r')
        img_ids = cPickle.load(f)
        f.close()

        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line, img_id in zip(data, img_ids):
                if line.split('/')[-1].split('.')[0] != img_id.split('/')[1]:
                    print 'WRONG!'
                    return
                event_id = img_id.split('/')[0]
                imgs[event_id].append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = {}
        for event_id in imgs:
            img_event = imgs[event_id]
            len_img = len(img_event)
            print len_img, img_event
            id_indicator = []
            count = 0
            features_this = []
            for i in xrange(len_img):
                id_indicator.append(count)
                img_1 = img_event[i]
                temp = caffe.io.load_image(img_1)
                input_1 = caffe.io.resize_image(temp, (256,256))
                caffe_in_1 = input_1 - mean_file[(2,1,0)]/255
                # print caffe_in_1
                for j in xrange(i + 1, len_img):
                    count += 1
                    img_2 = img_event[j]
                    temp = caffe.io.load_image(img_2)
                    input_2 = caffe.io.resize_image(temp, (256,256))
                    caffe_in_2 = input_2 - mean_file[(2,1,0)]/255
                    inputs = [caffe_in_1, caffe_in_2]
                    # print inputs
                    out_1 = net.predict_vote(inputs)#,oversample = False)
                    out_2 = net.predict_vote([caffe_in_2, caffe_in_1])#,oversample = False)
                    out = (out_1 - out_2) / 2
                    # print out
                    # print out[0, dict_name2[self.event_name]-1]
                    features_this.append(out[0, dict_name2[self.event_name]-1])
            for i in xrange(len_img):
                temp = []
                for j in xrange(i):
                    temp.append(-features_this[id_indicator[j] + i - j - 1])
                if i != len_img - 1:
                    for j in xrange(id_indicator[i], id_indicator[i+1]):
                        temp.append(features_this[j])
                print len_img, len(temp)
                img_this = img_event[i]
                features[img_this] = (np.sum(temp))
        features_new = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                features_new.append(features[line.split(' ')[0]])
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'perevent_sigmoid9_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features_new, f)
        f.close()
    def extract_feature_10_traintest_vote_sample(self, sample = 10):
        imgs = defaultdict(list)
        in_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/test_image_ids.cPickle'
        f = open(in_file,'r')
        img_ids = cPickle.load(f)
        f.close()

        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line, img_id in zip(data, img_ids):
                if line.split('/')[-1].split('.')[0] != img_id.split('/')[1]:
                    print 'WRONG!'
                    return
                event_id = img_id.split('/')[0]
                imgs[event_id].append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = {}
        for event_id in imgs:
            img_event = imgs[event_id]
            len_img = len(img_event)
            print len_img, img_event
            id_indicator = []
            count = 0
            # features_this = []
            for i in xrange(len_img):
                id_indicator.append(count)
                img_1 = img_event[i]
                temp = caffe.io.load_image(img_1)
                input_1 = caffe.io.resize_image(temp, (256,256))
                caffe_in_1 = input_1 - mean_file[(2,1,0)]/255
                # print caffe_in_1

                selected_idx = random.sample(range(len_img),sample)
                feature_this = []
                for j in selected_idx:
                    img_2 = img_event[j]
                    temp = caffe.io.load_image(img_2)
                    input_2 = caffe.io.resize_image(temp, (256,256))
                    caffe_in_2 = input_2 - mean_file[(2,1,0)]/255
                    inputs = [caffe_in_1, caffe_in_2]
                    # print inputs
                    out = net.predict_vote(inputs)
                    feature_this.append(out[0, dict_name2[self.event_name]-1])
                features[img_1] = np.mean(feature_this)

        features_new = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                features_new.append(features[line.split(' ')[0]])
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_'+str(sample)+'_perevent_sigmoid9_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features_new, f)
        f.close()

    def create_prototxt_traintest_vote(self):
        event_labels = []
        out_file_1 = '/home/feiyu1990/local/event_curation/baseline_all_0509/guru_vote_'+self.name+'_path.txt'
        out_file_2 = '/home/feiyu1990/local/event_curation/baseline_all_0509/guru_vote_'+self.name+'_path_p.txt'
        f1 = open(out_file_1, 'w')
        f2 = open(out_file_2, 'w')
        for event_name in dict_name2:
            event_label = dict_name2[event_name]
            temp = np.zeros((23,))
            temp[event_label-1] = 1
            imgs = defaultdict(list)
            in_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name+'/guru_'+self.name+'_path.txt'
            with open(in_file, 'r') as data:
                for line in data:
                    event_id = line.split('/')[-2]
                    imgs[event_id].append(line)
            for event_id in imgs:
                img_event = imgs[event_id]
                len_img = len(img_event)
                for i in xrange(len_img):
                    for j in xrange(len_img):
                        if i == j:
                            continue
                        event_labels.append(temp)
                        f1.write(img_event[i])
                        f2.write(img_event[j])
        f1.close()
        f2.close()
        print len(event_labels)

        event_labels = np.array(event_labels)
        f = h5py.File('/home/feiyu1990/local/event_curation/baseline_all_0509/guru_vote_'+self.name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()
    def extract_feature_10_face_traintest(self):
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/'+self.net_path+'/data/'+self.event_name + '/' +self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/'+'/snapshot/' + self.net_name + '.caffemodel'
        caffe.set_mode_gpu()

        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (55,55))
            caffe_in = input_2
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            if out.shape[1] == 23:
                features.append(out[0, dict_name2[self.event_name]-1])
            else:
                features.append(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid_10_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_10_recognition_traintest(self):
        # if os.path.exists('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle'):
        #     print '/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle already exists!'
        #     return
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        #if name == 'test':
        #    img_file = '/mnt/ilcompf2d0/project/yuwang/event_curation/
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/training/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/model/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            # print out.shape
            # if out.shape[1] == 23:
            #     features.append(out[0, dict_name2[self.event_name]-1])
                # print out
            # else:
            features.append(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        #for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_predict_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()

    def combine_event_feature_traintest(self, event_net_path, event_prediction_name, event_net_name):

        f = open('/home/feiyu1990/local/event_curation/'+ event_net_path+'/features/'+self.event_name + '_' + event_prediction_name+'_predict_'+ event_net_name + '.cPickle','r')
        event_recognition_feature = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + self.event_name+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()

        pred_comb_dict = defaultdict(list)
        for predict_score, img_id in zip(event_recognition_feature, all_event_ids):
            event_id = img_id.split('/')[0]
            pred_comb_dict[event_id].append([img_id, predict_score])
        prediction_album = dict()
        for album_name in pred_comb_dict:
            prediction_thisevent = np.zeros((23,))
            for i in pred_comb_dict[album_name]:
                prediction_ = np.array(i[1])
                prediction_thisevent += prediction_
            prediction_album[album_name] = prediction_thisevent

        print prediction_album

        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/snapshot/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (256,256))
            caffe_in = input_2 - mean_file[(2,1,0)]/255
            inputs = [caffe_in]
            out = net.predict(inputs)
            features.append(out[0])
            count += 1
            if count % 100 == 0:
                print self.event_name, out

        combined_importance_all = []
        for img_name, importance_prediction in zip(all_event_ids, features):
            event_id = img_name.split('/')[0]
            new_importance = np.sum(prediction_album[event_id] * importance_prediction)
            combined_importance_all.append(new_importance)
        print combined_importance_all
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+self.event_name + '_23_sigmoid9_10_segment_twoloss_fc300_iter_100000.cPickle','wb')
        cPickle.dump(features, f)
        f.close()

def extract_feature_aesthetic(event_name):
    img_list = root + 'baseline_all_0509/'+event_name+'/test_image_ids.cPickle'
    f = open(img_list, 'r')
    imgs = cPickle.load(f)
    f.close()
    for i in xrange(len(imgs)):
        temp = imgs[i]
        imgs[i] = temp.split('_')[1]

    img_dict = {}
    for i in imgs:
        img_dict[i] = np.Inf


    score_list = root + 'aesthetic/aesthetic_scores.txt'
    with open(score_list,'r') as data:
        for line in data:
            line = line[:-1]
            img_id = line.split('.')[0]
            # print img_id
            if img_id in img_dict:
                score = float(line.split(' ')[1])
                # print score
                img_dict[img_id] = score
    # print img_dict
    img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+event_name+'/guru_test_path.txt'
    feature = []
    with open(img_file, 'r') as data:
        for line in data:
            img_id = '/'.join(line.split('/')[-2:])
            img_id = img_id.split('.')[0]
            # print img_id
            feature.append(img_dict[img_id])

    f = open( root + 'aesthetic/'+event_name + '_aesthetic_score.cPickle','wb')
    cPickle.dump(feature, f)
    f.close()

class evaluation:
    def __init__(self, net_path, type, validation_name, val_id, face_type = None):
        # f = open(root + 'baseline_all_0509/test_event_abandoned_noILE.pkl','r')
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
        #self.evaluate_models = evaluate_models

        if type == 'worker_nonoverlap':
            self.evaluate_present_with_worker_nooverlap()
        if type == 'worker':
            self.evaluate_present_with_worker_nooverlap(abandon_overlap=False)
        if type == 'face':
            self.create_and_evaluate_face()
        if type == 'combine':
            self.evaluate_present_combine(self.validation_name)
        if type == 'evaluate_face':
            self.evaluate_face(face_type)
    def create_predict_dict_from_cpickle_multevent(self, validation_name, event_name, mat_name, event_index, multi_event = True):
        if 'val' in validation_name:
            validation_path = '/validation_'+ str(self.val_id) + '/' + validation_name
        else:
            validation_path = '/' + validation_name
        path = root+'baseline_all_0509/' + event_name+validation_path +'_image_ids.cPickle'
        #print path
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()
        f = open(mat_name + '.cPickle', 'r')
        predict_score = cPickle.load(f)
        f.close()

        f = open(root + 'baseline_all_0509/' + event_name+validation_path +'_ulr_dict.cPickle', 'r')
        test_url_dict = cPickle.load(f)
        f.close()

        prediction_dict = {}
        for (name_, score) in zip(all_event_ids, predict_score):
            event_name = name_.split('/')[0]
            if event_name in prediction_dict:
                if multi_event:
                    prediction_dict[event_name] += [[name_, test_url_dict[name_], score[event_index-1]]]
                else:
                    prediction_dict[event_name] += [[name_, test_url_dict[name_], score]]
            else:
                if multi_event:
                    prediction_dict[event_name] = [[name_, test_url_dict[name_], score[event_index-1]]]
                else:
                    prediction_dict[event_name] = [[name_, test_url_dict[name_], score]]


        f = open(mat_name + '_dict.cPickle','wb')
        cPickle.dump(prediction_dict, f)
        f.close()

    def permute_groundtruth(self, event_name):
        if 'val' in self.validation_name:
            validation_path = '/validation_'+ str(self.val_id)
        else:
            validation_path = '/'
        f = open(root + baseline_name + event_name+ validation_path +'/vgg_'+self.validation_name+'_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        ground_truth_new = {}
        for idx in ground_truth:
            temp = []
            for i in ground_truth[idx]:
                temp.append((i[0], i[1], i[2]+random.uniform(-0.02, 0.02)))
            ground_truth_new[idx] = temp
        f = open(root + baseline_name + event_name+ validation_path+'/vgg_'+self.validation_name+'_result_v2_permuted.cPickle','wb')
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

    def evaluate_rank_reweighted_permuted(self, event_name, model_names, permuted = '_permuted',abandon_overlap = False):
        if 'val' in self.validation_name:
            f = open(root + baseline_name + event_name+ self.validation_path + '/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
        else:
            f = open(root + baseline_name + event_name+ '/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
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
                if abandon_overlap and event_id in self.abandoned_test:
                    # print event_id
                    continue
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

    def evaluate_MAP_permuted(self, event_name, model_names, min_retrieval = 5, permuted = '_permuted',abandon_overlap = False):
        if TIE == False:
            maps = []
            n_ks = []
            if 'val' in self.validation_name:
                f = open(root + baseline_name + event_name+ self.validation_path + '/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
            else:
                f = open(root + baseline_name + event_name+ '/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
            ground_truth = cPickle.load(f)
            f.close()
            for model_name in model_names:
                APs = []
                f = open(root + model_name, 'r')
                predict_result = cPickle.load(f)
                f.close()
                for event_id in ground_truth:
                    # print event_id
                    # print self.abandoned_test
                    if abandon_overlap and event_id in self.abandoned_test:
                        # print event_id
                        continue

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
                    AP = self.average_precision(ground_rank, predict_rank, n_k)
            
                    APs.append([event_id, AP])
                    n_ks.append([n_k, len(ground_)])
                maps.append(sum([i[1] for i in APs])/len(APs))
                percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
            return maps, percent, [i[1] for i in APs]
        else:
            maps = []
            n_ks = []
            if 'val' in self.validation_name:
                f = open(root + baseline_name + event_name+ self.validation_path + '/vgg_'+self.validation_name+'_result_v2.cPickle','r')
            else:
                f = open(root + baseline_name + event_name+ '/vgg_'+self.validation_name+'_result_v2.cPickle','r')
            ground_truth = cPickle.load(f)
            f.close()
            for model_name in model_names:
                APs = []
                f = open(root + model_name, 'r')
                predict_result = cPickle.load(f)
                f.close()
                for event_id in ground_truth:
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
                maps.append(sum([i[1] for i in APs])/len(APs))
                percent = (float(sum([i[0] for i in n_ks]))/sum(i[1] for i in n_ks))
            return maps, percent, [i[1] for i in APs]

    def evaluate_top20_permuted(self, event_name, model_names, percent = 20, permuted = '_permuted',abandon_overlap = False):
        retval = []
        if 'val' in self.validation_name:
            f = open(root + baseline_name + event_name+ self.validation_path + '/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
        else:
            f = open(root + baseline_name + event_name+ '/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        for model_name in model_names:
            f = open(root +  model_name, 'r')
            predict_result = cPickle.load(f)
            f.close()
            count_all = 0; n_k_all = 0
            for event_id in ground_truth:
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
        if 'val' in self.validation_name:
            validation_path = self.validation_path
        else:
            validation_path = '/'


        f = open(root + baseline_name + event_name+ validation_path +'/vgg_'+self.validation_name+'_result_v2'+permuted+'.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()

        input_path = root + baseline_name + event_name+ validation_path + '/' + self.validation_name + '_image_ids.cPickle'
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
        input_path = root + baseline_name + event_name+ validation_path + '/' +self.validation_name+'.csv'
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

    def baseline_evaluation(self, event_name, permute = True,
                            model_names = [], evaluate_worker = True, worker_times = global_permutation_time, abandon_overlap = False):
            model_names_this = [event_name.join(i) for i in model_names]
            if permute:
                self.permute_groundtruth(event_name)
            retrievals = [[] for i in model_names_this]; percent = []; precision = [[] for i in model_names_this]
            reweighted = self.evaluate_rank_reweighted_permuted(event_name, model_names_this, permuted='_permuted', abandon_overlap = abandon_overlap)
            for i in xrange(6, 45, 5):
                temp, percent_temp, AP_temp = self.evaluate_MAP_permuted(event_name, model_names_this, min_retrieval=i, abandon_overlap = abandon_overlap)
                for j in xrange(len(temp)):
                    retrievals[j].append(temp[j])
                percent.append(percent_temp)
            for i in xrange(6, 45, 5):
                # temp = self.evaluate_top5_image_permuted(event_name, model_names_this, img_n=i, permuted='_permuted')
                temp = self.evaluate_top20_permuted(event_name, model_names_this, percent=i, permuted='_permuted', abandon_overlap = abandon_overlap)
                for j in xrange(len(temp)):
                    precision[j].append(temp[j])
            all_aps=[];all_reweighted=[];all_ps=[]
            if not evaluate_worker:
                return reweighted, percent, retrievals , precision, [], []


            for i in xrange(worker_times):
                all_nks, temp2, temp3, temp4 = self.amt_worker_result_predict_average(event_name, permuted='_permuted', abandon_overlap = abandon_overlap)
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


    def create_random_dict(self, validation_name, event_name, save_name):
        path = root+baseline_name + event_name+ '/'+validation_name+'_image_ids.cPickle'
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()
        f = open(root + baseline_name + event_name+ '/'+validation_name+'_ulr_dict.cPickle', 'r')
        test_url_dict = cPickle.load(f)
        f.close()
        score_random = [random.uniform(0,1) for i in xrange(len(all_event_ids))]
        f = open(root + baseline_name + event_name+ '/' +validation_name+ '_'+save_name.split('.')[0][:-5]+'.cPickle','wb')
        cPickle.dump(score_random, f)
        f.close()

        prediction_dict = {}
        for score, name_ in zip(score_random, all_event_ids):
            event_name_ = name_.split('/')[0]
            if event_name_ in prediction_dict:
                prediction_dict[event_name_] += [[name_, test_url_dict[name_], score]]
            else:
                prediction_dict[event_name_] = [[name_, test_url_dict[name_], score]]

        f = open(root + baseline_name + event_name+ '/'+validation_name+ '_' + save_name + '.cPickle','wb')
        cPickle.dump(prediction_dict, f)
        f.close()

    def evaluate_present_with_worker_nooverlap(self, abandon_overlap = True):
        print 'HERE!!!'
        # model_names = [['CNN_all_event/'+'/validation_'+str(self.val_id)+'/features/']]*len(self.evaluate_models)
        # for i in xrange(len(model_names)):
        #    model_names[i].append(self.evaluate_models[i])
        model_names = [
                        ['baseline_all_noblock/', '/'+self.validation_name+'_random_dict.cPickle']
# #                         # # ,['aesthetic/', '_aesthetic_score_dict.cPickle']
# #                         # ,[baseline_name, '/alexnet6k_predict_result_10_dict.cPickle']
# #                         # # ,['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_euclidean_iter_70000_dict.cPickle']
# #                         # # # ,['CNN_all_event_old/features/', '_test_sigmoid9_10_segment_iter_20000_dict.cPickle']
# #                         # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000_dict.cPickle']
# #                         # # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000_dict.cPickle']
# #                         # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_nosigmoid_iter_30000_dict.cPickle']
# #                         # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_sigmoid_iter_10000_dict.cPickle']
#                         ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_0.1_2time_iter_100000_dict.cPickle']
#                         ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_svm_iter_100000_dict.cPickle']
#                         ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_0.1_iter_100000_dict.cPickle']
#                         ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_svm_0.2_iter_100000_dict.cPickle']
#                         ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_svm_0.5_iter_100000_dict.cPickle']
# # #                         # # ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_svm_pretrain_iter_100000_dict.cPickle']

                        ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_30000_dict.cPickle']
                        ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_adagrad_iter_30000_dict.cPickle']

                        # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_noevent_8w_iter_200000_dict.cPickle']
                        ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_100000_dict.cPickle']
# # #                         ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_subsample0.8_iter_100000_dict.cPickle']
# # #                         ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_subsample0.5_iter_100000_dict.cPickle']
# #                         ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_100000_dict.cPickle']
# #
#                         ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_2time_iter_100000_dict.cPickle']
#                         ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_2time_iter_200000_dict.cPickle']
#                         ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_5time_iter_120000_dict.cPickle']
#                         ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_5time_iter_170000_dict.cPickle']
                        ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_5time_iter_200000_dict.cPickle']
#                         ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_5time_iter_230000_dict.cPickle']
#                         ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_5time_iter_260000_dict.cPickle']
#                         ,['CNN_all_event_old/features/', '_test_sigmoid9_10_segment_iter_70000_dict.cPickle']
#                         ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_fast_iter_10000_dict.cPickle']



                        # ,['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_oversample_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_2_7w_dict.cPickle']

                        #
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']


                       # #CD ..
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_noevent_iter_100000_dict.cPickle']
                       #  # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_iter_100000_dict.cPickle']
                       #  # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_0.1_iter_50000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_0.1_iter_100000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_0.01_iter_100000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_0.5_iter_100000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_iter_100000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_00501_0.4_iter_100000_dict.cPickle']
                       #  # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.1_iter_100000_dict.cPickle']
                       #  # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.4_iter_30000_dict.cPickle']
                       #  # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.5_iter_40000_dict.cPickle']
                       #  # # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.3_iter_40000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.3_iter_100000_dict.cPickle']
                       #  # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.5_iter_40000_dict.cPickle']
                       #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.5_iter_100000_dict.cPickle']
                       # #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.5_iter_100000_dict.cPickle']
                       # #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.3_iter_100000_dict.cPickle']
                       # #  ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.4_iter_100000_dict.cPickle']
                       # # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_00501_0.5_iter_100000_dict.cPickle']


                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_joint_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_joint_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_3_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_2_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_300000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_400000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_400000_new_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_10_perevent_sigmoid9_10_segment_joint_fc_iter_200000_dict.cPickle']
                        #
                        #
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_step10w_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_step10w_iter_250000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_8q_iter_100000_dict.cPickle']
                        # #
                        # # # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_10000_dict.cPickle']
                        # # # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_100000_dict.cPickle']
                        # # # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_130000_dict.cPickle']
                        # # # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_200000_dict.cPickle']
                        # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_80000_dict.cPickle']
                        # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_100000_dict.cPickle']
                        # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_120000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_150000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/',  '_sigmoid9_10_multi_event_dict.cPickle']
                        # ,['CNN_all_event_1009/features/',  '_sigmoid9_10_multi_event_multiply_dict.cPickle']
                        ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_differentsize_iter_110000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc100_8w_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc100_8w_iter_150000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_120000_dict.cPickle']
                        # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_130000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc200_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_0.5_iter_70000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc50_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc100_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc200_iter_70000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc200_iter_90000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc200_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_400_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_100000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc1000_iter_160000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc1000_15w_iter_170000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc1000_15w_iter_300000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc100_8w_iter_120000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc100_8w_iter_180000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc100_0.05_iter_100000_dict.cPickle']

                       # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc500_diffmargin_iter_50000_dict.cPickle']



                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_100000_dict.cPickle']
                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_100000_dict.cPickle']
                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_00501_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_00501_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_00501_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_00501_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_00501_iter_100000_dict.cPickle']

                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.3_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_step_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_step_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_poly_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_poly_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_cont_iter_2000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_cont_iter_26000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_cont_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_0.7_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_cont_step_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_new_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_new_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_new_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_new_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_cont_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_cont_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_cont_iter_30000_dict.cPickle']




                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_poly_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.1_iter_48000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_100000_dict.cPickle']


                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_100000_dict.cPickle']


                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_cont_iter_2000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_iter_90000_dict.cPickle']
                       ]

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
        for event_name in dict_name2:
            model_names_this = [event_name.join(ii) for ii in model_names]

            for model_name_this in model_names_this:
                try:
                    self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=True)
                except:
                    try:
                        self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=False)
                    except:
                        print 'Skipping creation of dict:', model_name_this

            percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
            for i in xrange(permutation_times):
                    # if i %10 == 0:
                    #     print i
                    # self.create_random_dict(self.validation_name, event_name, 'random_dict')
                    reweighted, percent, retrievals , precision, mean_aps, mean_ps = self.baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times, abandon_overlap=abandon_overlap)
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

            f = open(root + event_name.join(model_names[0]),'r')
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
                precision_models[i].append([j*len_ for j in precision_model_average[i]])
            print '*WORKER*'
            print ', '.join(["%.3f" % v for v in retrievals_worker_average])
            print ', '.join(["%.3f" % v for v in precision_worker_average])
            print '\n'
            retrieval_worker_all.append([i*len_ for i in retrievals_worker_average])
            precision_worker_all.append([i*len_ for i in precision_worker_average])
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


        print 'Worker'
        #print retrieval_worker_all
        temp = np.array(precision_worker_all)
        temp1 = np.sum(temp, axis=0)
        print [i/len_all for i in temp1]

        temp = np.array(retrieval_worker_all)
        temp1 = np.sum(temp, axis=0)
        print [i/len_all for i in temp1]

    def evaluate_present_with_worker(self):
        print 'HERE!!!'
        # model_names = [['CNN_all_event/'+'/validation_'+str(self.val_id)+'/features/']]*len(self.evaluate_models)
        # for i in xrange(len(model_names)):
        #    model_names[i].append(self.evaluate_models[i])
        model_names = [
                        ['baseline_all_noblock/', '/'+self.validation_name+'_random_dict.cPickle']
        #                 # ,['baseline_all_0509/', '/'+self.validation_name+'_kmeans_prediction_5_dict.cPickle']
        #                 # ,['baseline_all_0509/', '/'+self.validation_name+'_kmeans_prediction_10_dict.cPickle']
        #                 # ,['baseline_all_0509/', '/'+self.validation_name+'_kmeans_prediction_20percent_dict.cPickle']
        # #                 ,['aesthetic/', '_aesthetic_score_dict.cPickle']
        #                 ,[baseline_name, '/alexnet6k_predict_result_10_dict.cPickle']
        # #                 #
        # #                 ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_svm_iter_100000_dict.cPickle']
        # #                 ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_0.1_iter_100000_dict.cPickle']
        #                 ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_0.1_2time_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_svm_pretrain_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_euclidean_iter_20000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_euclidean_iter_70000_dict.cPickle']
        #                 #
        #                 # ,['CNN_all_event_0.1/features/', '_test_sigmoid9_10_0.3_2time_iter_100000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/', '_test_sigmoid9_10_segment_iter_70000_dict.cPickle']
                        ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_2_7w_dict.cPickle']


                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_110000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_131000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_210000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_230000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_230000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_250000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_327000_dict.cPickle']
                        # #
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_43000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_105000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_150000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_210000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_250000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_335000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_polylarge_iter_400000_dict.cPickle']
                        #
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_step_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_step_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_step_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_step_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_step_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_step_iter_100000_dict.cPickle']
                        # # #
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_327000_dict.cPickle']
                        #
                        #

                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_cont_iter_2000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_100000_dict.cPickle']

                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_70000_dict.cPickle']


                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_100000_dict.cPickle']
                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.3_iter_80000_dict.cPickle']
                        # # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_100000_dict.cPickle']
                        # # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_cont_iter_2000_dict.cPickle']
                        # #             # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_90000_dict.cPickle']
                        # # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']
                        # # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.1_iter_48000_dict.cPickle']

                        #
                        # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_noevent_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_0.1_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.3_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.5_iter_40000_dict.cPickle']
                        # # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_iter_50000_dict.cPickle']
                        # # ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_iter_50000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_noevent_iter_100000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_iter_100000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_svm_0.1_iter_100000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_iter_100000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.1_iter_100000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.4_iter_30000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_fromnoevent_0.5_iter_40000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.3_iter_40000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.3_iter_100000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.5_iter_40000_dict.cPickle']
                        ,['CNN_all_event_vgg/features/', '_' + self.validation_name + '_sigmoid9_10_VGG_segment_0.5_iter_100000_dict.cPickle']



                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_50000_224_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_largelr_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_test_sigmoid9_10_segment_googlenet_quick_1_iter_327000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']


                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_poly_iter_30000_dict.cPickle']

                        #
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.3_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.3_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_70000_dict.cPickle']

                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.3_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_50000_224_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_0.1_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_svm_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_l1_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_l1_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_l1_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_largelr_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_largelr_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_largelr_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_largelr_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.5_iter_80000_dict.cPickle']

                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.2_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_0.3_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_69000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_segment_googlenet_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_50000_dict.cPickle']
                        # # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_google_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_5w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.1_iter_48000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_step_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_step_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_cont_step_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_00501_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_l1_00501_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_l1_00501_iter_30000_dict.cPickle']

                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_google/features/', '_' + self.validation_name + '_sigmoid9_10_fromnoevent_googlenet_0.5_iter_80000_dict.cPickle']
                        #

                        # #

                        # ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_2round_iter_750000_dict.cPickle']
                        # ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_2time_iter_10000_dict.cPickle']
        #                 ['CNN_all_event_old/features/', '_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_30000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/', '_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_50000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/', '_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_70000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_2time_iter_100000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_2time_iter_200000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_100000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_180000_dict.cPickle']
        #                 ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_differentsize_iter_130000_dict.cPickle']



        #                 ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_poly_iter_40000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_poly_iter_100000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_largepoly_iter_100000_dict.cPickle']


                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_poly_iter_100000_dict.cPickle']

        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_fast_iter_10000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_iter_20000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_iter_40000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_iter_50000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_iter_80000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_2_iter_20000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_2_iter_50000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_2_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_frombegin_iter_30000_dict.cPickle']
        #                 # ['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_super_segment_superevent_frombegin_iter_100000_dict.cPickle']
        #
        #                 ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_nosigmoid_iter_30000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_euclidean_sigmoid_iter_10000_dict.cPickle']       #                 #
        #                 ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name+'_sigmoid9_10_segment_noevent_iter_100000_dict.cPickle']

                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_10w_gamma0.1_iter_120000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_10w_gamma0.1_iter_160000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_10w_gamma0.1_iter_180000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_new_10w_gamma0.1_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_joint_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_joint_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_3_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_2_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_300000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_400000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_iter_400000_new_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_10_perevent_sigmoid9_10_segment_joint_fc_iter_200000_dict.cPickle']


                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_step10w_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_step10w_iter_250000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_8q_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_130000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_120000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_150000_dict.cPickle']





                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + 'perevent_sigmoid9_10_segment_joint_fc_hidden1q_8q_iter_200000_dict.cPickle']

                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_iter_10000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_iter_30000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_iter_50000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_iter_70000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_2w_iter_70000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_fromsuper_2w_iter_100000_dict.cPickle']
        #                 ['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_20000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_30000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_40000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_50000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_70000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_100000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_90000_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_iter_130000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_poly_10w_iter_100000_dict.cPickle']

                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_10000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_20000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_30000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_40000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_60000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_70000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_80000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_90000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_smallpoly_10w_iter_100000_dict.cPickle']


                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_50000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_150000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_200000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromsuper_3_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromsuper_frombegin_iter_90000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_nosigmoid_iter_100000_dict.cPickle']
        #
        #                 #
        #                 ['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_test_combined_10_new_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_test_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_2_7w_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_2_10w_dict.cPickle']
        #                 # ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_2_7w10w_dict.cPickle']
        #                 ,['CNN_all_event_1009/features/', '_test_face_combined_thetapre_10w_dict.cPickle']
        #                 #
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_from_validation_noevent_iter_100000_dict.cPickle']
        #                 # ['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_finetune_iter_60000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_80000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_70000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000_dict.cPickle']
        #                 # ,['CNN_all_event_1009/validation_'+str(self.val_id)+'/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_iter_50000_dict.cPickle']
                       ]



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
        for event_name in dict_name2:
            model_names_this = [event_name.join(ii) for ii in model_names]

            for model_name_this in model_names_this:
                try:
                    self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=True)
                except:
                    try:
                        self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=False)
                    except:
                        print 'Skipping creation of dict:', model_name_this

            percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
            for i in xrange(permutation_times):
                    # if i %10 == 0:
                    #     print i
                    # self.create_random_dict(self.validation_name, event_name, 'random_dict')
                    reweighted, percent, retrievals , precision, mean_aps, mean_ps = self.baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times)
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

            f = open(root + event_name.join(model_names[0]),'r')
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
                precision_models[i].append([j*len_ for j in precision_model_average[i]])
            print '*WORKER*'
            print ', '.join(["%.3f" % v for v in retrievals_worker_average])
            print ', '.join(["%.3f" % v for v in precision_worker_average])
            print '\n'
            retrieval_worker_all.append([i*len_ for i in retrievals_worker_average])
            precision_worker_all.append([i*len_ for i in precision_worker_average])
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


        print 'Worker'
        #print retrieval_worker_all
        temp = np.array(precision_worker_all)
        temp1 = np.sum(temp, axis=0)
        print [i/len_all for i in temp1]

        temp = np.array(retrieval_worker_all)
        temp1 = np.sum(temp, axis=0)
        print [i/len_all for i in temp1]

    def grid_search_face(self, event_name, permute, validation_name, times = global_permutation_time):
        alphas = np.arange(0,1.5,0.05)
        all_results = {}
        for j in xrange(times):
            if permute:
                self.permute_groundtruth(event_name)
            for alpha in alphas:
                        retrieval = []
                        #create_face(event_name, validation_name,'_'+validation_name + '_sigmoidcropped_importance_allevent_iter_100000.cPickle', '_'+validation_name+'_sigmoid9segment_iter_70000.cPickle', validation_name + '_combine_face', alpha, beta)
                        self.create_face('slope','/validation_'+str(self.val_id) + '/', event_name,  '_val_validation' + face_model, '_'+validation_name+combine_face_model, validation_name + '_combine_face', 0, 0, alpha, '_val_training' + face_model)
                        #create_face(event_name, validation,'_test_sigmoidcropped_importance_decay_groundtruth_iter_100000.cPickle', 'test_combine_face', alpha, beta)
                        for i in xrange(6, 45, 5):
                            temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path + '/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                            retrieval.append(temp[0])
                        if alpha in all_results:
                            all_results[alpha].append((retrieval, sum(retrieval)))
                        else:
                            all_results[alpha] = [(retrieval, sum(retrieval))]
        for alpha in all_results:
            all_results[alpha] = np.sum([i[1] for i in all_results[alpha]])
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print sorted_all_result
        #if sorted_all_result[0][0] == 0:
        #    return 0
        return sorted_all_result
    def grid_search_face_3(self, event_name, use_theta, permute, validation_name, times = global_permutation_time, theta_pre = 1.0):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        if use_theta:
            theta = theta_pre
            # thetas = np.arange(0,1.05,0.05)
            # all_results = {}
            # for j in xrange(times):
            #     if permute:
            #         self.permute_groundtruth(event_name)
            #     for theta in thetas:
            #                 retrieval = []
            #                 self.create_face('slope','/validation_'+str(self.val_id) + '/', event_name,  '_'+validation_name + face_model, '_'+validation_name+combine_face_model, validation_name + '_combine_face', 0, 0, theta, '_val_training' + face_model)
            #                 for i in xrange(6, 45, 5):
            #                     temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
            #                     retrieval.append(temp[0])
            #                 if theta in all_results:
            #                     all_results[theta].append((retrieval, sum(retrieval)))
            #                 else:
            #                     all_results[theta] = [(retrieval, sum(retrieval))]
            # for alpha in all_results:
            #     all_results[alpha] = np.sum([i[1] for i in all_results[alpha]])
            # sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
            # print sorted_all_result
            # if sorted_all_result[0][0] == 0:
            #     return (0, 0, 0)
            # theta = sorted_all_result[0][0]
        else:
            theta = 1.0
        alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1.05,0.05)
        all_results = {}
        all_results_std = {}

        if theta == 0.0:
            retval = [((0.0, 0.0, 0.0), 0.0)]
            return retval
        try:
            self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root + self.net_path+'/validation_'+str(self.val_id)+'/features/'+event_name+'_'+validation_name+combine_face_model[:-8], dict_name2[event_name], multi_event=True)
        except:
            self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root+ self.net_path+'/validation_'+str(self.val_id)+'/features/'+event_name+'_'+validation_name+combine_face_model[:-8], dict_name2[event_name], multi_event=False)


        for j in xrange(times):
            ap_original = []
            if permute:
                self.permute_groundtruth(event_name)
            retrieval = []
            for i in xrange(6, 45, 5):
                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+event_name+'_'+validation_name+combine_face_model[:-8]+'_dict.cPickle'], min_retrieval= i)
                retrieval.append(temp[0])
                ap_original.append(APs)
            if (0, 0, 0) not in all_results:
                all_results[(0, 0, 0)] = sum(retrieval)
            else:
                all_results[(0, 0, 0)] += sum(retrieval)
            for alpha in alphas:
                for beta in betas:
                    if beta <= 0.2+alpha:
                        continue
                    retrieval = []
                    self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                    std_ = 0
                    for i in xrange(6, 45, 5):
                            temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                            std_ += np.std(APs)
                            retrieval.append(temp[0])
                    if (alpha, beta, theta) not in all_results:
                            all_results[(alpha, beta, theta)] = sum(retrieval)
                            all_results_std[(alpha, beta, theta)] = std_
                    else:
                            all_results[(alpha, beta, theta)] += sum(retrieval)
                            all_results_std[(alpha, beta, theta)] += std_

        baseline = all_results[(0,0,0)] / times
        sorted_all_result_std = sorted(all_results_std.items(), key=operator.itemgetter(1), reverse=True)
        # abandoned_ = []
        # for i in xrange(30):
        #     abandoned_.append((sorted_all_result_std[i][0], all_results[sorted_all_result_std[i][0]]))
        #     all_results.pop(sorted_all_result_std[i][0], None)
            #print sorted_all_result_std[i][0]
        #print 'ABANDONED:', abandoned_

        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        '''
        print sorted_all_result
        if sorted_all_result[0][1] / times - baseline > 0.01:
            print sorted_all_result[0][0]
            return sorted_all_result[0][0]
        else:
            print (0, 0, 0)
            return (0, 0, 0)
        '''
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            # if sorted_all_result[i][1] / times - baseline < 0.01:
            #     break
            retval.append(sorted_all_result[i])
        print retval
        return retval
        # if sorted_all_result[0][1] / times - baseline < 0.01:
        #     return [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        #
        # temp_ = [(sorted_all_result[0][0],sorted_all_result[0][0][1] - sorted_all_result[0][0][0]) ]
        # for i in xrange(1, len(sorted_all_result)):
        #     if abs(sorted_all_result[i-1][1] - sorted_all_result[i][1]) < 10**(-5):
        #         temp_.append((sorted_all_result[i][0], sorted_all_result[i][0][1] - sorted_all_result[i][0][0]))
        #     else:
        #         break
        # sorted_temp = sorted(temp_, key=operator.itemgetter(1))
        # if len(sorted_temp) >= 3:
        #     #print sorted_temp
        #     return [i[0] for i in sorted_temp]
        # else:
        #     temp = [i[0] for i in sorted_temp]
        #     if len(sorted_temp) == 1:
        #         if sorted_all_result[1][1] / times - baseline < 0.01:
        #             temp.extend([(0, 0, 0), (0, 0, 0)])
        #         elif sorted_all_result[2][1] / times - baseline < 0.01:
        #             temp.extend([sorted_all_result[1][0], (0, 0, 0), (0, 0, 0)])
        #         else:
        #             temp.extend([sorted_all_result[1][0], sorted_all_result[2][0]])
        #         return temp
        #     else:
        #         if sorted_all_result[2][1] / times - baseline < 0.01:
        #             temp.extend([(1,1,1)])
        #         else:
        #             temp.extend([sorted_all_result[2][0]])
        #         return temp
    def grid_search_face_2(self, event_name, use_theta, permute, validation_name, times = global_permutation_time, theta_pre = 1.0):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        if use_theta:
            theta = theta_pre
        else:
            theta = 1.0
        alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for alpha in alphas:
                    beta = 1.0
                    retrieval = []
                    self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                    std_ = 0
                    for i in xrange(6, 45, 5):
                            temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                            std_ += np.std(APs)
                            retrieval.append(temp[0])
                    if (alpha, beta, theta) not in all_results:
                            all_results[(alpha, beta, theta)] = sum(retrieval)
                            # all_results_std[(alpha, beta, theta)] = std_
                    else:
                            all_results[(alpha, beta, theta)] += sum(retrieval)
                            # all_results_std[(alpha, beta, theta)] += std_

            for beta in betas:
                    alpha = 0.0
                    retrieval = []
                    self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                    std_ = 0
                    for i in xrange(6, 45, 5):
                            temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                            std_ += np.std(APs)
                            retrieval.append(temp[0])
                    if (alpha, beta, theta) not in all_results:
                            all_results[(alpha, beta, theta)] = sum(retrieval)
                    else:
                            all_results[(alpha, beta, theta)] += sum(retrieval)

        baseline = all_results[(0,0,theta)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval
    def grid_search_face_35(self, event_name, permute, validation_name, times = global_permutation_time):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        thetas = np.arange(0,1.05,0.05)
        alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for theta in thetas:
                for alpha in alphas:
                        beta = 1.0
                        retrieval = []
                        self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                                # all_results_std[(alpha, beta, theta)] = std_
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)
                                # all_results_std[(alpha, beta, theta)] += std_

                for beta in betas:
                        alpha = 0.0
                        retrieval = []
                        self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)

        baseline = all_results[(0,0,0)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval
    def grid_search_face_25(self, event_name, permute, validation_name, times = global_permutation_time):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        thetas = np.arange(0,1.05,0.05)
        # alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for theta in thetas:
                for beta in betas:
                        alpha = 0.0
                        retrieval = []
                        self.create_face('power','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)

        baseline = all_results[(0,0,0)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval
    def grid_search_face_15(self, event_name, permute, validation_name, times = global_permutation_time):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        thetas = np.arange(0,1.5,0.05)
        # alphas = np.arange(0,1.05,0.05)
        # betas = np.arange(0,1,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for theta in thetas:
                        beta = 1.0
                        alpha = 0.0
                        retrieval = []
                        self.create_face('power','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)

        baseline = all_results[(0,1,0)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval
    def grid_search_face_betatheta(self, event_name, permute, validation_name, times = global_permutation_time):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        thetas = np.arange(0,1.05,0.05)
        # alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for theta in thetas:
                for beta in betas:
                        alpha = 0.0
                        retrieval = []
                        self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)

        baseline = all_results[(0,0,0)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval
    def grid_search_face_thetaprepower(self, event_name, use_theta, permute, validation_name, times = global_permutation_time, theta_pre = 1.0):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        if use_theta:
            theta = theta_pre
            # thetas = np.arange(0,1.05,0.05)
            # all_results = {}
            # for j in xrange(times):
            #     if permute:
            #         self.permute_groundtruth(event_name)
            #     for theta in thetas:
            #                 retrieval = []
            #                 self.create_face('slope','/validation_'+str(self.val_id) + '/', event_name,  '_'+validation_name + face_model, '_'+validation_name+combine_face_model, validation_name + '_combine_face', 0, 0, theta, '_val_training' + face_model)
            #                 for i in xrange(6, 45, 5):
            #                     temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
            #                     retrieval.append(temp[0])
            #                 if theta in all_results:
            #                     all_results[theta].append((retrieval, sum(retrieval)))
            #                 else:
            #                     all_results[theta] = [(retrieval, sum(retrieval))]
            # for alpha in all_results:
            #     all_results[alpha] = np.sum([i[1] for i in all_results[alpha]])
            # sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
            # print sorted_all_result
            # if sorted_all_result[0][0] == 0:
            #     return (0, 0, 0)
            # theta = sorted_all_result[0][0]
        else:
            theta = 1.0
        alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1.05,0.05)
        all_results = {}
        all_results_std = {}

        if theta == 0.0:
            retval = [((0.0, 0.0, 0.0), 0.0)]
            return retval
        try:
            self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root + self.net_path+'/validation_'+str(self.val_id)+'/features/'+event_name+'_'+validation_name+combine_face_model[:-8], dict_name2[event_name], multi_event=True)
        except:
            self.create_predict_dict_from_cpickle_multevent(self.validation_name,event_name, root+ self.net_path+'/validation_'+str(self.val_id)+'/features/'+event_name+'_'+validation_name+combine_face_model[:-8], dict_name2[event_name], multi_event=False)


        for j in xrange(times):
            ap_original = []
            if permute:
                self.permute_groundtruth(event_name)
            retrieval = []
            for i in xrange(6, 45, 5):
                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+event_name+'_'+validation_name+combine_face_model[:-8]+'_dict.cPickle'], min_retrieval= i)
                retrieval.append(temp[0])
                ap_original.append(APs)
            if (0, 0, 0) not in all_results:
                all_results[(0, 0, 0)] = sum(retrieval)
            else:
                all_results[(0, 0, 0)] += sum(retrieval)
            for alpha in alphas:
                for beta in betas:
                    if beta <= alpha:
                        continue
                    retrieval = []
                    self.create_face('power','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                    std_ = 0
                    for i in xrange(6, 45, 5):
                            temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                            std_ += np.std(APs)
                            retrieval.append(temp[0])
                    if (alpha, beta, theta) not in all_results:
                            all_results[(alpha, beta, theta)] = sum(retrieval)
                            all_results_std[(alpha, beta, theta)] = std_
                    else:
                            all_results[(alpha, beta, theta)] += sum(retrieval)
                            all_results_std[(alpha, beta, theta)] += std_

        baseline = all_results[(0,0,0)] / times
        sorted_all_result_std = sorted(all_results_std.items(), key=operator.itemgetter(1), reverse=True)
        # abandoned_ = []
        # for i in xrange(30):
        #     abandoned_.append((sorted_all_result_std[i][0], all_results[sorted_all_result_std[i][0]]))
        #     all_results.pop(sorted_all_result_std[i][0], None)
            #print sorted_all_result_std[i][0]
        #print 'ABANDONED:', abandoned_

        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        '''
        print sorted_all_result
        if sorted_all_result[0][1] / times - baseline > 0.01:
            print sorted_all_result[0][0]
            return sorted_all_result[0][0]
        else:
            print (0, 0, 0)
            return (0, 0, 0)
        '''
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            # if sorted_all_result[i][1] / times - baseline < 0.01:
            #     break
            retval.append(sorted_all_result[i])
        print retval
        return retval
        # if sorted_all_result[0][1] / times - baseline < 0.01:
        #     return [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        #
        # temp_ = [(sorted_all_result[0][0],sorted_all_result[0][0][1] - sorted_all_result[0][0][0]) ]
        # for i in xrange(1, len(sorted_all_result)):
        #     if abs(sorted_all_result[i-1][1] - sorted_all_result[i][1]) < 10**(-5):
        #         temp_.append((sorted_all_result[i][0], sorted_all_result[i][0][1] - sorted_all_result[i][0][0]))
        #     else:
        #         break
        # sorted_temp = sorted(temp_, key=operator.itemgetter(1))
        # if len(sorted_temp) >= 3:
        #     #print sorted_temp
        #     return [i[0] for i in sorted_temp]
        # else:
        #     temp = [i[0] for i in sorted_temp]
        #     if len(sorted_temp) == 1:
        #         if sorted_all_result[1][1] / times - baseline < 0.01:
        #             temp.extend([(0, 0, 0), (0, 0, 0)])
        #         elif sorted_all_result[2][1] / times - baseline < 0.01:
        #             temp.extend([sorted_all_result[1][0], (0, 0, 0), (0, 0, 0)])
        #         else:
        #             temp.extend([sorted_all_result[1][0], sorted_all_result[2][0]])
        #         return temp
        #     else:
        #         if sorted_all_result[2][1] / times - baseline < 0.01:
        #             temp.extend([(1,1,1)])
        #         else:
        #             temp.extend([sorted_all_result[2][0]])
        #         return temp
    def grid_search_face_beta_05_theta(self, event_name, permute, validation_name, times = global_permutation_time):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        thetas = np.arange(0,1.05,0.05)
        # alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0.5,1.05,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for theta in thetas:
                for beta in betas:
                        alpha = beta - 0.5
                        retrieval = []
                        self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)
                # beta = 0.0
                # alpha = 0.0
                # retrieval = []
                # self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                # std_ = 0
                # for i in xrange(6, 45, 5):
                #                 temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                #                 std_ += np.std(APs)
                #                 retrieval.append(temp[0])
                # if (alpha, beta, theta) not in all_results:
                #                 all_results[(alpha, beta, theta)] = sum(retrieval)
                # else:
                #                 all_results[(alpha, beta, theta)] += sum(retrieval)
        baseline = all_results[(0,0.5,0)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval
    def grid_search_face_full(self, event_name, permute, validation_name, times = global_permutation_time):
        print self.net_path+'/validation_'+str(self.val_id) + '/'+'/features/' + event_name+ '_'+validation_name+combine_face_model
        print root + 'face_heatmap' + '/validation_'+str(self.val_id) + '/features/'+event_name+ '_'+validation_name + face_model
        thetas = np.arange(0,1.05,0.05)
        alphas = np.arange(0,1.05,0.05)
        betas = np.arange(0,1.05,0.05)
        all_results = {}
        # all_results_std = {}

        for j in xrange(times):
            self.permute_groundtruth(event_name)
            for theta in thetas:
                for alpha in alphas:
                    for beta in betas:
                        if beta <= alpha + 0.2:
                            continue
                        retrieval = []
                        self.create_face('else','/validation_'+str(self.val_id) + '/', event_name, '_'+validation_name + face_model,  '_'+validation_name+combine_face_model, validation_name + '_combine_face', alpha, beta, theta, '_val_training' + face_model)
                        std_ = 0
                        for i in xrange(6, 45, 5):
                                temp, percent_temp, APs = self.evaluate_MAP_permuted(event_name, [self.net_path+'/validation_'+str(self.val_id)+'/features/'+validation_name+'_combine_face'+'_dict.cPickle'], min_retrieval= i)
                                std_ += np.std(APs)
                                retrieval.append(temp[0])
                        if (alpha, beta, theta) not in all_results:
                                all_results[(alpha, beta, theta)] = sum(retrieval)
                                # all_results_std[(alpha, beta, theta)] = std_
                        else:
                                all_results[(alpha, beta, theta)] += sum(retrieval)
                                # all_results_std[(alpha, beta, theta)] += std_

        baseline = all_results[(0,1,0)] / times
        sorted_all_result = sorted(all_results.items(), key=operator.itemgetter(1), reverse=True)
        print event_name, sorted_all_result[0][1] / times - baseline
        #print sorted_all_result
        retval = []
        for i in xrange(len(sorted_all_result)):
            retval.append(sorted_all_result[i])
        print retval
        return retval


    def create_face(self, type,validation_path, event_name, face_model_name, original_model_name, name, alpha, beta, theta, training_model_name):
        f = open(root + 'face_heatmap' + validation_path + '/features/'+event_name+ face_model_name,'r')
        # print 'face_model: ', root + 'face_heatmap' + validation_path + '/features/'+event_name+ face_model_name
        face = cPickle.load(f)
        f.close()
        try:
            face = [i[0][dict_name2[event_name]-1] for i in face]
        except:
            pass

        f = open(root + self.net_path + validation_path+ '/features/'+event_name+original_model_name,'r')
        feature = cPickle.load(f)
        # print 'original model: ',root + self.net_path + validation_path+ '/features/'+event_name+original_model_name
        f.close()
        try:
            feature = [i[0][dict_name2[event_name]-1] for i in feature]
        except:
            pass

        # f = open(root + 'face_heatmap/' + validation_path + '/features/'+event_name+training_model_name,'r')
        # face_training = cPickle.load(f)
        # f.close()
        # print 'training_facemodel: ', root + 'face_heatmap/' + validation_path + '/features/'+event_name+training_model_name
        # try:
        #     face_training = [i[0][dict_name2[event_name]-1] for i in face_training]
        # except:
        #     pass
        # min_ = np.min(face_training)
        # max_ = np.max(face_training)
        # alpha = alpha *(max_-min_) + min_
        # beta = beta *(max_-min_) + min_

        if type == 'cut_off':
            feature_new = [min(beta, max(alpha, float(j)))*float(i) for (i,j) in zip(feature, face)]
        elif type == 'slope':
            feature_new = [(float(j)*theta)+float(i) for (i,j) in zip(feature, face)]
        elif type == 'power':
            feature_new = [(min(beta, max(alpha, float(j)))**theta)*float(i) for (i,j) in zip(feature, face)]
        else:
            feature_new = [min(beta, max(alpha, float(j)))*theta+float(i) for (i,j) in zip(feature, face)]
            # print feature
            # print face
            # print feature_new
        f = open(root + self.net_path+validation_path+'/features/' + name + '.cPickle','w')
        # print root + self.net_path+validation_path+'/features/' + name + '.cPickle'
        cPickle.dump(feature_new,f)
        f.close()
        self.create_predict_dict_from_cpickle_multevent(self.validation_name, event_name, root + self.net_path+validation_path+'/features/'+name, dict_name2[event_name], multi_event=False)
    def evaluate_present_face(self, dict_from_validation2, type, combine_name):

        model_names = [
                       [self.net_path+self.validation_path+'/features/', '_test'+combine_face_model[:-8]+'_dict.cPickle']
                       ,[self.net_path+self.validation_path+'/features/', '_test_face_combined_'+combine_name+'_dict.cPickle']
                       ]
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

        for event_name in dict_from_validation2:
            this_param = dict_from_validation2[event_name]
            #this_param_2 = dict_from_validation1[event_name]
            print event_name
            print this_param
            #print this_param_2
            if type == 'alpha':
                self.create_face('else', self.validation_path, event_name, '_test' + face_model, '_test'+combine_face_model, event_name + '_test_face_combined_'+combine_name, this_param[0][0], this_param[0][1], this_param[0][2], '_training' + face_model)
            else:
                self.create_face('slope', self.validation_path, event_name, '_test' + face_model, '_test'+combine_face_model, event_name + '_test_face_combined_'+combine_name, 0, 0, this_param[0], '_test' + face_model)
            #self.create_face('cut_off', event_name, '_val_validation' + face_model, '_val_validation'+combine_face_model, event_name + '_val_validation_face_combined', this_param_2[0], this_param_2[1], this_param_2[2], '_val_training' + face_model)

            # f = open(root + 'CNN_all_event_old/features/'+event_name+'_val_test_face_combined.cPickle','r')
            # a_temp = cPickle.load(f)
            # f.close()
            # f = open(root + 'CNN_all_event_old/features/'+event_name+'_val_test_face_combined_2.cPickle','r')
            # b_temp = cPickle.load(f)
            # f.close()
            # f = open(root + 'CNN_all_event_old/features/'+event_name+'_val_test_face_combined_3.cPickle','w')
            # cPickle.dump([i*j for i, j in zip(a_temp, b_temp)],f)
            # f.close()

            f = open(root + baseline_name + event_name + '/vgg_test_result_v2.cPickle','r')
            temp = cPickle.load(f)
            f.close()

            len_ = len(temp)
            #print temp.keys()
            len_all += len_
            model_names_this = [event_name.join(ii) for ii in model_names]
            for model_name_this in model_names_this:
                try:
                    self.create_predict_dict_from_cpickle_multevent('test', event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=True)
                except:
                    try:
                        self.create_predict_dict_from_cpickle_multevent('test', event_name, root + ('_').join(model_name_this.split('_')[:-1]), dict_name2[event_name], multi_event=False)
                    except:
                        print 'Skipping creation of dict:', model_name_this


            percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
            for i in xrange(permutation_times):
                    if i %10 == 0:
                        print i
                    reweighted, percent, retrievals , precision, mean_aps, mean_ps = self.baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times)
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
                print ', '.join(["%.4f" % v for v in precision_model_average[i]])
                print ', '.join(["%.4f" % v for v in retrievals_model_average[i]])
                retrieval_models[i].append([j*len_ for j in retrievals_model_average[i]])
                precision_models[i].append([j*len_ for j in precision_model_average[i]])
            print '*WORKER*'
            print ', '.join(["%.4f" % v for v in retrievals_worker_average])
            print ', '.join(["%.4f" % v for v in precision_worker_average])
            print '\n'
            retrieval_worker_all.append([i*len_ for i in retrievals_worker_average])
            precision_worker_all.append([i*len_ for i in precision_worker_average])
        print '*********************************'
        print '*********************************'
        for i in xrange(len(retrieval_models)):
            print model_names[i]
            #print retrieval_models[i]
            temp = np.array(retrieval_models[i])
            #print temp
            #print len_all
            temp1 = np.sum(temp, axis=0)
            print [j/len_all for j in temp1]
        for i in xrange(len(precision_models)):
            print model_names[i]
            #print retrieval_models[i]
            temp = np.array(precision_models[i])
            temp1 = np.sum(temp, axis=0)
            print [j/len_all for j in temp1]
        print 'Worker'
        #print retrieval_worker_all
        temp = np.array(retrieval_worker_all)
        temp1 = np.sum(temp, axis=0)
        print [i/len_all for i in temp1]
    def create_and_evaluate_face(self):

        dict_from_validation = {}
        # dict_from_validation = {'PersonalSports': [[((0.40000000000000002, 0.55000000000000004, 1.0), 14.263289935890167), ((0.5, 0.55000000000000004, 1.0), 14.261991183820982), ((0.35000000000000003, 0.55000000000000004, 1.0), 14.259958113014259), ((0.10000000000000001, 0.55000000000000004, 1.0), 14.259958113014259), ((0.30000000000000004, 0.55000000000000004, 1.0), 14.259958113014259), ((0.0, 0.55000000000000004, 1.0), 14.259958113014259), ((0.050000000000000003, 0.55000000000000004, 1.0), 14.259958113014259), ((0.20000000000000001, 0.55000000000000004, 1.0), 14.259958113014259), ((0.25, 0.55000000000000004, 1.0), 14.259958113014259), ((0.15000000000000002, 0.55000000000000004, 1.0), 14.259958113014259), ((0.45000000000000001, 0.55000000000000004, 1.0), 14.248415830875679)], [((0.5, 0.55000000000000004, 1.0), 11.612876593690233), ((0.60000000000000009, 0.85000000000000009, 1.0), 11.595416357968451), ((0.60000000000000009, 0.80000000000000004, 1.0), 11.595416357968451), ((0.60000000000000009, 0.95000000000000007, 1.0), 11.595416357968451), ((0.60000000000000009, 0.90000000000000002, 1.0), 11.595416357968451), ((0.60000000000000009, 1.0, 1.0), 11.595416357968451), ((0.5, 0.90000000000000002, 1.0), 11.553274828313574), ((0.5, 0.85000000000000009, 1.0), 11.553274828313574), ((0.5, 0.95000000000000007, 1.0), 11.553274828313574), ((0.5, 1.0, 1.0), 11.553274828313574), ((0.5, 0.80000000000000004, 1.0), 11.553274828313574), ((0.40000000000000002, 0.5, 1.0), 11.436667412826685), ((0.45000000000000001, 0.5, 1.0), 11.426734992378375), ((0.65000000000000002, 0.80000000000000004, 1.0), 11.40828406186992), ((0.65000000000000002, 1.0, 1.0), 11.40828406186992), ((0.65000000000000002, 0.85000000000000009, 1.0), 11.40828406186992), ((0.65000000000000002, 0.90000000000000002, 1.0), 11.40828406186992), ((0.65000000000000002, 0.95000000000000007, 1.0), 11.40828406186992), ((0.45000000000000001, 0.85000000000000009, 1.0), 11.351614862068127), ((0.45000000000000001, 0.95000000000000007, 1.0), 11.351614862068127), ((0.45000000000000001, 0.90000000000000002, 1.0), 11.351614862068127), ((0.45000000000000001, 0.80000000000000004, 1.0), 11.351614862068127), ((0.45000000000000001, 1.0, 1.0), 11.351614862068127), ((0.35000000000000003, 0.5, 1.0), 11.333195336116063), ((0.55000000000000004, 0.80000000000000004, 1.0), 11.304803431783853), ((0.55000000000000004, 0.85000000000000009, 1.0), 11.304803431783853), ((0.55000000000000004, 1.0, 1.0), 11.304803431783853), ((0.55000000000000004, 0.95000000000000007, 1.0), 11.304803431783853), ((0.55000000000000004, 0.90000000000000002, 1.0), 11.304803431783853), ((0.60000000000000009, 0.75, 1.0), 11.276407751280045), ((0.30000000000000004, 0.5, 1.0), 11.26326671516737), ((0.25, 0.5, 1.0), 11.256483734064712), ((0.15000000000000002, 0.5, 1.0), 11.254318262141176), ((0.0, 0.5, 1.0), 11.254318262141176), ((0.20000000000000001, 0.5, 1.0), 11.254318262141176), ((0.10000000000000001, 0.5, 1.0), 11.254318262141176), ((0.050000000000000003, 0.5, 1.0), 11.254318262141176), ((0.5, 0.65000000000000002, 1.0), 11.253028044844278), ((0.40000000000000002, 0.90000000000000002, 1.0), 11.247693359063211), ((0.40000000000000002, 1.0, 1.0), 11.247693359063211), ((0.40000000000000002, 0.85000000000000009, 1.0), 11.247693359063211), ((0.40000000000000002, 0.95000000000000007, 1.0), 11.247693359063211), ((0.40000000000000002, 0.80000000000000004, 1.0), 11.247693359063211), ((0.5, 0.75, 1.0), 11.24293897443386), ((0.5, 0.70000000000000007, 1.0), 11.235751529242988), ((0.35000000000000003, 0.80000000000000004, 1.0), 11.214919653037292), ((0.35000000000000003, 1.0, 1.0), 11.214919653037292), ((0.35000000000000003, 0.95000000000000007, 1.0), 11.214919653037292), ((0.35000000000000003, 0.85000000000000009, 1.0), 11.214919653037292), ((0.35000000000000003, 0.90000000000000002, 1.0), 11.214919653037292), ((0.45000000000000001, 0.55000000000000004, 1.0), 11.198812433064164), ((0.5, 0.60000000000000009, 1.0), 11.150250841808637), ((0.30000000000000004, 0.85000000000000009, 1.0), 11.135661567461517), ((0.30000000000000004, 0.90000000000000002, 1.0), 11.135661567461517), ((0.30000000000000004, 1.0, 1.0), 11.135661567461517), ((0.30000000000000004, 0.95000000000000007, 1.0), 11.135661567461517), ((0.30000000000000004, 0.80000000000000004, 1.0), 11.135661567461517), ((0.25, 0.95000000000000007, 1.0), 11.133712274940692), ((0.25, 0.80000000000000004, 1.0), 11.133712274940692), ((0.25, 1.0, 1.0), 11.133712274940692), ((0.25, 0.90000000000000002, 1.0), 11.133712274940692), ((0.25, 0.85000000000000009, 1.0), 11.133712274940692), ((0.0, 0.90000000000000002, 1.0), 11.131093706145656), ((0.10000000000000001, 0.95000000000000007, 1.0), 11.131093706145656), ((0.20000000000000001, 0.80000000000000004, 1.0), 11.131093706145656), ((0.050000000000000003, 0.85000000000000009, 1.0), 11.131093706145656), ((0.15000000000000002, 0.90000000000000002, 1.0), 11.131093706145656), ((0.10000000000000001, 0.90000000000000002, 1.0), 11.131093706145656), ((0.15000000000000002, 0.95000000000000007, 1.0), 11.131093706145656), ((0.0, 0.95000000000000007, 1.0), 11.131093706145656), ((0.0, 0.80000000000000004, 1.0), 11.131093706145656), ((0.20000000000000001, 0.85000000000000009, 1.0), 11.131093706145656), ((0.20000000000000001, 1.0, 1.0), 11.131093706145656), ((0.15000000000000002, 0.80000000000000004, 1.0), 11.131093706145656), ((0.050000000000000003, 1.0, 1.0), 11.131093706145656), ((0.10000000000000001, 0.80000000000000004, 1.0), 11.131093706145656), ((0.050000000000000003, 0.95000000000000007, 1.0), 11.131093706145656), ((0.0, 1.0, 1.0), 11.131093706145656), ((0.10000000000000001, 0.85000000000000009, 1.0), 11.131093706145656), ((0.050000000000000003, 0.80000000000000004, 1.0), 11.131093706145656), ((0.20000000000000001, 0.95000000000000007, 1.0), 11.131093706145656), ((0.20000000000000001, 0.90000000000000002, 1.0), 11.131093706145656), ((0.050000000000000003, 0.90000000000000002, 1.0), 11.131093706145656), ((0.15000000000000002, 0.85000000000000009, 1.0), 11.131093706145656), ((0.10000000000000001, 1.0, 1.0), 11.131093706145656), ((0.15000000000000002, 1.0, 1.0), 11.131093706145656), ((0.0, 0.85000000000000009, 1.0), 11.131093706145656), ((0.40000000000000002, 0.55000000000000004, 1.0), 11.09715239432681), ((0.65000000000000002, 0.75, 1.0), 11.094352945613224), ((0.35000000000000003, 0.55000000000000004, 1.0), 11.079004039009503), ((0.45000000000000001, 0.70000000000000007, 1.0), 11.064979579286044), ((0.45000000000000001, 0.65000000000000002, 1.0), 11.06166399741472), ((0.45000000000000001, 0.75, 1.0), 11.046605323822124), ((0.55000000000000004, 0.70000000000000007, 1.0), 11.026688927661624), ((0.60000000000000009, 0.70000000000000007, 1.0), 11.025142613078213), ((0.45000000000000001, 0.60000000000000009, 1.0), 11.000485279767673), ((0.55000000000000004, 0.75, 1.0), 10.995166660154746), ((0.30000000000000004, 0.55000000000000004, 1.0), 10.979201943877657), ((0.25, 0.55000000000000004, 1.0), 10.975939690268797), ((0.10000000000000001, 0.55000000000000004, 1.0), 10.973591648447073), ((0.0, 0.55000000000000004, 1.0), 10.973591648447073), ((0.050000000000000003, 0.55000000000000004, 1.0), 10.973591648447073), ((0.20000000000000001, 0.55000000000000004, 1.0), 10.973591648447073), ((0.15000000000000002, 0.55000000000000004, 1.0), 10.973591648447073), ((0.40000000000000002, 0.70000000000000007, 1.0), 10.96462981358526), ((0.40000000000000002, 0.65000000000000002, 1.0), 10.961483297540898), ((0.35000000000000003, 0.70000000000000007, 1.0), 10.943896138756553), ((0.40000000000000002, 0.45000000000000001, 1.0), 10.94186625373982), ((0.35000000000000003, 0.65000000000000002, 1.0), 10.940656191204722), ((0.40000000000000002, 0.75, 1.0), 10.9391102717265), ((0.35000000000000003, 0.75, 1.0), 10.920800851414866), ((0.70000000000000007, 1.0, 1.0), 10.91244664536292), ((0.70000000000000007, 0.80000000000000004, 1.0), 10.91244664536292), ((0.70000000000000007, 0.90000000000000002, 1.0), 10.91244664536292), ((0.70000000000000007, 0.85000000000000009, 1.0), 10.91244664536292), ((0.70000000000000007, 0.95000000000000007, 1.0), 10.91244664536292), ((0.35000000000000003, 0.45000000000000001, 1.0), 10.910661043523321), ((0.40000000000000002, 0.60000000000000009, 1.0), 10.894482309296407), ((0.35000000000000003, 0.60000000000000009, 1.0), 10.87446075851579), ((0.30000000000000004, 0.70000000000000007, 1.0), 10.84996891258291), ((0.30000000000000004, 0.65000000000000002, 1.0), 10.84672896503108), ((0.25, 0.70000000000000007, 1.0), 10.84557008471897), ((0.15000000000000002, 0.70000000000000007, 1.0), 10.843222042897244), ((0.10000000000000001, 0.70000000000000007, 1.0), 10.843222042897244), ((0.0, 0.70000000000000007, 1.0), 10.843222042897244), ((0.20000000000000001, 0.70000000000000007, 1.0), 10.843222042897244), ((0.050000000000000003, 0.70000000000000007, 1.0), 10.843222042897244), ((0.25, 0.65000000000000002, 1.0), 10.842330137167146), ((0.050000000000000003, 0.65000000000000002, 1.0), 10.83998209534542), ((0.0, 0.65000000000000002, 1.0), 10.83998209534542), ((0.10000000000000001, 0.65000000000000002, 1.0), 10.83998209534542), ((0.20000000000000001, 0.65000000000000002, 1.0), 10.83998209534542), ((0.15000000000000002, 0.65000000000000002, 1.0), 10.83998209534542), ((0.30000000000000004, 0.75, 1.0), 10.82601476760276), ((0.25, 0.75, 1.0), 10.822022372487366), ((0.10000000000000001, 0.75, 1.0), 10.81967433066564), ((0.20000000000000001, 0.75, 1.0), 10.81967433066564), ((0.0, 0.75, 1.0), 10.81967433066564), ((0.050000000000000003, 0.75, 1.0), 10.81967433066564), ((0.15000000000000002, 0.75, 1.0), 10.81967433066564), ((0.30000000000000004, 0.60000000000000009, 1.0), 10.779625583437069), ((0.55000000000000004, 0.65000000000000002, 1.0), 10.779604955357621), ((0.25, 0.60000000000000009, 1.0), 10.775226755573133), ((0.050000000000000003, 0.60000000000000009, 1.0), 10.772878713751407), ((0.15000000000000002, 0.60000000000000009, 1.0), 10.772878713751407), ((0.20000000000000001, 0.60000000000000009, 1.0), 10.772878713751407), ((0.10000000000000001, 0.60000000000000009, 1.0), 10.772878713751407), ((0.0, 0.60000000000000009, 1.0), 10.772878713751407), ((0.25, 0.45000000000000001, 1.0), 10.771236867542276), ((0.10000000000000001, 0.45000000000000001, 1.0), 10.77022992309783), ((0.15000000000000002, 0.45000000000000001, 1.0), 10.77022992309783), ((0.20000000000000001, 0.45000000000000001, 1.0), 10.77022992309783), ((0.0, 0.45000000000000001, 1.0), 10.77022992309783), ((0.050000000000000003, 0.45000000000000001, 1.0), 10.77022992309783), ((0.30000000000000004, 0.45000000000000001, 1.0), 10.769048040333754), ((0.65000000000000002, 0.70000000000000007, 1.0), 10.764280951235925), ((0.70000000000000007, 0.75, 1.0), 10.761335534251806)], [((0.75, 0.85000000000000009, 1.0), 16.28899107311131), ((0.75, 0.90000000000000002, 1.0), 16.189758008412422), ((0.75, 0.95000000000000007, 1.0), 16.188472433238154), ((0.75, 1.0, 1.0), 16.188472433238154), ((0.80000000000000004, 0.90000000000000002, 1.0), 16.029979861234022), ((0.80000000000000004, 0.95000000000000007, 1.0), 16.027574670315243), ((0.80000000000000004, 1.0, 1.0), 16.027574670315243), ((0.75, 0.80000000000000004, 1.0), 15.8356857912416), ((0.60000000000000009, 0.65000000000000002, 1.0), 15.799547983274126), ((0.55000000000000004, 0.85000000000000009, 1.0), 15.739941867819011), ((0.55000000000000004, 0.60000000000000009, 1.0), 15.73626537901541), ((0.5, 0.65000000000000002, 1.0), 15.735936904602003), ((0.5, 0.85000000000000009, 1.0), 15.735314378872433), ((0.45000000000000001, 0.85000000000000009, 1.0), 15.729797671004647), ((0.30000000000000004, 0.85000000000000009, 1.0), 15.721847475798453), ((0.050000000000000003, 0.85000000000000009, 1.0), 15.721847475798453), ((0.20000000000000001, 0.85000000000000009, 1.0), 15.721847475798453), ((0.10000000000000001, 0.85000000000000009, 1.0), 15.721847475798453), ((0.25, 0.85000000000000009, 1.0), 15.721847475798453), ((0.15000000000000002, 0.85000000000000009, 1.0), 15.721847475798453), ((0.0, 0.85000000000000009, 1.0), 15.721847475798453), ((0.45000000000000001, 0.65000000000000002, 1.0), 15.721802124408917), ((0.35000000000000003, 0.85000000000000009, 1.0), 15.721191960180503), ((0.40000000000000002, 0.85000000000000009, 1.0), 15.720895768385578), ((0.55000000000000004, 0.65000000000000002, 1.0), 15.718432721292482), ((0.050000000000000003, 0.65000000000000002, 1.0), 15.702918309142994), ((0.0, 0.65000000000000002, 1.0), 15.702918309142994), ((0.10000000000000001, 0.65000000000000002, 1.0), 15.702918309142994), ((0.30000000000000004, 0.65000000000000002, 1.0), 15.702918309142994), ((0.20000000000000001, 0.65000000000000002, 1.0), 15.702918309142994), ((0.15000000000000002, 0.65000000000000002, 1.0), 15.702918309142994), ((0.25, 0.65000000000000002, 1.0), 15.702918309142994), ((0.35000000000000003, 0.65000000000000002, 1.0), 15.699545852957968), ((0.40000000000000002, 0.65000000000000002, 1.0), 15.698175426888582), ((0.60000000000000009, 0.85000000000000009, 1.0), 15.677468473264044), ((0.80000000000000004, 0.85000000000000009, 1.0), 15.669369773168405), ((0.65000000000000002, 0.85000000000000009, 1.0), 15.66603112920033), ((0.45000000000000001, 0.60000000000000009, 1.0), 15.660520634042513), ((0.5, 0.60000000000000009, 1.0), 15.654319099513936), ((0.050000000000000003, 0.60000000000000009, 1.0), 15.636439102085859), ((0.15000000000000002, 0.60000000000000009, 1.0), 15.636439102085859), ((0.20000000000000001, 0.60000000000000009, 1.0), 15.636439102085859), ((0.10000000000000001, 0.60000000000000009, 1.0), 15.636439102085859), ((0.30000000000000004, 0.60000000000000009, 1.0), 15.636439102085859), ((0.0, 0.60000000000000009, 1.0), 15.636439102085859), ((0.25, 0.60000000000000009, 1.0), 15.636439102085859), ((0.35000000000000003, 0.60000000000000009, 1.0), 15.63528225619704), ((0.40000000000000002, 0.60000000000000009, 1.0), 15.633601100051598), ((0.70000000000000007, 0.85000000000000009, 1.0), 15.631412399508323), ((0.55000000000000004, 0.90000000000000002, 1.0), 15.550146504991867), ((0.55000000000000004, 1.0, 1.0), 15.547521303110823), ((0.55000000000000004, 0.95000000000000007, 1.0), 15.547521303110823), ((0.5, 0.90000000000000002, 1.0), 15.543689153552954), ((0.5, 0.95000000000000007, 1.0), 15.541063951671909), ((0.5, 1.0, 1.0), 15.541063951671909), ((0.45000000000000001, 0.90000000000000002, 1.0), 15.538129562567686), ((0.45000000000000001, 0.95000000000000007, 1.0), 15.535325509312742), ((0.45000000000000001, 1.0, 1.0), 15.535325509312742), ((0.5, 0.55000000000000004, 1.0), 15.534999482130866), ((0.0, 0.90000000000000002, 1.0), 15.530008992739445), ((0.30000000000000004, 0.90000000000000002, 1.0), 15.530008992739445), ((0.15000000000000002, 0.90000000000000002, 1.0), 15.530008992739445), ((0.10000000000000001, 0.90000000000000002, 1.0), 15.530008992739445), ((0.25, 0.90000000000000002, 1.0), 15.530008992739445), ((0.20000000000000001, 0.90000000000000002, 1.0), 15.530008992739445), ((0.050000000000000003, 0.90000000000000002, 1.0), 15.530008992739445), ((0.10000000000000001, 0.45000000000000001, 1.0), 15.529532025393681), ((0.15000000000000002, 0.45000000000000001, 1.0), 15.529532025393681), ((0.20000000000000001, 0.45000000000000001, 1.0), 15.529532025393681), ((0.25, 0.45000000000000001, 1.0), 15.529532025393681), ((0.0, 0.45000000000000001, 1.0), 15.529532025393681), ((0.050000000000000003, 0.45000000000000001, 1.0), 15.529532025393681), ((0.30000000000000004, 0.45000000000000001, 1.0), 15.529532025393681), ((0.35000000000000003, 0.90000000000000002, 1.0), 15.5293534771215), ((0.40000000000000002, 0.90000000000000002, 1.0), 15.529057285326573), ((0.10000000000000001, 0.95000000000000007, 1.0), 15.5272049394845), ((0.25, 0.95000000000000007, 1.0), 15.5272049394845), ((0.15000000000000002, 0.95000000000000007, 1.0), 15.5272049394845), ((0.0, 0.95000000000000007, 1.0), 15.5272049394845), ((0.25, 1.0, 1.0), 15.5272049394845), ((0.20000000000000001, 1.0, 1.0), 15.5272049394845), ((0.050000000000000003, 1.0, 1.0), 15.5272049394845), ((0.050000000000000003, 0.95000000000000007, 1.0), 15.5272049394845), ((0.0, 1.0, 1.0), 15.5272049394845), ((0.30000000000000004, 1.0, 1.0), 15.5272049394845), ((0.20000000000000001, 0.95000000000000007, 1.0), 15.5272049394845), ((0.30000000000000004, 0.95000000000000007, 1.0), 15.5272049394845), ((0.10000000000000001, 1.0, 1.0), 15.5272049394845), ((0.15000000000000002, 1.0, 1.0), 15.5272049394845), ((0.35000000000000003, 1.0, 1.0), 15.526549423866554), ((0.35000000000000003, 0.95000000000000007, 1.0), 15.526549423866554), ((0.40000000000000002, 1.0, 1.0), 15.526253232071621), ((0.40000000000000002, 0.95000000000000007, 1.0), 15.526253232071621), ((0.45000000000000001, 0.5, 1.0), 15.524628044290559), ((0.35000000000000003, 0.45000000000000001, 1.0), 15.523308688114385), ((0.40000000000000002, 0.45000000000000001, 1.0), 15.519576558617203), ((0.70000000000000007, 0.90000000000000002, 1.0), 15.5170412095946), ((0.70000000000000007, 1.0, 1.0), 15.515377086406758), ((0.70000000000000007, 0.95000000000000007, 1.0), 15.515377086406758), ((0.55000000000000004, 0.70000000000000007, 1.0), 15.506349269620705), ((0.5, 0.70000000000000007, 1.0), 15.502627734157338)]], 'Museum': [[], [((0.30000000000000004, 0.55000000000000004, 1.0), 14.83383550208047), ((0.30000000000000004, 0.5, 1.0), 14.820492148930454), ((0.35000000000000003, 0.60000000000000009, 1.0), 14.751164904601712), ((0.35000000000000003, 0.55000000000000004, 1.0), 14.72264287674295), ((0.25, 0.5, 1.0), 14.696001048888874), ((0.15000000000000002, 0.5, 1.0), 14.696001048888874), ((0.0, 0.5, 1.0), 14.696001048888874), ((0.20000000000000001, 0.5, 1.0), 14.696001048888874), ((0.10000000000000001, 0.5, 1.0), 14.696001048888874), ((0.050000000000000003, 0.5, 1.0), 14.696001048888874), ((0.10000000000000001, 0.45000000000000001, 1.0), 14.576128758563666), ((0.15000000000000002, 0.45000000000000001, 1.0), 14.576128758563666), ((0.20000000000000001, 0.45000000000000001, 1.0), 14.576128758563666), ((0.25, 0.45000000000000001, 1.0), 14.576128758563666), ((0.0, 0.45000000000000001, 1.0), 14.576128758563666), ((0.050000000000000003, 0.45000000000000001, 1.0), 14.576128758563666), ((0.30000000000000004, 0.60000000000000009, 1.0), 14.556205965992287), ((0.10000000000000001, 0.55000000000000004, 1.0), 14.522992707779027), ((0.0, 0.55000000000000004, 1.0), 14.522992707779027), ((0.050000000000000003, 0.55000000000000004, 1.0), 14.522992707779027), ((0.20000000000000001, 0.55000000000000004, 1.0), 14.522992707779027), ((0.25, 0.55000000000000004, 1.0), 14.522992707779027), ((0.15000000000000002, 0.55000000000000004, 1.0), 14.522992707779027), ((0.30000000000000004, 0.75, 1.0), 14.49006839985472), ((0.30000000000000004, 0.85000000000000009, 1.0), 14.49006839985472), ((0.30000000000000004, 0.90000000000000002, 1.0), 14.49006839985472), ((0.30000000000000004, 0.65000000000000002, 1.0), 14.49006839985472), ((0.30000000000000004, 0.70000000000000007, 1.0), 14.49006839985472), ((0.30000000000000004, 1.0, 1.0), 14.49006839985472), ((0.30000000000000004, 0.95000000000000007, 1.0), 14.49006839985472), ((0.30000000000000004, 0.80000000000000004, 1.0), 14.49006839985472), ((0.35000000000000003, 0.80000000000000004, 1.0), 14.484743891752126), ((0.35000000000000003, 0.65000000000000002, 1.0), 14.484743891752126), ((0.35000000000000003, 1.0, 1.0), 14.484743891752126), ((0.35000000000000003, 0.75, 1.0), 14.484743891752126), ((0.35000000000000003, 0.95000000000000007, 1.0), 14.484743891752126), ((0.35000000000000003, 0.85000000000000009, 1.0), 14.484743891752126), ((0.35000000000000003, 0.90000000000000002, 1.0), 14.484743891752126), ((0.35000000000000003, 0.70000000000000007, 1.0), 14.484743891752126), ((0.40000000000000002, 0.60000000000000009, 1.0), 14.479063340289448), ((0.40000000000000002, 0.90000000000000002, 1.0), 14.456047467273574), ((0.40000000000000002, 0.75, 1.0), 14.456047467273574), ((0.40000000000000002, 1.0, 1.0), 14.456047467273574), ((0.40000000000000002, 0.70000000000000007, 1.0), 14.456047467273574), ((0.40000000000000002, 0.85000000000000009, 1.0), 14.456047467273574), ((0.40000000000000002, 0.95000000000000007, 1.0), 14.456047467273574), ((0.40000000000000002, 0.80000000000000004, 1.0), 14.456047467273574), ((0.40000000000000002, 0.65000000000000002, 1.0), 14.456047467273574), ((0.050000000000000003, 0.60000000000000009, 1.0), 14.43556989584832), ((0.15000000000000002, 0.60000000000000009, 1.0), 14.43556989584832), ((0.20000000000000001, 0.60000000000000009, 1.0), 14.43556989584832), ((0.10000000000000001, 0.60000000000000009, 1.0), 14.43556989584832), ((0.0, 0.60000000000000009, 1.0), 14.43556989584832), ((0.25, 0.60000000000000009, 1.0), 14.43556989584832), ((0.45000000000000001, 0.75, 1.0), 14.4284010198057), ((0.45000000000000001, 0.85000000000000009, 1.0), 14.4284010198057), ((0.45000000000000001, 0.95000000000000007, 1.0), 14.4284010198057), ((0.45000000000000001, 0.70000000000000007, 1.0), 14.4284010198057), ((0.45000000000000001, 0.65000000000000002, 1.0), 14.4284010198057), ((0.45000000000000001, 0.90000000000000002, 1.0), 14.4284010198057), ((0.45000000000000001, 0.80000000000000004, 1.0), 14.4284010198057), ((0.45000000000000001, 1.0, 1.0), 14.4284010198057), ((0.050000000000000003, 0.65000000000000002, 1.0), 14.369432329710751), ((0.0, 0.90000000000000002, 1.0), 14.369432329710751), ((0.10000000000000001, 0.95000000000000007, 1.0), 14.369432329710751), ((0.15000000000000002, 0.70000000000000007, 1.0), 14.369432329710751), ((0.0, 0.65000000000000002, 1.0), 14.369432329710751), ((0.10000000000000001, 0.70000000000000007, 1.0), 14.369432329710751), ((0.10000000000000001, 0.75, 1.0), 14.369432329710751), ((0.20000000000000001, 0.80000000000000004, 1.0), 14.369432329710751), ((0.050000000000000003, 0.85000000000000009, 1.0), 14.369432329710751), ((0.25, 0.75, 1.0), 14.369432329710751), ((0.20000000000000001, 0.75, 1.0), 14.369432329710751), ((0.25, 0.95000000000000007, 1.0), 14.369432329710751), ((0.10000000000000001, 0.65000000000000002, 1.0), 14.369432329710751), ((0.15000000000000002, 0.90000000000000002, 1.0), 14.369432329710751), ((0.25, 0.70000000000000007, 1.0), 14.369432329710751), ((0.10000000000000001, 0.90000000000000002, 1.0), 14.369432329710751), ((0.15000000000000002, 0.95000000000000007, 1.0), 14.369432329710751), ((0.0, 0.75, 1.0), 14.369432329710751), ((0.0, 0.95000000000000007, 1.0), 14.369432329710751), ((0.0, 0.80000000000000004, 1.0), 14.369432329710751), ((0.25, 0.80000000000000004, 1.0), 14.369432329710751), ((0.050000000000000003, 0.75, 1.0), 14.369432329710751), ((0.20000000000000001, 0.65000000000000002, 1.0), 14.369432329710751), ((0.25, 1.0, 1.0), 14.369432329710751), ((0.20000000000000001, 0.85000000000000009, 1.0), 14.369432329710751), ((0.20000000000000001, 1.0, 1.0), 14.369432329710751), ((0.15000000000000002, 0.80000000000000004, 1.0), 14.369432329710751), ((0.15000000000000002, 0.75, 1.0), 14.369432329710751), ((0.0, 0.70000000000000007, 1.0), 14.369432329710751), ((0.050000000000000003, 1.0, 1.0), 14.369432329710751), ((0.10000000000000001, 0.80000000000000004, 1.0), 14.369432329710751), ((0.15000000000000002, 0.65000000000000002, 1.0), 14.369432329710751), ((0.050000000000000003, 0.95000000000000007, 1.0), 14.369432329710751), ((0.0, 1.0, 1.0), 14.369432329710751), ((0.25, 0.90000000000000002, 1.0), 14.369432329710751), ((0.10000000000000001, 0.85000000000000009, 1.0), 14.369432329710751), ((0.050000000000000003, 0.80000000000000004, 1.0), 14.369432329710751), ((0.20000000000000001, 0.95000000000000007, 1.0), 14.369432329710751), ((0.20000000000000001, 0.70000000000000007, 1.0), 14.369432329710751), ((0.20000000000000001, 0.90000000000000002, 1.0), 14.369432329710751), ((0.050000000000000003, 0.90000000000000002, 1.0), 14.369432329710751), ((0.25, 0.65000000000000002, 1.0), 14.369432329710751), ((0.25, 0.85000000000000009, 1.0), 14.369432329710751), ((0.15000000000000002, 0.85000000000000009, 1.0), 14.369432329710751), ((0.10000000000000001, 1.0, 1.0), 14.369432329710751), ((0.050000000000000003, 0.70000000000000007, 1.0), 14.369432329710751), ((0.15000000000000002, 1.0, 1.0), 14.369432329710751), ((0.0, 0.85000000000000009, 1.0), 14.369432329710751), ((0.30000000000000004, 0.45000000000000001, 1.0), 14.16190725352909), ((0.35000000000000003, 0.5, 1.0), 14.04572464800825), ((0.30000000000000004, 0.40000000000000002, 1.0), 13.909084811902293), ((0.40000000000000002, 0.55000000000000004, 1.0), 13.879159832787156), ((0.45000000000000001, 0.60000000000000009, 1.0), 13.853344330463296), ((0.20000000000000001, 0.35000000000000003, 1.0), 13.824980138967751), ((0.25, 0.35000000000000003, 1.0), 13.824980138967751), ((0.0, 0.35000000000000003, 1.0), 13.824980138967751), ((0.050000000000000003, 0.35000000000000003, 1.0), 13.824980138967751), ((0.15000000000000002, 0.35000000000000003, 1.0), 13.824980138967751), ((0.10000000000000001, 0.35000000000000003, 1.0), 13.824980138967751), ((0.15000000000000002, 0.40000000000000002, 1.0), 13.79321693662449), ((0.0, 0.40000000000000002, 1.0), 13.79321693662449), ((0.10000000000000001, 0.40000000000000002, 1.0), 13.79321693662449), ((0.20000000000000001, 0.40000000000000002, 1.0), 13.79321693662449), ((0.050000000000000003, 0.40000000000000002, 1.0), 13.79321693662449), ((0.25, 0.40000000000000002, 1.0), 13.79321693662449), ((0.35000000000000003, 0.45000000000000001, 1.0), 13.79290220638145), ((0.5, 0.90000000000000002, 1.0), 13.792156857512037), ((0.5, 0.85000000000000009, 1.0), 13.792156857512037), ((0.5, 0.65000000000000002, 1.0), 13.792156857512037), ((0.5, 0.70000000000000007, 1.0), 13.792156857512037), ((0.5, 0.95000000000000007, 1.0), 13.792156857512037), ((0.5, 0.75, 1.0), 13.792156857512037), ((0.5, 1.0, 1.0), 13.792156857512037), ((0.5, 0.80000000000000004, 1.0), 13.792156857512037), ((0.30000000000000004, 0.35000000000000003, 1.0), 13.694746942721986), ((0.35000000000000003, 0.40000000000000002, 1.0), 13.688447414363633), ((0.40000000000000002, 0.5, 1.0), 13.675422948460202), ((0.25, 0.30000000000000004, 1.0), 13.653634688401606), ((0.15000000000000002, 0.30000000000000004, 1.0), 13.653634688401606), ((0.050000000000000003, 0.30000000000000004, 1.0), 13.653634688401606), ((0.0, 0.30000000000000004, 1.0), 13.653634688401606), ((0.20000000000000001, 0.30000000000000004, 1.0), 13.653634688401606), ((0.10000000000000001, 0.30000000000000004, 1.0), 13.653634688401606), ((0.45000000000000001, 0.55000000000000004, 1.0), 13.651362101899354), ((0.40000000000000002, 0.45000000000000001, 1.0), 13.62451124034261), ((0.45000000000000001, 0.5, 1.0), 13.62451124034261)], [((0.60000000000000009, 0.95000000000000007, 1.0), 17.25040756355289), ((0.60000000000000009, 0.90000000000000002, 1.0), 17.25040756355289), ((0.60000000000000009, 1.0, 1.0), 17.25040756355289), ((0.60000000000000009, 0.85000000000000009, 1.0), 17.233740896886225), ((0.55000000000000004, 0.85000000000000009, 1.0), 17.231524679944325), ((0.55000000000000004, 1.0, 1.0), 17.231524679944325), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.231524679944325), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.231524679944325), ((0.55000000000000004, 0.80000000000000004, 1.0), 17.21485801327766), ((0.5, 0.90000000000000002, 1.0), 17.196219720589383), ((0.5, 0.85000000000000009, 1.0), 17.196219720589383), ((0.5, 0.95000000000000007, 1.0), 17.196219720589383), ((0.5, 1.0, 1.0), 17.196219720589383), ((0.0, 0.90000000000000002, 1.0), 17.192306211108303), ((0.40000000000000002, 0.90000000000000002, 1.0), 17.192306211108303), ((0.10000000000000001, 0.95000000000000007, 1.0), 17.192306211108303), ((0.40000000000000002, 1.0, 1.0), 17.192306211108303), ((0.35000000000000003, 1.0, 1.0), 17.192306211108303), ((0.30000000000000004, 0.85000000000000009, 1.0), 17.192306211108303), ((0.050000000000000003, 0.85000000000000009, 1.0), 17.192306211108303), ((0.30000000000000004, 0.90000000000000002, 1.0), 17.192306211108303), ((0.25, 0.95000000000000007, 1.0), 17.192306211108303), ((0.15000000000000002, 0.90000000000000002, 1.0), 17.192306211108303), ((0.45000000000000001, 0.85000000000000009, 1.0), 17.192306211108303), ((0.10000000000000001, 0.90000000000000002, 1.0), 17.192306211108303), ((0.40000000000000002, 0.85000000000000009, 1.0), 17.192306211108303), ((0.45000000000000001, 0.95000000000000007, 1.0), 17.192306211108303), ((0.15000000000000002, 0.95000000000000007, 1.0), 17.192306211108303), ((0.0, 0.95000000000000007, 1.0), 17.192306211108303), ((0.25, 1.0, 1.0), 17.192306211108303), ((0.35000000000000003, 0.95000000000000007, 1.0), 17.192306211108303), ((0.20000000000000001, 0.85000000000000009, 1.0), 17.192306211108303), ((0.20000000000000001, 1.0, 1.0), 17.192306211108303), ((0.40000000000000002, 0.95000000000000007, 1.0), 17.192306211108303), ((0.050000000000000003, 1.0, 1.0), 17.192306211108303), ((0.35000000000000003, 0.85000000000000009, 1.0), 17.192306211108303), ((0.35000000000000003, 0.90000000000000002, 1.0), 17.192306211108303), ((0.050000000000000003, 0.95000000000000007, 1.0), 17.192306211108303), ((0.0, 1.0, 1.0), 17.192306211108303), ((0.45000000000000001, 0.90000000000000002, 1.0), 17.192306211108303), ((0.25, 0.90000000000000002, 1.0), 17.192306211108303), ((0.30000000000000004, 1.0, 1.0), 17.192306211108303), ((0.10000000000000001, 0.85000000000000009, 1.0), 17.192306211108303), ((0.20000000000000001, 0.95000000000000007, 1.0), 17.192306211108303), ((0.20000000000000001, 0.90000000000000002, 1.0), 17.192306211108303), ((0.050000000000000003, 0.90000000000000002, 1.0), 17.192306211108303), ((0.25, 0.85000000000000009, 1.0), 17.192306211108303), ((0.30000000000000004, 0.95000000000000007, 1.0), 17.192306211108303), ((0.15000000000000002, 0.85000000000000009, 1.0), 17.192306211108303), ((0.10000000000000001, 1.0, 1.0), 17.192306211108303), ((0.15000000000000002, 1.0, 1.0), 17.192306211108303), ((0.0, 0.85000000000000009, 1.0), 17.192306211108303), ((0.45000000000000001, 1.0, 1.0), 17.192306211108303), ((0.5, 0.80000000000000004, 1.0), 17.179553053922714), ((0.35000000000000003, 0.80000000000000004, 1.0), 17.175639544441633), ((0.20000000000000001, 0.80000000000000004, 1.0), 17.175639544441633), ((0.0, 0.80000000000000004, 1.0), 17.175639544441633), ((0.25, 0.80000000000000004, 1.0), 17.175639544441633), ((0.15000000000000002, 0.80000000000000004, 1.0), 17.175639544441633), ((0.10000000000000001, 0.80000000000000004, 1.0), 17.175639544441633), ((0.050000000000000003, 0.80000000000000004, 1.0), 17.175639544441633), ((0.45000000000000001, 0.80000000000000004, 1.0), 17.175639544441633), ((0.40000000000000002, 0.80000000000000004, 1.0), 17.175639544441633), ((0.30000000000000004, 0.80000000000000004, 1.0), 17.175639544441633), ((0.60000000000000009, 0.80000000000000004, 1.0), 17.035799428632256), ((0.55000000000000004, 0.75, 1.0), 17.01691654502369), ((0.5, 0.75, 1.0), 16.981986211043367), ((0.60000000000000009, 0.75, 1.0), 16.979359507622526), ((0.40000000000000002, 0.75, 1.0), 16.97807270156229), ((0.45000000000000001, 0.75, 1.0), 16.97807270156229), ((0.30000000000000004, 0.75, 1.0), 16.97807270156229), ((0.10000000000000001, 0.75, 1.0), 16.97807270156229), ((0.25, 0.75, 1.0), 16.97807270156229), ((0.20000000000000001, 0.75, 1.0), 16.97807270156229), ((0.35000000000000003, 0.75, 1.0), 16.97807270156229), ((0.0, 0.75, 1.0), 16.97807270156229), ((0.050000000000000003, 0.75, 1.0), 16.97807270156229), ((0.15000000000000002, 0.75, 1.0), 16.97807270156229), ((0.55000000000000004, 0.70000000000000007, 1.0), 16.967913989382392), ((0.60000000000000009, 0.70000000000000007, 1.0), 16.945130865760046), ((0.5, 0.70000000000000007, 1.0), 16.932983655402076), ((0.55000000000000004, 0.65000000000000002, 1.0), 16.930068589737253), ((0.15000000000000002, 0.70000000000000007, 1.0), 16.929353293006464), ((0.10000000000000001, 0.70000000000000007, 1.0), 16.929353293006464), ((0.40000000000000002, 0.70000000000000007, 1.0), 16.929353293006464), ((0.25, 0.70000000000000007, 1.0), 16.929353293006464), ((0.30000000000000004, 0.70000000000000007, 1.0), 16.929353293006464), ((0.45000000000000001, 0.70000000000000007, 1.0), 16.929353293006464), ((0.0, 0.70000000000000007, 1.0), 16.929353293006464), ((0.35000000000000003, 0.70000000000000007, 1.0), 16.929353293006464), ((0.20000000000000001, 0.70000000000000007, 1.0), 16.929353293006464), ((0.050000000000000003, 0.70000000000000007, 1.0), 16.929353293006464), ((0.5, 0.65000000000000002, 1.0), 16.894801389066313), ((0.050000000000000003, 0.65000000000000002, 1.0), 16.891171026670705), ((0.0, 0.65000000000000002, 1.0), 16.891171026670705), ((0.35000000000000003, 0.65000000000000002, 1.0), 16.891171026670705), ((0.10000000000000001, 0.65000000000000002, 1.0), 16.891171026670705), ((0.30000000000000004, 0.65000000000000002, 1.0), 16.891171026670705), ((0.20000000000000001, 0.65000000000000002, 1.0), 16.891171026670705), ((0.45000000000000001, 0.65000000000000002, 1.0), 16.891171026670705), ((0.15000000000000002, 0.65000000000000002, 1.0), 16.891171026670705), ((0.25, 0.65000000000000002, 1.0), 16.891171026670705), ((0.40000000000000002, 0.65000000000000002, 1.0), 16.891171026670705), ((0.70000000000000007, 1.0, 1.0), 16.785189930220056), ((0.70000000000000007, 0.95000000000000007, 1.0), 16.785189930220056), ((0.65000000000000002, 1.0, 1.0), 16.75864362902854), ((0.65000000000000002, 0.95000000000000007, 1.0), 16.75864362902854), ((0.65000000000000002, 0.90000000000000002, 1.0), 16.741976962361875), ((0.75, 0.95000000000000007, 1.0), 16.591960164993104), ((0.75, 1.0, 1.0), 16.591960164993104), ((0.70000000000000007, 0.90000000000000002, 1.0), 16.587248461966084), ((0.65000000000000002, 0.85000000000000009, 1.0), 16.544035494107906), ((0.80000000000000004, 0.95000000000000007, 1.0), 16.535405459777685), ((0.75, 0.90000000000000002, 1.0), 16.535405459777685), ((0.80000000000000004, 1.0, 1.0), 16.535405459777685), ((0.70000000000000007, 0.85000000000000009, 1.0), 16.53069375675067), ((0.85000000000000009, 1.0, 1.0), 16.49298943086612), ((0.75, 0.85000000000000009, 1.0), 16.49298943086612), ((0.85000000000000009, 0.95000000000000007, 1.0), 16.49298943086612), ((0.80000000000000004, 0.90000000000000002, 1.0), 16.49298943086612)]], 'UrbanTrip': [[((0.60000000000000009, 0.85000000000000009, 1.0), 18.815038562245977), ((0.65000000000000002, 1.0, 1.0), 18.815038562245977), ((0.60000000000000009, 0.95000000000000007, 1.0), 18.815038562245977), ((0.60000000000000009, 0.90000000000000002, 1.0), 18.815038562245977), ((0.60000000000000009, 1.0, 1.0), 18.815038562245977), ((0.65000000000000002, 0.90000000000000002, 1.0), 18.815038562245977), ((0.65000000000000002, 0.95000000000000007, 1.0), 18.815038562245977), ((0.55000000000000004, 0.80000000000000004, 1.0), 18.794149184028075), ((0.55000000000000004, 0.85000000000000009, 1.0), 18.794149184028075), ((0.55000000000000004, 1.0, 1.0), 18.794149184028075), ((0.55000000000000004, 0.95000000000000007, 1.0), 18.794149184028075), ((0.55000000000000004, 0.90000000000000002, 1.0), 18.794149184028075), ((0.5, 0.90000000000000002, 1.0), 18.786670612804343), ((0.5, 0.85000000000000009, 1.0), 18.786670612804343), ((0.5, 0.95000000000000007, 1.0), 18.786670612804343), ((0.5, 0.75, 1.0), 18.786670612804343), ((0.5, 1.0, 1.0), 18.786670612804343), ((0.5, 0.80000000000000004, 1.0), 18.786670612804343), ((0.25, 0.75, 1.0), 18.738311102396295), ((0.25, 0.95000000000000007, 1.0), 18.738311102396295), ((0.25, 0.70000000000000007, 1.0), 18.738311102396295), ((0.25, 0.80000000000000004, 1.0), 18.738311102396295), ((0.25, 1.0, 1.0), 18.738311102396295), ((0.25, 0.90000000000000002, 1.0), 18.738311102396295), ((0.25, 0.65000000000000002, 1.0), 18.738311102396295), ((0.25, 0.85000000000000009, 1.0), 18.738311102396295), ((0.20000000000000001, 0.80000000000000004, 1.0), 18.737460700230102), ((0.20000000000000001, 0.75, 1.0), 18.737460700230102), ((0.20000000000000001, 0.65000000000000002, 1.0), 18.737460700230102), ((0.20000000000000001, 0.85000000000000009, 1.0), 18.737460700230102), ((0.20000000000000001, 1.0, 1.0), 18.737460700230102), ((0.20000000000000001, 0.95000000000000007, 1.0), 18.737460700230102), ((0.20000000000000001, 0.70000000000000007, 1.0), 18.737460700230102), ((0.20000000000000001, 0.90000000000000002, 1.0), 18.737460700230102), ((0.050000000000000003, 0.65000000000000002, 1.0), 18.736624902167094), ((0.0, 0.90000000000000002, 1.0), 18.736624902167094), ((0.10000000000000001, 0.95000000000000007, 1.0), 18.736624902167094), ((0.15000000000000002, 0.70000000000000007, 1.0), 18.736624902167094), ((0.0, 0.65000000000000002, 1.0), 18.736624902167094), ((0.10000000000000001, 0.70000000000000007, 1.0), 18.736624902167094), ((0.10000000000000001, 0.75, 1.0), 18.736624902167094), ((0.050000000000000003, 0.85000000000000009, 1.0), 18.736624902167094), ((0.10000000000000001, 0.65000000000000002, 1.0), 18.736624902167094), ((0.15000000000000002, 0.90000000000000002, 1.0), 18.736624902167094), ((0.10000000000000001, 0.90000000000000002, 1.0), 18.736624902167094), ((0.15000000000000002, 0.95000000000000007, 1.0), 18.736624902167094), ((0.0, 0.75, 1.0), 18.736624902167094), ((0.0, 0.95000000000000007, 1.0), 18.736624902167094), ((0.0, 0.80000000000000004, 1.0), 18.736624902167094), ((0.050000000000000003, 0.75, 1.0), 18.736624902167094), ((0.15000000000000002, 0.80000000000000004, 1.0), 18.736624902167094), ((0.15000000000000002, 0.75, 1.0), 18.736624902167094), ((0.0, 0.70000000000000007, 1.0), 18.736624902167094), ((0.050000000000000003, 1.0, 1.0), 18.736624902167094), ((0.10000000000000001, 0.80000000000000004, 1.0), 18.736624902167094), ((0.15000000000000002, 0.65000000000000002, 1.0), 18.736624902167094), ((0.050000000000000003, 0.95000000000000007, 1.0), 18.736624902167094), ((0.0, 1.0, 1.0), 18.736624902167094), ((0.10000000000000001, 0.85000000000000009, 1.0), 18.736624902167094), ((0.050000000000000003, 0.80000000000000004, 1.0), 18.736624902167094), ((0.050000000000000003, 0.90000000000000002, 1.0), 18.736624902167094), ((0.15000000000000002, 0.85000000000000009, 1.0), 18.736624902167094), ((0.10000000000000001, 1.0, 1.0), 18.736624902167094), ((0.050000000000000003, 0.70000000000000007, 1.0), 18.736624902167094), ((0.15000000000000002, 1.0, 1.0), 18.736624902167094), ((0.0, 0.85000000000000009, 1.0), 18.736624902167094), ((0.30000000000000004, 0.75, 1.0), 18.7351736281404), ((0.30000000000000004, 0.85000000000000009, 1.0), 18.7351736281404), ((0.30000000000000004, 0.90000000000000002, 1.0), 18.7351736281404), ((0.30000000000000004, 0.65000000000000002, 1.0), 18.7351736281404), ((0.30000000000000004, 0.70000000000000007, 1.0), 18.7351736281404), ((0.30000000000000004, 1.0, 1.0), 18.7351736281404), ((0.30000000000000004, 0.95000000000000007, 1.0), 18.7351736281404), ((0.30000000000000004, 0.80000000000000004, 1.0), 18.7351736281404), ((0.35000000000000003, 0.80000000000000004, 1.0), 18.732664092283038), ((0.35000000000000003, 0.65000000000000002, 1.0), 18.732664092283038), ((0.35000000000000003, 1.0, 1.0), 18.732664092283038), ((0.35000000000000003, 0.75, 1.0), 18.732664092283038), ((0.35000000000000003, 0.95000000000000007, 1.0), 18.732664092283038), ((0.35000000000000003, 0.85000000000000009, 1.0), 18.732664092283038), ((0.35000000000000003, 0.90000000000000002, 1.0), 18.732664092283038), ((0.35000000000000003, 0.70000000000000007, 1.0), 18.732664092283038), ((0.45000000000000001, 0.75, 1.0), 18.68845837013581), ((0.45000000000000001, 0.85000000000000009, 1.0), 18.68845837013581), ((0.45000000000000001, 0.95000000000000007, 1.0), 18.68845837013581), ((0.45000000000000001, 0.70000000000000007, 1.0), 18.68845837013581), ((0.45000000000000001, 0.90000000000000002, 1.0), 18.68845837013581), ((0.45000000000000001, 0.80000000000000004, 1.0), 18.68845837013581), ((0.45000000000000001, 1.0, 1.0), 18.68845837013581), ((0.40000000000000002, 0.90000000000000002, 1.0), 18.68553826074206), ((0.40000000000000002, 0.75, 1.0), 18.68553826074206), ((0.40000000000000002, 1.0, 1.0), 18.68553826074206), ((0.40000000000000002, 0.70000000000000007, 1.0), 18.68553826074206), ((0.40000000000000002, 0.85000000000000009, 1.0), 18.68553826074206), ((0.40000000000000002, 0.95000000000000007, 1.0), 18.68553826074206), ((0.40000000000000002, 0.80000000000000004, 1.0), 18.68553826074206), ((0.40000000000000002, 0.65000000000000002, 1.0), 18.68553826074206), ((0.35000000000000003, 0.40000000000000002, 1.0), 18.4622666466425), ((0.25, 0.40000000000000002, 1.0), 18.461237889004547), ((0.20000000000000001, 0.40000000000000002, 1.0), 18.460387486838357), ((0.15000000000000002, 0.40000000000000002, 1.0), 18.459551688775342), ((0.0, 0.40000000000000002, 1.0), 18.459551688775342), ((0.10000000000000001, 0.40000000000000002, 1.0), 18.459551688775342), ((0.050000000000000003, 0.40000000000000002, 1.0), 18.459551688775342), ((0.30000000000000004, 0.40000000000000002, 1.0), 18.458948178096414), ((0.40000000000000002, 0.45000000000000001, 1.0), 18.42295452701683), ((0.85000000000000009, 1.0, 1.0), 18.41503856224597), ((0.75, 0.85000000000000009, 1.0), 18.41503856224597), ((0.65000000000000002, 0.75, 1.0), 18.41503856224597), ((0.85000000000000009, 0.90000000000000002, 1.0), 18.41503856224597), ((0.70000000000000007, 1.0, 1.0), 18.41503856224597), ((0.70000000000000007, 0.75, 1.0), 18.41503856224597), ((0.65000000000000002, 0.80000000000000004, 1.0), 18.41503856224597), ((0.60000000000000009, 0.70000000000000007, 1.0), 18.41503856224597), ((0.70000000000000007, 0.80000000000000004, 1.0), 18.41503856224597), ((0.80000000000000004, 0.95000000000000007, 1.0), 18.41503856224597), ((0.70000000000000007, 0.90000000000000002, 1.0), 18.41503856224597), ((0.60000000000000009, 0.80000000000000004, 1.0), 18.41503856224597), ((0.75, 0.95000000000000007, 1.0), 18.41503856224597), ((0.75, 0.90000000000000002, 1.0), 18.41503856224597), ((0.60000000000000009, 0.65000000000000002, 1.0), 18.41503856224597), ((0.90000000000000002, 1.0, 1.0), 18.41503856224597), ((0.65000000000000002, 0.85000000000000009, 1.0), 18.41503856224597), ((0.75, 0.80000000000000004, 1.0), 18.41503856224597), ((0.85000000000000009, 0.95000000000000007, 1.0), 18.41503856224597), ((0.60000000000000009, 0.75, 1.0), 18.41503856224597), ((0.70000000000000007, 0.85000000000000009, 1.0), 18.41503856224597), ((0.80000000000000004, 0.90000000000000002, 1.0), 18.41503856224597), ((0.80000000000000004, 0.85000000000000009, 1.0), 18.41503856224597), ((0.90000000000000002, 0.95000000000000007, 1.0), 18.41503856224597), ((0.75, 1.0, 1.0), 18.41503856224597), ((0.70000000000000007, 0.95000000000000007, 1.0), 18.41503856224597), ((0.80000000000000004, 1.0, 1.0), 18.41503856224597), ((0.65000000000000002, 0.70000000000000007, 1.0), 18.41503856224597), ((0.55000000000000004, 0.60000000000000009, 1.0), 18.413050592871514), ((0.5, 0.55000000000000004, 1.0), 18.41243195697994), ((0.45000000000000001, 0.5, 1.0), 18.403538824678286), ((0.55000000000000004, 0.65000000000000002, 1.0), 18.39797657356975), ((0.5, 0.60000000000000009, 1.0), 18.39735793767818), ((0.55000000000000004, 0.70000000000000007, 1.0), 18.39414918402808), ((0.55000000000000004, 0.75, 1.0), 18.39414918402808), ((0.5, 0.65000000000000002, 1.0), 18.389898817666538), ((0.5, 0.70000000000000007, 1.0), 18.386670612804345), ((0.25, 0.45000000000000001, 1.0), 18.37769808494326), ((0.20000000000000001, 0.45000000000000001, 1.0), 18.376847682777065), ((0.10000000000000001, 0.45000000000000001, 1.0), 18.376011884714053), ((0.15000000000000002, 0.45000000000000001, 1.0), 18.376011884714053), ((0.0, 0.45000000000000001, 1.0), 18.376011884714053), ((0.050000000000000003, 0.45000000000000001, 1.0), 18.376011884714053), ((0.30000000000000004, 0.45000000000000001, 1.0), 18.374560610687364), ((0.35000000000000003, 0.45000000000000001, 1.0), 18.372051074830004), ((0.25, 0.5, 1.0), 18.356565606097174), ((0.20000000000000001, 0.5, 1.0), 18.355715203930984), ((0.15000000000000002, 0.5, 1.0), 18.35487940586797), ((0.0, 0.5, 1.0), 18.35487940586797), ((0.10000000000000001, 0.5, 1.0), 18.35487940586797), ((0.050000000000000003, 0.5, 1.0), 18.35487940586797), ((0.30000000000000004, 0.5, 1.0), 18.35342813184128), ((0.35000000000000003, 0.5, 1.0), 18.35091859598392)], [((0.45000000000000001, 0.60000000000000009, 1.0), 18.66018514521009), ((0.45000000000000001, 0.75, 1.0), 18.652970137995084), ((0.45000000000000001, 0.85000000000000009, 1.0), 18.652970137995084), ((0.45000000000000001, 0.95000000000000007, 1.0), 18.652970137995084), ((0.45000000000000001, 0.70000000000000007, 1.0), 18.652970137995084), ((0.45000000000000001, 0.65000000000000002, 1.0), 18.652970137995084), ((0.45000000000000001, 0.90000000000000002, 1.0), 18.652970137995084), ((0.45000000000000001, 0.80000000000000004, 1.0), 18.652970137995084), ((0.45000000000000001, 1.0, 1.0), 18.652970137995084), ((0.40000000000000002, 0.55000000000000004, 1.0), 18.617686070860138), ((0.40000000000000002, 0.90000000000000002, 1.0), 18.60729892788424), ((0.40000000000000002, 0.75, 1.0), 18.60729892788424), ((0.40000000000000002, 1.0, 1.0), 18.60729892788424), ((0.40000000000000002, 0.60000000000000009, 1.0), 18.60729892788424), ((0.40000000000000002, 0.70000000000000007, 1.0), 18.60729892788424), ((0.40000000000000002, 0.85000000000000009, 1.0), 18.60729892788424), ((0.40000000000000002, 0.95000000000000007, 1.0), 18.60729892788424), ((0.40000000000000002, 0.80000000000000004, 1.0), 18.60729892788424), ((0.40000000000000002, 0.65000000000000002, 1.0), 18.60729892788424), ((0.45000000000000001, 0.55000000000000004, 1.0), 18.600201425226366), ((0.30000000000000004, 0.40000000000000002, 1.0), 18.587824963515754)], []], 'Zoo': [[], [], [((0.0, 0.80000000000000004, 1.0), 16.133847146567287), ((0.15000000000000002, 0.80000000000000004, 1.0), 16.133847146567287), ((0.10000000000000001, 0.80000000000000004, 1.0), 16.133847146567287), ((0.050000000000000003, 0.80000000000000004, 1.0), 16.133847146567287), ((0.20000000000000001, 0.80000000000000004, 1.0), 16.126023884708935), ((0.25, 0.80000000000000004, 1.0), 16.121899953085006), ((0.30000000000000004, 0.80000000000000004, 1.0), 16.120863333758912), ((0.10000000000000001, 0.55000000000000004, 1.0), 16.11798948017167), ((0.0, 0.55000000000000004, 1.0), 16.11798948017167), ((0.050000000000000003, 0.55000000000000004, 1.0), 16.11798948017167), ((0.15000000000000002, 0.55000000000000004, 1.0), 16.11798948017167), ((0.20000000000000001, 0.55000000000000004, 1.0), 16.110166218313324), ((0.0, 0.90000000000000002, 1.0), 16.109402702122843), ((0.050000000000000003, 0.85000000000000009, 1.0), 16.109402702122843), ((0.15000000000000002, 0.90000000000000002, 1.0), 16.109402702122843), ((0.10000000000000001, 0.90000000000000002, 1.0), 16.109402702122843), ((0.10000000000000001, 0.85000000000000009, 1.0), 16.109402702122843), ((0.050000000000000003, 0.90000000000000002, 1.0), 16.109402702122843), ((0.15000000000000002, 0.85000000000000009, 1.0), 16.109402702122843), ((0.0, 0.85000000000000009, 1.0), 16.109402702122843), ((0.25, 0.55000000000000004, 1.0), 16.10604228668939), ((0.30000000000000004, 0.55000000000000004, 1.0), 16.105005667363297), ((0.20000000000000001, 0.85000000000000009, 1.0), 16.101579440264494), ((0.20000000000000001, 0.90000000000000002, 1.0), 16.101579440264494), ((0.25, 0.90000000000000002, 1.0), 16.097455508640557), ((0.25, 0.85000000000000009, 1.0), 16.097455508640557), ((0.30000000000000004, 0.85000000000000009, 1.0), 16.096418889314467), ((0.30000000000000004, 0.90000000000000002, 1.0), 16.096418889314467), ((0.35000000000000003, 0.80000000000000004, 1.0), 16.09249095527745), ((0.35000000000000003, 0.55000000000000004, 1.0), 16.076623715242302), ((0.10000000000000001, 0.75, 1.0), 16.073976776196915), ((0.0, 0.75, 1.0), 16.073976776196915), ((0.050000000000000003, 0.75, 1.0), 16.073976776196915), ((0.15000000000000002, 0.75, 1.0), 16.073976776196915), ((0.45000000000000001, 0.85000000000000009, 1.0), 16.069675957953393), ((0.35000000000000003, 0.85000000000000009, 1.0), 16.068046510833007), ((0.35000000000000003, 0.90000000000000002, 1.0), 16.068046510833007), ((0.20000000000000001, 0.75, 1.0), 16.066153514338566), ((0.25, 0.75, 1.0), 16.062029582714633), ((0.30000000000000004, 0.75, 1.0), 16.060992963388543), ((0.10000000000000001, 0.95000000000000007, 1.0), 16.05940270212284), ((0.15000000000000002, 0.95000000000000007, 1.0), 16.05940270212284), ((0.0, 0.95000000000000007, 1.0), 16.05940270212284), ((0.050000000000000003, 1.0, 1.0), 16.05940270212284), ((0.050000000000000003, 0.95000000000000007, 1.0), 16.05940270212284), ((0.0, 1.0, 1.0), 16.05940270212284), ((0.10000000000000001, 1.0, 1.0), 16.05940270212284), ((0.15000000000000002, 1.0, 1.0), 16.05940270212284), ((0.20000000000000001, 1.0, 1.0), 16.051579440264494), ((0.20000000000000001, 0.95000000000000007, 1.0), 16.051579440264494), ((0.25, 0.95000000000000007, 1.0), 16.04745550864056), ((0.25, 1.0, 1.0), 16.04745550864056), ((0.30000000000000004, 1.0, 1.0), 16.04641888931447), ((0.30000000000000004, 0.95000000000000007, 1.0), 16.04641888931447), ((0.45000000000000001, 0.90000000000000002, 1.0), 16.045231513508945), ((0.15000000000000002, 0.5, 1.0), 16.037890603193667), ((0.0, 0.5, 1.0), 16.037890603193667), ((0.10000000000000001, 0.5, 1.0), 16.037890603193667), ((0.050000000000000003, 0.5, 1.0), 16.037890603193667), ((0.050000000000000003, 0.60000000000000009, 1.0), 16.035856742018403), ((0.15000000000000002, 0.60000000000000009, 1.0), 16.035856742018403), ((0.10000000000000001, 0.60000000000000009, 1.0), 16.035856742018403), ((0.0, 0.60000000000000009, 1.0), 16.035856742018403), ((0.35000000000000003, 0.75, 1.0), 16.03262058490708), ((0.20000000000000001, 0.5, 1.0), 16.030067341335315), ((0.20000000000000001, 0.60000000000000009, 1.0), 16.02803348016005), ((0.25, 0.5, 1.0), 16.02594340971138), ((0.40000000000000002, 0.55000000000000004, 1.0), 16.025001974826477), ((0.30000000000000004, 0.5, 1.0), 16.024906790385288), ((0.25, 0.60000000000000009, 1.0), 16.02390954853612), ((0.30000000000000004, 0.60000000000000009, 1.0), 16.022872929210028), ((0.35000000000000003, 1.0, 1.0), 16.018046510833006), ((0.35000000000000003, 0.95000000000000007, 1.0), 16.018046510833006), ((0.15000000000000002, 0.70000000000000007, 1.0), 16.013152193372335), ((0.10000000000000001, 0.70000000000000007, 1.0), 16.013152193372335), ((0.0, 0.70000000000000007, 1.0), 16.013152193372335), ((0.050000000000000003, 0.70000000000000007, 1.0), 16.013152193372335), ((0.45000000000000001, 0.80000000000000004, 1.0), 16.010972254249687), ((0.20000000000000001, 0.70000000000000007, 1.0), 16.005328931513983), ((0.25, 0.70000000000000007, 1.0), 16.00120499989005), ((0.30000000000000004, 0.70000000000000007, 1.0), 16.000168380563956), ((0.35000000000000003, 0.5, 1.0), 15.996524838264296), ((0.45000000000000001, 0.95000000000000007, 1.0), 15.995231513508948), ((0.45000000000000001, 1.0, 1.0), 15.995231513508948), ((0.35000000000000003, 0.60000000000000009, 1.0), 15.994500550728567), ((0.5, 0.90000000000000002, 1.0), 15.983199880828789), ((0.40000000000000002, 0.80000000000000004, 1.0), 15.979880605783904), ((0.35000000000000003, 0.70000000000000007, 1.0), 15.971796002082499), ((0.40000000000000002, 0.90000000000000002, 1.0), 15.955436161339458), ((0.40000000000000002, 0.85000000000000009, 1.0), 15.955436161339458), ((0.050000000000000003, 0.65000000000000002, 1.0), 15.952598900545228), ((0.0, 0.65000000000000002, 1.0), 15.952598900545228), ((0.10000000000000001, 0.65000000000000002, 1.0), 15.952598900545228), ((0.15000000000000002, 0.65000000000000002, 1.0), 15.952598900545228), ((0.20000000000000001, 0.65000000000000002, 1.0), 15.944775638686881), ((0.25, 0.65000000000000002, 1.0), 15.940651707062948), ((0.30000000000000004, 0.65000000000000002, 1.0), 15.939615087736854), ((0.40000000000000002, 0.5, 1.0), 15.932558456654087), ((0.5, 0.85000000000000009, 1.0), 15.929125806754712), ((0.45000000000000001, 0.75, 1.0), 15.926465131742564), ((0.40000000000000002, 0.75, 1.0), 15.915380605783902), ((0.35000000000000003, 0.65000000000000002, 1.0), 15.911242709255395), ((0.45000000000000001, 0.65000000000000002, 1.0), 15.909397528192924), ((0.5, 0.95000000000000007, 1.0), 15.908755436384345), ((0.5, 1.0, 1.0), 15.908755436384345), ((0.40000000000000002, 1.0, 1.0), 15.905436161339457), ((0.40000000000000002, 0.95000000000000007, 1.0), 15.905436161339457), ((0.45000000000000001, 0.55000000000000004, 1.0), 15.897496917925231), ((0.45000000000000001, 0.60000000000000009, 1.0), 15.894308181114782), ((0.55000000000000004, 0.95000000000000007, 1.0), 15.874664921431929), ((0.55000000000000004, 0.90000000000000002, 1.0), 15.870590847357857), ((0.5, 0.70000000000000007, 1.0), 15.86388137444888), ((0.45000000000000001, 0.70000000000000007, 1.0), 15.862025626215758), ((0.5, 0.80000000000000004, 1.0), 15.856014695643603), ((0.40000000000000002, 0.70000000000000007, 1.0), 15.850354964758262), ((0.55000000000000004, 1.0, 1.0), 15.850220476987483), ((0.5, 0.60000000000000009, 1.0), 15.84804580273817), ((0.40000000000000002, 0.60000000000000009, 1.0), 15.840995362308657), ((0.5, 0.65000000000000002, 1.0), 15.826120296432245), ((0.5, 0.75, 1.0), 15.801262558891466), ((0.55000000000000004, 0.85000000000000009, 1.0), 15.797479736246743), ((0.55000000000000004, 0.75, 1.0), 15.794629416001687), ((0.55000000000000004, 0.70000000000000007, 1.0), 15.793886081558352), ((0.55000000000000004, 0.65000000000000002, 1.0), 15.78952742615792), ((0.40000000000000002, 0.65000000000000002, 1.0), 15.756973650599171), ((0.55000000000000004, 0.80000000000000004, 1.0), 15.74614640291341), ((0.10000000000000001, 0.45000000000000001, 1.0), 15.198724196047007), ((0.15000000000000002, 0.45000000000000001, 1.0), 15.198724196047007), ((0.0, 0.45000000000000001, 1.0), 15.198724196047007), ((0.050000000000000003, 0.45000000000000001, 1.0), 15.198724196047007), ((0.20000000000000001, 0.45000000000000001, 1.0), 15.190900934188658), ((0.25, 0.45000000000000001, 1.0), 15.186777002564728), ((0.30000000000000004, 0.45000000000000001, 1.0), 15.185740383238635), ((0.35000000000000003, 0.45000000000000001, 1.0), 15.157358431117641), ((0.60000000000000009, 1.0, 1.0), 15.134712322969246), ((0.40000000000000002, 0.45000000000000001, 1.0), 15.121190643146209), ((0.45000000000000001, 0.5, 1.0), 15.113121098849993), ((0.5, 0.55000000000000004, 1.0), 15.089404411034163), ((0.60000000000000009, 0.95000000000000007, 1.0), 15.080638248895172), ((0.60000000000000009, 0.90000000000000002, 1.0), 15.074193804450728), ((0.60000000000000009, 0.80000000000000004, 1.0), 15.073109241190725), ((0.60000000000000009, 0.65000000000000002, 1.0), 15.068649441012548), ((0.60000000000000009, 0.70000000000000007, 1.0), 15.066615281679509), ((0.60000000000000009, 0.75, 1.0), 15.064655469036955), ((0.55000000000000004, 0.60000000000000009, 1.0), 15.041725395602647), ((0.60000000000000009, 0.85000000000000009, 1.0), 15.022860471117394)]], 'BeachTrip': [[((0.5, 0.60000000000000009, 1.0), 19.369223640079113), ((0.45000000000000001, 0.60000000000000009, 1.0), 19.32016551723274), ((0.5, 0.65000000000000002, 1.0), 19.298879297990116), ((0.40000000000000002, 0.60000000000000009, 1.0), 19.261038533105758), ((0.35000000000000003, 0.60000000000000009, 1.0), 19.257863929931155), ((0.45000000000000001, 0.65000000000000002, 1.0), 19.24253009164091), ((0.55000000000000004, 0.65000000000000002, 1.0), 19.2266555820588), ((0.55000000000000004, 0.60000000000000009, 1.0), 19.2042764392076)], [((0.40000000000000002, 0.75, 1.0), 17.632379841406557), ((0.30000000000000004, 0.75, 1.0), 17.632379841406557), ((0.10000000000000001, 0.75, 1.0), 17.632379841406557), ((0.25, 0.75, 1.0), 17.632379841406557), ((0.20000000000000001, 0.75, 1.0), 17.632379841406557), ((0.35000000000000003, 0.75, 1.0), 17.632379841406557), ((0.0, 0.75, 1.0), 17.632379841406557), ((0.050000000000000003, 0.75, 1.0), 17.632379841406557), ((0.15000000000000002, 0.75, 1.0), 17.632379841406557), ((0.45000000000000001, 0.75, 1.0), 17.63178693837463), ((0.15000000000000002, 0.70000000000000007, 1.0), 17.629430735234457), ((0.10000000000000001, 0.70000000000000007, 1.0), 17.629430735234457), ((0.40000000000000002, 0.70000000000000007, 1.0), 17.629430735234457), ((0.25, 0.70000000000000007, 1.0), 17.629430735234457), ((0.30000000000000004, 0.70000000000000007, 1.0), 17.629430735234457), ((0.0, 0.70000000000000007, 1.0), 17.629430735234457), ((0.35000000000000003, 0.70000000000000007, 1.0), 17.629430735234457), ((0.20000000000000001, 0.70000000000000007, 1.0), 17.629430735234457), ((0.050000000000000003, 0.70000000000000007, 1.0), 17.629430735234457), ((0.45000000000000001, 0.70000000000000007, 1.0), 17.62895808960052), ((0.5, 0.75, 1.0), 17.6285433257929), ((0.5, 0.70000000000000007, 1.0), 17.623670107180505), ((0.55000000000000004, 0.70000000000000007, 1.0), 17.623670107180505), ((0.55000000000000004, 0.75, 1.0), 17.62325534337288), ((0.60000000000000009, 0.75, 1.0), 17.62325534337288), ((0.35000000000000003, 0.80000000000000004, 1.0), 17.558354754705512), ((0.20000000000000001, 0.80000000000000004, 1.0), 17.558354754705512), ((0.0, 0.80000000000000004, 1.0), 17.558354754705512), ((0.25, 0.80000000000000004, 1.0), 17.558354754705512), ((0.15000000000000002, 0.80000000000000004, 1.0), 17.558354754705512), ((0.10000000000000001, 0.80000000000000004, 1.0), 17.558354754705512), ((0.050000000000000003, 0.80000000000000004, 1.0), 17.558354754705512), ((0.45000000000000001, 0.80000000000000004, 1.0), 17.558354754705512), ((0.40000000000000002, 0.80000000000000004, 1.0), 17.558354754705512), ((0.30000000000000004, 0.80000000000000004, 1.0), 17.558354754705512), ((0.5, 0.80000000000000004, 1.0), 17.554248996291115), ((0.55000000000000004, 0.80000000000000004, 1.0), 17.553442225323053), ((0.65000000000000002, 0.80000000000000004, 1.0), 17.548154242903035), ((0.60000000000000009, 0.80000000000000004, 1.0), 17.548154242903035), ((0.60000000000000009, 0.70000000000000007, 1.0), 17.53221971729261), ((0.65000000000000002, 0.75, 1.0), 17.531804953484993), ((0.050000000000000003, 0.65000000000000002, 1.0), 17.491345678457407), ((0.0, 0.65000000000000002, 1.0), 17.491345678457407), ((0.35000000000000003, 0.65000000000000002, 1.0), 17.491345678457407), ((0.10000000000000001, 0.65000000000000002, 1.0), 17.491345678457407), ((0.30000000000000004, 0.65000000000000002, 1.0), 17.491345678457407), ((0.20000000000000001, 0.65000000000000002, 1.0), 17.491345678457407), ((0.15000000000000002, 0.65000000000000002, 1.0), 17.491345678457407), ((0.25, 0.65000000000000002, 1.0), 17.491345678457407), ((0.40000000000000002, 0.65000000000000002, 1.0), 17.491345678457407), ((0.5, 0.65000000000000002, 1.0), 17.49064940503386), ((0.45000000000000001, 0.65000000000000002, 1.0), 17.49064940503386), ((0.55000000000000004, 0.65000000000000002, 1.0), 17.477430307662978), ((0.30000000000000004, 0.85000000000000009, 1.0), 17.474817429142963), ((0.050000000000000003, 0.85000000000000009, 1.0), 17.474817429142963), ((0.45000000000000001, 0.85000000000000009, 1.0), 17.474817429142963), ((0.40000000000000002, 0.85000000000000009, 1.0), 17.474817429142963), ((0.20000000000000001, 0.85000000000000009, 1.0), 17.474817429142963), ((0.35000000000000003, 0.85000000000000009, 1.0), 17.474817429142963), ((0.10000000000000001, 0.85000000000000009, 1.0), 17.474817429142963), ((0.25, 0.85000000000000009, 1.0), 17.474817429142963), ((0.15000000000000002, 0.85000000000000009, 1.0), 17.474817429142963), ((0.0, 0.85000000000000009, 1.0), 17.474817429142963), ((0.0, 0.90000000000000002, 1.0), 17.473244733629826), ((0.40000000000000002, 0.90000000000000002, 1.0), 17.473244733629826), ((0.10000000000000001, 0.95000000000000007, 1.0), 17.473244733629826), ((0.40000000000000002, 1.0, 1.0), 17.473244733629826), ((0.35000000000000003, 1.0, 1.0), 17.473244733629826), ((0.30000000000000004, 0.90000000000000002, 1.0), 17.473244733629826), ((0.25, 0.95000000000000007, 1.0), 17.473244733629826), ((0.15000000000000002, 0.90000000000000002, 1.0), 17.473244733629826), ((0.10000000000000001, 0.90000000000000002, 1.0), 17.473244733629826), ((0.45000000000000001, 0.95000000000000007, 1.0), 17.473244733629826), ((0.15000000000000002, 0.95000000000000007, 1.0), 17.473244733629826), ((0.0, 0.95000000000000007, 1.0), 17.473244733629826), ((0.25, 1.0, 1.0), 17.473244733629826), ((0.35000000000000003, 0.95000000000000007, 1.0), 17.473244733629826), ((0.20000000000000001, 1.0, 1.0), 17.473244733629826), ((0.40000000000000002, 0.95000000000000007, 1.0), 17.473244733629826), ((0.050000000000000003, 1.0, 1.0), 17.473244733629826), ((0.35000000000000003, 0.90000000000000002, 1.0), 17.473244733629826), ((0.050000000000000003, 0.95000000000000007, 1.0), 17.473244733629826), ((0.0, 1.0, 1.0), 17.473244733629826), ((0.45000000000000001, 0.90000000000000002, 1.0), 17.473244733629826), ((0.25, 0.90000000000000002, 1.0), 17.473244733629826), ((0.30000000000000004, 1.0, 1.0), 17.473244733629826), ((0.20000000000000001, 0.95000000000000007, 1.0), 17.473244733629826), ((0.20000000000000001, 0.90000000000000002, 1.0), 17.473244733629826), ((0.050000000000000003, 0.90000000000000002, 1.0), 17.473244733629826), ((0.30000000000000004, 0.95000000000000007, 1.0), 17.473244733629826), ((0.10000000000000001, 1.0, 1.0), 17.473244733629826), ((0.15000000000000002, 1.0, 1.0), 17.473244733629826), ((0.45000000000000001, 1.0, 1.0), 17.473244733629826), ((0.5, 0.85000000000000009, 1.0), 17.471051529323926), ((0.5, 0.90000000000000002, 1.0), 17.46962462234342), ((0.5, 0.95000000000000007, 1.0), 17.46962462234342), ((0.5, 1.0, 1.0), 17.46962462234342), ((0.55000000000000004, 0.85000000000000009, 1.0), 17.46936172395934), ((0.60000000000000009, 0.85000000000000009, 1.0), 17.468554952991283), ((0.55000000000000004, 1.0, 1.0), 17.46847878520573), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.46847878520573), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.46847878520573), ((0.70000000000000007, 0.75, 1.0), 17.468271801031385), ((0.65000000000000002, 0.70000000000000007, 1.0), 17.467952923480365), ((0.60000000000000009, 0.95000000000000007, 1.0), 17.46760069164471), ((0.60000000000000009, 0.90000000000000002, 1.0), 17.46760069164471), ((0.60000000000000009, 1.0, 1.0), 17.46760069164471), ((0.65000000000000002, 0.85000000000000009, 1.0), 17.46326697057126), ((0.65000000000000002, 1.0, 1.0), 17.4625363370143), ((0.65000000000000002, 0.90000000000000002, 1.0), 17.4625363370143), ((0.65000000000000002, 0.95000000000000007, 1.0), 17.4625363370143), ((0.75, 0.95000000000000007, 1.0), 17.457554874819767), ((0.75, 0.90000000000000002, 1.0), 17.457554874819767), ((0.75, 1.0, 1.0), 17.457554874819767), ((0.75, 0.80000000000000004, 1.0), 17.456058839776613), ((0.70000000000000007, 0.80000000000000004, 1.0), 17.44816342080165), ((0.70000000000000007, 1.0, 1.0), 17.444725038392207), ((0.70000000000000007, 0.90000000000000002, 1.0), 17.444725038392207), ((0.70000000000000007, 0.95000000000000007, 1.0), 17.444725038392207), ((0.70000000000000007, 0.85000000000000009, 1.0), 17.44108100741301), ((0.050000000000000003, 0.60000000000000009, 1.0), 17.42755416693862), ((0.40000000000000002, 0.60000000000000009, 1.0), 17.42755416693862), ((0.15000000000000002, 0.60000000000000009, 1.0), 17.42755416693862), ((0.45000000000000001, 0.60000000000000009, 1.0), 17.42755416693862), ((0.20000000000000001, 0.60000000000000009, 1.0), 17.42755416693862), ((0.10000000000000001, 0.60000000000000009, 1.0), 17.42755416693862), ((0.30000000000000004, 0.60000000000000009, 1.0), 17.42755416693862), ((0.0, 0.60000000000000009, 1.0), 17.42755416693862), ((0.35000000000000003, 0.60000000000000009, 1.0), 17.42755416693862), ((0.25, 0.60000000000000009, 1.0), 17.42755416693862), ((0.5, 0.60000000000000009, 1.0), 17.414335069567738), ((0.60000000000000009, 0.65000000000000002, 1.0), 17.41316351385073), ((0.55000000000000004, 0.60000000000000009, 1.0), 17.41316351385073)], []], 'PersonalMusicActivity': [[((0.55000000000000004, 0.80000000000000004, 1.0), 16.18038813131797), ((0.55000000000000004, 1.0, 1.0), 16.169746596926434), ((0.55000000000000004, 0.95000000000000007, 1.0), 16.169746596926434), ((0.55000000000000004, 0.90000000000000002, 1.0), 16.169746596926434), ((0.55000000000000004, 0.75, 1.0), 16.166730723910565), ((0.55000000000000004, 0.85000000000000009, 1.0), 16.08502437470421), ((0.5, 0.70000000000000007, 1.0), 16.0250223029943), ((0.40000000000000002, 0.65000000000000002, 1.0), 15.866903460221968), ((0.30000000000000004, 0.65000000000000002, 1.0), 15.853226168406968), ((0.35000000000000003, 0.65000000000000002, 1.0), 15.85155643616658), ((0.45000000000000001, 0.65000000000000002, 1.0), 15.825875567422337), ((0.40000000000000002, 0.70000000000000007, 1.0), 15.815206043881695), ((0.050000000000000003, 0.65000000000000002, 1.0), 15.804900275549825), ((0.0, 0.65000000000000002, 1.0), 15.804900275549825), ((0.10000000000000001, 0.65000000000000002, 1.0), 15.804900275549825), ((0.20000000000000001, 0.65000000000000002, 1.0), 15.804900275549825), ((0.15000000000000002, 0.65000000000000002, 1.0), 15.804900275549825), ((0.25, 0.65000000000000002, 1.0), 15.804900275549825), ((0.30000000000000004, 0.70000000000000007, 1.0), 15.801528752066694), ((0.35000000000000003, 0.70000000000000007, 1.0), 15.799859019826311), ((0.5, 0.90000000000000002, 1.0), 15.797869482270054), ((0.5, 0.95000000000000007, 1.0), 15.797869482270054), ((0.5, 1.0, 1.0), 15.797869482270054), ((0.45000000000000001, 0.70000000000000007, 1.0), 15.774178151082062), ((0.15000000000000002, 0.70000000000000007, 1.0), 15.75320285920955), ((0.10000000000000001, 0.70000000000000007, 1.0), 15.75320285920955), ((0.25, 0.70000000000000007, 1.0), 15.75320285920955), ((0.0, 0.70000000000000007, 1.0), 15.75320285920955), ((0.20000000000000001, 0.70000000000000007, 1.0), 15.75320285920955), ((0.050000000000000003, 0.70000000000000007, 1.0), 15.75320285920955), ((0.5, 0.75, 1.0), 15.71689726004783), ((0.5, 0.85000000000000009, 1.0), 15.682591704492276), ((0.5, 0.80000000000000004, 1.0), 15.653425037825608), ((0.60000000000000009, 0.70000000000000007, 1.0), 15.49371599276558), ((0.45000000000000001, 0.95000000000000007, 1.0), 15.461388405554219), ((0.45000000000000001, 0.90000000000000002, 1.0), 15.461388405554219), ((0.45000000000000001, 1.0, 1.0), 15.461388405554219), ((0.40000000000000002, 0.90000000000000002, 1.0), 15.454576792181015), ((0.40000000000000002, 1.0, 1.0), 15.454576792181015), ((0.40000000000000002, 0.95000000000000007, 1.0), 15.454576792181015), ((0.30000000000000004, 0.90000000000000002, 1.0), 15.440899500366015), ((0.30000000000000004, 1.0, 1.0), 15.440899500366015), ((0.30000000000000004, 0.95000000000000007, 1.0), 15.440899500366015), ((0.35000000000000003, 1.0, 1.0), 15.439229768125628), ((0.35000000000000003, 0.95000000000000007, 1.0), 15.439229768125628), ((0.35000000000000003, 0.90000000000000002, 1.0), 15.439229768125628), ((0.0, 0.90000000000000002, 1.0), 15.39257360750887), ((0.10000000000000001, 0.95000000000000007, 1.0), 15.39257360750887), ((0.25, 0.95000000000000007, 1.0), 15.39257360750887), ((0.15000000000000002, 0.90000000000000002, 1.0), 15.39257360750887), ((0.10000000000000001, 0.90000000000000002, 1.0), 15.39257360750887), ((0.15000000000000002, 0.95000000000000007, 1.0), 15.39257360750887), ((0.0, 0.95000000000000007, 1.0), 15.39257360750887), ((0.25, 1.0, 1.0), 15.39257360750887), ((0.20000000000000001, 1.0, 1.0), 15.39257360750887), ((0.050000000000000003, 1.0, 1.0), 15.39257360750887), ((0.050000000000000003, 0.95000000000000007, 1.0), 15.39257360750887), ((0.0, 1.0, 1.0), 15.39257360750887), ((0.25, 0.90000000000000002, 1.0), 15.39257360750887), ((0.20000000000000001, 0.95000000000000007, 1.0), 15.39257360750887), ((0.20000000000000001, 0.90000000000000002, 1.0), 15.39257360750887), ((0.050000000000000003, 0.90000000000000002, 1.0), 15.39257360750887), ((0.10000000000000001, 1.0, 1.0), 15.39257360750887), ((0.15000000000000002, 1.0, 1.0), 15.39257360750887), ((0.45000000000000001, 0.85000000000000009, 1.0), 15.346110627776442), ((0.40000000000000002, 0.85000000000000009, 1.0), 15.339299014403235), ((0.30000000000000004, 0.85000000000000009, 1.0), 15.325621722588235), ((0.35000000000000003, 0.85000000000000009, 1.0), 15.323951990347851), ((0.45000000000000001, 0.80000000000000004, 1.0), 15.316943961109773), ((0.45000000000000001, 0.75, 1.0), 15.315832849998664), ((0.40000000000000002, 0.80000000000000004, 1.0), 15.310132347736571), ((0.40000000000000002, 0.75, 1.0), 15.30902123662546), ((0.30000000000000004, 0.80000000000000004, 1.0), 15.29645505592157), ((0.30000000000000004, 0.75, 1.0), 15.295343944810458), ((0.35000000000000003, 0.80000000000000004, 1.0), 15.294785323681184), ((0.35000000000000003, 0.75, 1.0), 15.293674212570073), ((0.050000000000000003, 0.85000000000000009, 1.0), 15.277295829731093), ((0.20000000000000001, 0.85000000000000009, 1.0), 15.277295829731093), ((0.10000000000000001, 0.85000000000000009, 1.0), 15.277295829731093), ((0.25, 0.85000000000000009, 1.0), 15.277295829731093), ((0.15000000000000002, 0.85000000000000009, 1.0), 15.277295829731093), ((0.0, 0.85000000000000009, 1.0), 15.277295829731093), ((0.20000000000000001, 0.80000000000000004, 1.0), 15.248129163064425), ((0.0, 0.80000000000000004, 1.0), 15.248129163064425), ((0.25, 0.80000000000000004, 1.0), 15.248129163064425), ((0.15000000000000002, 0.80000000000000004, 1.0), 15.248129163064425), ((0.10000000000000001, 0.80000000000000004, 1.0), 15.248129163064425), ((0.050000000000000003, 0.80000000000000004, 1.0), 15.248129163064425), ((0.10000000000000001, 0.75, 1.0), 15.247018051953315), ((0.25, 0.75, 1.0), 15.247018051953315), ((0.20000000000000001, 0.75, 1.0), 15.247018051953315), ((0.0, 0.75, 1.0), 15.247018051953315), ((0.050000000000000003, 0.75, 1.0), 15.247018051953315), ((0.15000000000000002, 0.75, 1.0), 15.247018051953315), ((0.55000000000000004, 0.65000000000000002, 1.0), 15.19068638024717), ((0.55000000000000004, 0.70000000000000007, 1.0), 15.121591405556957), ((0.60000000000000009, 0.85000000000000009, 1.0), 15.107787862254114), ((0.60000000000000009, 0.80000000000000004, 1.0), 15.094130454846706), ((0.60000000000000009, 0.75, 1.0), 15.083204528920781), ((0.60000000000000009, 0.95000000000000007, 1.0), 15.012424105640356), ((0.60000000000000009, 0.90000000000000002, 1.0), 15.012424105640356), ((0.60000000000000009, 1.0, 1.0), 15.012424105640356), ((0.5, 0.65000000000000002, 1.0), 14.89600491826263), ((0.5, 0.60000000000000009, 1.0), 14.889480697637428), ((0.60000000000000009, 0.65000000000000002, 1.0), 14.880142909167985)], [((0.60000000000000009, 0.70000000000000007, 1.0), 18.59109361893929), ((0.60000000000000009, 0.80000000000000004, 1.0), 18.459481280470424), ((0.60000000000000009, 0.85000000000000009, 1.0), 18.454353075342222), ((0.60000000000000009, 0.95000000000000007, 1.0), 18.454353075342222), ((0.60000000000000009, 0.90000000000000002, 1.0), 18.454353075342222), ((0.60000000000000009, 1.0, 1.0), 18.454353075342222), ((0.15000000000000002, 0.70000000000000007, 1.0), 18.412520738868114), ((0.10000000000000001, 0.70000000000000007, 1.0), 18.412520738868114), ((0.25, 0.70000000000000007, 1.0), 18.412520738868114), ((0.30000000000000004, 0.70000000000000007, 1.0), 18.412520738868114), ((0.0, 0.70000000000000007, 1.0), 18.412520738868114), ((0.20000000000000001, 0.70000000000000007, 1.0), 18.412520738868114), ((0.050000000000000003, 0.70000000000000007, 1.0), 18.412520738868114), ((0.35000000000000003, 0.70000000000000007, 1.0), 18.380027098249474), ((0.20000000000000001, 0.80000000000000004, 1.0), 18.263481142825704), ((0.0, 0.80000000000000004, 1.0), 18.263481142825704), ((0.25, 0.80000000000000004, 1.0), 18.263481142825704), ((0.15000000000000002, 0.80000000000000004, 1.0), 18.263481142825704), ((0.10000000000000001, 0.80000000000000004, 1.0), 18.263481142825704), ((0.050000000000000003, 0.80000000000000004, 1.0), 18.263481142825704), ((0.30000000000000004, 0.80000000000000004, 1.0), 18.263481142825704), ((0.0, 0.90000000000000002, 1.0), 18.254709213001142), ((0.10000000000000001, 0.95000000000000007, 1.0), 18.254709213001142), ((0.30000000000000004, 0.85000000000000009, 1.0), 18.254709213001142), ((0.050000000000000003, 0.85000000000000009, 1.0), 18.254709213001142), ((0.30000000000000004, 0.90000000000000002, 1.0), 18.254709213001142), ((0.25, 0.95000000000000007, 1.0), 18.254709213001142), ((0.15000000000000002, 0.90000000000000002, 1.0), 18.254709213001142), ((0.10000000000000001, 0.90000000000000002, 1.0), 18.254709213001142), ((0.15000000000000002, 0.95000000000000007, 1.0), 18.254709213001142), ((0.0, 0.95000000000000007, 1.0), 18.254709213001142), ((0.25, 1.0, 1.0), 18.254709213001142), ((0.20000000000000001, 0.85000000000000009, 1.0), 18.254709213001142), ((0.20000000000000001, 1.0, 1.0), 18.254709213001142), ((0.050000000000000003, 1.0, 1.0), 18.254709213001142), ((0.050000000000000003, 0.95000000000000007, 1.0), 18.254709213001142), ((0.0, 1.0, 1.0), 18.254709213001142), ((0.25, 0.90000000000000002, 1.0), 18.254709213001142), ((0.30000000000000004, 1.0, 1.0), 18.254709213001142), ((0.10000000000000001, 0.85000000000000009, 1.0), 18.254709213001142), ((0.20000000000000001, 0.95000000000000007, 1.0), 18.254709213001142), ((0.20000000000000001, 0.90000000000000002, 1.0), 18.254709213001142), ((0.050000000000000003, 0.90000000000000002, 1.0), 18.254709213001142), ((0.25, 0.85000000000000009, 1.0), 18.254709213001142), ((0.30000000000000004, 0.95000000000000007, 1.0), 18.254709213001142), ((0.15000000000000002, 0.85000000000000009, 1.0), 18.254709213001142), ((0.10000000000000001, 1.0, 1.0), 18.254709213001142), ((0.15000000000000002, 1.0, 1.0), 18.254709213001142), ((0.0, 0.85000000000000009, 1.0), 18.254709213001142), ((0.35000000000000003, 0.80000000000000004, 1.0), 18.23098750220706), ((0.35000000000000003, 1.0, 1.0), 18.222215572382503), ((0.35000000000000003, 0.95000000000000007, 1.0), 18.222215572382503), ((0.35000000000000003, 0.85000000000000009, 1.0), 18.222215572382503), ((0.35000000000000003, 0.90000000000000002, 1.0), 18.222215572382503), ((0.45000000000000001, 0.70000000000000007, 1.0), 18.183388838617795), ((0.40000000000000002, 0.70000000000000007, 1.0), 18.178220678000546), ((0.45000000000000001, 0.80000000000000004, 1.0), 18.03445144118087), ((0.40000000000000002, 0.80000000000000004, 1.0), 18.02928328056362), ((0.45000000000000001, 0.85000000000000009, 1.0), 18.025679511356305), ((0.45000000000000001, 0.95000000000000007, 1.0), 18.025679511356305), ((0.45000000000000001, 0.90000000000000002, 1.0), 18.025679511356305), ((0.45000000000000001, 1.0, 1.0), 18.025679511356305), ((0.40000000000000002, 0.90000000000000002, 1.0), 18.020511350739064), ((0.40000000000000002, 1.0, 1.0), 18.020511350739064), ((0.40000000000000002, 0.85000000000000009, 1.0), 18.020511350739064), ((0.40000000000000002, 0.95000000000000007, 1.0), 18.020511350739064), ((0.5, 0.70000000000000007, 1.0), 17.904423577242273), ((0.55000000000000004, 0.70000000000000007, 1.0), 17.84963465194987), ((0.60000000000000009, 0.75, 1.0), 17.83448128047042), ((0.10000000000000001, 0.55000000000000004, 1.0), 17.755410362429124), ((0.0, 0.55000000000000004, 1.0), 17.755410362429124), ((0.050000000000000003, 0.55000000000000004, 1.0), 17.755410362429124), ((0.20000000000000001, 0.55000000000000004, 1.0), 17.755410362429124), ((0.25, 0.55000000000000004, 1.0), 17.755410362429124), ((0.15000000000000002, 0.55000000000000004, 1.0), 17.755410362429124), ((0.5, 0.80000000000000004, 1.0), 17.751657374852783), ((0.5, 0.90000000000000002, 1.0), 17.742885445028218), ((0.5, 0.85000000000000009, 1.0), 17.742885445028218), ((0.5, 0.95000000000000007, 1.0), 17.742885445028218), ((0.5, 1.0, 1.0), 17.742885445028218), ((0.35000000000000003, 0.55000000000000004, 1.0), 17.737127989798378), ((0.30000000000000004, 0.55000000000000004, 1.0), 17.72746918595854), ((0.55000000000000004, 0.80000000000000004, 1.0), 17.696977334577795), ((0.55000000000000004, 0.85000000000000009, 1.0), 17.689762327362786), ((0.55000000000000004, 1.0, 1.0), 17.689762327362786), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.689762327362786), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.689762327362786), ((0.30000000000000004, 0.75, 1.0), 17.6384811428257), ((0.10000000000000001, 0.75, 1.0), 17.6384811428257), ((0.25, 0.75, 1.0), 17.6384811428257), ((0.20000000000000001, 0.75, 1.0), 17.6384811428257), ((0.0, 0.75, 1.0), 17.6384811428257), ((0.050000000000000003, 0.75, 1.0), 17.6384811428257), ((0.15000000000000002, 0.75, 1.0), 17.6384811428257), ((0.35000000000000003, 0.75, 1.0), 17.60598750220706), ((0.050000000000000003, 0.65000000000000002, 1.0), 17.591244048211376), ((0.0, 0.65000000000000002, 1.0), 17.591244048211376), ((0.10000000000000001, 0.65000000000000002, 1.0), 17.591244048211376), ((0.30000000000000004, 0.65000000000000002, 1.0), 17.591244048211376), ((0.20000000000000001, 0.65000000000000002, 1.0), 17.591244048211376), ((0.15000000000000002, 0.65000000000000002, 1.0), 17.591244048211376), ((0.25, 0.65000000000000002, 1.0), 17.591244048211376), ((0.35000000000000003, 0.65000000000000002, 1.0), 17.558750407592733), ((0.40000000000000002, 0.55000000000000004, 1.0), 17.541691481861868), ((0.45000000000000001, 0.75, 1.0), 17.40945144118087), ((0.40000000000000002, 0.75, 1.0), 17.40428328056362), ((0.45000000000000001, 0.55000000000000004, 1.0), 17.36382047905384), ((0.45000000000000001, 0.65000000000000002, 1.0), 17.362112147961057), ((0.40000000000000002, 0.65000000000000002, 1.0), 17.35992017782), ((0.60000000000000009, 0.65000000000000002, 1.0), 17.312483101303986), ((0.75, 0.85000000000000009, 1.0), 17.293429373853204), ((0.75, 0.95000000000000007, 1.0), 17.293429373853204), ((0.75, 0.90000000000000002, 1.0), 17.293429373853204), ((0.75, 1.0, 1.0), 17.293429373853204), ((0.65000000000000002, 0.80000000000000004, 1.0), 17.215560064195298), ((0.65000000000000002, 1.0, 1.0), 17.215560064195298), ((0.65000000000000002, 0.85000000000000009, 1.0), 17.215560064195298), ((0.65000000000000002, 0.90000000000000002, 1.0), 17.215560064195298), ((0.65000000000000002, 0.95000000000000007, 1.0), 17.215560064195298), ((0.5, 0.75, 1.0), 17.12665737485278), ((0.70000000000000007, 0.80000000000000004, 1.0), 17.0980674033579), ((0.70000000000000007, 1.0, 1.0), 17.09375705853031), ((0.70000000000000007, 0.90000000000000002, 1.0), 17.09375705853031), ((0.70000000000000007, 0.85000000000000009, 1.0), 17.09375705853031), ((0.70000000000000007, 0.95000000000000007, 1.0), 17.09375705853031), ((0.55000000000000004, 0.65000000000000002, 1.0), 17.07854548926466), ((0.55000000000000004, 0.75, 1.0), 17.07197733457779), ((0.5, 0.65000000000000002, 1.0), 17.06882559279996), ((0.35000000000000003, 0.5, 1.0), 17.054871467366812), ((0.25, 0.5, 1.0), 17.052977527972875), ((0.15000000000000002, 0.5, 1.0), 17.052977527972875), ((0.0, 0.5, 1.0), 17.052977527972875), ((0.20000000000000001, 0.5, 1.0), 17.052977527972875), ((0.10000000000000001, 0.5, 1.0), 17.052977527972875), ((0.30000000000000004, 0.5, 1.0), 17.052977527972875), ((0.050000000000000003, 0.5, 1.0), 17.052977527972875), ((0.40000000000000002, 0.5, 1.0), 17.036271053362587), ((0.050000000000000003, 0.60000000000000009, 1.0), 17.032672668544297), ((0.15000000000000002, 0.60000000000000009, 1.0), 17.032672668544297), ((0.20000000000000001, 0.60000000000000009, 1.0), 17.032672668544297), ((0.10000000000000001, 0.60000000000000009, 1.0), 17.032672668544297), ((0.30000000000000004, 0.60000000000000009, 1.0), 17.032672668544297), ((0.0, 0.60000000000000009, 1.0), 17.032672668544297), ((0.25, 0.60000000000000009, 1.0), 17.032672668544297), ((0.35000000000000003, 0.60000000000000009, 1.0), 16.97021604635622), ((0.65000000000000002, 0.70000000000000007, 1.0), 16.847218580175408), ((0.45000000000000001, 0.60000000000000009, 1.0), 16.801787521540724), ((0.40000000000000002, 0.60000000000000009, 1.0), 16.80154328965176), ((0.10000000000000001, 0.45000000000000001, 1.0), 16.799377892931336), ((0.15000000000000002, 0.45000000000000001, 1.0), 16.799377892931336), ((0.20000000000000001, 0.45000000000000001, 1.0), 16.799377892931336), ((0.25, 0.45000000000000001, 1.0), 16.799377892931336), ((0.0, 0.45000000000000001, 1.0), 16.799377892931336), ((0.050000000000000003, 0.45000000000000001, 1.0), 16.799377892931336), ((0.30000000000000004, 0.45000000000000001, 1.0), 16.799377892931336), ((0.35000000000000003, 0.45000000000000001, 1.0), 16.78142910799979), ((0.15000000000000002, 0.40000000000000002, 1.0), 16.771527838396498), ((0.0, 0.40000000000000002, 1.0), 16.771527838396498), ((0.10000000000000001, 0.40000000000000002, 1.0), 16.771527838396498), ((0.20000000000000001, 0.40000000000000002, 1.0), 16.771527838396498), ((0.050000000000000003, 0.40000000000000002, 1.0), 16.771527838396498), ((0.25, 0.40000000000000002, 1.0), 16.771527838396498), ((0.30000000000000004, 0.40000000000000002, 1.0), 16.766909611795516), ((0.45000000000000001, 0.5, 1.0), 16.753339665489968)], [((0.60000000000000009, 0.85000000000000009, 1.0), 19.716766333537365), ((0.60000000000000009, 0.80000000000000004, 1.0), 19.716766333537365), ((0.60000000000000009, 0.95000000000000007, 1.0), 19.716766333537365), ((0.60000000000000009, 0.90000000000000002, 1.0), 19.716766333537365), ((0.60000000000000009, 1.0, 1.0), 19.716766333537365), ((0.55000000000000004, 0.80000000000000004, 1.0), 19.64426896061395), ((0.55000000000000004, 0.85000000000000009, 1.0), 19.64426896061395), ((0.55000000000000004, 1.0, 1.0), 19.64426896061395), ((0.55000000000000004, 0.95000000000000007, 1.0), 19.64426896061395), ((0.55000000000000004, 0.90000000000000002, 1.0), 19.64426896061395), ((0.55000000000000004, 0.75, 1.0), 19.560935627280617), ((0.60000000000000009, 0.75, 1.0), 19.49454411131514), ((0.35000000000000003, 0.80000000000000004, 1.0), 19.4944622319529), ((0.0, 0.90000000000000002, 1.0), 19.4944622319529), ((0.40000000000000002, 0.90000000000000002, 1.0), 19.4944622319529), ((0.10000000000000001, 0.95000000000000007, 1.0), 19.4944622319529), ((0.5, 0.90000000000000002, 1.0), 19.4944622319529), ((0.40000000000000002, 1.0, 1.0), 19.4944622319529), ((0.35000000000000003, 1.0, 1.0), 19.4944622319529), ((0.30000000000000004, 0.85000000000000009, 1.0), 19.4944622319529), ((0.20000000000000001, 0.80000000000000004, 1.0), 19.4944622319529), ((0.050000000000000003, 0.85000000000000009, 1.0), 19.4944622319529), ((0.5, 0.85000000000000009, 1.0), 19.4944622319529), ((0.30000000000000004, 0.90000000000000002, 1.0), 19.4944622319529), ((0.25, 0.95000000000000007, 1.0), 19.4944622319529), ((0.15000000000000002, 0.90000000000000002, 1.0), 19.4944622319529), ((0.45000000000000001, 0.85000000000000009, 1.0), 19.4944622319529), ((0.5, 0.95000000000000007, 1.0), 19.4944622319529), ((0.10000000000000001, 0.90000000000000002, 1.0), 19.4944622319529), ((0.40000000000000002, 0.85000000000000009, 1.0), 19.4944622319529), ((0.45000000000000001, 0.95000000000000007, 1.0), 19.4944622319529), ((0.15000000000000002, 0.95000000000000007, 1.0), 19.4944622319529), ((0.0, 0.95000000000000007, 1.0), 19.4944622319529), ((0.0, 0.80000000000000004, 1.0), 19.4944622319529), ((0.25, 0.80000000000000004, 1.0), 19.4944622319529), ((0.25, 1.0, 1.0), 19.4944622319529), ((0.35000000000000003, 0.95000000000000007, 1.0), 19.4944622319529), ((0.20000000000000001, 0.85000000000000009, 1.0), 19.4944622319529), ((0.20000000000000001, 1.0, 1.0), 19.4944622319529), ((0.15000000000000002, 0.80000000000000004, 1.0), 19.4944622319529), ((0.40000000000000002, 0.95000000000000007, 1.0), 19.4944622319529), ((0.050000000000000003, 1.0, 1.0), 19.4944622319529), ((0.35000000000000003, 0.85000000000000009, 1.0), 19.4944622319529), ((0.10000000000000001, 0.80000000000000004, 1.0), 19.4944622319529), ((0.5, 1.0, 1.0), 19.4944622319529), ((0.35000000000000003, 0.90000000000000002, 1.0), 19.4944622319529), ((0.050000000000000003, 0.95000000000000007, 1.0), 19.4944622319529), ((0.0, 1.0, 1.0), 19.4944622319529), ((0.45000000000000001, 0.90000000000000002, 1.0), 19.4944622319529), ((0.25, 0.90000000000000002, 1.0), 19.4944622319529), ((0.30000000000000004, 1.0, 1.0), 19.4944622319529), ((0.10000000000000001, 0.85000000000000009, 1.0), 19.4944622319529), ((0.050000000000000003, 0.80000000000000004, 1.0), 19.4944622319529), ((0.20000000000000001, 0.95000000000000007, 1.0), 19.4944622319529), ((0.5, 0.80000000000000004, 1.0), 19.4944622319529), ((0.20000000000000001, 0.90000000000000002, 1.0), 19.4944622319529), ((0.45000000000000001, 0.80000000000000004, 1.0), 19.4944622319529), ((0.050000000000000003, 0.90000000000000002, 1.0), 19.4944622319529), ((0.40000000000000002, 0.80000000000000004, 1.0), 19.4944622319529), ((0.25, 0.85000000000000009, 1.0), 19.4944622319529), ((0.30000000000000004, 0.95000000000000007, 1.0), 19.4944622319529), ((0.15000000000000002, 0.85000000000000009, 1.0), 19.4944622319529), ((0.10000000000000001, 1.0, 1.0), 19.4944622319529), ((0.30000000000000004, 0.80000000000000004, 1.0), 19.4944622319529), ((0.15000000000000002, 1.0, 1.0), 19.4944622319529), ((0.0, 0.85000000000000009, 1.0), 19.4944622319529), ((0.45000000000000001, 1.0, 1.0), 19.4944622319529), ((0.40000000000000002, 0.75, 1.0), 19.411128898619566), ((0.45000000000000001, 0.75, 1.0), 19.411128898619566), ((0.30000000000000004, 0.75, 1.0), 19.411128898619566), ((0.10000000000000001, 0.75, 1.0), 19.411128898619566), ((0.25, 0.75, 1.0), 19.411128898619566), ((0.20000000000000001, 0.75, 1.0), 19.411128898619566), ((0.35000000000000003, 0.75, 1.0), 19.411128898619566), ((0.0, 0.75, 1.0), 19.411128898619566), ((0.5, 0.75, 1.0), 19.411128898619566), ((0.050000000000000003, 0.75, 1.0), 19.411128898619566), ((0.15000000000000002, 0.75, 1.0), 19.411128898619566), ((0.55000000000000004, 0.70000000000000007, 1.0), 18.953974278261896), ((0.55000000000000004, 0.60000000000000009, 1.0), 18.77969556580785), ((0.15000000000000002, 0.70000000000000007, 1.0), 18.71612742614406), ((0.10000000000000001, 0.70000000000000007, 1.0), 18.71612742614406), ((0.40000000000000002, 0.70000000000000007, 1.0), 18.71612742614406), ((0.5, 0.70000000000000007, 1.0), 18.71612742614406), ((0.25, 0.70000000000000007, 1.0), 18.71612742614406), ((0.30000000000000004, 0.70000000000000007, 1.0), 18.71612742614406), ((0.45000000000000001, 0.70000000000000007, 1.0), 18.71612742614406), ((0.0, 0.70000000000000007, 1.0), 18.71612742614406), ((0.35000000000000003, 0.70000000000000007, 1.0), 18.71612742614406), ((0.20000000000000001, 0.70000000000000007, 1.0), 18.71612742614406), ((0.050000000000000003, 0.70000000000000007, 1.0), 18.71612742614406), ((0.60000000000000009, 0.65000000000000002, 1.0), 18.5135991251451), ((0.5, 0.55000000000000004, 1.0), 18.5035201260213), ((0.5, 0.60000000000000009, 1.0), 18.35963449592535), ((0.65000000000000002, 1.0, 1.0), 18.340353563191563), ((0.65000000000000002, 0.85000000000000009, 1.0), 18.340353563191563), ((0.65000000000000002, 0.90000000000000002, 1.0), 18.340353563191563), ((0.65000000000000002, 0.95000000000000007, 1.0), 18.340353563191563), ((0.050000000000000003, 0.60000000000000009, 1.0), 18.301142432433288), ((0.40000000000000002, 0.60000000000000009, 1.0), 18.301142432433288), ((0.15000000000000002, 0.60000000000000009, 1.0), 18.301142432433288), ((0.45000000000000001, 0.60000000000000009, 1.0), 18.301142432433288), ((0.20000000000000001, 0.60000000000000009, 1.0), 18.301142432433288), ((0.10000000000000001, 0.60000000000000009, 1.0), 18.301142432433288), ((0.30000000000000004, 0.60000000000000009, 1.0), 18.301142432433288), ((0.0, 0.60000000000000009, 1.0), 18.301142432433288), ((0.35000000000000003, 0.60000000000000009, 1.0), 18.301142432433288), ((0.25, 0.60000000000000009, 1.0), 18.301142432433288), ((0.55000000000000004, 0.65000000000000002, 1.0), 18.291822279489832), ((0.60000000000000009, 0.70000000000000007, 1.0), 18.275686188852227), ((0.35000000000000003, 0.55000000000000004, 1.0), 18.253889551569298), ((0.10000000000000001, 0.55000000000000004, 1.0), 18.253889551569298), ((0.30000000000000004, 0.55000000000000004, 1.0), 18.253889551569298), ((0.0, 0.55000000000000004, 1.0), 18.253889551569298), ((0.050000000000000003, 0.55000000000000004, 1.0), 18.253889551569298), ((0.45000000000000001, 0.55000000000000004, 1.0), 18.253889551569298), ((0.20000000000000001, 0.55000000000000004, 1.0), 18.253889551569298), ((0.25, 0.55000000000000004, 1.0), 18.253889551569298), ((0.15000000000000002, 0.55000000000000004, 1.0), 18.253889551569298), ((0.40000000000000002, 0.55000000000000004, 1.0), 18.253889551569298), ((0.70000000000000007, 1.0, 1.0), 18.19936614855905), ((0.70000000000000007, 0.90000000000000002, 1.0), 18.19936614855905), ((0.70000000000000007, 0.85000000000000009, 1.0), 18.19936614855905), ((0.70000000000000007, 0.95000000000000007, 1.0), 18.19936614855905), ((0.65000000000000002, 0.80000000000000004, 1.0), 18.17368689652489), ((0.050000000000000003, 0.65000000000000002, 1.0), 18.098358821755387), ((0.0, 0.65000000000000002, 1.0), 18.098358821755387), ((0.35000000000000003, 0.65000000000000002, 1.0), 18.098358821755387), ((0.10000000000000001, 0.65000000000000002, 1.0), 18.098358821755387), ((0.5, 0.65000000000000002, 1.0), 18.098358821755387), ((0.30000000000000004, 0.65000000000000002, 1.0), 18.098358821755387), ((0.20000000000000001, 0.65000000000000002, 1.0), 18.098358821755387), ((0.45000000000000001, 0.65000000000000002, 1.0), 18.098358821755387), ((0.15000000000000002, 0.65000000000000002, 1.0), 18.098358821755387), ((0.25, 0.65000000000000002, 1.0), 18.098358821755387), ((0.40000000000000002, 0.65000000000000002, 1.0), 18.098358821755387), ((0.75, 0.85000000000000009, 1.0), 17.020588705589876), ((0.75, 0.95000000000000007, 1.0), 17.020588705589876), ((0.75, 0.90000000000000002, 1.0), 17.020588705589876), ((0.75, 1.0, 1.0), 17.020588705589876), ((0.70000000000000007, 0.80000000000000004, 1.0), 17.003598952791855), ((0.65000000000000002, 0.75, 1.0), 16.977919700757695)]], 'Christmas': [[((0.5, 0.60000000000000009, 1.0), 23.365395796355923), ((0.5, 0.55000000000000004, 1.0), 23.29431225380778), ((0.40000000000000002, 0.5, 1.0), 23.292617519033907), ((0.45000000000000001, 0.55000000000000004, 1.0), 23.26124973477358), ((0.40000000000000002, 0.45000000000000001, 1.0), 23.24502732048419), ((0.55000000000000004, 0.65000000000000002, 1.0), 23.19962214343368), ((0.5, 0.65000000000000002, 1.0), 23.166077355537478), ((0.55000000000000004, 0.75, 1.0), 23.149139488833374), ((0.55000000000000004, 0.70000000000000007, 1.0), 23.14152410421799), ((0.55000000000000004, 0.80000000000000004, 1.0), 23.12126070095459), ((0.55000000000000004, 0.85000000000000009, 1.0), 23.12126070095459), ((0.55000000000000004, 1.0, 1.0), 23.12126070095459), ((0.55000000000000004, 0.95000000000000007, 1.0), 23.12126070095459), ((0.55000000000000004, 0.90000000000000002, 1.0), 23.12126070095459), ((0.35000000000000003, 0.45000000000000001, 1.0), 23.100052588976776), ((0.45000000000000001, 0.5, 1.0), 23.097805711504147), ((0.5, 0.75, 1.0), 23.092301154886282), ((0.55000000000000004, 0.60000000000000009, 1.0), 23.086271943809667), ((0.45000000000000001, 0.60000000000000009, 1.0), 23.08553726789655), ((0.5, 0.70000000000000007, 1.0), 23.084765135350256), ((0.5, 0.80000000000000004, 1.0), 23.060973292933415), ((0.5, 0.90000000000000002, 1.0), 23.02639136251815), ((0.5, 0.85000000000000009, 1.0), 23.02639136251815), ((0.5, 0.95000000000000007, 1.0), 23.02639136251815), ((0.5, 1.0, 1.0), 23.02639136251815), ((0.45000000000000001, 0.65000000000000002, 1.0), 22.902293289652576), ((0.65000000000000002, 0.70000000000000007, 1.0), 22.895299284185082)], [], [((0.35000000000000003, 0.40000000000000002, 1.0), 21.862044593980258), ((0.15000000000000002, 0.40000000000000002, 1.0), 21.813138360030862), ((0.0, 0.40000000000000002, 1.0), 21.813138360030862), ((0.10000000000000001, 0.40000000000000002, 1.0), 21.813138360030862), ((0.20000000000000001, 0.40000000000000002, 1.0), 21.813138360030862), ((0.050000000000000003, 0.40000000000000002, 1.0), 21.813138360030862), ((0.25, 0.40000000000000002, 1.0), 21.813138360030862), ((0.30000000000000004, 0.40000000000000002, 1.0), 21.811847650691178), ((0.20000000000000001, 0.35000000000000003, 1.0), 21.63396337739011), ((0.0, 0.35000000000000003, 1.0), 21.63396337739011), ((0.050000000000000003, 0.35000000000000003, 1.0), 21.63396337739011), ((0.15000000000000002, 0.35000000000000003, 1.0), 21.63396337739011), ((0.10000000000000001, 0.35000000000000003, 1.0), 21.63396337739011), ((0.25, 0.35000000000000003, 1.0), 21.633954673701364), ((0.30000000000000004, 0.35000000000000003, 1.0), 21.63169849127671)]], 'PersonalArtActivity': [[((0.0, 0.25, 1.0), 13.52169787400948), ((0.050000000000000003, 0.25, 1.0), 13.52169787400948), ((0.10000000000000001, 0.25, 1.0), 13.519567148147003), ((0.15000000000000002, 0.25, 1.0), 13.517114401697548), ((0.20000000000000001, 0.25, 1.0), 13.51372870809008), ((0.0, 0.20000000000000001, 1.0), 13.512070521767981), ((0.050000000000000003, 0.20000000000000001, 1.0), 13.512070521767981), ((0.050000000000000003, 0.15000000000000002, 1.0), 13.512017393306097), ((0.0, 0.15000000000000002, 1.0), 13.512017393306097), ((0.10000000000000001, 0.20000000000000001, 1.0), 13.509617775318524)], [], [((0.65000000000000002, 0.75, 1.0), 16.954095563813613), ((0.70000000000000007, 0.80000000000000004, 1.0), 16.92951434678168), ((0.70000000000000007, 0.75, 1.0), 16.92855731553168), ((0.65000000000000002, 0.80000000000000004, 1.0), 16.80337334006566), ((0.65000000000000002, 1.0, 1.0), 16.793618650310975), ((0.65000000000000002, 0.85000000000000009, 1.0), 16.793618650310975), ((0.65000000000000002, 0.90000000000000002, 1.0), 16.793618650310975), ((0.65000000000000002, 0.95000000000000007, 1.0), 16.793618650310975), ((0.65000000000000002, 0.70000000000000007, 1.0), 16.788726117231313), ((0.70000000000000007, 1.0, 1.0), 16.7722429455304), ((0.70000000000000007, 0.90000000000000002, 1.0), 16.7722429455304), ((0.70000000000000007, 0.85000000000000009, 1.0), 16.7722429455304), ((0.70000000000000007, 0.95000000000000007, 1.0), 16.7722429455304), ((0.60000000000000009, 0.70000000000000007, 1.0), 16.68348904265604), ((0.60000000000000009, 0.75, 1.0), 16.604562291828916), ((0.60000000000000009, 0.80000000000000004, 1.0), 16.602151343578086), ((0.60000000000000009, 0.85000000000000009, 1.0), 16.567217567037165), ((0.60000000000000009, 0.95000000000000007, 1.0), 16.567217567037165), ((0.60000000000000009, 0.90000000000000002, 1.0), 16.567217567037165), ((0.60000000000000009, 1.0, 1.0), 16.567217567037165), ((0.60000000000000009, 0.65000000000000002, 1.0), 16.561161076744924), ((0.55000000000000004, 0.60000000000000009, 1.0), 16.200736682735712), ((0.55000000000000004, 0.65000000000000002, 1.0), 16.198058850892018), ((0.55000000000000004, 0.75, 1.0), 16.090238272346866), ((0.55000000000000004, 0.70000000000000007, 1.0), 16.068169483622416), ((0.55000000000000004, 0.80000000000000004, 1.0), 16.043937372665503), ((0.55000000000000004, 0.85000000000000009, 1.0), 16.02674229261328), ((0.55000000000000004, 1.0, 1.0), 16.02674229261328), ((0.55000000000000004, 0.95000000000000007, 1.0), 16.02674229261328), ((0.55000000000000004, 0.90000000000000002, 1.0), 16.02674229261328), ((0.5, 0.60000000000000009, 1.0), 16.025866110223944), ((0.5, 0.70000000000000007, 1.0), 15.857878469510734), ((0.5, 0.65000000000000002, 1.0), 15.845332635210447), ((0.15000000000000002, 0.40000000000000002, 1.0), 15.825602758389655), ((0.0, 0.40000000000000002, 1.0), 15.825602758389655), ((0.10000000000000001, 0.40000000000000002, 1.0), 15.825602758389655), ((0.20000000000000001, 0.40000000000000002, 1.0), 15.825602758389655), ((0.050000000000000003, 0.40000000000000002, 1.0), 15.825602758389655), ((0.25, 0.40000000000000002, 1.0), 15.825557713344612), ((0.30000000000000004, 0.40000000000000002, 1.0), 15.82375943511521), ((0.35000000000000003, 0.40000000000000002, 1.0), 15.823553668881738), ((0.30000000000000004, 0.60000000000000009, 1.0), 15.805045499247843), ((0.050000000000000003, 0.60000000000000009, 1.0), 15.80219133258118), ((0.15000000000000002, 0.60000000000000009, 1.0), 15.80219133258118), ((0.20000000000000001, 0.60000000000000009, 1.0), 15.80219133258118), ((0.10000000000000001, 0.60000000000000009, 1.0), 15.80219133258118), ((0.0, 0.60000000000000009, 1.0), 15.80219133258118), ((0.25, 0.60000000000000009, 1.0), 15.80219133258118), ((0.35000000000000003, 0.60000000000000009, 1.0), 15.801759241047554), ((0.45000000000000001, 0.60000000000000009, 1.0), 15.79695378459503), ((0.40000000000000002, 0.60000000000000009, 1.0), 15.77981906456835)]], 'GroupActivity': [[((0.70000000000000007, 1.0, 1.0), 18.37915892134071), ((0.70000000000000007, 0.90000000000000002, 1.0), 18.37915892134071), ((0.70000000000000007, 0.85000000000000009, 1.0), 18.37915892134071), ((0.70000000000000007, 0.95000000000000007, 1.0), 18.37915892134071), ((0.75, 0.85000000000000009, 1.0), 18.05878148144125), ((0.75, 0.95000000000000007, 1.0), 18.05878148144125), ((0.75, 0.90000000000000002, 1.0), 18.05878148144125), ((0.75, 1.0, 1.0), 18.05878148144125), ((0.80000000000000004, 0.95000000000000007, 1.0), 17.895426331300385), ((0.80000000000000004, 0.90000000000000002, 1.0), 17.895426331300385), ((0.80000000000000004, 0.85000000000000009, 1.0), 17.895426331300385), ((0.80000000000000004, 1.0, 1.0), 17.895426331300385), ((0.70000000000000007, 0.80000000000000004, 1.0), 17.85183465830216)], [((0.55000000000000004, 0.75, 1.0), 25.611779986847075), ((0.5, 0.75, 1.0), 25.506478958366266), ((0.40000000000000002, 0.75, 1.0), 25.473961894787458), ((0.25, 0.75, 1.0), 25.446983375632293), ((0.10000000000000001, 0.75, 1.0), 25.44611757476649), ((0.20000000000000001, 0.75, 1.0), 25.44611757476649), ((0.0, 0.75, 1.0), 25.44611757476649), ((0.050000000000000003, 0.75, 1.0), 25.44611757476649), ((0.15000000000000002, 0.75, 1.0), 25.44611757476649), ((0.30000000000000004, 0.75, 1.0), 25.433897791529727), ((0.45000000000000001, 0.75, 1.0), 25.396913525708356), ((0.35000000000000003, 0.75, 1.0), 25.380742697297404), ((0.55000000000000004, 0.80000000000000004, 1.0), 25.368954371752558), ((0.55000000000000004, 0.85000000000000009, 1.0), 25.361403417206134), ((0.55000000000000004, 1.0, 1.0), 25.361403417206134), ((0.55000000000000004, 0.95000000000000007, 1.0), 25.361403417206134), ((0.55000000000000004, 0.90000000000000002, 1.0), 25.361403417206134), ((0.5, 0.90000000000000002, 1.0), 25.25430514462024), ((0.5, 0.85000000000000009, 1.0), 25.25430514462024), ((0.5, 0.95000000000000007, 1.0), 25.25430514462024), ((0.5, 1.0, 1.0), 25.25430514462024), ((0.5, 0.80000000000000004, 1.0), 25.25430514462024), ((0.40000000000000002, 0.90000000000000002, 1.0), 25.228817170230972), ((0.40000000000000002, 1.0, 1.0), 25.228817170230972), ((0.40000000000000002, 0.85000000000000009, 1.0), 25.228817170230972), ((0.40000000000000002, 0.95000000000000007, 1.0), 25.228817170230972), ((0.40000000000000002, 0.80000000000000004, 1.0), 25.228817170230972), ((0.25, 0.95000000000000007, 1.0), 25.2018386510758), ((0.25, 0.80000000000000004, 1.0), 25.2018386510758), ((0.25, 1.0, 1.0), 25.2018386510758), ((0.25, 0.90000000000000002, 1.0), 25.2018386510758), ((0.25, 0.85000000000000009, 1.0), 25.2018386510758), ((0.0, 0.90000000000000002, 1.0), 25.200972850209997), ((0.10000000000000001, 0.95000000000000007, 1.0), 25.200972850209997), ((0.20000000000000001, 0.80000000000000004, 1.0), 25.200972850209997), ((0.050000000000000003, 0.85000000000000009, 1.0), 25.200972850209997), ((0.15000000000000002, 0.90000000000000002, 1.0), 25.200972850209997), ((0.10000000000000001, 0.90000000000000002, 1.0), 25.200972850209997), ((0.15000000000000002, 0.95000000000000007, 1.0), 25.200972850209997), ((0.0, 0.95000000000000007, 1.0), 25.200972850209997), ((0.0, 0.80000000000000004, 1.0), 25.200972850209997), ((0.20000000000000001, 0.85000000000000009, 1.0), 25.200972850209997), ((0.20000000000000001, 1.0, 1.0), 25.200972850209997), ((0.15000000000000002, 0.80000000000000004, 1.0), 25.200972850209997), ((0.050000000000000003, 1.0, 1.0), 25.200972850209997), ((0.10000000000000001, 0.80000000000000004, 1.0), 25.200972850209997), ((0.050000000000000003, 0.95000000000000007, 1.0), 25.200972850209997), ((0.0, 1.0, 1.0), 25.200972850209997), ((0.10000000000000001, 0.85000000000000009, 1.0), 25.200972850209997), ((0.050000000000000003, 0.80000000000000004, 1.0), 25.200972850209997), ((0.20000000000000001, 0.95000000000000007, 1.0), 25.200972850209997), ((0.20000000000000001, 0.90000000000000002, 1.0), 25.200972850209997), ((0.050000000000000003, 0.90000000000000002, 1.0), 25.200972850209997), ((0.15000000000000002, 0.85000000000000009, 1.0), 25.200972850209997), ((0.10000000000000001, 1.0, 1.0), 25.200972850209997), ((0.15000000000000002, 1.0, 1.0), 25.200972850209997), ((0.0, 0.85000000000000009, 1.0), 25.200972850209997), ((0.30000000000000004, 0.85000000000000009, 1.0), 25.18875306697324), ((0.30000000000000004, 0.90000000000000002, 1.0), 25.18875306697324), ((0.30000000000000004, 1.0, 1.0), 25.18875306697324), ((0.30000000000000004, 0.95000000000000007, 1.0), 25.18875306697324), ((0.30000000000000004, 0.80000000000000004, 1.0), 25.18875306697324), ((0.45000000000000001, 0.85000000000000009, 1.0), 25.151768801151867), ((0.45000000000000001, 0.95000000000000007, 1.0), 25.151768801151867), ((0.45000000000000001, 0.90000000000000002, 1.0), 25.151768801151867), ((0.45000000000000001, 0.80000000000000004, 1.0), 25.151768801151867), ((0.45000000000000001, 1.0, 1.0), 25.151768801151867), ((0.35000000000000003, 0.80000000000000004, 1.0), 25.13559797274092), ((0.35000000000000003, 1.0, 1.0), 25.13559797274092), ((0.35000000000000003, 0.95000000000000007, 1.0), 25.13559797274092), ((0.35000000000000003, 0.85000000000000009, 1.0), 25.13559797274092), ((0.35000000000000003, 0.90000000000000002, 1.0), 25.13559797274092), ((0.5, 0.70000000000000007, 1.0), 24.89753489658264), ((0.40000000000000002, 0.70000000000000007, 1.0), 24.876175837688244), ((0.25, 0.70000000000000007, 1.0), 24.849197318533072), ((0.15000000000000002, 0.70000000000000007, 1.0), 24.848331517667273), ((0.10000000000000001, 0.70000000000000007, 1.0), 24.848331517667273), ((0.0, 0.70000000000000007, 1.0), 24.848331517667273), ((0.20000000000000001, 0.70000000000000007, 1.0), 24.848331517667273), ((0.050000000000000003, 0.70000000000000007, 1.0), 24.848331517667273), ((0.30000000000000004, 0.70000000000000007, 1.0), 24.83611173443051), ((0.45000000000000001, 0.70000000000000007, 1.0), 24.796249220664382), ((0.35000000000000003, 0.70000000000000007, 1.0), 24.782956640198186), ((0.60000000000000009, 0.75, 1.0), 24.514105259566072), ((0.60000000000000009, 0.85000000000000009, 1.0), 24.386567032875686), ((0.60000000000000009, 0.80000000000000004, 1.0), 24.386567032875686), ((0.60000000000000009, 0.95000000000000007, 1.0), 24.386567032875686), ((0.60000000000000009, 0.90000000000000002, 1.0), 24.386567032875686), ((0.60000000000000009, 1.0, 1.0), 24.386567032875686), ((0.65000000000000002, 1.0, 1.0), 24.175968549784102), ((0.65000000000000002, 0.85000000000000009, 1.0), 24.175968549784102), ((0.65000000000000002, 0.90000000000000002, 1.0), 24.175968549784102), ((0.65000000000000002, 0.95000000000000007, 1.0), 24.175968549784102), ((0.65000000000000002, 0.80000000000000004, 1.0), 24.042848321173174), ((0.75, 0.95000000000000007, 1.0), 23.980120732643407), ((0.75, 0.90000000000000002, 1.0), 23.980120732643407), ((0.75, 1.0, 1.0), 23.980120732643407), ((0.70000000000000007, 1.0, 1.0), 23.95474668970826), ((0.70000000000000007, 0.90000000000000002, 1.0), 23.95474668970826), ((0.70000000000000007, 0.85000000000000009, 1.0), 23.95474668970826), ((0.70000000000000007, 0.95000000000000007, 1.0), 23.95474668970826), ((0.40000000000000002, 0.65000000000000002, 1.0), 23.947330375571514), ((0.75, 0.85000000000000009, 1.0), 23.90393025645293), ((0.25, 0.65000000000000002, 1.0), 23.896466746816948), ((0.050000000000000003, 0.65000000000000002, 1.0), 23.895600945951145), ((0.0, 0.65000000000000002, 1.0), 23.895600945951145), ((0.10000000000000001, 0.65000000000000002, 1.0), 23.895600945951145), ((0.20000000000000001, 0.65000000000000002, 1.0), 23.895600945951145), ((0.15000000000000002, 0.65000000000000002, 1.0), 23.895600945951145), ((0.30000000000000004, 0.65000000000000002, 1.0), 23.883381162714375), ((0.25, 0.40000000000000002, 1.0), 23.8370167055026), ((0.15000000000000002, 0.40000000000000002, 1.0), 23.836231776146242), ((0.0, 0.40000000000000002, 1.0), 23.836231776146242), ((0.10000000000000001, 0.40000000000000002, 1.0), 23.836231776146242), ((0.20000000000000001, 0.40000000000000002, 1.0), 23.836231776146242), ((0.050000000000000003, 0.40000000000000002, 1.0), 23.836231776146242), ((0.35000000000000003, 0.65000000000000002, 1.0), 23.830226068482062), ((0.30000000000000004, 0.40000000000000002, 1.0), 23.82108775978975), ((0.70000000000000007, 0.80000000000000004, 1.0), 23.81688651196713)], [((0.70000000000000007, 0.85000000000000009, 1.0), 20.921116070538655), ((0.70000000000000007, 0.80000000000000004, 1.0), 20.919284313642404), ((0.70000000000000007, 1.0, 1.0), 20.898981621992014), ((0.70000000000000007, 0.95000000000000007, 1.0), 20.898981621992014), ((0.65000000000000002, 0.75, 1.0), 20.88970574255239), ((0.65000000000000002, 0.80000000000000004, 1.0), 20.88454821650535), ((0.70000000000000007, 0.90000000000000002, 1.0), 20.883229440594633), ((0.65000000000000002, 1.0, 1.0), 20.85248686965871), ((0.65000000000000002, 0.95000000000000007, 1.0), 20.85248686965871), ((0.65000000000000002, 0.85000000000000009, 1.0), 20.846975242161996), ((0.65000000000000002, 0.90000000000000002, 1.0), 20.82972390605981), ((0.60000000000000009, 0.95000000000000007, 1.0), 20.76590466596786), ((0.60000000000000009, 1.0, 1.0), 20.76590466596786), ((0.60000000000000009, 0.75, 1.0), 20.763393025031533), ((0.60000000000000009, 0.80000000000000004, 1.0), 20.756073843982687), ((0.60000000000000009, 0.85000000000000009, 1.0), 20.754570959190946), ((0.60000000000000009, 0.90000000000000002, 1.0), 20.74490652348534), ((0.55000000000000004, 0.60000000000000009, 1.0), 20.605917921105576), ((0.45000000000000001, 0.60000000000000009, 1.0), 20.565782498873766), ((0.60000000000000009, 0.65000000000000002, 1.0), 20.52963592989439), ((0.60000000000000009, 0.70000000000000007, 1.0), 20.49985919415231), ((0.70000000000000007, 0.75, 1.0), 20.499665602384542), ((0.5, 0.65000000000000002, 1.0), 20.489663187103627), ((0.45000000000000001, 0.80000000000000004, 1.0), 20.48539839624486), ((0.40000000000000002, 0.80000000000000004, 1.0), 20.481985981127735), ((0.45000000000000001, 0.5, 1.0), 20.474844256750018), ((0.45000000000000001, 0.85000000000000009, 1.0), 20.470056689818463), ((0.40000000000000002, 0.85000000000000009, 1.0), 20.466644274701338), ((0.35000000000000003, 0.80000000000000004, 1.0), 20.46558910020646), ((0.45000000000000001, 0.95000000000000007, 1.0), 20.462959908303333), ((0.45000000000000001, 1.0, 1.0), 20.462959908303333), ((0.65000000000000002, 0.70000000000000007, 1.0), 20.46004150303867), ((0.40000000000000002, 1.0, 1.0), 20.459547493186204), ((0.40000000000000002, 0.95000000000000007, 1.0), 20.459547493186204), ((0.35000000000000003, 0.85000000000000009, 1.0), 20.450247393780064), ((0.30000000000000004, 0.80000000000000004, 1.0), 20.44464644896212), ((0.35000000000000003, 1.0, 1.0), 20.443150612264933), ((0.35000000000000003, 0.95000000000000007, 1.0), 20.443150612264933), ((0.45000000000000001, 0.90000000000000002, 1.0), 20.441961765820814), ((0.40000000000000002, 0.90000000000000002, 1.0), 20.438549350703685), ((0.40000000000000002, 0.45000000000000001, 1.0), 20.436814807763255), ((0.30000000000000004, 0.85000000000000009, 1.0), 20.42930474253572), ((0.75, 0.95000000000000007, 1.0), 20.42864265893553), ((0.75, 1.0, 1.0), 20.42864265893553), ((0.30000000000000004, 1.0, 1.0), 20.42220796102059), ((0.30000000000000004, 0.95000000000000007, 1.0), 20.42220796102059), ((0.35000000000000003, 0.90000000000000002, 1.0), 20.422152469782418), ((0.35000000000000003, 0.40000000000000002, 1.0), 20.419640832501045), ((0.75, 0.90000000000000002, 1.0), 20.41534524217389), ((0.55000000000000004, 0.75, 1.0), 20.413946665998747), ((0.75, 0.85000000000000009, 1.0), 20.404743557385864), ((0.30000000000000004, 0.90000000000000002, 1.0), 20.401209818538078), ((0.45000000000000001, 0.75, 1.0), 20.39990448159586), ((0.75, 0.80000000000000004, 1.0), 20.397453234400313), ((0.40000000000000002, 0.75, 1.0), 20.39700911558187), ((0.30000000000000004, 0.35000000000000003, 1.0), 20.392976974672713), ((0.80000000000000004, 0.85000000000000009, 1.0), 20.38286659545427), ((0.35000000000000003, 0.75, 1.0), 20.380612234660603), ((0.45000000000000001, 0.65000000000000002, 1.0), 20.379873197586832), ((0.5, 0.55000000000000004, 1.0), 20.3600491173543), ((0.30000000000000004, 0.75, 1.0), 20.359669583416263), ((0.25, 0.80000000000000004, 1.0), 20.354649551903403), ((0.050000000000000003, 0.30000000000000004, 1.0), 20.34992579022031), ((0.0, 0.30000000000000004, 1.0), 20.34992579022031), ((0.10000000000000001, 0.30000000000000004, 1.0), 20.34992579022031), ((0.15000000000000002, 0.30000000000000004, 1.0), 20.346399101767147), ((0.0, 0.80000000000000004, 1.0), 20.346361659540293), ((0.15000000000000002, 0.80000000000000004, 1.0), 20.346361659540293), ((0.10000000000000001, 0.80000000000000004, 1.0), 20.346361659540293), ((0.050000000000000003, 0.80000000000000004, 1.0), 20.346361659540293), ((0.20000000000000001, 0.80000000000000004, 1.0), 20.34190849184352), ((0.25, 0.85000000000000009, 1.0), 20.33930784547701), ((0.5, 0.60000000000000009, 1.0), 20.33675943328826), ((0.25, 0.95000000000000007, 1.0), 20.332211063961875), ((0.25, 1.0, 1.0), 20.332211063961875), ((0.40000000000000002, 0.60000000000000009, 1.0), 20.33177972485449), ((0.050000000000000003, 0.85000000000000009, 1.0), 20.331019953113902), ((0.10000000000000001, 0.85000000000000009, 1.0), 20.331019953113902), ((0.15000000000000002, 0.85000000000000009, 1.0), 20.331019953113902), ((0.0, 0.85000000000000009, 1.0), 20.331019953113902), ((0.80000000000000004, 0.90000000000000002, 1.0), 20.3282019095354), ((0.80000000000000004, 0.95000000000000007, 1.0), 20.32818884312072), ((0.80000000000000004, 1.0, 1.0), 20.32818884312072), ((0.20000000000000001, 0.85000000000000009, 1.0), 20.32656678541713), ((0.10000000000000001, 0.95000000000000007, 1.0), 20.323923171598768), ((0.15000000000000002, 0.95000000000000007, 1.0), 20.323923171598768), ((0.0, 0.95000000000000007, 1.0), 20.323923171598768), ((0.050000000000000003, 1.0, 1.0), 20.323923171598768), ((0.050000000000000003, 0.95000000000000007, 1.0), 20.323923171598768), ((0.0, 1.0, 1.0), 20.323923171598768), ((0.10000000000000001, 1.0, 1.0), 20.323923171598768), ((0.15000000000000002, 1.0, 1.0), 20.323923171598768), ((0.55000000000000004, 0.65000000000000002, 1.0), 20.31956908041041), ((0.20000000000000001, 1.0, 1.0), 20.319470003901994), ((0.20000000000000001, 0.95000000000000007, 1.0), 20.319470003901994), ((0.20000000000000001, 0.30000000000000004, 1.0), 20.31565007574934), ((0.85000000000000009, 1.0, 1.0), 20.31546227572145), ((0.85000000000000009, 0.95000000000000007, 1.0), 20.31546227572145), ((0.10000000000000001, 0.25, 1.0), 20.314377820091046), ((0.0, 0.25, 1.0), 20.314377820091046), ((0.050000000000000003, 0.25, 1.0), 20.314377820091046), ((0.25, 0.90000000000000002, 1.0), 20.31121292147936)]], 'Wedding': [[((0.60000000000000009, 0.65000000000000002, 1.0), 22.43821039927611)], [((0.40000000000000002, 0.85000000000000009, 1.0), 22.56233044835697), ((0.40000000000000002, 0.90000000000000002, 1.0), 22.525663781690305), ((0.35000000000000003, 0.80000000000000004, 1.0), 22.525045988425045), ((0.35000000000000003, 0.85000000000000009, 1.0), 22.51960719111125), ((0.40000000000000002, 1.0, 1.0), 22.515293411319934), ((0.40000000000000002, 0.95000000000000007, 1.0), 22.515293411319934), ((0.30000000000000004, 0.5, 1.0), 22.48673967093923), ((0.35000000000000003, 1.0, 1.0), 22.472570154074212), ((0.35000000000000003, 0.95000000000000007, 1.0), 22.472570154074212), ((0.35000000000000003, 0.90000000000000002, 1.0), 22.472570154074212), ((0.40000000000000002, 0.80000000000000004, 1.0), 22.409412030236634), ((0.30000000000000004, 0.55000000000000004, 1.0), 22.375662637554093), ((0.40000000000000002, 0.75, 1.0), 22.36647980852316), ((0.35000000000000003, 0.75, 1.0), 22.361248056300184), ((0.35000000000000003, 0.5, 1.0), 22.356553537334122), ((0.35000000000000003, 0.60000000000000009, 1.0), 22.35211515677303), ((0.30000000000000004, 0.85000000000000009, 1.0), 22.33221017929945), ((0.35000000000000003, 0.55000000000000004, 1.0), 22.33113459427036), ((0.30000000000000004, 0.80000000000000004, 1.0), 22.328065643279917), ((0.30000000000000004, 0.60000000000000009, 1.0), 22.32497147297046), ((0.40000000000000002, 0.70000000000000007, 1.0), 22.321688320987974), ((0.35000000000000003, 0.70000000000000007, 1.0), 22.308269889926805), ((0.40000000000000002, 0.5, 1.0), 22.30825453305253), ((0.35000000000000003, 0.65000000000000002, 1.0), 22.30122036751452), ((0.40000000000000002, 0.60000000000000009, 1.0), 22.29842386997179), ((0.30000000000000004, 0.65000000000000002, 1.0), 22.289716029628842), ((0.40000000000000002, 0.65000000000000002, 1.0), 22.286774917486092), ((0.30000000000000004, 0.90000000000000002, 1.0), 22.268136105225377), ((0.30000000000000004, 1.0, 1.0), 22.263537636564408), ((0.30000000000000004, 0.95000000000000007, 1.0), 22.263537636564408), ((0.45000000000000001, 0.90000000000000002, 1.0), 22.2552436807324), ((0.30000000000000004, 0.75, 1.0), 22.219778606242876), ((0.45000000000000001, 0.95000000000000007, 1.0), 22.21857701406573), ((0.45000000000000001, 1.0, 1.0), 22.208206643695366), ((0.40000000000000002, 0.55000000000000004, 1.0), 22.200819163091843), ((0.30000000000000004, 0.70000000000000007, 1.0), 22.158970500225976), ((0.35000000000000003, 0.45000000000000001, 1.0), 22.11482445550376), ((0.45000000000000001, 0.85000000000000009, 1.0), 22.097791161029882), ((0.55000000000000004, 1.0, 1.0), 22.094797723665906), ((0.25, 0.5, 1.0), 22.08072253547785), ((0.45000000000000001, 0.55000000000000004, 1.0), 22.070304702947933), ((0.45000000000000001, 0.80000000000000004, 1.0), 22.05840906577725), ((0.30000000000000004, 0.45000000000000001, 1.0), 22.047469182199382), ((0.45000000000000001, 0.75, 1.0), 22.020372382442947), ((0.60000000000000009, 1.0, 1.0), 22.010061338565073), ((0.5, 0.95000000000000007, 1.0), 22.00151336987887), ((0.45000000000000001, 0.5, 1.0), 21.97687839819152), ((0.45000000000000001, 0.65000000000000002, 1.0), 21.973429114260703), ((0.45000000000000001, 0.70000000000000007, 1.0), 21.966662007840362), ((0.60000000000000009, 0.95000000000000007, 1.0), 21.96535234385608), ((0.5, 1.0, 1.0), 21.964846703212206), ((0.25, 0.45000000000000001, 1.0), 21.959459372846478), ((0.45000000000000001, 0.60000000000000009, 1.0), 21.958765564145324), ((0.65000000000000002, 1.0, 1.0), 21.957289703619992), ((0.55000000000000004, 0.95000000000000007, 1.0), 21.933519945888133), ((0.70000000000000007, 1.0, 1.0), 21.93320966574965), ((0.60000000000000009, 0.90000000000000002, 1.0), 21.927623504307064), ((0.40000000000000002, 0.45000000000000001, 1.0), 21.924099474824953), ((0.35000000000000003, 0.40000000000000002, 1.0), 21.9223126440029)], [((0.30000000000000004, 0.45000000000000001, 1.0), 22.984778405071534), ((0.35000000000000003, 0.5, 1.0), 22.955807051875635), ((0.35000000000000003, 0.55000000000000004, 1.0), 22.85222608101165), ((0.30000000000000004, 0.55000000000000004, 1.0), 22.849007527732294), ((0.30000000000000004, 0.40000000000000002, 1.0), 22.838877375352194), ((0.25, 0.45000000000000001, 1.0), 22.8205188110543), ((0.30000000000000004, 0.5, 1.0), 22.802303740242728), ((0.35000000000000003, 0.60000000000000009, 1.0), 22.799839154713503), ((0.40000000000000002, 0.55000000000000004, 1.0), 22.796214460700924), ((0.45000000000000001, 0.75, 1.0), 22.782037216600738), ((0.35000000000000003, 0.45000000000000001, 1.0), 22.780348755585862), ((0.45000000000000001, 0.85000000000000009, 1.0), 22.765533628052815), ((0.45000000000000001, 0.95000000000000007, 1.0), 22.765533628052815), ((0.45000000000000001, 0.90000000000000002, 1.0), 22.765533628052815), ((0.45000000000000001, 0.80000000000000004, 1.0), 22.765533628052815), ((0.45000000000000001, 1.0, 1.0), 22.765533628052815), ((0.25, 0.40000000000000002, 1.0), 22.759678134119397), ((0.35000000000000003, 0.40000000000000002, 1.0), 22.754841675150846), ((0.40000000000000002, 0.5, 1.0), 22.750569324631808), ((0.35000000000000003, 0.65000000000000002, 1.0), 22.737668723302306), ((0.45000000000000001, 0.70000000000000007, 1.0), 22.721865336433336), ((0.40000000000000002, 0.45000000000000001, 1.0), 22.68141589361442), ((0.45000000000000001, 0.55000000000000004, 1.0), 22.67486007278362), ((0.45000000000000001, 0.65000000000000002, 1.0), 22.67295890989144), ((0.25, 0.55000000000000004, 1.0), 22.672234672635696), ((0.30000000000000004, 0.60000000000000009, 1.0), 22.668032826225335), ((0.40000000000000002, 0.70000000000000007, 1.0), 22.650504851305982), ((0.25, 0.5, 1.0), 22.636213203558544), ((0.45000000000000001, 0.5, 1.0), 22.60586535489176), ((0.35000000000000003, 0.70000000000000007, 1.0), 22.59686174833851), ((0.45000000000000001, 0.60000000000000009, 1.0), 22.595163619012986), ((0.40000000000000002, 0.60000000000000009, 1.0), 22.594723232395353), ((0.35000000000000003, 0.80000000000000004, 1.0), 22.578923984587934), ((0.35000000000000003, 0.75, 1.0), 22.578923984587934), ((0.40000000000000002, 0.65000000000000002, 1.0), 22.5653605115548), ((0.35000000000000003, 1.0, 1.0), 22.554973477878814), ((0.35000000000000003, 0.95000000000000007, 1.0), 22.554973477878814), ((0.35000000000000003, 0.85000000000000009, 1.0), 22.554973477878814), ((0.35000000000000003, 0.90000000000000002, 1.0), 22.554973477878814), ((0.40000000000000002, 0.75, 1.0), 22.51620015764168), ((0.40000000000000002, 0.80000000000000004, 1.0), 22.51620015764168), ((0.40000000000000002, 0.90000000000000002, 1.0), 22.49224965093255), ((0.40000000000000002, 1.0, 1.0), 22.49224965093255), ((0.40000000000000002, 0.85000000000000009, 1.0), 22.49224965093255), ((0.40000000000000002, 0.95000000000000007, 1.0), 22.49224965093255), ((0.25, 0.35000000000000003, 1.0), 22.472221320286206), ((0.30000000000000004, 0.35000000000000003, 1.0), 22.464075757655575), ((0.30000000000000004, 0.65000000000000002, 1.0), 22.436741477886574), ((0.30000000000000004, 0.70000000000000007, 1.0), 22.42237929766017), ((0.30000000000000004, 0.75, 1.0), 22.39661567977833), ((0.30000000000000004, 0.80000000000000004, 1.0), 22.39661567977833), ((0.20000000000000001, 0.45000000000000001, 1.0), 22.37787712135303), ((0.30000000000000004, 0.85000000000000009, 1.0), 22.3726651730692), ((0.30000000000000004, 0.90000000000000002, 1.0), 22.3726651730692), ((0.30000000000000004, 1.0, 1.0), 22.3726651730692), ((0.30000000000000004, 0.95000000000000007, 1.0), 22.3726651730692), ((0.25, 0.30000000000000004, 1.0), 22.34043986738912), ((0.25, 0.65000000000000002, 1.0), 22.335158291134388), ((0.25, 0.70000000000000007, 1.0), 22.306381606875856), ((0.5, 0.75, 1.0), 22.301803440678217), ((0.5, 0.90000000000000002, 1.0), 22.292979419001526), ((0.5, 0.85000000000000009, 1.0), 22.292979419001526), ((0.5, 0.95000000000000007, 1.0), 22.292979419001526), ((0.5, 1.0, 1.0), 22.292979419001526), ((0.5, 0.80000000000000004, 1.0), 22.292979419001526), ((0.25, 0.60000000000000009, 1.0), 22.2892431203689), ((0.25, 0.75, 1.0), 22.278506799690703), ((0.25, 0.80000000000000004, 1.0), 22.278506799690703), ((0.20000000000000001, 0.40000000000000002, 1.0), 22.268713526878265), ((0.25, 0.95000000000000007, 1.0), 22.254556292981576), ((0.25, 1.0, 1.0), 22.254556292981576), ((0.25, 0.90000000000000002, 1.0), 22.254556292981576), ((0.25, 0.85000000000000009, 1.0), 22.254556292981576), ((0.5, 0.70000000000000007, 1.0), 22.254514176393876), ((0.5, 0.55000000000000004, 1.0), 22.22325856926816), ((0.15000000000000002, 0.20000000000000001, 1.0), 22.218551104695294), ((0.5, 0.60000000000000009, 1.0), 22.199095897516024), ((0.5, 0.65000000000000002, 1.0), 22.15230024484425), ((0.20000000000000001, 0.25, 1.0), 22.135792777327147), ((0.20000000000000001, 0.5, 1.0), 22.09730871640047), ((0.15000000000000002, 0.45000000000000001, 1.0), 22.09203862172009), ((0.55000000000000004, 0.80000000000000004, 1.0), 22.088592515592207), ((0.55000000000000004, 0.85000000000000009, 1.0), 22.079768493915523), ((0.55000000000000004, 1.0, 1.0), 22.079768493915523), ((0.55000000000000004, 0.95000000000000007, 1.0), 22.079768493915523), ((0.55000000000000004, 0.90000000000000002, 1.0), 22.079768493915523), ((0.10000000000000001, 0.15000000000000002, 1.0), 22.073933193462274)]], 'ReligiousActivity': [[], [((0.30000000000000004, 0.45000000000000001, 1.0), 20.078900856527156), ((0.30000000000000004, 0.5, 1.0), 20.067959534262137), ((0.10000000000000001, 0.45000000000000001, 1.0), 20.04349901099446), ((0.0, 0.45000000000000001, 1.0), 20.04349901099446), ((0.050000000000000003, 0.45000000000000001, 1.0), 20.04349901099446), ((0.15000000000000002, 0.45000000000000001, 1.0), 20.041882979966665), ((0.20000000000000001, 0.45000000000000001, 1.0), 20.034355255586526), ((0.25, 0.45000000000000001, 1.0), 20.020441089631547), ((0.0, 0.5, 1.0), 19.968700948809445), ((0.10000000000000001, 0.5, 1.0), 19.968700948809445), ((0.050000000000000003, 0.5, 1.0), 19.968700948809445), ((0.15000000000000002, 0.5, 1.0), 19.967203494481797), ((0.20000000000000001, 0.5, 1.0), 19.960580895806753), ((0.25, 0.5, 1.0), 19.94903158809709), ((0.35000000000000003, 0.5, 1.0), 19.741801655412406), ((0.35000000000000003, 0.45000000000000001, 1.0), 19.643590269673407), ((0.40000000000000002, 0.45000000000000001, 1.0), 19.56637933030836), ((0.40000000000000002, 0.5, 1.0), 19.542088265010648), ((0.30000000000000004, 0.65000000000000002, 1.0), 19.429288693997155), ((0.30000000000000004, 0.70000000000000007, 1.0), 19.393316028886172), ((0.30000000000000004, 0.75, 1.0), 19.376219763175516), ((0.30000000000000004, 0.85000000000000009, 1.0), 19.350845515352283), ((0.30000000000000004, 0.90000000000000002, 1.0), 19.350845515352283), ((0.30000000000000004, 1.0, 1.0), 19.350845515352283), ((0.30000000000000004, 0.95000000000000007, 1.0), 19.350845515352283), ((0.30000000000000004, 0.80000000000000004, 1.0), 19.350845515352283), ((0.050000000000000003, 0.65000000000000002, 1.0), 19.329742671158456), ((0.0, 0.65000000000000002, 1.0), 19.329742671158456), ((0.10000000000000001, 0.65000000000000002, 1.0), 19.329742671158456), ((0.15000000000000002, 0.65000000000000002, 1.0), 19.328245216830815), ((0.20000000000000001, 0.65000000000000002, 1.0), 19.3227049557606), ((0.30000000000000004, 0.55000000000000004, 1.0), 19.32015842939076), ((0.25, 0.65000000000000002, 1.0), 19.31371426391276), ((0.10000000000000001, 0.70000000000000007, 1.0), 19.30341722822737), ((0.0, 0.70000000000000007, 1.0), 19.30341722822737), ((0.050000000000000003, 0.70000000000000007, 1.0), 19.30341722822737), ((0.30000000000000004, 0.60000000000000009, 1.0), 19.30228889782242), ((0.15000000000000002, 0.70000000000000007, 1.0), 19.30191977389973), ((0.20000000000000001, 0.70000000000000007, 1.0), 19.296379512829514), ((0.25, 0.70000000000000007, 1.0), 19.288054798155592), ((0.10000000000000001, 0.75, 1.0), 19.283678845544863), ((0.0, 0.75, 1.0), 19.283678845544863), ((0.050000000000000003, 0.75, 1.0), 19.283678845544863), ((0.15000000000000002, 0.75, 1.0), 19.282181391217218), ((0.20000000000000001, 0.75, 1.0), 19.276641130147006), ((0.25, 0.75, 1.0), 19.268316415473087), ((0.0, 0.90000000000000002, 1.0), 19.25830459772164), ((0.10000000000000001, 0.95000000000000007, 1.0), 19.25830459772164), ((0.050000000000000003, 0.85000000000000009, 1.0), 19.25830459772164), ((0.10000000000000001, 0.90000000000000002, 1.0), 19.25830459772164), ((0.0, 0.95000000000000007, 1.0), 19.25830459772164), ((0.0, 0.80000000000000004, 1.0), 19.25830459772164), ((0.050000000000000003, 1.0, 1.0), 19.25830459772164), ((0.10000000000000001, 0.80000000000000004, 1.0), 19.25830459772164), ((0.050000000000000003, 0.95000000000000007, 1.0), 19.25830459772164), ((0.0, 1.0, 1.0), 19.25830459772164), ((0.10000000000000001, 0.85000000000000009, 1.0), 19.25830459772164), ((0.050000000000000003, 0.80000000000000004, 1.0), 19.25830459772164), ((0.050000000000000003, 0.90000000000000002, 1.0), 19.25830459772164), ((0.10000000000000001, 1.0, 1.0), 19.25830459772164), ((0.0, 0.85000000000000009, 1.0), 19.25830459772164), ((0.15000000000000002, 0.90000000000000002, 1.0), 19.256807143393992), ((0.15000000000000002, 0.95000000000000007, 1.0), 19.256807143393992), ((0.15000000000000002, 0.80000000000000004, 1.0), 19.256807143393992), ((0.15000000000000002, 0.85000000000000009, 1.0), 19.256807143393992), ((0.15000000000000002, 1.0, 1.0), 19.256807143393992), ((0.20000000000000001, 0.80000000000000004, 1.0), 19.25126688232378), ((0.20000000000000001, 0.85000000000000009, 1.0), 19.25126688232378), ((0.20000000000000001, 1.0, 1.0), 19.25126688232378), ((0.20000000000000001, 0.95000000000000007, 1.0), 19.25126688232378), ((0.20000000000000001, 0.90000000000000002, 1.0), 19.25126688232378), ((0.25, 0.95000000000000007, 1.0), 19.242942167649858), ((0.25, 0.80000000000000004, 1.0), 19.242942167649858), ((0.25, 1.0, 1.0), 19.242942167649858), ((0.25, 0.90000000000000002, 1.0), 19.242942167649858), ((0.25, 0.85000000000000009, 1.0), 19.242942167649858), ((0.050000000000000003, 0.60000000000000009, 1.0), 19.217725015302573), ((0.10000000000000001, 0.60000000000000009, 1.0), 19.217725015302573), ((0.0, 0.60000000000000009, 1.0), 19.217725015302573), ((0.15000000000000002, 0.60000000000000009, 1.0), 19.21622756097493), ((0.10000000000000001, 0.55000000000000004, 1.0), 19.215340222856316), ((0.0, 0.55000000000000004, 1.0), 19.215340222856316), ((0.050000000000000003, 0.55000000000000004, 1.0), 19.215340222856316), ((0.15000000000000002, 0.55000000000000004, 1.0), 19.21384276852867), ((0.20000000000000001, 0.60000000000000009, 1.0), 19.210390099238854), ((0.20000000000000001, 0.55000000000000004, 1.0), 19.207677361230257), ((0.25, 0.60000000000000009, 1.0), 19.200650183070366), ((0.25, 0.55000000000000004, 1.0), 19.19793744506177), ((0.35000000000000003, 0.65000000000000002, 1.0), 19.183153037276107), ((0.35000000000000003, 0.70000000000000007, 1.0), 19.16179621345985), ((0.35000000000000003, 0.75, 1.0), 19.138163240921497), ((0.35000000000000003, 0.80000000000000004, 1.0), 19.10187487586626), ((0.35000000000000003, 1.0, 1.0), 19.10187487586626), ((0.35000000000000003, 0.95000000000000007, 1.0), 19.10187487586626), ((0.35000000000000003, 0.85000000000000009, 1.0), 19.10187487586626), ((0.35000000000000003, 0.90000000000000002, 1.0), 19.10187487586626), ((0.35000000000000003, 0.55000000000000004, 1.0), 19.0660067049642), ((0.40000000000000002, 0.65000000000000002, 1.0), 19.055518854276965), ((0.35000000000000003, 0.60000000000000009, 1.0), 19.054460955023934), ((0.0, 0.40000000000000002, 1.0), 19.039495972944433), ((0.10000000000000001, 0.40000000000000002, 1.0), 19.039495972944433), ((0.050000000000000003, 0.40000000000000002, 1.0), 19.039495972944433), ((0.15000000000000002, 0.40000000000000002, 1.0), 19.03781530067553), ((0.40000000000000002, 0.70000000000000007, 1.0), 19.037348481422264), ((0.20000000000000001, 0.40000000000000002, 1.0), 19.029161422483533), ((0.40000000000000002, 0.75, 1.0), 19.022320166905818), ((0.25, 0.40000000000000002, 1.0), 19.015247256528554), ((0.40000000000000002, 0.90000000000000002, 1.0), 18.98952371071819), ((0.40000000000000002, 1.0, 1.0), 18.98952371071819), ((0.40000000000000002, 0.85000000000000009, 1.0), 18.98952371071819), ((0.40000000000000002, 0.95000000000000007, 1.0), 18.98952371071819), ((0.40000000000000002, 0.80000000000000004, 1.0), 18.98952371071819), ((0.30000000000000004, 0.40000000000000002, 1.0), 18.913809602734602), ((0.40000000000000002, 0.60000000000000009, 1.0), 18.906722412623377), ((0.40000000000000002, 0.55000000000000004, 1.0), 18.875362077528848), ((0.45000000000000001, 0.5, 1.0), 18.442738233310976), ((0.35000000000000003, 0.40000000000000002, 1.0), 18.34707971070537), ((0.45000000000000001, 0.65000000000000002, 1.0), 18.12013060292581), ((0.45000000000000001, 0.70000000000000007, 1.0), 18.115368698163902), ((0.45000000000000001, 0.75, 1.0), 18.096196307698996), ((0.45000000000000001, 0.80000000000000004, 1.0), 18.07841865744969), ((0.45000000000000001, 0.85000000000000009, 1.0), 18.07218082189831), ((0.45000000000000001, 0.95000000000000007, 1.0), 18.07218082189831), ((0.45000000000000001, 0.90000000000000002, 1.0), 18.07218082189831), ((0.45000000000000001, 1.0, 1.0), 18.07218082189831), ((0.45000000000000001, 0.60000000000000009, 1.0), 17.924974263021255), ((0.45000000000000001, 0.55000000000000004, 1.0), 17.858622351956903), ((0.0, 0.35000000000000003, 1.0), 17.825235255078), ((0.050000000000000003, 0.35000000000000003, 1.0), 17.825235255078), ((0.10000000000000001, 0.35000000000000003, 1.0), 17.825235255078), ((0.15000000000000002, 0.35000000000000003, 1.0), 17.82288528953824), ((0.20000000000000001, 0.35000000000000003, 1.0), 17.817035625638695), ((0.25, 0.35000000000000003, 1.0), 17.793812126865223), ((0.60000000000000009, 0.75, 1.0), 17.464586906675034), ((0.60000000000000009, 0.85000000000000009, 1.0), 17.46409720030297), ((0.60000000000000009, 0.80000000000000004, 1.0), 17.46409720030297), ((0.60000000000000009, 0.95000000000000007, 1.0), 17.46409720030297), ((0.60000000000000009, 0.90000000000000002, 1.0), 17.46409720030297), ((0.60000000000000009, 1.0, 1.0), 17.46409720030297), ((0.10000000000000001, 0.25, 1.0), 17.451570883913906), ((0.0, 0.25, 1.0), 17.451570883913906), ((0.050000000000000003, 0.25, 1.0), 17.451570883913906), ((0.15000000000000002, 0.25, 1.0), 17.447112873925946), ((0.60000000000000009, 0.70000000000000007, 1.0), 17.38786612745425), ((0.20000000000000001, 0.25, 1.0), 17.379321880841935), ((0.5, 0.55000000000000004, 1.0), 17.353900308729237), ((0.0, 0.20000000000000001, 1.0), 17.34412304075178), ((0.050000000000000003, 0.20000000000000001, 1.0), 17.34412304075178), ((0.10000000000000001, 0.20000000000000001, 1.0), 17.339665030763815), ((0.65000000000000002, 0.75, 1.0), 17.32791617045996), ((0.65000000000000002, 0.80000000000000004, 1.0), 17.326524704782777), ((0.65000000000000002, 1.0, 1.0), 17.32603499841072), ((0.65000000000000002, 0.85000000000000009, 1.0), 17.32603499841072), ((0.65000000000000002, 0.90000000000000002, 1.0), 17.32603499841072), ((0.65000000000000002, 0.95000000000000007, 1.0), 17.32603499841072), ((0.60000000000000009, 0.65000000000000002, 1.0), 17.297102578789524), ((0.65000000000000002, 0.70000000000000007, 1.0), 17.297102578789524), ((0.15000000000000002, 0.20000000000000001, 1.0), 17.296606976698463), ((0.55000000000000004, 0.60000000000000009, 1.0), 17.28038559670612)], [((0.40000000000000002, 0.55000000000000004, 1.0), 14.032148968882105), ((0.35000000000000003, 0.55000000000000004, 1.0), 14.031764353497488), ((0.30000000000000004, 0.55000000000000004, 1.0), 14.031077540310676), ((0.10000000000000001, 0.55000000000000004, 1.0), 14.030213485011137), ((0.0, 0.55000000000000004, 1.0), 14.030213485011137), ((0.050000000000000003, 0.55000000000000004, 1.0), 14.030213485011137), ((0.20000000000000001, 0.55000000000000004, 1.0), 14.030213485011137), ((0.25, 0.55000000000000004, 1.0), 14.030213485011137), ((0.15000000000000002, 0.55000000000000004, 1.0), 14.030213485011137), ((0.40000000000000002, 0.60000000000000009, 1.0), 13.990715863393095), ((0.35000000000000003, 0.60000000000000009, 1.0), 13.99033124800848), ((0.30000000000000004, 0.60000000000000009, 1.0), 13.989644434821667), ((0.050000000000000003, 0.60000000000000009, 1.0), 13.98878037952213), ((0.15000000000000002, 0.60000000000000009, 1.0), 13.98878037952213), ((0.20000000000000001, 0.60000000000000009, 1.0), 13.98878037952213), ((0.10000000000000001, 0.60000000000000009, 1.0), 13.98878037952213), ((0.0, 0.60000000000000009, 1.0), 13.98878037952213), ((0.25, 0.60000000000000009, 1.0), 13.98878037952213), ((0.45000000000000001, 0.55000000000000004, 1.0), 13.956346300211303), ((0.45000000000000001, 0.60000000000000009, 1.0), 13.92814482283503), ((0.55000000000000004, 0.65000000000000002, 1.0), 13.836208374050896), ((0.5, 0.60000000000000009, 1.0), 13.832520923571055), ((0.60000000000000009, 0.65000000000000002, 1.0), 13.333396898431486), ((0.40000000000000002, 0.5, 1.0), 13.316734972239544), ((0.35000000000000003, 0.5, 1.0), 13.31635035685493), ((0.30000000000000004, 0.5, 1.0), 13.315663543668116), ((0.25, 0.5, 1.0), 13.314799488368577), ((0.15000000000000002, 0.5, 1.0), 13.314799488368577), ((0.0, 0.5, 1.0), 13.314799488368577), ((0.20000000000000001, 0.5, 1.0), 13.314799488368577), ((0.10000000000000001, 0.5, 1.0), 13.314799488368577), ((0.050000000000000003, 0.5, 1.0), 13.314799488368577), ((0.55000000000000004, 0.60000000000000009, 1.0), 13.312867085448106), ((0.5, 0.55000000000000004, 1.0), 13.268222181786024), ((0.45000000000000001, 0.5, 1.0), 13.248880482855027), ((0.60000000000000009, 0.70000000000000007, 1.0), 13.228144093286438)]], 'Graduation': [[], [((0.5, 0.95000000000000007, 1.0), 17.747684496881043), ((0.5, 1.0, 1.0), 17.747684496881043), ((0.5, 0.90000000000000002, 1.0), 17.71731671651326), ((0.5, 0.80000000000000004, 1.0), 17.71678234827019), ((0.5, 0.85000000000000009, 1.0), 17.708632268245477), ((0.5, 0.75, 1.0), 17.70269590200605), ((0.5, 0.70000000000000007, 1.0), 17.695045884871774), ((0.45000000000000001, 0.95000000000000007, 1.0), 17.523183891452387), ((0.45000000000000001, 1.0, 1.0), 17.523183891452387), ((0.55000000000000004, 1.0, 1.0), 17.50868226413891), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.50868226413891), ((0.55000000000000004, 0.70000000000000007, 1.0), 17.50604365185447), ((0.45000000000000001, 0.90000000000000002, 1.0), 17.492816111084604), ((0.45000000000000001, 0.80000000000000004, 1.0), 17.49118346672652), ((0.45000000000000001, 0.75, 1.0), 17.487300394604773), ((0.45000000000000001, 0.70000000000000007, 1.0), 17.484318723390963), ((0.45000000000000001, 0.85000000000000009, 1.0), 17.484131662816825), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.47831448377113), ((0.5, 0.65000000000000002, 1.0), 17.473216915243864), ((0.55000000000000004, 0.85000000000000009, 1.0), 17.46963003550335), ((0.55000000000000004, 0.80000000000000004, 1.0), 17.467842166540827), ((0.5, 0.60000000000000009, 1.0), 17.45417411041972), ((0.55000000000000004, 0.75, 1.0), 17.44642747093161), ((0.10000000000000001, 0.95000000000000007, 1.0), 17.324104042204446), ((0.10000000000000001, 1.0, 1.0), 17.324104042204446), ((0.0, 0.95000000000000007, 1.0), 17.323922488101324), ((0.050000000000000003, 1.0, 1.0), 17.323922488101324), ((0.050000000000000003, 0.95000000000000007, 1.0), 17.323922488101324), ((0.0, 1.0, 1.0), 17.323922488101324), ((0.10000000000000001, 0.90000000000000002, 1.0), 17.293736261836663), ((0.0, 0.90000000000000002, 1.0), 17.29355470773354), ((0.050000000000000003, 0.90000000000000002, 1.0), 17.29355470773354), ((0.10000000000000001, 0.80000000000000004, 1.0), 17.29212902799617), ((0.0, 0.80000000000000004, 1.0), 17.29194747389305), ((0.050000000000000003, 0.80000000000000004, 1.0), 17.29194747389305), ((0.10000000000000001, 0.70000000000000007, 1.0), 17.28910592698159), ((0.0, 0.70000000000000007, 1.0), 17.28892437287847), ((0.050000000000000003, 0.70000000000000007, 1.0), 17.28892437287847), ((0.10000000000000001, 0.75, 1.0), 17.28626255212143), ((0.0, 0.75, 1.0), 17.286080998018306), ((0.050000000000000003, 0.75, 1.0), 17.286080998018306), ((0.10000000000000001, 0.85000000000000009, 1.0), 17.28505181356888), ((0.050000000000000003, 0.85000000000000009, 1.0), 17.28487025946576), ((0.0, 0.85000000000000009, 1.0), 17.28487025946576), ((0.15000000000000002, 0.95000000000000007, 1.0), 17.26204103881456), ((0.15000000000000002, 1.0, 1.0), 17.26204103881456), ((0.45000000000000001, 0.65000000000000002, 1.0), 17.246603268850766), ((0.15000000000000002, 0.90000000000000002, 1.0), 17.23167325844678), ((0.15000000000000002, 0.80000000000000004, 1.0), 17.230066024606288), ((0.15000000000000002, 0.70000000000000007, 1.0), 17.227042923591704), ((0.15000000000000002, 0.75, 1.0), 17.224199548731544), ((0.15000000000000002, 0.85000000000000009, 1.0), 17.222988810178993), ((0.55000000000000004, 0.65000000000000002, 1.0), 17.20723502963997), ((0.45000000000000001, 0.60000000000000009, 1.0), 17.17230616279221), ((0.20000000000000001, 1.0, 1.0), 17.075181067495684), ((0.20000000000000001, 0.95000000000000007, 1.0), 17.075181067495684), ((0.10000000000000001, 0.65000000000000002, 1.0), 17.051157044686256), ((0.050000000000000003, 0.65000000000000002, 1.0), 17.050975490583134), ((0.0, 0.65000000000000002, 1.0), 17.050975490583134), ((0.10000000000000001, 0.45000000000000001, 1.0), 17.04939307106573), ((0.0, 0.45000000000000001, 1.0), 17.049211516962608), ((0.050000000000000003, 0.45000000000000001, 1.0), 17.049211516962608), ((0.5, 0.55000000000000004, 1.0), 17.04751525851211), ((0.20000000000000001, 0.90000000000000002, 1.0), 17.0448132871279), ((0.20000000000000001, 0.80000000000000004, 1.0), 17.04320605328741), ((0.20000000000000001, 0.70000000000000007, 1.0), 17.037515613732477), ((0.20000000000000001, 0.75, 1.0), 17.03733957741267), ((0.20000000000000001, 0.85000000000000009, 1.0), 17.036128838860122), ((0.15000000000000002, 0.65000000000000002, 1.0), 16.989094041296372), ((0.15000000000000002, 0.45000000000000001, 1.0), 16.987040284015972), ((0.55000000000000004, 0.60000000000000009, 1.0), 16.92244645627706), ((0.30000000000000004, 1.0, 1.0), 16.889863462866604), ((0.30000000000000004, 0.95000000000000007, 1.0), 16.889863462866604), ((0.45000000000000001, 0.5, 1.0), 16.863715545874534), ((0.30000000000000004, 0.90000000000000002, 1.0), 16.859495682498824), ((0.30000000000000004, 0.80000000000000004, 1.0), 16.85788844865833), ((0.30000000000000004, 0.70000000000000007, 1.0), 16.85627931211028), ((0.30000000000000004, 0.75, 1.0), 16.853436425467883), ((0.30000000000000004, 0.85000000000000009, 1.0), 16.850811234231045), ((0.10000000000000001, 0.60000000000000009, 1.0), 16.846700134824047), ((0.050000000000000003, 0.60000000000000009, 1.0), 16.846518580720925), ((0.0, 0.60000000000000009, 1.0), 16.846518580720925), ((0.40000000000000002, 1.0, 1.0), 16.837001244598635), ((0.40000000000000002, 0.95000000000000007, 1.0), 16.837001244598635), ((0.20000000000000001, 0.45000000000000001, 1.0), 16.818473656470662), ((0.60000000000000009, 0.70000000000000007, 1.0), 16.817795045002285), ((0.40000000000000002, 0.70000000000000007, 1.0), 16.80667872840439), ((0.40000000000000002, 0.90000000000000002, 1.0), 16.806633464230856), ((0.40000000000000002, 0.80000000000000004, 1.0), 16.80253309721369), ((0.20000000000000001, 0.65000000000000002, 1.0), 16.800443946520705), ((0.40000000000000002, 0.85000000000000009, 1.0), 16.797949015963074), ((0.40000000000000002, 0.75, 1.0), 16.796811942273436), ((0.15000000000000002, 0.60000000000000009, 1.0), 16.784637131434163), ((0.60000000000000009, 0.95000000000000007, 1.0), 16.74957760709475), ((0.60000000000000009, 1.0, 1.0), 16.74957760709475), ((0.30000000000000004, 0.45000000000000001, 1.0), 16.734924905734655), ((0.60000000000000009, 0.90000000000000002, 1.0), 16.719209826726967), ((0.60000000000000009, 0.80000000000000004, 1.0), 16.70384634449028), ((0.60000000000000009, 0.85000000000000009, 1.0), 16.699616747858443), ((0.60000000000000009, 0.75, 1.0), 16.69400113166652), ((0.25, 0.95000000000000007, 1.0), 16.689810252097683), ((0.25, 1.0, 1.0), 16.689810252097683), ((0.10000000000000001, 0.5, 1.0), 16.677875186251207), ((0.0, 0.5, 1.0), 16.677693632148085), ((0.050000000000000003, 0.5, 1.0), 16.677693632148085), ((0.40000000000000002, 0.45000000000000001, 1.0), 16.668314732216952), ((0.25, 0.90000000000000002, 1.0), 16.659442471729903), ((0.25, 0.70000000000000007, 1.0), 16.65785685523701), ((0.25, 0.80000000000000004, 1.0), 16.65783523788941), ((0.25, 0.85000000000000009, 1.0), 16.650758023462124), ((0.25, 0.75, 1.0), 16.649139721566584), ((0.35000000000000003, 1.0, 1.0), 16.643833973567155), ((0.35000000000000003, 0.95000000000000007, 1.0), 16.643833973567155), ((0.65000000000000002, 1.0, 1.0), 16.62676884801454), ((0.65000000000000002, 0.95000000000000007, 1.0), 16.62676884801454), ((0.15000000000000002, 0.5, 1.0), 16.626546083300756), ((0.10000000000000001, 0.40000000000000002, 1.0), 16.621438950125363), ((0.0, 0.40000000000000002, 1.0), 16.621257396022237), ((0.050000000000000003, 0.40000000000000002, 1.0), 16.621257396022237), ((0.30000000000000004, 0.65000000000000002, 1.0), 16.618390226035313), ((0.35000000000000003, 0.70000000000000007, 1.0), 16.61701021797298), ((0.35000000000000003, 0.90000000000000002, 1.0), 16.61346619319938), ((0.35000000000000003, 0.80000000000000004, 1.0), 16.611858959358887), ((0.20000000000000001, 0.60000000000000009, 1.0), 16.610992065242673), ((0.60000000000000009, 0.65000000000000002, 1.0), 16.60716427934296), ((0.35000000000000003, 0.85000000000000009, 1.0), 16.604781744931596), ((0.35000000000000003, 0.75, 1.0), 16.602678981363834), ((0.65000000000000002, 0.80000000000000004, 1.0), 16.59919225052608), ((0.65000000000000002, 0.90000000000000002, 1.0), 16.595339166584854), ((0.65000000000000002, 0.85000000000000009, 1.0), 16.593429514034177), ((0.65000000000000002, 0.75, 1.0), 16.59233737233236), ((0.40000000000000002, 0.65000000000000002, 1.0), 16.58165872911421), ((0.15000000000000002, 0.40000000000000002, 1.0), 16.576679337932475), ((0.45000000000000001, 0.55000000000000004, 1.0), 16.566906423546026), ((0.35000000000000003, 0.45000000000000001, 1.0), 16.532824511341236), ((0.20000000000000001, 0.35000000000000003, 1.0), 16.51390248879384), ((0.35000000000000003, 0.40000000000000002, 1.0), 16.50619868855786), ((0.65000000000000002, 0.70000000000000007, 1.0), 16.480892344750742), ((0.25, 0.45000000000000001, 1.0), 16.473769926154734), ((0.20000000000000001, 0.40000000000000002, 1.0), 16.4589127018181), ((0.0, 0.35000000000000003, 1.0), 16.443271721108662), ((0.050000000000000003, 0.35000000000000003, 1.0), 16.443271721108662), ((0.10000000000000001, 0.35000000000000003, 1.0), 16.442712925414828), ((0.20000000000000001, 0.5, 1.0), 16.430092916011706), ((0.30000000000000004, 0.35000000000000003, 1.0), 16.428448326234573), ((0.30000000000000004, 0.60000000000000009, 1.0), 16.423694202575444), ((0.15000000000000002, 0.35000000000000003, 1.0), 16.419797813816654), ((0.25, 0.65000000000000002, 1.0), 16.41840143615992), ((0.25, 0.40000000000000002, 1.0), 16.412015811026038), ((0.35000000000000003, 0.65000000000000002, 1.0), 16.38906763823347), ((0.40000000000000002, 0.60000000000000009, 1.0), 16.388295830000377), ((0.30000000000000004, 0.40000000000000002, 1.0), 16.347446564906452), ((0.30000000000000004, 0.5, 1.0), 16.292464985542203), ((0.10000000000000001, 0.55000000000000004, 1.0), 16.289428481421115), ((0.0, 0.55000000000000004, 1.0), 16.289246927317993), ((0.050000000000000003, 0.55000000000000004, 1.0), 16.289246927317993), ((0.15000000000000002, 0.55000000000000004, 1.0), 16.226909505182867), ((0.40000000000000002, 0.5, 1.0), 16.21706193801647), ((0.25, 0.60000000000000009, 1.0), 16.21592315508978), ((0.25, 0.35000000000000003, 1.0), 16.203099033828313), ((0.35000000000000003, 0.60000000000000009, 1.0), 16.199142507083042), ((0.050000000000000003, 0.15000000000000002, 1.0), 16.17115712958423), ((0.0, 0.15000000000000002, 1.0), 16.17115712958423), ((0.10000000000000001, 0.20000000000000001, 1.0), 16.16781714253315), ((0.0, 0.20000000000000001, 1.0), 16.16762474234075), ((0.050000000000000003, 0.20000000000000001, 1.0), 16.16762474234075), ((0.10000000000000001, 0.15000000000000002, 1.0), 16.162646435183014), ((0.10000000000000001, 0.25, 1.0), 16.13175154635275), ((0.0, 0.25, 1.0), 16.131559146160345), ((0.050000000000000003, 0.25, 1.0), 16.131559146160345), ((0.70000000000000007, 0.80000000000000004, 1.0), 16.104696556318714), ((0.70000000000000007, 1.0, 1.0), 16.10341523803124), ((0.70000000000000007, 0.95000000000000007, 1.0), 16.10341523803124), ((0.70000000000000007, 0.90000000000000002, 1.0), 16.09865545423105), ((0.70000000000000007, 0.85000000000000009, 1.0), 16.097238679250513), ((0.15000000000000002, 0.30000000000000004, 1.0), 16.094179456694427), ((0.75, 0.95000000000000007, 1.0), 16.092379831731595), ((0.75, 1.0, 1.0), 16.092379831731595), ((0.15000000000000002, 0.20000000000000001, 1.0), 16.090813457073555), ((0.75, 0.90000000000000002, 1.0), 16.090095852437127), ((0.75, 0.85000000000000009, 1.0), 16.08772611695804), ((0.20000000000000001, 0.25, 1.0), 16.082265060728194), ((0.25, 0.5, 1.0), 16.072375550120555), ((0.0, 0.30000000000000004, 1.0), 16.070903115540695), ((0.10000000000000001, 0.30000000000000004, 1.0), 16.070523502415984), ((0.050000000000000003, 0.30000000000000004, 1.0), 16.07034194831286), ((0.35000000000000003, 0.5, 1.0), 16.070218577435792), ((0.20000000000000001, 0.30000000000000004, 1.0), 16.047954843549615), ((0.20000000000000001, 0.55000000000000004, 1.0), 16.046506669844728), ((0.25, 0.30000000000000004, 1.0), 16.04324426147785), ((0.15000000000000002, 0.25, 1.0), 16.010976487701374), ((0.40000000000000002, 0.55000000000000004, 1.0), 15.988681397515238), ((0.70000000000000007, 0.75, 1.0), 15.965081846766058)], [((0.5, 0.60000000000000009, 1.0), 21.09239142143483), ((0.45000000000000001, 0.60000000000000009, 1.0), 20.90047754700769), ((0.30000000000000004, 0.35000000000000003, 1.0), 20.887333896217754), ((0.55000000000000004, 0.60000000000000009, 1.0), 20.863259468870236), ((0.55000000000000004, 0.65000000000000002, 1.0), 20.831445914117843), ((0.65000000000000002, 0.85000000000000009, 1.0), 20.81568645786887), ((0.65000000000000002, 1.0, 1.0), 20.80456947655958), ((0.65000000000000002, 0.90000000000000002, 1.0), 20.80456947655958), ((0.65000000000000002, 0.95000000000000007, 1.0), 20.80456947655958), ((0.35000000000000003, 0.40000000000000002, 1.0), 20.77409448463425), ((0.55000000000000004, 0.85000000000000009, 1.0), 20.76949712869979), ((0.55000000000000004, 1.0, 1.0), 20.765766715594378), ((0.55000000000000004, 0.95000000000000007, 1.0), 20.765766715594378), ((0.55000000000000004, 0.90000000000000002, 1.0), 20.765766715594378), ((0.55000000000000004, 0.70000000000000007, 1.0), 20.76189167126967), ((0.70000000000000007, 0.85000000000000009, 1.0), 20.760645056511137), ((0.65000000000000002, 0.80000000000000004, 1.0), 20.75949413167654), ((0.70000000000000007, 0.90000000000000002, 1.0), 20.75011799098407), ((0.70000000000000007, 1.0, 1.0), 20.740800225416304), ((0.70000000000000007, 0.95000000000000007, 1.0), 20.740800225416304), ((0.65000000000000002, 0.75, 1.0), 20.738151013232578)]], 'CasualFamilyGather': [[((0.80000000000000004, 0.90000000000000002, 1.0), 23.9413425896647), ((0.70000000000000007, 0.90000000000000002, 1.0), 23.935729216862953), ((0.70000000000000007, 0.85000000000000009, 1.0), 23.87108527614758), ((0.65000000000000002, 0.90000000000000002, 1.0), 23.806738244120616), ((0.55000000000000004, 0.60000000000000009, 1.0), 23.8006000406569), ((0.85000000000000009, 0.95000000000000007, 1.0), 23.791359155307394), ((0.65000000000000002, 0.85000000000000009, 1.0), 23.775515022897398), ((0.5, 0.55000000000000004, 1.0), 23.717080791609895), ((0.90000000000000002, 1.0, 1.0), 23.708747137437182), ((0.5, 0.60000000000000009, 1.0), 23.704536180203323), ((0.80000000000000004, 0.85000000000000009, 1.0), 23.693259649487132), ((0.45000000000000001, 0.60000000000000009, 1.0), 23.688562712192127), ((0.45000000000000001, 0.55000000000000004, 1.0), 23.640775302981762), ((0.60000000000000009, 0.85000000000000009, 1.0), 23.639730761678077), ((0.75, 0.85000000000000009, 1.0), 23.62978691556059), ((0.85000000000000009, 0.90000000000000002, 1.0), 23.61037726254557), ((0.45000000000000001, 0.5, 1.0), 23.604709240246514), ((0.85000000000000009, 1.0, 1.0), 23.600882964831204), ((0.70000000000000007, 1.0, 1.0), 23.592188531357973), ((0.70000000000000007, 0.95000000000000007, 1.0), 23.592188531357973), ((0.60000000000000009, 0.90000000000000002, 1.0), 23.559550622467327), ((0.65000000000000002, 1.0, 1.0), 23.531864704961365), ((0.65000000000000002, 0.95000000000000007, 1.0), 23.531864704961365), ((0.40000000000000002, 0.5, 1.0), 23.491048978961874), ((0.70000000000000007, 0.80000000000000004, 1.0), 23.45815853033568), ((0.35000000000000003, 0.60000000000000009, 1.0), 23.453823570742845), ((0.40000000000000002, 0.55000000000000004, 1.0), 23.45301273682537), ((0.30000000000000004, 0.60000000000000009, 1.0), 23.451593955764345), ((0.25, 0.60000000000000009, 1.0), 23.45024054974534), ((0.40000000000000002, 0.60000000000000009, 1.0), 23.449860224134028), ((0.60000000000000009, 0.65000000000000002, 1.0), 23.449230366090003), ((0.60000000000000009, 0.70000000000000007, 1.0), 23.449132995275292), ((0.15000000000000002, 0.60000000000000009, 1.0), 23.4486944767707), ((0.20000000000000001, 0.60000000000000009, 1.0), 23.4486944767707), ((0.10000000000000001, 0.60000000000000009, 1.0), 23.4479729760492), ((0.050000000000000003, 0.60000000000000009, 1.0), 23.446622081081284), ((0.0, 0.60000000000000009, 1.0), 23.446622081081284), ((0.90000000000000002, 0.95000000000000007, 1.0), 23.43569637867214)], [((0.0, 0.35000000000000003, 1.0), 20.16211649446192), ((0.050000000000000003, 0.35000000000000003, 1.0), 20.16211649446192), ((0.10000000000000001, 0.35000000000000003, 1.0), 20.16211649446192), ((0.15000000000000002, 0.35000000000000003, 1.0), 20.1571629340904), ((0.40000000000000002, 0.45000000000000001, 1.0), 20.151473714992843), ((0.35000000000000003, 0.40000000000000002, 1.0), 20.056864204633037), ((0.20000000000000001, 0.35000000000000003, 1.0), 20.05269499715895), ((0.25, 0.35000000000000003, 1.0), 20.012960863623086), ((0.30000000000000004, 0.35000000000000003, 1.0), 19.989714151208293), ((0.35000000000000003, 0.45000000000000001, 1.0), 19.419847616460398), ((0.15000000000000002, 0.40000000000000002, 1.0), 19.383312186281294), ((0.0, 0.40000000000000002, 1.0), 19.383312186281294), ((0.10000000000000001, 0.40000000000000002, 1.0), 19.383312186281294), ((0.050000000000000003, 0.40000000000000002, 1.0), 19.383312186281294), ((0.20000000000000001, 0.40000000000000002, 1.0), 19.365624125549758), ((0.25, 0.45000000000000001, 1.0), 19.311475613219972), ((0.10000000000000001, 0.45000000000000001, 1.0), 19.304796736541093), ((0.15000000000000002, 0.45000000000000001, 1.0), 19.304796736541093), ((0.0, 0.45000000000000001, 1.0), 19.304796736541093), ((0.050000000000000003, 0.45000000000000001, 1.0), 19.304796736541093), ((0.20000000000000001, 0.45000000000000001, 1.0), 19.297877729622094), ((0.25, 0.40000000000000002, 1.0), 19.274599808490503), ((0.050000000000000003, 0.30000000000000004, 1.0), 19.25040924265981), ((0.0, 0.30000000000000004, 1.0), 19.25040924265981), ((0.10000000000000001, 0.30000000000000004, 1.0), 19.25040924265981), ((0.30000000000000004, 0.40000000000000002, 1.0), 19.239634726571637), ((0.60000000000000009, 0.65000000000000002, 1.0), 19.23413908309297), ((0.15000000000000002, 0.30000000000000004, 1.0), 19.207976179691038), ((0.30000000000000004, 0.45000000000000001, 1.0), 19.20101993387221), ((0.10000000000000001, 0.25, 1.0), 19.197287630360723), ((0.0, 0.25, 1.0), 19.197287630360723), ((0.050000000000000003, 0.25, 1.0), 19.197287630360723), ((0.15000000000000002, 0.25, 1.0), 19.193001916075005), ((0.0, 0.20000000000000001, 1.0), 19.189986097307308), ((0.050000000000000003, 0.20000000000000001, 1.0), 19.189986097307308), ((0.10000000000000001, 0.20000000000000001, 1.0), 19.189986097307308), ((0.15000000000000002, 0.20000000000000001, 1.0), 19.188687396008607), ((0.70000000000000007, 0.75, 1.0), 19.185241396396638), ((0.20000000000000001, 0.30000000000000004, 1.0), 19.173966154628378), ((0.20000000000000001, 0.25, 1.0), 19.169037459478968), ((0.25, 0.30000000000000004, 1.0), 19.16828557977972), ((0.60000000000000009, 0.85000000000000009, 1.0), 18.972295638852906), ((0.60000000000000009, 0.80000000000000004, 1.0), 18.972295638852906), ((0.60000000000000009, 0.95000000000000007, 1.0), 18.972295638852906), ((0.60000000000000009, 0.90000000000000002, 1.0), 18.972295638852906), ((0.60000000000000009, 1.0, 1.0), 18.972295638852906), ((0.60000000000000009, 0.75, 1.0), 18.972295638852906), ((0.050000000000000003, 0.15000000000000002, 1.0), 18.887507424019812), ((0.10000000000000001, 0.15000000000000002, 1.0), 18.887507424019812), ((0.0, 0.15000000000000002, 1.0), 18.887507424019812), ((0.65000000000000002, 0.70000000000000007, 1.0), 18.875710604981297), ((0.55000000000000004, 0.60000000000000009, 1.0), 18.77820890280214)], [((0.65000000000000002, 0.75, 1.0), 24.175554953348744), ((0.80000000000000004, 0.90000000000000002, 1.0), 24.01189672825101), ((0.80000000000000004, 0.85000000000000009, 1.0), 24.01071726997853), ((0.70000000000000007, 0.75, 1.0), 23.996247285802383), ((0.80000000000000004, 0.95000000000000007, 1.0), 23.964902680631962), ((0.80000000000000004, 1.0, 1.0), 23.949676413843193), ((0.60000000000000009, 0.70000000000000007, 1.0), 23.775142894711294), ((0.75, 0.80000000000000004, 1.0), 23.71712383300118), ((0.10000000000000001, 0.55000000000000004, 1.0), 23.561135800324788), ((0.0, 0.55000000000000004, 1.0), 23.561135800324788), ((0.050000000000000003, 0.55000000000000004, 1.0), 23.561135800324788), ((0.20000000000000001, 0.55000000000000004, 1.0), 23.561135800324788), ((0.25, 0.55000000000000004, 1.0), 23.561135800324788), ((0.15000000000000002, 0.55000000000000004, 1.0), 23.561135800324788), ((0.35000000000000003, 0.55000000000000004, 1.0), 23.55973306054944), ((0.30000000000000004, 0.55000000000000004, 1.0), 23.55973306054944), ((0.40000000000000002, 0.55000000000000004, 1.0), 23.55645925102563), ((0.25, 0.5, 1.0), 23.53992279563366), ((0.15000000000000002, 0.5, 1.0), 23.53992279563366), ((0.0, 0.5, 1.0), 23.53992279563366), ((0.20000000000000001, 0.5, 1.0), 23.53992279563366), ((0.35000000000000003, 0.5, 1.0), 23.53992279563366), ((0.10000000000000001, 0.5, 1.0), 23.53992279563366), ((0.40000000000000002, 0.5, 1.0), 23.53992279563366), ((0.30000000000000004, 0.5, 1.0), 23.53992279563366), ((0.050000000000000003, 0.5, 1.0), 23.53992279563366), ((0.45000000000000001, 0.55000000000000004, 1.0), 23.53651065795204), ((0.45000000000000001, 0.5, 1.0), 23.48018014857484), ((0.70000000000000007, 0.80000000000000004, 1.0), 23.478549638550447), ((0.5, 0.55000000000000004, 1.0), 23.447006106131308), ((0.10000000000000001, 0.45000000000000001, 1.0), 23.443194455174673), ((0.15000000000000002, 0.45000000000000001, 1.0), 23.443194455174673), ((0.35000000000000003, 0.45000000000000001, 1.0), 23.443194455174673), ((0.20000000000000001, 0.45000000000000001, 1.0), 23.443194455174673), ((0.15000000000000002, 0.40000000000000002, 1.0), 23.443194455174673), ((0.20000000000000001, 0.35000000000000003, 1.0), 23.443194455174673), ((0.0, 0.40000000000000002, 1.0), 23.443194455174673), ((0.25, 0.45000000000000001, 1.0), 23.443194455174673), ((0.25, 0.35000000000000003, 1.0), 23.443194455174673), ((0.10000000000000001, 0.40000000000000002, 1.0), 23.443194455174673), ((0.0, 0.35000000000000003, 1.0), 23.443194455174673), ((0.30000000000000004, 0.40000000000000002, 1.0), 23.443194455174673), ((0.0, 0.45000000000000001, 1.0), 23.443194455174673), ((0.050000000000000003, 0.45000000000000001, 1.0), 23.443194455174673), ((0.30000000000000004, 0.45000000000000001, 1.0), 23.443194455174673), ((0.20000000000000001, 0.40000000000000002, 1.0), 23.443194455174673), ((0.050000000000000003, 0.35000000000000003, 1.0), 23.443194455174673), ((0.050000000000000003, 0.40000000000000002, 1.0), 23.443194455174673), ((0.25, 0.40000000000000002, 1.0), 23.443194455174673), ((0.15000000000000002, 0.35000000000000003, 1.0), 23.443194455174673), ((0.10000000000000001, 0.35000000000000003, 1.0), 23.443194455174673), ((0.60000000000000009, 0.75, 1.0), 23.404700568439562), ((0.40000000000000002, 0.45000000000000001, 1.0), 23.385902788508005), ((0.30000000000000004, 0.35000000000000003, 1.0), 23.385902788508005), ((0.35000000000000003, 0.40000000000000002, 1.0), 23.385902788508005), ((0.10000000000000001, 0.25, 1.0), 23.37511566444785), ((0.0, 0.20000000000000001, 1.0), 23.37511566444785), ((0.15000000000000002, 0.30000000000000004, 1.0), 23.37511566444785), ((0.0, 0.25, 1.0), 23.37511566444785), ((0.050000000000000003, 0.25, 1.0), 23.37511566444785), ((0.050000000000000003, 0.20000000000000001, 1.0), 23.37511566444785), ((0.10000000000000001, 0.20000000000000001, 1.0), 23.37511566444785), ((0.15000000000000002, 0.25, 1.0), 23.37511566444785), ((0.050000000000000003, 0.30000000000000004, 1.0), 23.37511566444785), ((0.0, 0.30000000000000004, 1.0), 23.37511566444785), ((0.20000000000000001, 0.30000000000000004, 1.0), 23.37511566444785), ((0.10000000000000001, 0.30000000000000004, 1.0), 23.37511566444785), ((0.65000000000000002, 0.70000000000000007, 1.0), 23.374317689112218), ((0.65000000000000002, 0.80000000000000004, 1.0), 23.346199278065555), ((0.15000000000000002, 0.20000000000000001, 1.0), 23.32928233111452), ((0.25, 0.30000000000000004, 1.0), 23.32928233111452), ((0.20000000000000001, 0.25, 1.0), 23.32928233111452), ((0.75, 0.85000000000000009, 1.0), 23.311008191439054), ((0.60000000000000009, 0.65000000000000002, 1.0), 23.292122386726515), ((0.75, 1.0, 1.0), 23.26839926919598), ((0.75, 0.90000000000000002, 1.0), 23.26664636931808), ((0.70000000000000007, 0.85000000000000009, 1.0), 23.26353553468345), ((0.75, 0.95000000000000007, 1.0), 23.251420102529316)]], 'Architecture': [[], [], [((0.40000000000000002, 0.90000000000000002, 1.0), 16.890516753206246), ((0.40000000000000002, 0.75, 1.0), 16.890516753206246), ((0.40000000000000002, 1.0, 1.0), 16.890516753206246), ((0.40000000000000002, 0.70000000000000007, 1.0), 16.890516753206246), ((0.40000000000000002, 0.85000000000000009, 1.0), 16.890516753206246), ((0.40000000000000002, 0.95000000000000007, 1.0), 16.890516753206246), ((0.40000000000000002, 0.80000000000000004, 1.0), 16.890516753206246), ((0.40000000000000002, 0.60000000000000009, 1.0), 16.890516753206242), ((0.40000000000000002, 0.65000000000000002, 1.0), 16.890516753206242), ((0.45000000000000001, 0.60000000000000009, 1.0), 16.883278292831818), ((0.45000000000000001, 0.75, 1.0), 16.881274480671756), ((0.45000000000000001, 0.85000000000000009, 1.0), 16.881274480671756), ((0.45000000000000001, 0.95000000000000007, 1.0), 16.881274480671756), ((0.45000000000000001, 0.70000000000000007, 1.0), 16.881274480671756), ((0.45000000000000001, 0.65000000000000002, 1.0), 16.881274480671756), ((0.45000000000000001, 0.90000000000000002, 1.0), 16.881274480671756), ((0.45000000000000001, 0.80000000000000004, 1.0), 16.881274480671756), ((0.45000000000000001, 1.0, 1.0), 16.881274480671756), ((0.40000000000000002, 0.55000000000000004, 1.0), 16.868048298540824), ((0.5, 0.65000000000000002, 1.0), 16.86697336007755), ((0.5, 0.90000000000000002, 1.0), 16.864969547917493), ((0.5, 0.85000000000000009, 1.0), 16.864969547917493), ((0.5, 0.70000000000000007, 1.0), 16.864969547917493), ((0.5, 0.95000000000000007, 1.0), 16.864969547917493), ((0.5, 0.75, 1.0), 16.864969547917493), ((0.5, 1.0, 1.0), 16.864969547917493), ((0.5, 0.80000000000000004, 1.0), 16.864969547917493), ((0.55000000000000004, 0.80000000000000004, 1.0), 16.8602417365651), ((0.55000000000000004, 0.85000000000000009, 1.0), 16.8602417365651), ((0.55000000000000004, 1.0, 1.0), 16.8602417365651), ((0.55000000000000004, 0.70000000000000007, 1.0), 16.8602417365651), ((0.55000000000000004, 0.75, 1.0), 16.8602417365651), ((0.55000000000000004, 0.95000000000000007, 1.0), 16.8602417365651), ((0.55000000000000004, 0.90000000000000002, 1.0), 16.8602417365651), ((0.35000000000000003, 0.80000000000000004, 1.0), 16.819228037518126), ((0.35000000000000003, 0.65000000000000002, 1.0), 16.819228037518126), ((0.35000000000000003, 1.0, 1.0), 16.819228037518126), ((0.35000000000000003, 0.75, 1.0), 16.819228037518126), ((0.35000000000000003, 0.95000000000000007, 1.0), 16.819228037518126), ((0.35000000000000003, 0.85000000000000009, 1.0), 16.819228037518126), ((0.35000000000000003, 0.90000000000000002, 1.0), 16.819228037518126), ((0.35000000000000003, 0.70000000000000007, 1.0), 16.819228037518126), ((0.35000000000000003, 0.60000000000000009, 1.0), 16.819228037518126), ((0.5, 0.60000000000000009, 1.0), 16.81559437640905), ((0.45000000000000001, 0.55000000000000004, 1.0), 16.813560647350343), ((0.60000000000000009, 0.85000000000000009, 1.0), 16.81320312766642), ((0.60000000000000009, 0.70000000000000007, 1.0), 16.81320312766642), ((0.60000000000000009, 0.80000000000000004, 1.0), 16.81320312766642), ((0.60000000000000009, 0.95000000000000007, 1.0), 16.81320312766642), ((0.60000000000000009, 0.90000000000000002, 1.0), 16.81320312766642), ((0.60000000000000009, 1.0, 1.0), 16.81320312766642), ((0.60000000000000009, 0.75, 1.0), 16.81320312766642)]], 'ThemePark': [[((0.15000000000000002, 0.30000000000000004, 1.0), 19.014433789671436), ((0.15000000000000002, 0.25, 1.0), 18.94474559176004), ((0.15000000000000002, 0.20000000000000001, 1.0), 18.94433917024965), ((0.10000000000000001, 0.15000000000000002, 1.0), 18.9387256171361), ((0.25, 0.35000000000000003, 1.0), 18.935393832018995), ((0.20000000000000001, 0.35000000000000003, 1.0), 18.830182996522186), ((0.30000000000000004, 0.35000000000000003, 1.0), 18.810626388290757), ((0.30000000000000004, 0.40000000000000002, 1.0), 18.764020820026285), ((0.35000000000000003, 0.40000000000000002, 1.0), 18.747363683811002)], [((0.45000000000000001, 0.85000000000000009, 1.0), 16.201926281560006), ((0.45000000000000001, 0.95000000000000007, 1.0), 16.201926281560006), ((0.45000000000000001, 0.90000000000000002, 1.0), 16.201926281560006), ((0.45000000000000001, 1.0, 1.0), 16.201926281560006), ((0.40000000000000002, 0.90000000000000002, 1.0), 16.181680783051466), ((0.40000000000000002, 1.0, 1.0), 16.181680783051466), ((0.40000000000000002, 0.85000000000000009, 1.0), 16.181680783051466), ((0.40000000000000002, 0.95000000000000007, 1.0), 16.181680783051466), ((0.45000000000000001, 0.80000000000000004, 1.0), 16.156189338395645), ((0.35000000000000003, 1.0, 1.0), 16.152193627040848), ((0.35000000000000003, 0.95000000000000007, 1.0), 16.152193627040848), ((0.35000000000000003, 0.85000000000000009, 1.0), 16.152193627040848), ((0.35000000000000003, 0.90000000000000002, 1.0), 16.152193627040848), ((0.40000000000000002, 0.80000000000000004, 1.0), 16.135204592575274), ((0.45000000000000001, 0.75, 1.0), 16.13036923257554), ((0.30000000000000004, 0.85000000000000009, 1.0), 16.124295011141974), ((0.30000000000000004, 0.90000000000000002, 1.0), 16.124295011141974), ((0.30000000000000004, 1.0, 1.0), 16.124295011141974), ((0.30000000000000004, 0.95000000000000007, 1.0), 16.124295011141974), ((0.40000000000000002, 0.75, 1.0), 16.121420030363293), ((0.35000000000000003, 0.80000000000000004, 1.0), 16.105717436564657), ((0.25, 0.95000000000000007, 1.0), 16.099730562150107), ((0.25, 1.0, 1.0), 16.099730562150107), ((0.25, 0.90000000000000002, 1.0), 16.099730562150107), ((0.25, 0.85000000000000009, 1.0), 16.099730562150107), ((0.35000000000000003, 0.75, 1.0), 16.091932874352675), ((0.30000000000000004, 0.80000000000000004, 1.0), 16.077818820665787), ((0.20000000000000001, 0.85000000000000009, 1.0), 16.07066215914788), ((0.20000000000000001, 1.0, 1.0), 16.07066215914788), ((0.20000000000000001, 0.95000000000000007, 1.0), 16.07066215914788), ((0.20000000000000001, 0.90000000000000002, 1.0), 16.07066215914788), ((0.30000000000000004, 0.75, 1.0), 16.064034258453802), ((0.25, 0.80000000000000004, 1.0), 16.053254371673912), ((0.15000000000000002, 0.90000000000000002, 1.0), 16.046496022996447), ((0.15000000000000002, 0.95000000000000007, 1.0), 16.046496022996447), ((0.15000000000000002, 0.85000000000000009, 1.0), 16.046496022996447), ((0.15000000000000002, 1.0, 1.0), 16.046496022996447), ((0.25, 0.75, 1.0), 16.039519092616054), ((0.10000000000000001, 0.95000000000000007, 1.0), 16.03373378523421), ((0.10000000000000001, 0.90000000000000002, 1.0), 16.03373378523421), ((0.10000000000000001, 0.85000000000000009, 1.0), 16.03373378523421), ((0.10000000000000001, 1.0, 1.0), 16.03373378523421), ((0.0, 0.90000000000000002, 1.0), 16.02653940979271), ((0.050000000000000003, 0.85000000000000009, 1.0), 16.02653940979271), ((0.0, 0.95000000000000007, 1.0), 16.02653940979271), ((0.050000000000000003, 1.0, 1.0), 16.02653940979271), ((0.050000000000000003, 0.95000000000000007, 1.0), 16.02653940979271), ((0.0, 1.0, 1.0), 16.02653940979271), ((0.050000000000000003, 0.90000000000000002, 1.0), 16.02653940979271), ((0.0, 0.85000000000000009, 1.0), 16.02653940979271), ((0.20000000000000001, 0.80000000000000004, 1.0), 16.024185968671688), ((0.20000000000000001, 0.75, 1.0), 16.010450689613826), ((0.15000000000000002, 0.80000000000000004, 1.0), 16.00001983252026), ((0.10000000000000001, 0.80000000000000004, 1.0), 15.98725759475802), ((0.15000000000000002, 0.75, 1.0), 15.986284553462397), ((0.0, 0.80000000000000004, 1.0), 15.980063219316516), ((0.050000000000000003, 0.80000000000000004, 1.0), 15.980063219316516), ((0.10000000000000001, 0.75, 1.0), 15.973522315700162), ((0.0, 0.75, 1.0), 15.96632794025866), ((0.050000000000000003, 0.75, 1.0), 15.96632794025866), ((0.40000000000000002, 0.70000000000000007, 1.0), 15.88175071819398), ((0.35000000000000003, 0.70000000000000007, 1.0), 15.852263562183364), ((0.5, 0.85000000000000009, 1.0), 15.840308813381025), ((0.5, 0.90000000000000002, 1.0), 15.839569566069196), ((0.5, 0.95000000000000007, 1.0), 15.839569566069196), ((0.5, 1.0, 1.0), 15.839569566069196), ((0.30000000000000004, 0.70000000000000007, 1.0), 15.82436494628449), ((0.25, 0.70000000000000007, 1.0), 15.799849780446742), ((0.5, 0.75, 1.0), 15.79512362819584), ((0.5, 0.80000000000000004, 1.0), 15.79512362819584), ((0.45000000000000001, 0.70000000000000007, 1.0), 15.791665528871834), ((0.40000000000000002, 0.65000000000000002, 1.0), 15.790012622955885), ((0.20000000000000001, 0.70000000000000007, 1.0), 15.770781377444516), ((0.35000000000000003, 0.65000000000000002, 1.0), 15.760525466945268), ((0.15000000000000002, 0.70000000000000007, 1.0), 15.746615241293087), ((0.10000000000000001, 0.70000000000000007, 1.0), 15.733853003530848), ((0.30000000000000004, 0.65000000000000002, 1.0), 15.732626851046394), ((0.0, 0.70000000000000007, 1.0), 15.726658628089348), ((0.050000000000000003, 0.70000000000000007, 1.0), 15.726658628089348), ((0.25, 0.65000000000000002, 1.0), 15.708111685208648), ((0.40000000000000002, 0.5, 1.0), 15.704422811526497), ((0.20000000000000001, 0.65000000000000002, 1.0), 15.679043282206422), ((0.35000000000000003, 0.5, 1.0), 15.678313226932715), ((0.15000000000000002, 0.65000000000000002, 1.0), 15.65487714605499), ((0.30000000000000004, 0.5, 1.0), 15.650414611033842), ((0.10000000000000001, 0.65000000000000002, 1.0), 15.642114908292754), ((0.45000000000000001, 0.65000000000000002, 1.0), 15.636971084427392), ((0.050000000000000003, 0.65000000000000002, 1.0), 15.63492053285125), ((0.0, 0.65000000000000002, 1.0), 15.63492053285125), ((0.25, 0.5, 1.0), 15.634281618578267), ((0.5, 0.70000000000000007, 1.0), 15.631796908619124)], []], 'Sports': [[((0.40000000000000002, 0.60000000000000009, 1.0), 17.56838580874372), ((0.050000000000000003, 0.60000000000000009, 1.0), 17.55439465377806), ((0.10000000000000001, 0.60000000000000009, 1.0), 17.55439465377806), ((0.0, 0.60000000000000009, 1.0), 17.55439465377806), ((0.35000000000000003, 0.60000000000000009, 1.0), 17.553743480520566), ((0.15000000000000002, 0.60000000000000009, 1.0), 17.55260412288039), ((0.40000000000000002, 0.65000000000000002, 1.0), 17.550319969354348), ((0.20000000000000001, 0.60000000000000009, 1.0), 17.54784433101389), ((0.40000000000000002, 0.70000000000000007, 1.0), 17.54481731874137), ((0.10000000000000001, 0.70000000000000007, 1.0), 17.533044449225446), ((0.0, 0.70000000000000007, 1.0), 17.533044449225446), ((0.050000000000000003, 0.70000000000000007, 1.0), 17.533044449225446), ((0.15000000000000002, 0.70000000000000007, 1.0), 17.531253918327774), ((0.25, 0.60000000000000009, 1.0), 17.529989285208533), ((0.20000000000000001, 0.70000000000000007, 1.0), 17.52665947037662), ((0.050000000000000003, 0.65000000000000002, 1.0), 17.525246830177824), ((0.0, 0.65000000000000002, 1.0), 17.525246830177824), ((0.10000000000000001, 0.65000000000000002, 1.0), 17.525246830177824), ((0.30000000000000004, 0.60000000000000009, 1.0), 17.525036592484717), ((0.35000000000000003, 0.65000000000000002, 1.0), 17.52361970756816), ((0.15000000000000002, 0.65000000000000002, 1.0), 17.523456299280152), ((0.35000000000000003, 0.70000000000000007, 1.0), 17.520704697408252), ((0.20000000000000001, 0.65000000000000002, 1.0), 17.518707910442302), ((0.45000000000000001, 0.60000000000000009, 1.0), 17.517337205909296), ((0.25, 0.70000000000000007, 1.0), 17.51023813837299), ((0.30000000000000004, 0.70000000000000007, 1.0), 17.505322648030123), ((0.25, 0.65000000000000002, 1.0), 17.502286578438675), ((0.45000000000000001, 0.70000000000000007, 1.0), 17.498504983894936), ((0.30000000000000004, 0.65000000000000002, 1.0), 17.497353417999403), ((0.5, 0.60000000000000009, 1.0), 17.497140586986443), ((0.5, 0.70000000000000007, 1.0), 17.4909448087888), ((0.45000000000000001, 0.65000000000000002, 1.0), 17.490664847840517), ((0.5, 0.65000000000000002, 1.0), 17.482736156922513), ((0.55000000000000004, 0.60000000000000009, 1.0), 17.388713592852458), ((0.40000000000000002, 0.55000000000000004, 1.0), 17.384253108729474), ((0.10000000000000001, 0.55000000000000004, 1.0), 17.36865610666472), ((0.0, 0.55000000000000004, 1.0), 17.36865610666472), ((0.050000000000000003, 0.55000000000000004, 1.0), 17.36865610666472), ((0.55000000000000004, 0.65000000000000002, 1.0), 17.368197716298102), ((0.15000000000000002, 0.55000000000000004, 1.0), 17.366865575767044), ((0.20000000000000001, 0.55000000000000004, 1.0), 17.36195184301385), ((0.55000000000000004, 0.70000000000000007, 1.0), 17.356433232726896), ((0.35000000000000003, 0.55000000000000004, 1.0), 17.355791380844117), ((0.25, 0.55000000000000004, 1.0), 17.342462137133285), ((0.30000000000000004, 0.55000000000000004, 1.0), 17.337701752101776), ((0.45000000000000001, 0.55000000000000004, 1.0), 17.329042503861345), ((0.10000000000000001, 0.75, 1.0), 17.29181325977997), ((0.0, 0.75, 1.0), 17.29181325977997), ((0.050000000000000003, 0.75, 1.0), 17.29181325977997), ((0.40000000000000002, 0.75, 1.0), 17.291474856215235), ((0.15000000000000002, 0.75, 1.0), 17.2900227288823), ((0.20000000000000001, 0.75, 1.0), 17.285428280931143), ((0.35000000000000003, 0.75, 1.0), 17.279473507962777), ((0.5, 0.55000000000000004, 1.0), 17.275936877013883), ((0.25, 0.75, 1.0), 17.26900694892751), ((0.30000000000000004, 0.75, 1.0), 17.264091458584648), ((0.5, 0.75, 1.0), 17.253122710252416), ((0.45000000000000001, 0.75, 1.0), 17.243655739572006), ((0.55000000000000004, 0.75, 1.0), 17.20876465335881), ((0.050000000000000003, 0.30000000000000004, 1.0), 17.17696417941061), ((0.0, 0.30000000000000004, 1.0), 17.17696417941061), ((0.10000000000000001, 0.30000000000000004, 1.0), 17.17696417941061), ((0.15000000000000002, 0.30000000000000004, 1.0), 17.17446168851526)], [], []], 'Show': [[((0.40000000000000002, 0.55000000000000004, 1.0), 13.778126300191957), ((0.55000000000000004, 0.60000000000000009, 1.0), 13.767875109760666), ((0.55000000000000004, 0.65000000000000002, 1.0), 13.738081585752397), ((0.35000000000000003, 0.55000000000000004, 1.0), 13.728352171758505), ((0.40000000000000002, 0.60000000000000009, 1.0), 13.710051321085183), ((0.45000000000000001, 0.60000000000000009, 1.0), 13.69110917233827), ((0.40000000000000002, 0.45000000000000001, 1.0), 13.679466775155833), ((0.35000000000000003, 0.5, 1.0), 13.672515984802045), ((0.5, 0.60000000000000009, 1.0), 13.635892163734082), ((0.55000000000000004, 0.70000000000000007, 1.0), 13.627479037600875), ((0.5, 0.55000000000000004, 1.0), 13.62317513903638), ((0.30000000000000004, 0.45000000000000001, 1.0), 13.616374275994424), ((0.55000000000000004, 0.75, 1.0), 13.61118069058365), ((0.55000000000000004, 0.80000000000000004, 1.0), 13.610266520777845), ((0.55000000000000004, 0.85000000000000009, 1.0), 13.610266520777845), ((0.55000000000000004, 0.90000000000000002, 1.0), 13.610266520777845), ((0.55000000000000004, 0.95000000000000007, 1.0), 13.609041577140113), ((0.55000000000000004, 1.0, 1.0), 13.60842813176196), ((0.35000000000000003, 0.60000000000000009, 1.0), 13.581182747160472), ((0.25, 0.45000000000000001, 1.0), 13.57070991355264), ((0.30000000000000004, 0.5, 1.0), 13.566102495866273), ((0.30000000000000004, 0.55000000000000004, 1.0), 13.55477355591403), ((0.45000000000000001, 0.55000000000000004, 1.0), 13.540868817502304), ((0.45000000000000001, 0.5, 1.0), 13.538178851994232), ((0.40000000000000002, 0.5, 1.0), 13.536210318139304)], [((0.5, 1.0, 1.0), 17.897146675906622), ((0.45000000000000001, 1.0, 1.0), 17.879965757550963), ((0.55000000000000004, 1.0, 1.0), 17.875731962907846), ((0.5, 0.70000000000000007, 1.0), 17.875247157457213), ((0.40000000000000002, 1.0, 1.0), 17.86759886164261), ((0.45000000000000001, 0.70000000000000007, 1.0), 17.85733756027469), ((0.5, 0.75, 1.0), 17.856067769106836), ((0.5, 0.80000000000000004, 1.0), 17.854258277770768), ((0.5, 0.95000000000000007, 1.0), 17.85034733394134), ((0.40000000000000002, 0.70000000000000007, 1.0), 17.846659063291575), ((0.45000000000000001, 0.75, 1.0), 17.84336052499394), ((0.5, 0.85000000000000009, 1.0), 17.8391424854075), ((0.5, 0.90000000000000002, 1.0), 17.838295519280393), ((0.45000000000000001, 0.80000000000000004, 1.0), 17.837194824795738), ((0.45000000000000001, 0.95000000000000007, 1.0), 17.833166415585676), ((0.40000000000000002, 0.75, 1.0), 17.83142652951849), ((0.55000000000000004, 0.75, 1.0), 17.830874603336287), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.82893262094256), ((0.55000000000000004, 0.70000000000000007, 1.0), 17.82876503752184), ((0.55000000000000004, 0.80000000000000004, 1.0), 17.827736931010826), ((0.45000000000000001, 0.85000000000000009, 1.0), 17.825179561532998), ((0.40000000000000002, 0.80000000000000004, 1.0), 17.82482792888738), ((0.45000000000000001, 0.90000000000000002, 1.0), 17.82111460092473), ((0.40000000000000002, 0.95000000000000007, 1.0), 17.820799519677323), ((0.55000000000000004, 0.85000000000000009, 1.0), 17.817582668115016), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.812921930747233), ((0.40000000000000002, 0.85000000000000009, 1.0), 17.812812665624644), ((0.40000000000000002, 0.90000000000000002, 1.0), 17.80874770501638), ((0.60000000000000009, 1.0, 1.0), 17.742759782497867), ((0.60000000000000009, 0.75, 1.0), 17.741298375803805), ((0.60000000000000009, 0.95000000000000007, 1.0), 17.716129208339517), ((0.60000000000000009, 0.80000000000000004, 1.0), 17.709931263731807), ((0.60000000000000009, 0.70000000000000007, 1.0), 17.701406928196686), ((0.60000000000000009, 0.90000000000000002, 1.0), 17.69348440826943), ((0.60000000000000009, 0.85000000000000009, 1.0), 17.693418037737903), ((0.5, 0.65000000000000002, 1.0), 17.68245498007144), ((0.40000000000000002, 0.5, 1.0), 17.668093806418685), ((0.5, 0.60000000000000009, 1.0), 17.665788477794564), ((0.45000000000000001, 0.65000000000000002, 1.0), 17.659958438510436), ((0.40000000000000002, 0.65000000000000002, 1.0), 17.649972582219963), ((0.35000000000000003, 1.0, 1.0), 17.64407951734869), ((0.30000000000000004, 1.0, 1.0), 17.640289233326392), ((0.35000000000000003, 0.5, 1.0), 17.63842833902843), ((0.45000000000000001, 0.55000000000000004, 1.0), 17.634284156419355), ((0.25, 1.0, 1.0), 17.631131659513816), ((0.20000000000000001, 1.0, 1.0), 17.62954675614479), ((0.30000000000000004, 0.5, 1.0), 17.62947501510504), ((0.45000000000000001, 0.60000000000000009, 1.0), 17.628822115864523), ((0.40000000000000002, 0.60000000000000009, 1.0), 17.62778633075543), ((0.15000000000000002, 1.0, 1.0), 17.62722308200592), ((0.050000000000000003, 1.0, 1.0), 17.62649770224828), ((0.0, 1.0, 1.0), 17.62649770224828), ((0.10000000000000001, 1.0, 1.0), 17.62649770224828), ((0.25, 0.5, 1.0), 17.621781235093867), ((0.35000000000000003, 0.70000000000000007, 1.0), 17.621765404359437), ((0.30000000000000004, 0.70000000000000007, 1.0), 17.61797512033714), ((0.20000000000000001, 0.5, 1.0), 17.613007436714945), ((0.25, 0.70000000000000007, 1.0), 17.60881754652456), ((0.35000000000000003, 0.75, 1.0), 17.60790718522457), ((0.15000000000000002, 0.5, 1.0), 17.607570916088147), ((0.20000000000000001, 0.70000000000000007, 1.0), 17.607232643155534), ((0.10000000000000001, 0.5, 1.0), 17.605996404649787), ((0.15000000000000002, 0.70000000000000007, 1.0), 17.604908969016662), ((0.0, 0.5, 1.0), 17.604521862191636), ((0.050000000000000003, 0.5, 1.0), 17.604521862191636), ((0.10000000000000001, 0.70000000000000007, 1.0), 17.604183589259023), ((0.0, 0.70000000000000007, 1.0), 17.604183589259023), ((0.050000000000000003, 0.70000000000000007, 1.0), 17.604183589259023), ((0.40000000000000002, 0.55000000000000004, 1.0), 17.60417003755852), ((0.30000000000000004, 0.75, 1.0), 17.60411690120228), ((0.35000000000000003, 0.55000000000000004, 1.0), 17.601681282812017), ((0.35000000000000003, 0.80000000000000004, 1.0), 17.60130858459346), ((0.55000000000000004, 0.65000000000000002, 1.0), 17.601136019491566), ((0.30000000000000004, 0.80000000000000004, 1.0), 17.597518300571167), ((0.35000000000000003, 0.95000000000000007, 1.0), 17.597280175383403), ((0.25, 0.75, 1.0), 17.594959327389695), ((0.30000000000000004, 0.55000000000000004, 1.0), 17.59403395431639), ((0.30000000000000004, 0.95000000000000007, 1.0), 17.59348989136111), ((0.20000000000000001, 0.75, 1.0), 17.593374424020666), ((0.15000000000000002, 0.75, 1.0), 17.591050749881795), ((0.10000000000000001, 0.75, 1.0), 17.59032537012416), ((0.0, 0.75, 1.0), 17.59032537012416), ((0.050000000000000003, 0.75, 1.0), 17.59032537012416), ((0.35000000000000003, 0.85000000000000009, 1.0), 17.589293321330725), ((0.25, 0.80000000000000004, 1.0), 17.588360726758584), ((0.25, 0.55000000000000004, 1.0), 17.58810292307618), ((0.20000000000000001, 0.80000000000000004, 1.0), 17.586775823389562), ((0.30000000000000004, 0.85000000000000009, 1.0), 17.585503037308428), ((0.35000000000000003, 0.90000000000000002, 1.0), 17.58522836072246), ((0.15000000000000002, 0.80000000000000004, 1.0), 17.584452149250687), ((0.25, 0.95000000000000007, 1.0), 17.58433231754853), ((0.0, 0.80000000000000004, 1.0), 17.58372676949305), ((0.10000000000000001, 0.80000000000000004, 1.0), 17.58372676949305), ((0.050000000000000003, 0.80000000000000004, 1.0), 17.58372676949305), ((0.20000000000000001, 0.95000000000000007, 1.0), 17.582747414179508), ((0.20000000000000001, 0.55000000000000004, 1.0), 17.58266640244938), ((0.30000000000000004, 0.90000000000000002, 1.0), 17.581438076700167), ((0.15000000000000002, 0.55000000000000004, 1.0), 17.581091891011017), ((0.15000000000000002, 0.95000000000000007, 1.0), 17.580423740040633), ((0.10000000000000001, 0.95000000000000007, 1.0), 17.579698360282993), ((0.0, 0.95000000000000007, 1.0), 17.579698360282993), ((0.050000000000000003, 0.95000000000000007, 1.0), 17.579698360282993), ((0.10000000000000001, 0.55000000000000004, 1.0), 17.57961734855287), ((0.0, 0.55000000000000004, 1.0), 17.57961734855287), ((0.050000000000000003, 0.55000000000000004, 1.0), 17.57961734855287), ((0.25, 0.85000000000000009, 1.0), 17.576345463495848), ((0.20000000000000001, 0.85000000000000009, 1.0), 17.574760560126823), ((0.15000000000000002, 0.85000000000000009, 1.0), 17.572436885987948), ((0.25, 0.90000000000000002, 1.0), 17.572280502887587), ((0.050000000000000003, 0.85000000000000009, 1.0), 17.571711506230315), ((0.10000000000000001, 0.85000000000000009, 1.0), 17.571711506230315), ((0.0, 0.85000000000000009, 1.0), 17.571711506230315), ((0.20000000000000001, 0.90000000000000002, 1.0), 17.57069559951856), ((0.15000000000000002, 0.90000000000000002, 1.0), 17.56837192537969), ((0.0, 0.90000000000000002, 1.0), 17.56764654562205), ((0.10000000000000001, 0.90000000000000002, 1.0), 17.56764654562205), ((0.050000000000000003, 0.90000000000000002, 1.0), 17.56764654562205), ((0.35000000000000003, 0.45000000000000001, 1.0), 17.560457290177613), ((0.65000000000000002, 0.75, 1.0), 17.55848194203775), ((0.55000000000000004, 0.60000000000000009, 1.0), 17.55104391782837), ((0.40000000000000002, 0.45000000000000001, 1.0), 17.546760916022425), ((0.65000000000000002, 1.0, 1.0), 17.54559810357224), ((0.30000000000000004, 0.45000000000000001, 1.0), 17.525546133055165), ((0.65000000000000002, 0.80000000000000004, 1.0), 17.520667354099004), ((0.45000000000000001, 0.5, 1.0), 17.51907997194604), ((0.25, 0.45000000000000001, 1.0), 17.51803212788157), ((0.65000000000000002, 0.95000000000000007, 1.0), 17.517520793803197)], [((0.45000000000000001, 0.60000000000000009, 1.0), 15.66312824373844), ((0.45000000000000001, 0.65000000000000002, 1.0), 15.659401864297775), ((0.45000000000000001, 0.55000000000000004, 1.0), 15.512487532789425), ((0.45000000000000001, 0.75, 1.0), 15.47179868015371), ((0.45000000000000001, 0.85000000000000009, 1.0), 15.47179868015371), ((0.45000000000000001, 0.95000000000000007, 1.0), 15.47179868015371), ((0.45000000000000001, 0.70000000000000007, 1.0), 15.47179868015371), ((0.45000000000000001, 0.90000000000000002, 1.0), 15.47179868015371), ((0.45000000000000001, 0.80000000000000004, 1.0), 15.47179868015371), ((0.45000000000000001, 1.0, 1.0), 15.47179868015371)]], 'Halloween': [[((0.40000000000000002, 0.5, 1.0), 19.02655372356299), ((0.40000000000000002, 0.55000000000000004, 1.0), 18.871563582973423), ((0.40000000000000002, 0.60000000000000009, 1.0), 18.84457328796708), ((0.40000000000000002, 0.45000000000000001, 1.0), 18.659315597510034), ((0.55000000000000004, 0.65000000000000002, 1.0), 18.624675906175295), ((0.55000000000000004, 0.70000000000000007, 1.0), 18.616474231267645), ((0.5, 0.60000000000000009, 1.0), 18.60339259743442), ((0.45000000000000001, 0.60000000000000009, 1.0), 18.536561297486028), ((0.5, 0.65000000000000002, 1.0), 18.53405278608717), ((0.45000000000000001, 0.55000000000000004, 1.0), 18.51140453780296), ((0.5, 0.70000000000000007, 1.0), 18.47448672388223), ((0.45000000000000001, 0.65000000000000002, 1.0), 18.446171296634272), ((0.10000000000000001, 0.15000000000000002, 1.0), 18.400916137442504), ((0.050000000000000003, 0.15000000000000002, 1.0), 18.400119094978795), ((0.0, 0.15000000000000002, 1.0), 18.400119094978795), ((0.60000000000000009, 0.65000000000000002, 1.0), 18.396213380541646), ((0.40000000000000002, 0.65000000000000002, 1.0), 18.380540311466103), ((0.45000000000000001, 0.5, 1.0), 18.369544313849133), ((0.55000000000000004, 0.75, 1.0), 18.335938582121695), ((0.60000000000000009, 0.70000000000000007, 1.0), 18.327934314922082), ((0.35000000000000003, 0.5, 1.0), 18.315613924739562)], [((0.55000000000000004, 0.70000000000000007, 1.0), 21.153649001444045), ((0.5, 0.70000000000000007, 1.0), 21.063883171789715), ((0.55000000000000004, 0.85000000000000009, 1.0), 21.054476725835233), ((0.55000000000000004, 0.80000000000000004, 1.0), 21.037810059168564), ((0.55000000000000004, 0.75, 1.0), 21.034898378256884), ((0.5, 0.80000000000000004, 1.0), 20.95570696528017), ((0.5, 0.75, 1.0), 20.94395387886042), ((0.60000000000000009, 0.70000000000000007, 1.0), 20.939369525641318), ((0.65000000000000002, 0.70000000000000007, 1.0), 20.939044214896814), ((0.10000000000000001, 0.15000000000000002, 1.0), 20.835475737967176), ((0.050000000000000003, 0.15000000000000002, 1.0), 20.83380907130051), ((0.0, 0.15000000000000002, 1.0), 20.83380907130051), ((0.65000000000000002, 0.80000000000000004, 1.0), 20.8264423660353), ((0.60000000000000009, 0.85000000000000009, 1.0), 20.825034100219682), ((0.65000000000000002, 0.85000000000000009, 1.0), 20.824477599826473), ((0.60000000000000009, 0.75, 1.0), 20.823518884746782), ((0.60000000000000009, 0.80000000000000004, 1.0), 20.821546445898697), ((0.5, 0.85000000000000009, 1.0), 20.813373631946842), ((0.65000000000000002, 0.75, 1.0), 20.80622808032102), ((0.65000000000000002, 0.90000000000000002, 1.0), 20.79134548502769), ((0.65000000000000002, 1.0, 1.0), 20.78991691359912), ((0.65000000000000002, 0.95000000000000007, 1.0), 20.78991691359912), ((0.60000000000000009, 0.95000000000000007, 1.0), 20.78281187799746), ((0.60000000000000009, 0.90000000000000002, 1.0), 20.78281187799746), ((0.60000000000000009, 1.0, 1.0), 20.78281187799746), ((0.15000000000000002, 0.20000000000000001, 1.0), 20.77578963956653), ((0.0, 0.20000000000000001, 1.0), 20.68794910705285), ((0.050000000000000003, 0.20000000000000001, 1.0), 20.68794910705285), ((0.10000000000000001, 0.20000000000000001, 1.0), 20.68794910705285), ((0.55000000000000004, 0.65000000000000002, 1.0), 20.655015939200773), ((0.55000000000000004, 1.0, 1.0), 20.650217466575974), ((0.55000000000000004, 0.95000000000000007, 1.0), 20.650217466575974), ((0.55000000000000004, 0.90000000000000002, 1.0), 20.650217466575974), ((0.60000000000000009, 0.65000000000000002, 1.0), 20.636836485150447), ((0.5, 0.65000000000000002, 1.0), 20.620387585479335), ((0.70000000000000007, 1.0, 1.0), 20.578051767310335), ((0.70000000000000007, 0.95000000000000007, 1.0), 20.578051767310335), ((0.70000000000000007, 0.85000000000000009, 1.0), 20.578049990726257), ((0.70000000000000007, 0.80000000000000004, 1.0), 20.575978840628792), ((0.70000000000000007, 0.90000000000000002, 1.0), 20.575178591103825), ((0.70000000000000007, 0.75, 1.0), 20.469554070655636)], [((0.55000000000000004, 0.75, 1.0), 17.818319559223948), ((0.5, 0.70000000000000007, 1.0), 17.655833472597607), ((0.5, 0.75, 1.0), 17.610685252193466), ((0.55000000000000004, 0.70000000000000007, 1.0), 17.607654198304523), ((0.55000000000000004, 0.80000000000000004, 1.0), 17.569352944182757), ((0.55000000000000004, 0.85000000000000009, 1.0), 17.568204256269365), ((0.55000000000000004, 1.0, 1.0), 17.56608558100306), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.56608558100306), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.56608558100306), ((0.55000000000000004, 0.65000000000000002, 1.0), 17.503965982712565), ((0.5, 0.65000000000000002, 1.0), 17.46813180354641), ((0.5, 0.80000000000000004, 1.0), 17.461637789376255), ((0.5, 0.85000000000000009, 1.0), 17.439612377732058), ((0.5, 0.90000000000000002, 1.0), 17.437910718422707), ((0.5, 0.95000000000000007, 1.0), 17.437910718422707), ((0.5, 1.0, 1.0), 17.437910718422707), ((0.60000000000000009, 0.80000000000000004, 1.0), 17.436586746729137), ((0.60000000000000009, 0.85000000000000009, 1.0), 17.374482947792412), ((0.60000000000000009, 0.95000000000000007, 1.0), 17.373460068555246), ((0.60000000000000009, 0.90000000000000002, 1.0), 17.373460068555246), ((0.60000000000000009, 1.0, 1.0), 17.373460068555246), ((0.60000000000000009, 0.75, 1.0), 17.370811387166135), ((0.5, 0.60000000000000009, 1.0), 17.36665289398286), ((0.45000000000000001, 0.75, 1.0), 17.34272079219484), ((0.65000000000000002, 0.85000000000000009, 1.0), 17.3363634930142), ((0.65000000000000002, 1.0, 1.0), 17.33511499746692), ((0.65000000000000002, 0.90000000000000002, 1.0), 17.33511499746692), ((0.65000000000000002, 0.95000000000000007, 1.0), 17.33511499746692), ((0.75, 0.95000000000000007, 1.0), 17.300887264437083), ((0.75, 0.90000000000000002, 1.0), 17.300887264437083), ((0.75, 1.0, 1.0), 17.300887264437083), ((0.45000000000000001, 0.65000000000000002, 1.0), 17.29372170103623), ((0.70000000000000007, 0.85000000000000009, 1.0), 17.22456662638795), ((0.70000000000000007, 1.0, 1.0), 17.22356721919289), ((0.70000000000000007, 0.90000000000000002, 1.0), 17.22356721919289), ((0.70000000000000007, 0.95000000000000007, 1.0), 17.22356721919289), ((0.60000000000000009, 0.70000000000000007, 1.0), 17.22169354925131), ((0.45000000000000001, 0.70000000000000007, 1.0), 17.19079047855763), ((0.75, 0.85000000000000009, 1.0), 17.186403137452956), ((0.40000000000000002, 0.55000000000000004, 1.0), 17.180324038520617), ((0.40000000000000002, 0.60000000000000009, 1.0), 17.172916368748826), ((0.65000000000000002, 0.75, 1.0), 17.16825932970403)]], 'BusinessActivity': [[], [((0.45000000000000001, 0.5, 1.0), 15.173869859660414), ((0.0, 0.20000000000000001, 1.0), 15.082775767472263), ((0.050000000000000003, 0.20000000000000001, 1.0), 15.0724266278164), ((0.10000000000000001, 0.20000000000000001, 1.0), 15.061356552910375), ((0.5, 0.55000000000000004, 1.0), 15.059786840381504), ((0.0, 0.25, 1.0), 15.046063331015443), ((0.050000000000000003, 0.25, 1.0), 15.045746925383423), ((0.10000000000000001, 0.25, 1.0), 15.029167362617635), ((0.15000000000000002, 0.20000000000000001, 1.0), 15.016720271361988), ((0.55000000000000004, 0.60000000000000009, 1.0), 15.008526752804173), ((0.15000000000000002, 0.25, 1.0), 15.006143865236648), ((0.35000000000000003, 0.40000000000000002, 1.0), 15.001620097926729), ((0.20000000000000001, 0.25, 1.0), 14.999415577472195)], [((0.45000000000000001, 0.55000000000000004, 1.0), 15.264644042329573), ((0.40000000000000002, 0.55000000000000004, 1.0), 15.082587100202238), ((0.40000000000000002, 0.65000000000000002, 1.0), 14.998594280289264), ((0.35000000000000003, 0.55000000000000004, 1.0), 14.991636152795401), ((0.45000000000000001, 0.60000000000000009, 1.0), 14.959688624206896), ((0.30000000000000004, 0.55000000000000004, 1.0), 14.935472698959513), ((0.5, 0.60000000000000009, 1.0), 14.929732166932537), ((0.45000000000000001, 0.65000000000000002, 1.0), 14.920133640844512), ((0.40000000000000002, 0.75, 1.0), 14.91677299685524), ((0.25, 0.55000000000000004, 1.0), 14.916289012012339), ((0.40000000000000002, 0.90000000000000002, 1.0), 14.913340457172701), ((0.40000000000000002, 1.0, 1.0), 14.913340457172701), ((0.40000000000000002, 0.85000000000000009, 1.0), 14.913340457172701), ((0.40000000000000002, 0.95000000000000007, 1.0), 14.913340457172701), ((0.40000000000000002, 0.80000000000000004, 1.0), 14.913340457172701), ((0.20000000000000001, 0.55000000000000004, 1.0), 14.911675485931012), ((0.35000000000000003, 0.65000000000000002, 1.0), 14.907033097410826), ((0.15000000000000002, 0.55000000000000004, 1.0), 14.903917683691098), ((0.40000000000000002, 0.70000000000000007, 1.0), 14.900659628576694), ((0.10000000000000001, 0.55000000000000004, 1.0), 14.897024555725642), ((0.0, 0.55000000000000004, 1.0), 14.893910216706452), ((0.050000000000000003, 0.55000000000000004, 1.0), 14.893910216706452), ((0.30000000000000004, 0.65000000000000002, 1.0), 14.864656847518955), ((0.5, 0.55000000000000004, 1.0), 14.856393382416101), ((0.25, 0.65000000000000002, 1.0), 14.843063860883191), ((0.45000000000000001, 0.75, 1.0), 14.84285075225303), ((0.20000000000000001, 0.65000000000000002, 1.0), 14.839034832429192), ((0.45000000000000001, 0.85000000000000009, 1.0), 14.838230025757305), ((0.45000000000000001, 0.95000000000000007, 1.0), 14.838230025757305), ((0.45000000000000001, 0.90000000000000002, 1.0), 14.838230025757305), ((0.45000000000000001, 0.80000000000000004, 1.0), 14.838230025757305), ((0.45000000000000001, 1.0, 1.0), 14.838230025757305), ((0.15000000000000002, 0.65000000000000002, 1.0), 14.835758816331948), ((0.10000000000000001, 0.65000000000000002, 1.0), 14.827797042624614), ((0.45000000000000001, 0.70000000000000007, 1.0), 14.826484124636922), ((0.050000000000000003, 0.65000000000000002, 1.0), 14.824771721580701), ((0.0, 0.65000000000000002, 1.0), 14.824771721580701), ((0.40000000000000002, 0.60000000000000009, 1.0), 14.823613615760314), ((0.35000000000000003, 0.75, 1.0), 14.819667969998154), ((0.35000000000000003, 0.80000000000000004, 1.0), 14.81623543031561), ((0.35000000000000003, 1.0, 1.0), 14.81623543031561), ((0.35000000000000003, 0.95000000000000007, 1.0), 14.81623543031561), ((0.35000000000000003, 0.85000000000000009, 1.0), 14.81623543031561), ((0.35000000000000003, 0.90000000000000002, 1.0), 14.81623543031561), ((0.35000000000000003, 0.70000000000000007, 1.0), 14.806293301874462), ((0.30000000000000004, 0.75, 1.0), 14.776596857886114), ((0.30000000000000004, 0.85000000000000009, 1.0), 14.773164318203571), ((0.30000000000000004, 0.90000000000000002, 1.0), 14.773164318203571), ((0.30000000000000004, 1.0, 1.0), 14.773164318203571), ((0.30000000000000004, 0.95000000000000007, 1.0), 14.773164318203571), ((0.30000000000000004, 0.80000000000000004, 1.0), 14.773164318203571), ((0.30000000000000004, 0.70000000000000007, 1.0), 14.760667784479361), ((0.25, 0.75, 1.0), 14.757582053693996), ((0.25, 0.95000000000000007, 1.0), 14.75414951401146), ((0.25, 0.80000000000000004, 1.0), 14.75414951401146), ((0.25, 1.0, 1.0), 14.75414951401146), ((0.25, 0.90000000000000002, 1.0), 14.75414951401146), ((0.25, 0.85000000000000009, 1.0), 14.75414951401146), ((0.20000000000000001, 0.75, 1.0), 14.753251222512365), ((0.5, 0.65000000000000002, 1.0), 14.75266523824332), ((0.15000000000000002, 0.75, 1.0), 14.749973004160008), ((0.20000000000000001, 0.80000000000000004, 1.0), 14.749818682829824), ((0.20000000000000001, 0.85000000000000009, 1.0), 14.749818682829824), ((0.20000000000000001, 1.0, 1.0), 14.749818682829824), ((0.20000000000000001, 0.95000000000000007, 1.0), 14.749818682829824), ((0.20000000000000001, 0.90000000000000002, 1.0), 14.749818682829824), ((0.15000000000000002, 0.90000000000000002, 1.0), 14.74654046447747), ((0.15000000000000002, 0.95000000000000007, 1.0), 14.74654046447747), ((0.15000000000000002, 0.80000000000000004, 1.0), 14.74654046447747), ((0.15000000000000002, 0.85000000000000009, 1.0), 14.74654046447747), ((0.15000000000000002, 1.0, 1.0), 14.74654046447747), ((0.10000000000000001, 0.75, 1.0), 14.742011230452677), ((0.25, 0.70000000000000007, 1.0), 14.741652980287244), ((0.35000000000000003, 0.60000000000000009, 1.0), 14.740462555435595), ((0.0, 0.75, 1.0), 14.738985909408763), ((0.050000000000000003, 0.75, 1.0), 14.738985909408763), ((0.10000000000000001, 0.95000000000000007, 1.0), 14.738578690770137), ((0.10000000000000001, 0.90000000000000002, 1.0), 14.738578690770137), ((0.10000000000000001, 0.80000000000000004, 1.0), 14.738578690770137), ((0.10000000000000001, 0.85000000000000009, 1.0), 14.738578690770137), ((0.10000000000000001, 1.0, 1.0), 14.738578690770137), ((0.20000000000000001, 0.70000000000000007, 1.0), 14.737340283383615), ((0.0, 0.90000000000000002, 1.0), 14.735553369726224), ((0.050000000000000003, 0.85000000000000009, 1.0), 14.735553369726224), ((0.0, 0.95000000000000007, 1.0), 14.735553369726224), ((0.0, 0.80000000000000004, 1.0), 14.735553369726224), ((0.050000000000000003, 1.0, 1.0), 14.735553369726224), ((0.050000000000000003, 0.95000000000000007, 1.0), 14.735553369726224), ((0.0, 1.0, 1.0), 14.735553369726224), ((0.050000000000000003, 0.80000000000000004, 1.0), 14.735553369726224), ((0.050000000000000003, 0.90000000000000002, 1.0), 14.735553369726224), ((0.0, 0.85000000000000009, 1.0), 14.735553369726224), ((0.15000000000000002, 0.70000000000000007, 1.0), 14.73406206503126), ((0.10000000000000001, 0.70000000000000007, 1.0), 14.726100291323926), ((0.0, 0.70000000000000007, 1.0), 14.723074970280015), ((0.050000000000000003, 0.70000000000000007, 1.0), 14.723074970280015), ((0.30000000000000004, 0.60000000000000009, 1.0), 14.694352634678102), ((0.5, 0.90000000000000002, 1.0), 14.686211708551955), ((0.5, 0.85000000000000009, 1.0), 14.686211708551955), ((0.5, 0.95000000000000007, 1.0), 14.686211708551955), ((0.5, 0.75, 1.0), 14.686211708551955), ((0.5, 1.0, 1.0), 14.686211708551955), ((0.5, 0.80000000000000004, 1.0), 14.686211708551955), ((0.5, 0.70000000000000007, 1.0), 14.676133227494613), ((0.25, 0.60000000000000009, 1.0), 14.674548603137218), ((0.20000000000000001, 0.60000000000000009, 1.0), 14.667323059916708), ((0.15000000000000002, 0.60000000000000009, 1.0), 14.664053489077027), ((0.10000000000000001, 0.60000000000000009, 1.0), 14.65609171536969), ((0.050000000000000003, 0.60000000000000009, 1.0), 14.65306639432578), ((0.0, 0.60000000000000009, 1.0), 14.65306639432578), ((0.40000000000000002, 0.5, 1.0), 14.486643158107722), ((0.35000000000000003, 0.5, 1.0), 14.311871729597447), ((0.45000000000000001, 0.5, 1.0), 14.298807878106137), ((0.30000000000000004, 0.5, 1.0), 14.243250294897969), ((0.25, 0.5, 1.0), 14.218289169461013), ((0.20000000000000001, 0.5, 1.0), 14.21325019840544), ((0.15000000000000002, 0.5, 1.0), 14.207827259436648), ((0.10000000000000001, 0.5, 1.0), 14.19920546929378), ((0.0, 0.5, 1.0), 14.194863217342832), ((0.050000000000000003, 0.5, 1.0), 14.194863217342832)]], 'Protest': [[((0.60000000000000009, 0.95000000000000007, 1.0), 17.732735424481316), ((0.60000000000000009, 0.90000000000000002, 1.0), 17.732735424481316), ((0.60000000000000009, 1.0, 1.0), 17.688786397318), ((0.45000000000000001, 0.95000000000000007, 1.0), 17.587867604117147), ((0.45000000000000001, 0.90000000000000002, 1.0), 17.587867604117147), ((0.45000000000000001, 1.0, 1.0), 17.587867604117147), ((0.5, 0.90000000000000002, 1.0), 17.535772313844774), ((0.5, 0.95000000000000007, 1.0), 17.535772313844774), ((0.5, 1.0, 1.0), 17.535772313844774), ((0.40000000000000002, 0.90000000000000002, 1.0), 17.46250097562603), ((0.40000000000000002, 1.0, 1.0), 17.46250097562603), ((0.40000000000000002, 0.95000000000000007, 1.0), 17.46250097562603), ((0.65000000000000002, 1.0, 1.0), 17.453845327116024), ((0.65000000000000002, 0.90000000000000002, 1.0), 17.453845327116024), ((0.65000000000000002, 0.95000000000000007, 1.0), 17.453845327116024), ((0.20000000000000001, 1.0, 1.0), 17.42442003517122), ((0.20000000000000001, 0.95000000000000007, 1.0), 17.42442003517122), ((0.20000000000000001, 0.90000000000000002, 1.0), 17.42442003517122), ((0.0, 0.90000000000000002, 1.0), 17.424074958457936), ((0.10000000000000001, 0.95000000000000007, 1.0), 17.424074958457936), ((0.10000000000000001, 0.90000000000000002, 1.0), 17.424074958457936), ((0.0, 0.95000000000000007, 1.0), 17.424074958457936), ((0.050000000000000003, 1.0, 1.0), 17.424074958457936), ((0.050000000000000003, 0.95000000000000007, 1.0), 17.424074958457936), ((0.0, 1.0, 1.0), 17.424074958457936), ((0.050000000000000003, 0.90000000000000002, 1.0), 17.424074958457936), ((0.10000000000000001, 1.0, 1.0), 17.424074958457936), ((0.15000000000000002, 0.90000000000000002, 1.0), 17.423787519538706), ((0.15000000000000002, 0.95000000000000007, 1.0), 17.423787519538706), ((0.15000000000000002, 1.0, 1.0), 17.423787519538706), ((0.25, 0.95000000000000007, 1.0), 17.423604330130164), ((0.25, 1.0, 1.0), 17.423604330130164), ((0.25, 0.90000000000000002, 1.0), 17.423604330130164), ((0.30000000000000004, 0.90000000000000002, 1.0), 17.420721103727658), ((0.30000000000000004, 1.0, 1.0), 17.420721103727658), ((0.30000000000000004, 0.95000000000000007, 1.0), 17.420721103727658), ((0.45000000000000001, 0.80000000000000004, 1.0), 17.402709076420354), ((0.35000000000000003, 1.0, 1.0), 17.39688835624448), ((0.35000000000000003, 0.95000000000000007, 1.0), 17.39688835624448), ((0.35000000000000003, 0.90000000000000002, 1.0), 17.39688835624448), ((0.45000000000000001, 0.85000000000000009, 1.0), 17.388776253291102), ((0.5, 0.85000000000000009, 1.0), 17.354695979464864), ((0.5, 0.80000000000000004, 1.0), 17.354695979464864), ((0.55000000000000004, 0.90000000000000002, 1.0), 17.35344019661874), ((0.70000000000000007, 1.0, 1.0), 17.351645609628335), ((0.70000000000000007, 0.90000000000000002, 1.0), 17.351645609628335), ((0.70000000000000007, 0.95000000000000007, 1.0), 17.351645609628335), ((0.60000000000000009, 0.85000000000000009, 1.0), 17.350340042085932), ((0.60000000000000009, 0.80000000000000004, 1.0), 17.350340042085932), ((0.60000000000000009, 0.75, 1.0), 17.320428683603147), ((0.55000000000000004, 1.0, 1.0), 17.309491169455427), ((0.55000000000000004, 0.95000000000000007, 1.0), 17.309491169455427), ((0.40000000000000002, 0.80000000000000004, 1.0), 17.279156756070073), ((0.40000000000000002, 0.85000000000000009, 1.0), 17.265223932940824), ((0.65000000000000002, 0.80000000000000004, 1.0), 17.24296097337453), ((0.65000000000000002, 0.85000000000000009, 1.0), 17.24296097337453), ((0.20000000000000001, 0.80000000000000004, 1.0), 17.241075815615268), ((0.0, 0.80000000000000004, 1.0), 17.240730738901984), ((0.10000000000000001, 0.80000000000000004, 1.0), 17.240730738901984), ((0.050000000000000003, 0.80000000000000004, 1.0), 17.240730738901984), ((0.15000000000000002, 0.80000000000000004, 1.0), 17.240443299982754), ((0.25, 0.80000000000000004, 1.0), 17.240260110574212), ((0.30000000000000004, 0.80000000000000004, 1.0), 17.2373768841717), ((0.20000000000000001, 0.85000000000000009, 1.0), 17.227142992486016), ((0.050000000000000003, 0.85000000000000009, 1.0), 17.226797915772732), ((0.10000000000000001, 0.85000000000000009, 1.0), 17.226797915772732), ((0.0, 0.85000000000000009, 1.0), 17.226797915772732), ((0.15000000000000002, 0.85000000000000009, 1.0), 17.226510476853502), ((0.25, 0.85000000000000009, 1.0), 17.22632728744496), ((0.30000000000000004, 0.85000000000000009, 1.0), 17.22344406104245), ((0.35000000000000003, 0.80000000000000004, 1.0), 17.213544136688522), ((0.35000000000000003, 0.85000000000000009, 1.0), 17.199611313559274), ((0.60000000000000009, 0.70000000000000007, 1.0), 17.199003151463327), ((0.70000000000000007, 0.80000000000000004, 1.0), 17.17713863512136), ((0.55000000000000004, 0.60000000000000009, 1.0), 17.166168587909564)], [((0.60000000000000009, 0.65000000000000002, 1.0), 28.33578694970302), ((0.55000000000000004, 0.65000000000000002, 1.0), 28.29304591762387), ((0.5, 0.65000000000000002, 1.0), 28.29055875362524), ((0.45000000000000001, 0.65000000000000002, 1.0), 28.278481458939247), ((0.050000000000000003, 0.65000000000000002, 1.0), 28.276465329906994), ((0.0, 0.65000000000000002, 1.0), 28.276465329906994), ((0.35000000000000003, 0.65000000000000002, 1.0), 28.276465329906994), ((0.10000000000000001, 0.65000000000000002, 1.0), 28.276465329906994), ((0.30000000000000004, 0.65000000000000002, 1.0), 28.276465329906994), ((0.20000000000000001, 0.65000000000000002, 1.0), 28.276465329906994), ((0.15000000000000002, 0.65000000000000002, 1.0), 28.276465329906994), ((0.25, 0.65000000000000002, 1.0), 28.276465329906994), ((0.40000000000000002, 0.65000000000000002, 1.0), 28.276465329906994), ((0.85000000000000009, 1.0, 1.0), 28.270857072094017), ((0.80000000000000004, 1.0, 1.0), 28.19146497693687), ((0.90000000000000002, 1.0, 1.0), 28.15917906755887), ((0.75, 1.0, 1.0), 28.155551910411557), ((0.85000000000000009, 0.95000000000000007, 1.0), 28.14013144851125)], []], 'Cruise': [[], [((0.5, 0.55000000000000004, 1.0), 23.637581977178137)], [((0.65000000000000002, 0.75, 1.0), 16.240434828092987)]], 'Birthday': [[((0.40000000000000002, 0.45000000000000001, 1.0), 24.51358805395909), ((0.35000000000000003, 0.40000000000000002, 1.0), 24.501217622293474), ((0.35000000000000003, 0.45000000000000001, 1.0), 24.463502405681922), ((0.45000000000000001, 0.5, 1.0), 24.425197396293534), ((0.30000000000000004, 0.40000000000000002, 1.0), 24.37502396288676), ((0.40000000000000002, 0.5, 1.0), 24.330182589030088), ((0.40000000000000002, 0.55000000000000004, 1.0), 24.21440934942322), ((0.30000000000000004, 0.35000000000000003, 1.0), 24.15557897788001), ((0.25, 0.30000000000000004, 1.0), 24.12277447167107), ((0.75, 0.95000000000000007, 1.0), 24.1190386073295), ((0.75, 0.90000000000000002, 1.0), 24.1190386073295), ((0.75, 1.0, 1.0), 24.1190386073295), ((0.45000000000000001, 0.55000000000000004, 1.0), 24.0992493677005), ((0.15000000000000002, 0.30000000000000004, 1.0), 24.099030480850445), ((0.050000000000000003, 0.30000000000000004, 1.0), 24.099030480850445), ((0.0, 0.30000000000000004, 1.0), 24.099030480850445), ((0.10000000000000001, 0.30000000000000004, 1.0), 24.099030480850445)], [((0.40000000000000002, 0.5, 1.0), 22.466614766881126)], []], 'NatureTrip': [[((0.0, 0.5, 1.0), 15.794686624418292), ((0.10000000000000001, 0.5, 1.0), 15.794686624418292), ((0.050000000000000003, 0.5, 1.0), 15.794686624418292), ((0.15000000000000002, 0.5, 1.0), 15.794581399885324), ((0.20000000000000001, 0.5, 1.0), 15.792649401032143), ((0.25, 0.5, 1.0), 15.791407165007296), ((0.050000000000000003, 0.65000000000000002, 1.0), 15.790362773665867), ((0.0, 0.90000000000000002, 1.0), 15.790362773665867), ((0.10000000000000001, 0.95000000000000007, 1.0), 15.790362773665867), ((0.050000000000000003, 0.60000000000000009, 1.0), 15.790362773665867), ((0.10000000000000001, 0.55000000000000004, 1.0), 15.790362773665867), ((0.0, 0.65000000000000002, 1.0), 15.790362773665867), ((0.10000000000000001, 0.70000000000000007, 1.0), 15.790362773665867), ((0.10000000000000001, 0.75, 1.0), 15.790362773665867), ((0.050000000000000003, 0.85000000000000009, 1.0), 15.790362773665867), ((0.0, 0.55000000000000004, 1.0), 15.790362773665867), ((0.10000000000000001, 0.65000000000000002, 1.0), 15.790362773665867), ((0.10000000000000001, 0.90000000000000002, 1.0), 15.790362773665867), ((0.050000000000000003, 0.55000000000000004, 1.0), 15.790362773665867), ((0.0, 0.75, 1.0), 15.790362773665867), ((0.0, 0.95000000000000007, 1.0), 15.790362773665867), ((0.0, 0.80000000000000004, 1.0), 15.790362773665867), ((0.050000000000000003, 0.75, 1.0), 15.790362773665867), ((0.0, 0.70000000000000007, 1.0), 15.790362773665867), ((0.050000000000000003, 1.0, 1.0), 15.790362773665867), ((0.10000000000000001, 0.60000000000000009, 1.0), 15.790362773665867), ((0.10000000000000001, 0.80000000000000004, 1.0), 15.790362773665867), ((0.050000000000000003, 0.95000000000000007, 1.0), 15.790362773665867), ((0.0, 1.0, 1.0), 15.790362773665867), ((0.10000000000000001, 0.85000000000000009, 1.0), 15.790362773665867), ((0.050000000000000003, 0.80000000000000004, 1.0), 15.790362773665867), ((0.0, 0.60000000000000009, 1.0), 15.790362773665867), ((0.050000000000000003, 0.90000000000000002, 1.0), 15.790362773665867), ((0.10000000000000001, 1.0, 1.0), 15.790362773665867), ((0.050000000000000003, 0.70000000000000007, 1.0), 15.790362773665867), ((0.0, 0.85000000000000009, 1.0), 15.790362773665867), ((0.15000000000000002, 0.70000000000000007, 1.0), 15.790257549132903), ((0.15000000000000002, 0.60000000000000009, 1.0), 15.790257549132903), ((0.15000000000000002, 0.90000000000000002, 1.0), 15.790257549132903), ((0.15000000000000002, 0.95000000000000007, 1.0), 15.790257549132903), ((0.15000000000000002, 0.80000000000000004, 1.0), 15.790257549132903), ((0.15000000000000002, 0.75, 1.0), 15.790257549132903), ((0.15000000000000002, 0.55000000000000004, 1.0), 15.790257549132903), ((0.15000000000000002, 0.65000000000000002, 1.0), 15.790257549132903), ((0.15000000000000002, 0.85000000000000009, 1.0), 15.790257549132903), ((0.15000000000000002, 1.0, 1.0), 15.790257549132903), ((0.20000000000000001, 0.80000000000000004, 1.0), 15.788325550279723), ((0.20000000000000001, 0.75, 1.0), 15.788325550279723), ((0.20000000000000001, 0.65000000000000002, 1.0), 15.788325550279723), ((0.20000000000000001, 0.85000000000000009, 1.0), 15.788325550279723), ((0.20000000000000001, 1.0, 1.0), 15.788325550279723), ((0.20000000000000001, 0.55000000000000004, 1.0), 15.788325550279723), ((0.20000000000000001, 0.60000000000000009, 1.0), 15.788325550279723), ((0.20000000000000001, 0.95000000000000007, 1.0), 15.788325550279723), ((0.20000000000000001, 0.70000000000000007, 1.0), 15.788325550279723), ((0.20000000000000001, 0.90000000000000002, 1.0), 15.788325550279723), ((0.25, 0.75, 1.0), 15.787083314254875), ((0.25, 0.95000000000000007, 1.0), 15.787083314254875), ((0.25, 0.70000000000000007, 1.0), 15.787083314254875), ((0.25, 0.80000000000000004, 1.0), 15.787083314254875), ((0.25, 1.0, 1.0), 15.787083314254875), ((0.25, 0.55000000000000004, 1.0), 15.787083314254875), ((0.25, 0.90000000000000002, 1.0), 15.787083314254875), ((0.25, 0.65000000000000002, 1.0), 15.787083314254875), ((0.25, 0.85000000000000009, 1.0), 15.787083314254875), ((0.25, 0.60000000000000009, 1.0), 15.787083314254875), ((0.10000000000000001, 0.45000000000000001, 1.0), 15.784637150083102), ((0.0, 0.45000000000000001, 1.0), 15.784637150083102), ((0.050000000000000003, 0.45000000000000001, 1.0), 15.784637150083102), ((0.15000000000000002, 0.45000000000000001, 1.0), 15.784531925550137), ((0.20000000000000001, 0.45000000000000001, 1.0), 15.782599926696953), ((0.25, 0.45000000000000001, 1.0), 15.781357690672106), ((0.30000000000000004, 0.75, 1.0), 15.665352486332022), ((0.30000000000000004, 0.85000000000000009, 1.0), 15.665352486332022), ((0.30000000000000004, 0.55000000000000004, 1.0), 15.665352486332022), ((0.30000000000000004, 0.90000000000000002, 1.0), 15.665352486332022), ((0.30000000000000004, 0.65000000000000002, 1.0), 15.665352486332022), ((0.30000000000000004, 0.70000000000000007, 1.0), 15.665352486332022), ((0.30000000000000004, 0.60000000000000009, 1.0), 15.665352486332022), ((0.30000000000000004, 1.0, 1.0), 15.665352486332022), ((0.30000000000000004, 0.95000000000000007, 1.0), 15.665352486332022), ((0.30000000000000004, 0.80000000000000004, 1.0), 15.665352486332022), ((0.30000000000000004, 0.5, 1.0), 15.661711372442463), ((0.30000000000000004, 0.45000000000000001, 1.0), 15.655303011996835), ((0.35000000000000003, 0.80000000000000004, 1.0), 15.632608491650823), ((0.35000000000000003, 0.55000000000000004, 1.0), 15.632608491650823), ((0.35000000000000003, 0.65000000000000002, 1.0), 15.632608491650823), ((0.35000000000000003, 1.0, 1.0), 15.632608491650823), ((0.35000000000000003, 0.75, 1.0), 15.632608491650823), ((0.35000000000000003, 0.95000000000000007, 1.0), 15.632608491650823), ((0.35000000000000003, 0.85000000000000009, 1.0), 15.632608491650823), ((0.35000000000000003, 0.90000000000000002, 1.0), 15.632608491650823), ((0.35000000000000003, 0.70000000000000007, 1.0), 15.632608491650823), ((0.35000000000000003, 0.60000000000000009, 1.0), 15.632608491650823), ((0.35000000000000003, 0.5, 1.0), 15.626200131205195), ((0.40000000000000002, 0.90000000000000002, 1.0), 15.62098753425052), ((0.40000000000000002, 0.75, 1.0), 15.62098753425052), ((0.40000000000000002, 1.0, 1.0), 15.62098753425052), ((0.40000000000000002, 0.60000000000000009, 1.0), 15.62098753425052), ((0.40000000000000002, 0.70000000000000007, 1.0), 15.62098753425052), ((0.40000000000000002, 0.85000000000000009, 1.0), 15.62098753425052), ((0.40000000000000002, 0.95000000000000007, 1.0), 15.62098753425052), ((0.40000000000000002, 0.80000000000000004, 1.0), 15.62098753425052), ((0.40000000000000002, 0.65000000000000002, 1.0), 15.62098753425052), ((0.40000000000000002, 0.55000000000000004, 1.0), 15.62098753425052), ((0.0, 0.40000000000000002, 1.0), 14.841230556676507), ((0.10000000000000001, 0.40000000000000002, 1.0), 14.841230556676507), ((0.050000000000000003, 0.40000000000000002, 1.0), 14.841230556676507), ((0.15000000000000002, 0.40000000000000002, 1.0), 14.841125332143543), ((0.20000000000000001, 0.40000000000000002, 1.0), 14.839193333290362), ((0.25, 0.40000000000000002, 1.0), 14.837951097265517), ((0.45000000000000001, 0.75, 1.0), 14.673570207461273), ((0.45000000000000001, 0.85000000000000009, 1.0), 14.673570207461273), ((0.45000000000000001, 0.95000000000000007, 1.0), 14.673570207461273), ((0.45000000000000001, 0.70000000000000007, 1.0), 14.673570207461273), ((0.45000000000000001, 0.60000000000000009, 1.0), 14.673570207461273), ((0.45000000000000001, 0.65000000000000002, 1.0), 14.673570207461273), ((0.45000000000000001, 0.90000000000000002, 1.0), 14.673570207461273), ((0.45000000000000001, 0.80000000000000004, 1.0), 14.673570207461273), ((0.45000000000000001, 1.0, 1.0), 14.673570207461273), ((0.0, 0.35000000000000003, 1.0), 14.611068441595963), ((0.050000000000000003, 0.35000000000000003, 1.0), 14.611068441595963), ((0.10000000000000001, 0.35000000000000003, 1.0), 14.611068441595963), ((0.15000000000000002, 0.35000000000000003, 1.0), 14.610963217062999), ((0.20000000000000001, 0.35000000000000003, 1.0), 14.609031218209816), ((0.25, 0.35000000000000003, 1.0), 14.605500507872687), ((0.30000000000000004, 0.40000000000000002, 1.0), 14.517409494731398), ((0.35000000000000003, 0.45000000000000001, 1.0), 14.488306613939757), ((0.45000000000000001, 0.55000000000000004, 1.0), 14.48309401698508), ((0.40000000000000002, 0.5, 1.0), 14.48309401698508), ((0.5, 0.90000000000000002, 1.0), 14.478124020087543), ((0.5, 0.85000000000000009, 1.0), 14.478124020087543), ((0.5, 0.65000000000000002, 1.0), 14.478124020087543), ((0.5, 0.70000000000000007, 1.0), 14.478124020087543), ((0.5, 0.95000000000000007, 1.0), 14.478124020087543), ((0.5, 0.75, 1.0), 14.478124020087543), ((0.5, 0.60000000000000009, 1.0), 14.478124020087543), ((0.5, 1.0, 1.0), 14.478124020087543), ((0.5, 0.80000000000000004, 1.0), 14.478124020087543), ((0.30000000000000004, 0.35000000000000003, 1.0), 14.309666105756348), ((0.35000000000000003, 0.40000000000000002, 1.0), 14.29920963291524), ((0.40000000000000002, 0.45000000000000001, 1.0), 14.29399703596056), ((0.45000000000000001, 0.5, 1.0), 14.29399703596056), ((0.5, 0.55000000000000004, 1.0), 14.29399703596056), ((0.55000000000000004, 0.65000000000000002, 1.0), 14.267435224247516), ((0.55000000000000004, 0.80000000000000004, 1.0), 14.267435224247516), ((0.55000000000000004, 0.85000000000000009, 1.0), 14.267435224247516), ((0.55000000000000004, 1.0, 1.0), 14.267435224247516), ((0.55000000000000004, 0.70000000000000007, 1.0), 14.267435224247516), ((0.55000000000000004, 0.75, 1.0), 14.267435224247516), ((0.55000000000000004, 0.95000000000000007, 1.0), 14.267435224247516), ((0.55000000000000004, 0.90000000000000002, 1.0), 14.267435224247516), ((0.55000000000000004, 0.60000000000000009, 1.0), 14.267435224247516), ((0.050000000000000003, 0.30000000000000004, 1.0), 14.122554868921766), ((0.0, 0.30000000000000004, 1.0), 14.122554868921766), ((0.10000000000000001, 0.30000000000000004, 1.0), 14.122554868921766), ((0.15000000000000002, 0.30000000000000004, 1.0), 14.1224496443888), ((0.20000000000000001, 0.30000000000000004, 1.0), 14.12051764553562), ((0.25, 0.30000000000000004, 1.0), 14.115764787797868)], [], []]}
        # theta = {'PersonalSports': [0.050000000000000003], 'Museum': [0.60000000000000009], 'UrbanTrip': [0.15000000000000002], 'Zoo': [0.5], 'BeachTrip': [0.70000000000000007], 'PersonalMusicActivity': [0.95000000000000007], 'Christmas': [0.10000000000000001], 'PersonalArtActivity': [0.0], 'GroupActivity': [0.20000000000000001], 'Wedding': [0.15000000000000002], 'ReligiousActivity': [0.60000000000000009], 'Graduation': [0.10000000000000001], 'CasualFamilyGather': [0.10000000000000001], 'Architecture': [0.050000000000000003], 'ThemePark': [0.15000000000000002], 'Sports': [0.40000000000000002], 'Show': [0.70000000000000007], 'Halloween': [0.15000000000000002], 'BusinessActivity': [0.0], 'Protest': [0.20000000000000001], 'Cruise': [0.050000000000000003], 'Birthday': [0.0], 'NatureTrip': [0.5]}

        global combine_face_model
        # global face_model
        combine_face_model_list = ['_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle','_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle',
                                 '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle', '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle',
                                 '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                                 # '_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle','_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle',
                                 # '_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle', '_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle',
                                 # '_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle']
        val_ids = [0,1,2,3,4]#,0,1,2,3,4]
        for id, name in zip(val_ids, combine_face_model_list):
            self.val_id = id
            self.validation_path = '/validation_' + str(self.val_id) + '/'

            combine_face_model = name
            print combine_face_model

            if method == 'thetapre':
                dict_theta = {}
                f = open('face_combine_features_theta'+noqual_name+'_fromnoevent'+itera+'.cPickle','rb')
                dict_score = cPickle.load(f)
                f.close()
                for event_name in dict_name2:
                    temp = {}
                    this_dict = dict_score[event_name]
                    for i in this_dict:
                        for j in i:
                            if j[0] in temp:
                                temp[j[0]] += j[1]
                            else:
                                temp[j[0]] = j[1]
                    sorted_temp = sorted(temp.items(), key = operator.itemgetter(1), reverse=True)
                    dict_theta[event_name] = [sorted_temp[0][0]]
                print dict_theta
                for event_name in dict_name2:
                    if event_name not in dict_from_validation:
                        dict_from_validation[event_name] = [self.grid_search_face_3(event_name, permute=True, use_theta=True, validation_name='val_validation',theta_pre = dict_theta[event_name][0])]
                    else:
                        dict_from_validation[event_name].append(self.grid_search_face_3(event_name, permute=True, use_theta=True, validation_name='val_validation',theta_pre = dict_theta[event_name][0]))
            elif method == 'thetaprepower':
                dict_theta = {}
                f = open('face_combine_features_thetapower'+noqual_name+'_fromnoevent'+itera+'.cPickle','rb')
                dict_score = cPickle.load(f)
                f.close()
                for event_name in dict_name2:
                    temp = {}
                    this_dict = dict_score[event_name]
                    for i in this_dict:
                        for j in i:
                            if j[0] in temp:
                                temp[j[0]] += j[1]
                            else:
                                temp[j[0]] = j[1]
                    sorted_temp = sorted(temp.items(), key = operator.itemgetter(1), reverse=True)
                    dict_theta[event_name] = [sorted_temp[0][0][2]]
                print dict_theta
                for event_name in dict_name2:
                    if event_name not in dict_from_validation:
                        dict_from_validation[event_name] = [self.grid_search_face_thetaprepower(event_name,use_theta=True, permute=True, validation_name='val_validation',theta_pre = dict_theta[event_name][0])]
                    else:
                        dict_from_validation[event_name].append(self.grid_search_face_thetaprepower(event_name, use_theta=True, permute=True, validation_name='val_validation',theta_pre = dict_theta[event_name][0]))
            else:
                for event_name in dict_name2:
                    print event_name
                    #face_model = '_'+event_name.lower()+'_iter_30000_sigmoid1.cPickle'
                    #print face_model
                    if event_name not in dict_from_validation:
                        #print event_name
                        # dict_from_validation[event_name] = [self.grid_search_face_2(event_name,use_theta=True, permute=True, validation_name='val_validation', theta_pre=theta[event_name][0])]
                        if method == 'thetapower':
                            dict_from_validation[event_name] = [self.grid_search_face_15(event_name, permute=True, validation_name='val_validation')]
                        if method == 'alphathetapower':
                            dict_from_validation[event_name] = [self.grid_search_face_25(event_name, permute=True, validation_name='val_validation')]
                        if method == 'beta05theta':
                            dict_from_validation[event_name] = [self.grid_search_face_beta_05_theta(event_name, permute=True, validation_name='val_validation')]
                        if method == 'betatheta':
                            dict_from_validation[event_name] = [self.grid_search_face_betatheta(event_name, permute=True, validation_name='val_validation')]
                        if method == 'full':
                            dict_from_validation[event_name] = [self.grid_search_face_full(event_name, permute=True, validation_name='val_validation')]
                        if method == 'alphabeta':
                            dict_from_validation[event_name] = [self.grid_search_face_2(event_name,use_theta=False, permute=True, validation_name='val_validation')]
                        if method == 'theta':
                            dict_from_validation[event_name] = [self.grid_search_face(event_name, permute=True, validation_name='val_validation')]
                        if method == 'alphabetatheta':
                            dict_from_validation[event_name] = [self.grid_search_face_35(event_name, permute=True, validation_name='val_validation')]
                    else:
                        #print event_name
                        # dict_from_validation[event_name].append(self.grid_search_face_2(event_name, use_theta= True, permute=True, validation_name='val_validation',  theta_pre=theta[event_name][0]))
                        if method == 'full':
                            dict_from_validation[event_name].append(self.grid_search_face_full(event_name, permute=True, validation_name='val_validation'))
                        if method == 'thetapower':
                            dict_from_validation[event_name].append(self.grid_search_face_15(event_name, permute=True, validation_name='val_validation'))
                        if method == 'alphathetapower':
                            dict_from_validation[event_name].append(self.grid_search_face_25(event_name, permute=True, validation_name='val_validation'))
                        if method == 'beta05theta':
                            dict_from_validation[event_name].append(self.grid_search_face_beta_05_theta(event_name, permute=True, validation_name='val_validation'))
                        if method == 'betatheta':
                            dict_from_validation[event_name].append(self.grid_search_face_betatheta(event_name, permute=True, validation_name='val_validation'))
                        if method == 'alphabeta':
                            dict_from_validation[event_name].append(self.grid_search_face_2(event_name, use_theta= False, permute=True, validation_name='val_validation'))
                        if method == 'theta':
                            dict_from_validation[event_name].append(self.grid_search_face(event_name, permute=True, validation_name='val_validation'))
                        if method == 'alphabetatheta':
                            dict_from_validation[event_name] = [self.grid_search_face_35(event_name, permute=True, validation_name='val_validation')]
        print dict_from_validation
        f = open('face_combine_features_'+method+noqual_name+'_fromnoevent'+itera+'.cPickle','wb')
        cPickle.dump(dict_from_validation, f)
        f.close()
        '''
        for event_name in dict_name2:
             print event_name
             #dict_from_validation[event_name] = [self.grid_search_face(event_name, permute=True, validation_name='val_validation')]
             dict_from_validation[event_name] = [self.grid_search_face_3(event_name, use_theta=False, permute=True, validation_name='val_validation')]
        print dict_from_validation
        #dict_from_validation = {'PersonalSports': [(0.5, 0.55, 1.0)], 'Museum': [(0, 0, 0)], 'UrbanTrip': [(0.60, 0.85, 1.0)], 'Zoo': [(0, 0, 0)], 'BeachTrip': [(0.5, 0.60, 1.0)], 'PersonalMusicActivity': [(0.55, 0.80, 1.0)], 'Christmas': [(0.5, 0.60, 1.0)], 'PersonalArtActivity': [(0, 0, 0)], 'GroupActivity': [(0.75, 0.85, 1.0)], 'Wedding': [(0.60, 0.65, 1.0)], 'ReligiousActivity': [(0, 0, 0)], 'Graduation': [(0, 0, 0)], 'CasualFamilyGather': [(0.70, 0.90, 1.0)], 'Architecture': [(0, 0, 0)], 'ThemePark': [(0.30, 0.35, 1.0)], 'Sports': [(0.40, 0.60, 1.0)], 'Show': [(0, 0, 0)], 'Halloween': [(0.40, 0.5, 1.0)], 'BusinessActivity': [(0.5, 0.55, 1.0)], 'Protest': [(0.45, 0.95, 1.0)], 'Cruise': [(0, 0, 0)], 'Birthday': [(0.30, 0.35, 1.0)], 'NatureTrip': [(0.0, 0.5, 1.0)]}
        #dict_from_validation= {'PersonalSports': [0.0], 'Museum': [0.5], 'UrbanTrip': [0.75], 'Zoo': [0.70], 'BeachTrip': [0.30], 'PersonalMusicActivity': [0.60], 'Christmas': [0.25], 'PersonalArtActivity': [0.0], 'GroupActivity': [0.15], 'Wedding': [0.0], 'ReligiousActivity': [0.0], 'Graduation': [0.10], 'CasualFamilyGather': [0.20], 'Architecture': [0.0], 'ThemePark': [0.20], 'Sports': [0.30], 'Show': [0.10], 'Halloween': [0.050], 'BusinessActivity': [0.050], 'Protest': [0.90], 'Cruise': [0.35], 'Birthday': [0.20], 'NatureTrip': [0.95]}
        self.validation_name = 'test'
        self.evaluate_present_face(dict_from_validation)
        '''
    def evaluate_face(self, face_type):
        # dict_from_validation = {'PersonalSports': [(0.45000000000000001, 0.55000000000000004, 1.0)],
        #                         'Museum': [(0.35000000000000003, 0.65000000000000002, 1.0)],
        #                         'UrbanTrip': [(0,0,0)],
        #                         'Zoo': [(0.40000000000000002, 0.65000000000000002, 1.0)],
        #                         'BeachTrip': [(0.5, 0.80000000000000004, 1.0)],
        #                         'PersonalMusicActivity': [(0.45000000000000001, 0.65000000000000002, 1.0)],
        #                         'Christmas': [(0,0,0)],
        #                         'PersonalArtActivity': [(0,0,0)],
        #                         'GroupActivity': [(0.80000000000000004, 0.95000000000000007, 1.0)],
        #                         'Wedding': [(0.30000000000000004, 0.45000000000000001, 1.0)],
        #                         'ReligiousActivity': [(0.45000000000000001, 0.60000000000000009, 1.0)],
        #                         'Graduation': [(0.65000000000000002, 0.75, 1.0)],
        #                         'CasualFamilyGather': [(0.20000000000000001, 0.35000000000000003, 1.0)],
        #                         'Architecture': [(0,0,0)],
        #                         'ThemePark': [(0.40000000000000002, 0.65000000000000002, 1.0)],
        #                         'Sports':[(0,0,0)],'Show':[(0.55000000000000004, 0.65000000000000002, 1.0)],
        #                         'Halloween':[(0.55000000000000004, 0.65000000000000002, 1.0)],
        #                         'BusinessActivity':[(0,0,0)],'Protest':[(0.55000000000000004, 0.60000000000000009, 1.0)],
        #                         'Cruise': [(0,0,0)],'Birthday': [(0,0,0)], 'NatureTrip': [(0,0,0)]}
        # dict_from_validation = {'PersonalSports': [(1, 1, 1)], 'Museum': [(1, 1, 1)], 'UrbanTrip': [(0.30000000000000004, 0.80000000000000004, 1.0)],
        #                     'Zoo': [(1, 1, 1)], 'BeachTrip': [(1, 1, 1)], 'PersonalMusicActivity': [(1, 1, 1)], 'Christmas': [(0.050000000000000003, 0.70000000000000007, 1.0)],
        #                     'PersonalArtActivity': [(0.10000000000000001, 0.45000000000000001, 1.0)], 'GroupActivity': [(0.15000000000000002, 0.25, 1.0)],
        #                     'Wedding': [(0.65000000000000002, 0.95000000000000007, 1.0)], 'ReligiousActivity': [(1, 1, 1)], 'Graduation': [(0.45000000000000001, 0.5, 1.0)],
        #                     'CasualFamilyGather': [(0.0, 0.10000000000000001, 1.0)], 'Architecture': [(1, 1, 1)], 'ThemePark': [(0.55000000000000004, 0.70000000000000007, 1.0)],
        #                     'Sports': [(0.90000000000000002, 0.95000000000000007, 1.0)], 'Show': [(0.65000000000000002, 0.80000000000000004, 1.0)], 'Halloween': [(1, 1, 1)],
        #                     'BusinessActivity': [(1.0, 1.0, 1.0)], 'Protest': [(0.5, 0.65000000000000002, 1.0)], 'Cruise': [(1, 1, 1)],
        #                     'Birthday': [(0.75, 0.80000000000000004, 1.0)], 'NatureTrip': [(1, 1, 1)]
        #                     }
        #dict_from_validation = {'Wedding':[(0.20000000000000004, 0.25, 1.0)]}
        dict_from_validation = {}
        combine_name = face_type
        f = open(root + 'codes/face_combine_features_'+combine_name+noqual_name+'_fromnoevent'+itera+'.cPickle','r')
        dict_score = cPickle.load(f)
        f.close()
        for event_name in dict_name2:
            temp = {}
            temp1 = {}
            this_dict = dict_score[event_name]
            no_score = []
            for i in this_dict:
                for j in i:
                    if j[0] == (0,0,0):
                        no_score.append(j[1])
                    if j[0] in temp:
                        temp[j[0]].append(j[1])
                        temp1[j[0]] += j[1]
                    else:
                        temp[j[0]] = [j[1]]
                        temp1[j[0]] = j[1]
            sorted_temp = sorted(temp1.items(), key = operator.itemgetter(1), reverse=True)
            sorted_temp1 = []
            for ii in sorted_temp:
                # if np.sum([(i-j) > 0 for i,j in zip(temp[ii[0]], no_score)]) >= 4:
                    sorted_temp1.append((ii[0], temp[ii[0]]))
            # print sorted_temp
            # print sorted_temp1
            # print no_score
            if len(sorted_temp1) == 0:
                print event_name, 0.0
                continue
            print event_name, np.mean(sorted_temp1[0][1]) - np.mean(no_score), np.sum([(i-j) >= 0 for i,j in zip(sorted_temp1[0][1], no_score)])
            if (len(sorted_temp1[0][0]) == 3 and sorted_temp1[0][0][2] == 0) or (len(sorted_temp1[0][0]) == 1 and sorted_temp1[0][0][0] == 0):
                dict_from_validation[event_name]  = [(0,0,0)]
                continue
            if np.mean(sorted_temp1[0][1]) - np.mean(no_score) < 0.01:
                dict_from_validation[event_name]  = [(0,0,0)]
                continue
            dict_from_validation[event_name] = [sorted_temp1[0][0]]#, sorted_temp[0][1] - temp[0,0,0]]

            # else:
            #     dict_from_validation[event_name] = [0]
        print dict_from_validation
        # dict_from_validation['Wedding'] = [(0,1,dict_from_validation['Wedding'][0][2])]
        if len(dict_from_validation[dict_from_validation.keys()[0]][0]) == 1:
            print 'Using theta!'
            self.evaluate_present_face(dict_from_validation, 'theta',combine_name+itera)
        else:
            self.evaluate_present_face(dict_from_validation, 'alpha',combine_name+itera)
        # self.evaluate_present_face(dict_from_validation, 'theta')
    def combine_cues(self, event_type, names, event_index, save_name, validation_name):
        path = root+baseline_name+event_type+'/'+validation_name+'_image_ids.cPickle'
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()
        #n_combine = len(names)
        predict_score = []
        for name in names:
            f = open(root + name, 'r')
            predict_score.append(np.array(cPickle.load(f)))
            f.close()

        f = open(root +baseline_name+event_type+'/'+validation_name+'_ulr_dict.cPickle', 'r')
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

        f = open(root + 'CNN_all_event_1009/features/'+event_type +save_name +'_dict.cPickle','wb')
        cPickle.dump(prediction_dict, f)
        f.close()
        f = open(root + 'CNN_all_event_1009/features/'+event_type+save_name + '.cPickle','wb')
        cPickle.dump(np.mean(predict_score,axis=0), f)
        f.close()
    def combine_cues_ranking_euclidean(self, event_type, names, event_index, save_name, validation_name):
        path = root+baseline_name+event_type+'/'+validation_name+'_image_ids.cPickle'
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()
        #n_combine = len(names)
        predict_score = []
        f = open(root + names[0], 'r')
        predict_score.append(np.array(cPickle.load(f)))
        f.close()
        f = open(root + names[1], 'r')
        temp = np.array(cPickle.load(f))
        temp1 = temp / 80
        # temp1 = 1.0/(1.0+np.exp(0-temp))
        predict_score.append(temp1)
        f.close()

        f = open(root +baseline_name+event_type+'/'+validation_name+'_ulr_dict.cPickle', 'r')
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

        f = open(root + 'CNN_all_event_1009/features/'+event_type +save_name +'_dict.cPickle','wb')
        cPickle.dump(prediction_dict, f)
        f.close()
        f = open(root + 'CNN_all_event_1009/features/'+event_type+save_name + '.cPickle','wb')
        cPickle.dump(np.mean(predict_score,axis=0), f)
        f.close()
    def evaluate_present_combine(self, validation_name):
        # model_names_to_combine = [
                        # ['CNN_all_event_1009/validation_0/features/', '_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle']
                        # ,['CNN_all_event_1009/validation_1/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle']
                        # ,['CNN_all_event_1009/validation_2/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle']
                        # ,['CNN_all_event_1009/validation_3/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle']
                        # ,['CNN_all_event_1009/validation_4/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_70000.cPickle']
                        # ['CNN_all_event_1009/validation_0/features/', '_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                        # ,['CNN_all_event_1009/validation_1/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                        # ,['CNN_all_event_1009/validation_2/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                        # ,['CNN_all_event_1009/validation_3/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                        # ,['CNN_all_event_1009/validation_4/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                        # ,['CNN_all_event_1009/validation_4/features/','_'+validation_name+'_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle']
                      # ]
        model_names_to_combine = [
                        ['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_fromnoevent_iter_100000.cPickle']
                        ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000.cPickle']
        #                 ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_fromnoevent_3_iter_100000.cPickle']
                       ]
        combine_name = '_combined_10_ranking_euclidean'
        model_names = [['CNN_all_event_1009/features/','_' + validation_name+combine_name + '_dict.cPickle']]
        permutation_times = global_permutation_time
        worker_permutation_times = 1
        retrieval_models = []
        precision_models = []
        for i in model_names:
            retrieval_models.append([])
            precision_models.append([])
        retrieval_worker_all = []
        len_all = 0
        for event_name in dict_name2:
            self.combine_cues_ranking_euclidean(event_name,[event_name.join(i) for i in model_names_to_combine], dict_name2[event_name], '_' + validation_name+combine_name, validation_name)
            f = open(root + baseline_name + event_name + '/' + validation_name+'_event_id.cPickle','r')
            temp = cPickle.load(f)
            f.close()
            len_ = len(temp)
            len_all += len_
            percent_all = []; retrievals_model = []; precision_model = []; retrievals_worker = []; precision_worker = []
            for i in xrange(permutation_times):
                    if i %10 == 0:
                        print i

                    reweighted, percent, retrievals , precision, mean_aps, mean_ps = self.baseline_evaluation(event_name, True, model_names, worker_times=worker_permutation_times)
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
                precision_models[i].append([j*len_ for j in precision_model_average[i]])
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

        for i in xrange(len(precision_models)):
            print model_names[i]
            #print retrieval_models[i]
            temp = np.array(precision_models[i])
            temp1 = np.sum(temp, axis=0)
        #print temp1
            print [j/len_all for j in temp1]

        print 'Worker'
        #print retrieval_worker_all
        temp = np.array(retrieval_worker_all)
        temp1 = np.sum(temp, axis=0)
        print [i/len_all for i in temp1]


def evaluation_event_recognition(event_name):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + '_test_predict_event_recognition_iter_100000.cPickle','r')
    prediction = cPickle.load(f)
    f.close()

    f = open('/raid/yufeihome/event_curation/baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle' ,'r')
    ground_truth = cPickle.load(f)
    f.close()

    path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()

    ground_pred_comb = []
    for predict_score, img_id in zip(prediction, all_event_ids):
        event_id = img_id.split('/')[0]
        ground_this_event = ground_truth[event_id]
        for i in ground_this_event:
            if i[1] == img_id:
                ground_this = i[2]
                ground_pred_comb.append((ground_this, predict_score))

    prediction_label = [np.argmax(i) for i in prediction]

    prediction_dict = Counter(prediction_label)

    temp = [i == dict_name2[event_name] - 1 for i in prediction_label]
    # print temp
    accuracy_this = float(np.sum(temp)) / len(temp)
    if prediction_dict.most_common(2)[0][0] == dict_name2[event_name] - 1:
        print event_name, accuracy_this, prediction_dict.most_common(2)[-1][0]
    else:
        most_recognized = prediction_dict.most_common(2)[0][0]
        print event_name, accuracy_this,most_recognized,  float(np.sum([i == most_recognized for i in prediction_label])) / len(temp)
    important_threshold = 0.6
    selected_index = []
    for i in xrange(len(ground_pred_comb)):
        if ground_pred_comb[i][0] > important_threshold:
            selected_index.append(i)
    temp_2 = [temp[i] for i in selected_index]
    accuracy_this_new = float(np.sum(temp_2)) / len(temp_2)
    print 'Thre:', event_name, accuracy_this_new
    return accuracy_this_new, len(temp)

def evaluation_event_recognition_peralbum(event_name, threshold):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + '_test_predict_event_recognition_iter_100000.cPickle','r')
    prediction = cPickle.load(f)
    f.close()

    f = open('/raid/yufeihome/event_curation/baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle' ,'r')
    ground_truth = cPickle.load(f)
    f.close()

    path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()

    ground_pred_comb_dict = defaultdict(list)
    for predict_score, img_id in zip(prediction, all_event_ids):
        event_id = img_id.split('/')[0]
        ground_this_event = ground_truth[event_id]
        # ground_pred_comb_dict[event_id] = []
        for i in ground_this_event:
            if i[1] == img_id:
                importance_this = i[2]
                ground_pred_comb_dict[event_id].append([img_id, np.argmax(predict_score), importance_this])

    prediction_this_type = []
    for album_name in ground_pred_comb_dict:
        prediction_thisevent = np.zeros((23,))
        for i in ground_pred_comb_dict[album_name]:
            importance_score = i[2]
            # print prediction_thisevent, importance_score
            importance_score = (importance_score + 1) / 4
            if importance_score < threshold:
                # print importance_score
                continue
            prediction_thisevent[i[1]] += importance_score
        prediction_this_type.append(np.argmax(prediction_thisevent))

    temp = [i == dict_name2[event_name] - 1 for i in prediction_this_type]
    accuracy_this = float(np.sum(temp)) / len(temp)

    # print event_name, accuracy_this
    return accuracy_this, len(temp)

def evaluation_event_recognition_peralbum_weighted(event_name, threshold,net_name, oversample = True):
    if oversample:
        f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')
    else:
        f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + '_test_predict_event_recognition_iter_100000.cPickle','r')

    prediction = cPickle.load(f)
    f.close()

    # f = open('/raid/yufeihome/event_curation/baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle' ,'r')
    f = open('/raid/yufeihome/event_curation/CNN_all_event_1009/features/' + event_name + '_test_sigmoid9_10_segment_twoloss_fc300_diffweight_iter_100000_dict.cPickle' ,'r')
    ground_truth = cPickle.load(f)
    f.close()

    path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()

    ground_pred_comb_dict = defaultdict(list)
    for predict_score, img_id in zip(prediction, all_event_ids):
        event_id = img_id.split('/')[0]
        ground_this_event = ground_truth[event_id]
        # ground_pred_comb_dict[event_id] = []
        for i in ground_this_event:
            if i[0] == img_id:
                importance_this = i[2]
                ground_pred_comb_dict[event_id].append([img_id, predict_score, importance_this])
        # print ground_this_event

    prediction_this_type = []
    album_names = []
    for album_name in ground_pred_comb_dict:
        prediction_thisevent = np.zeros((23,))
        for i in ground_pred_comb_dict[album_name]:
            importance_score = i[2]
            prediction_ = np.array(i[1])
            # print prediction_thisevent, importance_score
            # importance_score = (importance_score + 1) / 4
            if importance_score < threshold:
                # print importance_score
                continue
            prediction_thisevent += importance_score * prediction_
        prediction_this_type.append(np.argmax(prediction_thisevent))
        album_names.append(album_name)

    temp = [i == dict_name2[event_name] - 1 for i in prediction_this_type]
    accuracy_this = float(np.sum(temp)) / len(temp)

    dict_name2_reverse = dict([(dict_name2[key] - 1, key) for key in dict_name2])
    predicted_label = [0]*23
    for i_this in xrange(23):
        predicted_label[i_this] = sum([i == i_this for i in prediction_this_type])
    str_to_print = ''
    for i in xrange(len(predicted_label)):
        if predicted_label[i] > 0 and i != dict_name2[event_name] - 1:
            albums_this = [album_names[ii] for ii in xrange(len(prediction_this_type)) if prediction_this_type[ii] == i]
            str_to_print += dict_name2_reverse[i]+':'+str(predicted_label[i])+ ' ' + str(albums_this) +', '
        if i == dict_name2[event_name] - 1:
            str_to_print += dict_name2_reverse[i]+':'+str(predicted_label[i])+', '
    print str_to_print
    print event_name, accuracy_this
    return accuracy_this, len(temp)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def evaluation_event_recognition_peralbum_noimportance(event_name, threshold,net_name, oversample = True):
    if oversample:
        f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')
    else:
        f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + '_test_predict_event_recognition_iter_100000.cPickle','r')

    prediction = cPickle.load(f)
    f.close()

    f = open('/raid/yufeihome/event_curation/baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle' ,'r')
    ground_truth = cPickle.load(f)
    f.close()

    path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()

    ground_pred_comb_dict = defaultdict(list)
    for predict_score, img_id in zip(prediction, all_event_ids):
        event_id = img_id.split('/')[0]
        ground_this_event = ground_truth[event_id]
        # ground_pred_comb_dict[event_id] = []
        for i in ground_this_event:
            if i[1] == img_id:
                importance_this = i[2]
                ground_pred_comb_dict[event_id].append([img_id, predict_score, importance_this])

    prediction_this_type = []
    album_names = []
    for album_name in ground_pred_comb_dict:
        prediction_thisevent = np.zeros((23,))
        for i in ground_pred_comb_dict[album_name]:
            # importance_score = i[2]
            prediction_ = np.array(i[1])
            # print prediction_thisevent, importance_score
            # importance_score = (importance_score + 1) / 4
            # if importance_score < threshold:
                # print importance_score
                # continue
            prediction_thisevent +=  prediction_
        prediction_this_type.append(np.argmax(prediction_thisevent))
        album_names.append(album_name)

    temp = [i == dict_name2[event_name] - 1 for i in prediction_this_type]
    accuracy_this = float(np.sum(temp)) / len(temp)

    dict_name2_reverse = dict([(dict_name2[key] - 1, key) for key in dict_name2])
    predicted_label = [0]*23
    for i_this in xrange(23):
        predicted_label[i_this] = sum([i == i_this for i in prediction_this_type])
    str_to_print = ''
    for i in xrange(len(predicted_label)):
        if predicted_label[i] > 0 and i != dict_name2[event_name] - 1:
            albums_this = [album_names[ii] for ii in xrange(len(prediction_this_type)) if prediction_this_type[ii] == i]
            str_to_print += dict_name2_reverse[i]+':'+str(predicted_label[i])+ ' ' + str(albums_this) +', '
        if i == dict_name2[event_name] - 1:
            str_to_print += dict_name2_reverse[i]+':'+str(predicted_label[i])+', '
    print str_to_print
    print event_name, accuracy_this
    temp = np.zeros((23,), dtype=float)
    for i in prediction_this_type:
        temp[i] += 1
    return accuracy_this, len(temp), temp / len(temp)

def extract_features_all(img_file_list, model_name, weight_name,blob_name,mean_file,out_file_name, img_dim = 227):
        imgs = []
        with open(img_file_list, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        caffe.set_mode_gpu()
        net = caffe.Net(model_name,
                    weight_name,
                    caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 2)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        net.blobs['data'].reshape(1,3,img_dim, img_dim)
        features = []
        count = 0
        for img in imgs:
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            print np.max(net.blobs['data'].data)
            out = net.forward()
            a = net.blobs[blob_name].data.copy()
            features.append(a)
            print img, count
            count += 1
            f = open(out_file_name,'wb')
            cPickle.dump(features, f)
            f.close()

def evaluate_top5image(event_name, model_names, img_n = 1, permuted = ''):
        retval = []
        f = open(root + baseline_name + event_name+ '/vgg_test_result_v2'+permuted+'.cPickle','r')
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
                threshold = ground_[img_n-1][1]
                n_k = len([i for i in ground_ if i[1] >= threshold])
                dict_wanted = set()
                for i in xrange(n_k):
                    dict_wanted.add(ground_[i][0])
                retrieved_count = 0
                for j in xrange(n_k):
                    if predict_[j][0] in dict_wanted:
                        retrieved_count += 1
                count_all += retrieved_count
                n_k_all += img_n
            retval.append([count_all , n_k_all])
            # retval.append(count_all)
        return retval

def amt_worker_result_predict_average(event_name, img_n = 5, permuted = ''):

        validation_path = '/'
        f = open(root + baseline_name + event_name+ validation_path +'/vgg_test_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()

        input_path = root + baseline_name + event_name+ validation_path + '/test_image_ids.cPickle'
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
        input_path = root + baseline_name + event_name+ validation_path + '/test.csv'
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
                        check_submission.append(score+random.uniform(-0.02, 0.02))
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
                # new_predict_rank = {}
                # for rank in rank_set:
                #     this_rank = [i for i in predict_rank if predict_rank[i] == rank]
                #     random.shuffle(this_rank)
                #     for i in xrange(len(this_rank)):
                #         new_predict_rank[this_rank[i]] = rank + i
                threshold = ground_[img_n-1][1]
                n_k = len([i for i in ground_ if i[1] >= threshold])
                n_ks.append([n_k, len(ground_)])
                #Precision3: mean precision
                n_k = len([i for i in ground_ if i[1] >= threshold])
                dict_wanted = set()
                for i in xrange(n_k):
                    dict_wanted.add(ground_[i][0])
                retrieved_count = 0
                for j in xrange(n_k):
                    if predict_[j][0] in dict_wanted:
                        retrieved_count += 1
                Ps.append([retrieved_count,img_n])
                all_ps.append(Ps)
            all_n_ks.append(n_ks)

        return all_n_ks, all_ps
def evaluate_top5image_worker(event_name, img_n = 5, permuted = '', worker_times = 50):
            all_ps = []
            for i in xrange(worker_times):
                all_nks, temp4 = amt_worker_result_predict_average(event_name, permuted='_permuted')
                all_ps.append([temp4])
            # print all_ps
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

            all_ps = all_ps_average[0]
            mean_ps1 = np.zeros(1);mean_ps2 = np.zeros(1)
            for i in xrange(len(all_ps)):
                for j in xrange(len(all_ps[i])):
                    mean_ps1[j] += all_ps[i][j][0]
                    mean_ps2[j] += all_ps[i][j][1]
            mean_ps = [mean_ps1[i]/mean_ps2[i] for i in xrange(len(mean_ps1))]
            return mean_ps1[0], mean_ps2[0]


def first5_percent(model_names = [['CNN_all_event_1009/features/','_test_combined_10_fromnoevent_2_10w_dict.cPickle'],
                                  ['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_noevent_iter_100000_dict.cPickle'],
                                  ['baseline_all_noblock/', '/test_random_dict.cPickle']]):
    retrieved_difference_all = []
    want_all = []
    for event_name in dict_name2:
        model_names_this = [i[0] + event_name + i[1] for i in model_names]
        retval = evaluate_top5image(event_name, model_names_this,img_n = 5)
        print event_name, retval
        retrieved_difference_all.append([retval[0][0],retval[1][0],retval[2][0]])
        want_all.append(retval[0][1])
    print np.sum([i[0] for i in retrieved_difference_all])
    print np.sum([i[1] for i in retrieved_difference_all])
    print np.sum([i[2] for i in retrieved_difference_all])
    print np.sum(want_all)

    retrieved_all = []
    want_all = []
    for event_name in dict_name2:
        temp = evaluate_top5image_worker(event_name)
        print temp
        retrieved_all.append(temp[0])
        want_all.append(temp[1])
        print event_name, temp
    print np.sum(retrieved_all) / np.sum(want_all)

if __name__ == '__main__':
    # b = create_CNN_training_prototxts(threshold = 0.0, val_name = 'training', folder_name='CNN_all_event_subsample0.3/', subsample = True, rate = 0.3, oversample = False)
    # b = create_CNN_training_prototxts(threshold = 0.0, val_name = 'training', folder_name='CNN_all_event_oversample0.3_3/', oversample = True)
    # b = create_CNN_training_prototxts_face(threshold = 0.0, val_name = 'training', folder_name='face_heatmap/')
    # c = create_CNN_training_prototxts_face(threshold = 0.0, val_name ="test", folder_name='face_heatmap/')
    # a = create_cross_validation()
    # a=create_CNN_training_prototxts(threshold = 0.0, val_name ="training", folder_name='CNN_all_event_new/')
    # a=create_CNN_training_prototxts(threshold = 0.0, val_name ="test", folder_name='CNN_all_event_new/')
    # b = create_CNN_training_prototxts(threshold = 0, val_name = 'val_training', folder_name='CNN_all_event_old2/')
    # c = create_CNN_training_prototxts(threshold = 0, val_name ="val_validation", folder_name='CNN_all_event_old2/')
    # c = create_CNN_training_prototxts(threshold = 0, val_name ="training", folder_name='CNN_all_event_subcategory2/')
    # c = create_CNN_training_prototxts(threshold = 0, val_name ="test", folder_name='CNN_all_event_subcategory2/')
    # a = create_cross_validation()
    # for event_name in dict_name2:
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_superevent_frombegin_iter_100000','python_deploy_siamese_superclass.prototxt','test', None, (256, 256))
       # a.extract_feature_10_superevent_traintest()

    # for event_name in dict_name2:
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_euclidean_sigmoid_iter_10000','python_deploy_euclidean.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    # for event_name in dict_name2:
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_euclidean_nosigmoid_2_iter_10000','python_deploy_euclidean.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       #  a = extract_features('CNN_all_event_1009', event_name, 'segment_joint_fc_iter_200000','python_deploy_joint_fc.prototxt','test', None, (256, 256))
       #  a.extract_feature_10_traintest_vote()
       #  a = extract_features('CNN_all_event_1009', event_name, 'segment_joint_fc_hidden1q_8q_iter_200000','python_deploy_joint_fc1q.prototxt','test', None, (256, 256))
       #  a.extract_feature_10_traintest_vote()
       #  a = extract_features('CNN_all_event_1009', event_name, 'segment_joint_fc_iter_200000','python_deploy_joint_fc.prototxt','test', None, (256, 256))
       #  a.extract_feature_10_traintest_vote_sample()


    # for event_name in dict_name2:
    #    a = extract_features('CNN_all_event_vgg', event_name, 'VGG_svm_0.1_iter_40000','VGG_deploy_segment.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_google', event_name, 'google_svm_0.1_iter_60000','python_deploy_googlenet_svm.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()    #
    #    if os.path.exists(root + 'CNN_all_event_1009/features/'+event_name+'_testperevent_sigmoid9_10_segment_joint_iter_400000.cPickle'):
    #        print root + 'CNN_all_event_1009/features/'+event_name+'_testperevent_sigmoid9_10_segment_joint_iter_400000.cPickle'
    #        continue
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_joint_iter_400000','python_deploy_joint.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest_vote_new()
    #    a = extract_features('CNN_all_event_google', event_name, 'segment_googlenet_quick_polylarge_iter_400000','python_deploy_googlenet_segment_1loss.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_google', event_name, 'segment_googlenet_iter_69000','python_deploy_googlenet_segment_1loss.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #    if os.path.exists(root + 'CNN_all_event_google/features/'+event_name+'_test_sigmoid9_10_segment_googlenet_largelr_iter_10000.cPickle'):
    #        print event_name
    #        continue
    #    if not os.path.exists(root + 'CNN_all_event_google/features/'+event_name+'_test_sigmoid9_10_segment_00501_iter_10000.cPickle'):
    #        a = extract_features('CNN_all_event_google', event_name, 'segment_00501_iter_10000','python_deploy_googlenet_segment_1loss.prototxt','test', None, (256, 256))
    #        a.extract_feature_10_traintest()
    #    if os.path.exists(root + 'CNN_all_event_google/features/'+event_name+'_test_sigmoid9_10_segment_00501_iter_60000.cPickle'):
    #        print event_name
    #        continue
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_subsample0.8_iter_100000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()

    #    a = extract_features('CNN_all_event_google', event_name, 'segment_googlenet_0.2_iter_50000','python_deploy_googlenet_segment_1loss.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()

    #    a = extract_features('CNN_all_event_google', event_name, 'segment_googlenet_l1_iter_30000','python_deploy_googlenet_segment_1loss.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    # for event_name in ['Sports']:
    #    if os.path.exists(root + 'CNN_all_event_google/features/'+event_name+'_test_sigmoid9_10_fromnoevent_googlenet_00501_cont_iter_20000.cPickle'):
    #        print event_name
    #        continue
    #    a = extract_features('CNN_all_event_google', event_name, 'fromnoevent_googlenet_00501_cont_iter_40000','python_deploy_googlenet_fromnoevent.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #   a = extract_features('CNN_all_event_google', event_name, 'google_noevent_0.5_iter_100000','python_deploy_googlenet_segment_noevent.prototxt','test', None, (256, 256))
    #   a.extract_feature_10_traintest()
    #
    #  for event_name in ['NatureTrip']:

       # if os.path.exists(root + 'CNN_all_event_google/features/'+event_name+'_test_sigmoid9_10_fromnoevent_googlenet_00501_iter_40000.cPickle'):
       #     print event_name
       #     continue
       # a = extract_features('CNN_all_event_google', event_name, 'fromnoevent_googlenet_00501_iter_40000','python_deploy_googlenet_fromnoevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()    #
    #
    #
    #
        # a = extract_features('CNN_all_event_google', event_name, 'google_svm_0.1_iter_100000','python_deploy_googlenet_svm.prototxt','test', None, (256, 256))
        # a.extract_feature_10_traintest()

    # for event_name in dict_name2:
    #     a = extract_features('CNN_all_event_1205', event_name, 'event_recognition_oversample_iter_80000','deploy.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_recognition_traintest()
    #    a = extract_features('CNN_all_event_google', event_name, 'google_noevent_iter_70000','python_deploy_googlenet_segment_noevent.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()

       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_20000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_30000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_40000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_60000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_70000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_80000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_poly_iter_90000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_10000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_20000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_30000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_40000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_50000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_60000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_70000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_80000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_90000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_smallpoly_10w_iter_100000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_new_iter_30000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_new_iter_50000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_new_iter_80000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_new_iter_100000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    # for event_name in dict_name2:
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_5time_iter_200000','python_deploy_siamese_new.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_5time_iter_120000','python_deploy_siamese_new.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_traintest()



    #
    #
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_superevent_2_iter_100000','python_deploy_siamese_superclass.prototxt','test', None, (256, 256))
    #    a.extract_feature_10_superevent_traintest()

    # for event_name in dict_name2:
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_2_iter_70000','python_deploy_siamese_old.prototxt','val_validation', None, (256, 256))
       # a.extract_feature_10_superevent_traintest()
    #    a.extract_feature_10(0)
    #    a.extract_feature_10(1)
    #    a.extract_feature_10(2)
    #    a.extract_feature_10(3)
    #    a.extract_feature_10(4)
    #     a = extract_features('CNN_all_event_old', event_name, 'segment_rmsprop_iter_40000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_8w_iter_200000','python_deploy_siamese_noevent.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_5time_iter_170000','python_deploy_siamese_new.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc100_8w_iter_150000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc100_8w_iter_120000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc200_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
       # a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc300_8w_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
       # a.extract_feature_10_traintest()
    #    b = extract_features('CNN_all_event_1009', event_name, 'segment_new_10w_gamma0.1_iter_140000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    b.extract_feature_10_traintest()
    #    b = extract_features('CNN_all_event_1009', event_name, 'segment_new_10w_gamma0.1_iter_160000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    b.extract_feature_10_traintest()
    #    b = extract_features('CNN_all_event_1009', event_name, 'segment_new_10w_gamma0.1_iter_180000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    b.extract_feature_10_traintest()
    #    b = extract_features('CNN_all_event_1009', event_name, 'segment_new_10w_gamma0.1_iter_200000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    b.extract_feature_10_traintest()


       # a.extract_feature_10(0)
       # a.extract_feature_10(1)
       # a.extract_feature_10(2)
       # a.extract_feature_10(3)
       # a.extract_feature_10(4)


        # a = extract_features('CNN_all_event_old', event_name, 'segment_3time_iter_50000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
        # a.extract_feature_10_traintest()
        # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_iter_200000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
        # a.extract_feature_10_traintest()
    # a.extract_feature_10(2)
    #    a.extract_feature_10(2)
    #    a = extract_features('CNN_all_event_1009', event_name, 'segment_iter_100000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #    a.extract_feature_10(4)
    #    a.extract_feature_10(2)
    #
    #     a = extract_features('CNN_all_event_1009',event_name, 'segment_iter_100000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
    #     a.extract_feature_10(3)
    #     a = extract_features('CNN_all_event_1009',event_name, 'segment_iter_100000','python_deploy_siamese_old.prototxt','val_validation', None, (256, 256))
    #     a.extract_feature_10(3)
    #
    #
    #     a = extract_features(event_name, 'segment_2_iter_80000','python_deploy_siamese_traintest.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_traintest()
    #
    # for event_name in dict_name2:
    #     a = extract_features('CNN_all_event_0.1', event_name, 'svm_0.2_iter_100000','python_deploy_siamese_svm.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_traintest()
    # for event_name in dict_name2:
    #     b = extract_features('face_heatmap', event_name, 'cropped_importance_allevent_iter_60000', 'lenet_deploy_allevent.prototxt', 'val_validation', ['sigmoid'],50)
    #     b.extract_feature_face(0)
    #     b.extract_feature_face(1)
    #     b.extract_feature_face(2)
    #     b.extract_feature_face(3)
    #     b.extract_feature_face(4)
    #     b = extract_features('face_heatmap', event_name, 'cropped_importance_allevent_iter_60000', 'lenet_deploy_allevent.prototxt', 'test', ['sigmoid'],50)
    #     b.extract_feature_face_traintest()

    # accuracy = 0;len_all = 0
    # for event_name in dict_name2:
    #     a,b = evaluation_event_recognition(event_name)
    #     accuracy += a * b
    #     len_all += b
    # print 'Overall accuracy: ', float(accuracy) / len_all


    #    a = extract_features('CNN_all_event_1009', event_name, 'from_validation_noevent_iter_100000','python_deploy_siamese_fromnoevent.prototxt','val_validation', None, (256, 256))
    #    a.extract_feature_10(0)
    #
    #     b = extract_features('face_heatmap', event_name, 'cropped_importance_allevent_iter_100000', 'lenet_deploy_allevent.prototxt', 'val_validation', ['sigmoid'],50)
    #     b.extract_feature_face(4)
    #     b.extract_feature_face(2)
    #     b = extract_features(event_name, 'cropped_importance_allevent_iter_100000', 'lenet_deploy_allevent.prototxt', 'test', ['sigmoid'],50)
    #     b.extract_feature_face(0)
    #
    # a = evaluation('bb','worker', 'test', 100)
    # a = evaluation('CNN_all_event_1009','worker', 'test', 0)
    # a = evaluation('CNN_all_event_1009','worker', 'test', 1)
    # a = evaluation('CNN_all_event_1009','worker', 'test', 2)
    # a = evaluation('CNN_all_event_1009','worker', 'test', 3)
    # a = evaluation('CNN_all_event_1009','worker', 'test', 4)
    # a = evaluation('CNN_all_event_1009','worker', 'val_validation', 1)
    # a = evaluation('CNN_all_event_1009','worker', 'val_validation', 3)
    # a = evaluation('CNN_all_event','worker', 'test', 3)
    TIE = True
    # global global_permutation_time
    # if TIE:
    #     global_permutation_time = 1
    # else:
    #     global_permutation_time = 50
    # a = evaluation('CNN_all_event_1009','worker', 'test',0)
    # a = evaluation('CNN_all_event_1009','worker', 'test',2)
    # a = evaluation('CNN_all_event_1009','combine','test',50)
    # a = evaluation('CNN_all_event_1009','worker', 'test',1)
    # a = evaluation('CNN_all_event_old', 'evaluate_face', 'test', 0)


    # itera = '_7w'
    itera = '_10w'
    # combine_face_model = '_combined_10_fromnoevent_2_7w.cPickle'
    combine_face_model = '_combined_10_fromnoevent_2_10w.cPickle'
    # combine_face_model = '_sigmoid9_10_segment_fromnoevent_iter_100000.cPickle'
    noqual_name = '_noqual'
    # noqual_name = ''
    # face_model = '_sigmoidcropped_importance_allevent_quality_iter_100000.cPickle'
    # face_model = '_sigmoidcropped_importance_allevent_iter_60000.cPickle'
    # a = evaluation('CNN_all_event_1009', 'evaluate_face', 'test', 0, face_type='thetapre')
    #
    # a = evaluation('CNN_all_event_1009','worker_nonoverlap', 'test',0)
    # a = evaluation('CNN_all_event_1009','worker', 'test',0)
    # first5_percent()
    # a = evaluation('CNN_all_event_1009','combine','test',50)

    # method =  'full'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method =  'beta05theta'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method =  'thetaprepower'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method = 'alphathetapower'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method = 'thetapower'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method = 'theta'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method = 'thetapre'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method = 'alphabetatheta'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)
    # method = 'alpha'
    # a = evaluation('CNN_all_event_1009', 'face', 'val_validation', 0)


    # evaluate_worker_agreement_topk(50)
    # worker_refine()
    # a = create_cross_validation()
    # print spearmanr([2,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2], [0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2])
    # for event_name in dict_name2:
    #     extract_feature_aesthetic(event_name)

    # create_event_recognition('CNN_all_event_1205')

    # for threshold in xrange(-10,10):
    #     accuracy = 0;len_all = 0
    #     for event_name in dict_name2:
    #         # a,b = evaluation_event_recognition_peralbum(event_name, float(threshold)/10)
    #         a,b = evaluation_event_recognition_peralbum_weighted(event_name, float(threshold)/200,'_test_predict_event_recognition_iter_100000', oversample = False)
    #         accuracy += a * b
    #         len_all += b
    #     print 'Threshold:',threshold, 'Overall accuracy: ', float(accuracy) / len_all

    #
    accuracy = 0;len_all = 0;threshold = 0.0
    confusion_matrix = np.zeros((23,23))
    for event_name in dict_name2:
        # a,b = evaluation_event_recognition_peralbum_weighted(event_name, threshold, '_test_predict_event_recognition_oversample_iter_80000')
        a,b = evaluation_event_recognition_peralbum_weighted(event_name, threshold, '_test_predict_event_recognition_iter_100000', oversample = False)
        # a,b,c= evaluation_event_recognition_peralbum_noimportance(event_name, threshold, '_test_predict_event_recognition_iter_100000', oversample = False)
        accuracy += a * b
        len_all += b
        # confusion_matrix[dict_name2[event_name] - 1,:] = c
    print 'Threshold:',threshold, 'Overall accuracy: ', float(accuracy) / len_all
    # print confusion_matrix


    # test_found = []; a = []
    # test_found_all = 0;  a_all = 0
    # for event_type in ['Birthday']:
    #     print event_type, len(test_found), len(a)
    #     f = open(event_type + '/test_event_id.cPickle','r')
    #     a = cPickle.load(f)
    #     f.close()
    #     f = open(event_type + '/training_event_id.cPickle','r')
    #     b = cPickle.load(f)
    #     f.close()
    #     c = [i.split('_')[1] for i in a]
    #     d = [i.split('_')[1] for i in b]
    #     test_found = []
    #     for i in c:
    #         a_all += 1
    #         if i in d:
    #             test_found.append(i)
    #             test_found_all+= 1
    # first5_percent()


    #
    # for event_name in dict_name2:
    #     print event_name
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc300_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.combine_event_feature_traintest('CNN_all_event_1205','test', 'event_recognition_iter_100000')

    # for event_name in dict_name2:
    #     # a = extract_features('CNN_all_event_old', event_name, 'segment_rmsprop_iter_10000','python_deploy_siamese_old2.prototxt','test', None, (256, 256))
    #     # a.extract_feature_10_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_adagrad_iter_30000','python_deploy_siamese_new.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_traintest()
    #
























