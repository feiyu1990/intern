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
global_permutation_time = 20


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

dict_name2_reverse = dict([(dict_name2[key], key) for key in dict_name2])
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


class create_cross_validation_corrected:
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
        # self.guru_find_valid_examples_all_reallabel_traintest_oversample()
        # self.guru_find_valid_examples_all_reallabel_traintest_highscore_importance()
        # self.guru_find_valid_examples_all_reallabel_traintest_highscore_importance_scaled()
        # self.guru_find_valid_examples_all_reallabel_traintest_highscore_importance_dummy()
        # self.guru_find_valid_examples_all_reallabel_traintest_highscore_importance_corrected()
        self.guru_find_valid_examples_all_reallabel_traintest_corrected()
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
        guru_find_valid_examples_all_reallabel_traintest_highscore_importance_dummy
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
    def guru_find_valid_examples_all_reallabel_traintest_highscore_importance(self, important_threshold = 0):
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
                if training_score_dict[id] <= important_threshold:
                    continue
                path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1), training_score_dict[id])
                print training_score_dict[id]
                training_img_lists.append(path)
            # for id in test_img_id:
            #     if training_score_dict[id] < important_threshold:
            #         continue
            #     path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
            #     test_img_lists.append(path)

        # random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list_importance.txt'
        # out_path2 = root + self.folder_name+'/data/test_list.txt'
        importance = []
        f1 = open(out_path1,'w')
        for i,j in training_img_lists:
            f1.write(i+'\n')
            importance.append(j)
        f1.close()
        importance = np.array(importance)
        f = h5py.File(root + self.folder_name+'/data/training_list_importance_label.h5','w')
        f.create_dataset("importance", data=importance)
        f.close()


        # f1 = open(out_path2,'w')
        # for i in test_img_lists:
        #     f1.write(i+'\n')
        # f1.close()
    def guru_find_valid_examples_all_reallabel_traintest_highscore_importance_corrected(self, important_threshold = 0):
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
                if training_score_dict[id] <= important_threshold:
                    continue
                if id.split('/')[0] in correct_list:
                    path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[correct_list[id.split('/')[0]]] - 1), training_score_dict[id])
                else:
                    path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1), training_score_dict[id])
                print training_score_dict[id]
                training_img_lists.append(path)
            # for id in test_img_id:
            #     if training_score_dict[id] < important_threshold:
            #         continue
            #     path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
            #     test_img_lists.append(path)

        # random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list_importance_corrected.txt'
        # out_path2 = root + self.folder_name+'/data/test_list.txt'
        importance = []
        f1 = open(out_path1,'w')
        for i,j in training_img_lists:
            f1.write(i+'\n')
            importance.append(j)
        f1.close()
        importance = np.array(importance)
        f = h5py.File(root + self.folder_name+'/data/training_list_importance_label_corrected.h5','w')
        f.create_dataset("importance", data=importance)
        f.close()
        f = open(root + self.folder_name+'/data/training_list_importance_label_corrected.txt','w')
        f.write(root + self.folder_name+'/data/training_list_importance_label_corrected.h5')
        f.close()

        # f1 = open(out_path2,'w')
        # for i in test_img_lists:
        #     f1.write(i+'\n')
        # f1.close()
    def guru_find_valid_examples_all_reallabel_traintest_corrected(self):
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
                if id.split('/')[0] in correct_list:
                    path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[correct_list[id.split('/')[0]]] - 1))
                else:
                    path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1))
                training_img_lists.append(path)
            for id in test_img_id:
                if id.split('/')[0] in correct_list:
                    path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[correct_list[id.split('/')[0]]] - 1))
                else:
                    path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1))
                test_img_lists.append(path)

        random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list.txt'
        f1 = open(out_path1,'w')
        for i in training_img_lists:
            f1.write(i+'\n')
        f1.close()

    def guru_find_valid_examples_all_reallabel_traintest_highscore_importance_scaled(self, important_threshold = 0):
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
                if training_score_dict[id] <= important_threshold:
                    continue
                path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1), training_score_dict[id])
                print training_score_dict[id]
                training_img_lists.append(path)
            # for id in test_img_id:
            #     if training_score_dict[id] < important_threshold:
            #         continue
            #     path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
            #     test_img_lists.append(path)

        # random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list_importance_scaled_3.txt'
        # out_path2 = root + self.folder_name+'/data/test_list.txt'
        importance = []
        f1 = open(out_path1,'w')
        for i,j in training_img_lists:
            f1.write(i+'\n')
            importance.append([float(j)**3])
        f1.close()
        importance = np.array(importance)
        print importance
        f = h5py.File(root + self.folder_name+'/data/training_list_importance_label_scaled_3.h5','w')
        f.create_dataset("importance", data=importance)
        f.close()
        f = open(root + self.folder_name+'/data/training_list_importance_label_scaled_3.txt','w')
        f.write(root + self.folder_name+'/data/training_list_importance_label_scaled_3.h5')
        f.close()

        # f1 = open(out_path2,'w')
        # for i in test_img_lists:
        #     f1.write(i+'\n')
        # f1.close()
    def guru_find_valid_examples_all_reallabel_traintest_highscore_importance_dummy(self, important_threshold = 0):
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
                if training_score_dict[id] <= important_threshold:
                    continue
                path = ('/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1), training_score_dict[id])
                # print training_score_dict[id]
                training_img_lists.append(path)
            # for id in test_img_id:
            #     if training_score_dict[id] < important_threshold:
            #         continue
            #     path = '/home/feiyu1990/local/event_curation/curation_images/' + event_name + '/' + id.split('_')[1] + '.jpg ' + str(dict_name2[event_name] - 1)
            #     test_img_lists.append(path)

        # random.shuffle(test_img_lists)
        random.shuffle(training_img_lists)
        out_path1 = root + self.folder_name+'/data/training_list_importance_dummy.txt'
        # out_path2 = root + self.folder_name+'/data/test_list.txt'
        importance = []
        f1 = open(out_path1,'w')
        for i,j in training_img_lists:
            f1.write(i+'\n')
            importance.append(float(1))
        f1.close()
        importance = np.array(importance)
        print importance
        f = h5py.File(root + self.folder_name+'/data/training_list_importance_label_dummy.h5','w')
        f.create_dataset("importance", data=importance)
        f.close()
        f = open(root + self.folder_name+'/data/training_list_importance_label_dummy.txt','w')
        f.write(root + self.folder_name+'/data/training_list_importance_label_dummy.h5')
        f.close()

        # f1 = open(out_path2,'w')
        # for i in test_img_lists:
        #     f1.write(i+'\n')
        # f1.close()


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

    def guru_find_valid_examples_all_reallabel_traintest(self, oversample=False):
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
            if count % 100 == 0:
                print self.event_name, out
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
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+self.event_name+'/guru_'+self.name+'_path.txt'
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
            if count % 100 == 0:
                print dict_name2[event_name], np.argmax(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        #for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_predict_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()
    def extract_feature_10_recognition_traintest_fc7(self):
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_0509/'+self.event_name+'/guru_'+self.name+'_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/training/' + self.model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+self.net_path+'/model/' + self.net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()

        net = caffe.Net(model_name, weight_name, caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        transformer.set_mean('data', mean_file.mean(1).mean(1))
        # print net.blobs['data']
        net.blobs['data'].reshape(1,3,self.img_size, self.img_size)
        features = []
        count = 0
        for img in imgs:
            count += 1
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            net.forward()
            a = net.blobs['fc7'].data.copy()
            if count % 100 == 0:
                # print img
                # print np.max(net.blobs['data'].data), np.min(net.blobs['data'].data)
                print a[0]
            features.append(a[0])


        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_fc7_'+ self.net_name + '.cPickle', 'wb')
        cPickle.dump(features, f)
        f.close()

    def extract_feature_10_recognition_traintest_multilabel(self):
        # if os.path.exists('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle'):
        #     print '/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_sigmoid9_10_'+ self.net_name + '.cPickle already exists!'
        #     return
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+self.event_name+'/guru_'+self.name+'_path.txt'
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
            if count % 100 == 0:
                print dict_name2[event_name], np.argmax(out[0])
            #print img, count, out[0, dict_name2[self.event_name]-1]
            count += 1
        #for i in xrange(len(blob_names)):
        f = open('/home/feiyu1990/local/event_curation/'+self.net_path+'/features/'+self.event_name + '_' +self.name+'_predict_'+ self.net_name + '.cPickle','wb')
        cPickle.dump(features, f)
        f.close()


    def extract_feature_10_recognition_traintest_smallcrop(self):
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
        # mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(1)
        caffe.set_mode_gpu()
        img_dims = self.img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        features = []
        count = 0
        mean_pixel = np.array([123, 117, 104])
        for img in imgs:
            temp = caffe.io.load_image(img)
            input_2 = caffe.io.resize_image(temp, (320,320))
            caffe_in = input_2 - mean_pixel/255
            inputs = [caffe_in]
            out = net.predict(inputs)#,oversample = False)
            # print out.shape
            # if out.shape[1] == 23:
            #     features.append(out[0, dict_name2[self.event_name]-1])
                # print out
            # else:
            features.append(out[0])
            if count % 100 == 0:
                print dict_name2[event_name], np.argmax(out[0])
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
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+self.event_name + '_23_sigmoid9_10_segment_joint_iter_100000.cPickle','wb')
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


def extract_feature_10_recognition_traintest_fc7(img_file, net_path, model_name, net_name, save_path):
        imgs = []
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/training/' + model_name
        weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/model/' + net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()

        net = caffe.Net(model_name, weight_name, caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        transformer.set_mean('data', mean_file.mean(1).mean(1))
        # print net.blobs['data']
        net.blobs['data'].reshape(1,3,227, 227)
        features = []
        count = 0
        for img in imgs:
            count += 1
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
            net.forward()
            a = net.blobs['fc7'].data.copy()
            if count % 100 == 0:
                # print img
                # print np.max(net.blobs['data'].data), np.min(net.blobs['data'].data)
                print count, a[0]
            features.append(a[0])

        np.save(save_path, np.array(features))


def extract_feature_10_recognition_traintest_multilabel(net_path, model_name_, net_name, img_size):
    event_prediction_dict = defaultdict(list)
    for event_name in dict_name2.keys() + ['multi_label']:
        print event_name
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+event_name+'/guru_test_path.txt'
        with open(img_file, 'r') as data:
            for line in data:
                imgs.append(line.split(' ')[0])
        model_name = '/home/feiyu1990/local/event_curation/'+net_path+'/training/' + model_name_
        weight_name = '/home/feiyu1990/local/event_curation/'+net_path+'/model/' + net_name + '.caffemodel'
        mean_file = np.load('/home/feiyu1990/local/caffe-mine-test/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        img_dims = img_size
        raw_scale = 255
        channel_swap = (2,1,0)
        net = caffe.Classifier(model_name, weight_name, image_dims=img_dims, raw_scale = raw_scale, channel_swap = channel_swap)

        with open('/home/feiyu1990/local/event_curation/baseline_all_correction_multi/'+event_name+'/test_image_ids.cPickle') as f:
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
                print dict_name2[event_name], np.argmax(out[0])
            count += 1
    f = open('/home/feiyu1990/local/event_curation/'+net_path+'/features/test_predict_'+ net_name + '_dict.pkl','wb')
    cPickle.dump(event_prediction_dict, f)
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
        if type == 'none':
            return
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
        model_names = [['baseline_all_noblock/', '/'+self.validation_name+'_random_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_00501_0.5_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_23_23_VGG_fromnoevent_00501_0.4_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_00501_0.1_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.2_5w_iter_30000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.2_5w_iter_50000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.1_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.1_iter_80000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.3_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.4_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.4_iter_30000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.5_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_fromnoevent_0.5_iter_40000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_noevent_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_noevent_iter_40000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_iter_50000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_0.3_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_0.3_iter_40000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_0.5_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_0.5_iter_40000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_23_23_VGG_segment_00501_0.5_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_segment_00501_0.4_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_23_23_VGG_segment_00501_0.1_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_svm_0.01_iter_100000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_svm_0.1_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_svm_0.1_iter_50000_dict.cPickle']
            ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_svm_0.5_iter_100000_dict.cPickle']
            # ,['CNN_all_event_vgg/features/','_test_sigmoid9_10_VGG_svm_iter_100000_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_poly_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct2_power_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_power_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_power1.5_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_power2_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_power2.5_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_power4_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_correct_power5_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_sigmoid_power_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_multiply_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_largest_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_add_largest_dict.cPickle']
            # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_noevent_iter_100000_dict.cPickle']
            # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_noevent_8w_iter_200000_dict.cPickle']
            # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_noevent_twoloss_fc300_iter_100000_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_top2_power2_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_diffweight_top2_power2_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_segment_5time_top2_power2_dict.cPickle']
            # ,['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_5time_iter_200000_dict.cPickle']
            # ,['CNN_all_event_old/features/', '_test_sigmoid9_10_segment_3time_iter_100000_dict.cPickle']
            # # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_diffweight_iter_100000_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct2_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct2_power3_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct2_power2_dict.cPickle']
            # # # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_correct2_power2_round2_dict.cPickle']


            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_expand_balanced_joint_fc300_sigmoid_correct2_power2_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct3_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct3_power2_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct3_power_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct10_power_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_event_joint_fc300_sigmoid_correct_power_dict.cPickle']
            # ,['CNN_all_event_1009/features/','_23_sigmoid9_10_segment_twoloss_fc300_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_test_combined_10_fromnoevent_2_7w_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_'+self.validation_name+'_sigmoid9_10_segment_fromnoevent_3_iter_100000_dict.cPickle']
            #
            #                       ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_twoloss_fc300_diffweight_iter_100000_dict.cPickle']
            #                       ,['CNN_all_event_1009/features/','_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2time_iter_100000_dict.cPickle']
            #                       ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_fromnoevent_twoloss_fc300_iter_100000_dict.cPickle']
            #                       # ,['CNN_all_event_1009/features/','_test_event_joint_fc300_diffweight_sigmoid_correct2_power2_round2_dict.cPickle']
            #                       # ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_twoloss_fc300_iter_100000_dict.cPickle'],
            #                       # ['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_fc_iter_100000_dic,t.cPickle'],
            #                       ['CNN_all_event_old/features/', '_test_sigmoid9_10_segment_3time_iter_100000_dict.cPickle'],
            #                       ,['CNN_all_event_1009/features/', '_test_combined_new_dict.cPickle']
                                  # ,['CNN_all_event_1009/features/', '_test_sigmoid9_23_segment_fromnoevent_iter_100000_corrected_dict.cPickle']
            #                       ,['CNN_all_event_1009/features/', '_test_combined_new_em_100_dict.cPickle']
            #
            #
            #             # ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_rmsprop_2_iter_100000_dict.cPickle']
            #             # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_nag_iter_100000_dict.cPickle']
            #             # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_adagrad_iter_100000_dict.cPickle']
            #             # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_adadelta_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_3time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_5time_iter_200000_dict.cPickle']
            #             # ,['CNN_all_event_old/features/','_'+self.validation_name+'_sigmoid9_10_segment_differentsize_iter_110000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_fc_iter_100000_dict.cPickle']
            #             # ,['CNN_all_event_1009/features/','_'+self.validation_name+'_sigmoid9_10_segment_fc_iter_190000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_iter_120000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_iter_190000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_0.5_iter_120000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_8w_0.5_iter_190000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_2time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_3time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_4time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_real_iter_140000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_real_2time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_real_2time_iter_150000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_0.75_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_0.75_real_iter_200000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_2time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_diffweight_iter_200000_corrected_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_3_real_iter_100000_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_em_dict.cPickle']
                        # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_start_dict.cPickle']
            # #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_iter_150000_dict.cPickle']
            # #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_iter_200000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc500_diffweight_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc500_diffweight_real_iter_200000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc500_diffweight_2_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc500_diffweight_2_real_iter_200000_dict.cPickle']
            # #                        #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_two                        ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_fc300_diffweight_real_iter_100000_dict.cPickle']
            # # #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_iter_50000_dict.cPickle']
            # #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_diffweight_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_diffweight_real_iter_200000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_diffweight_2_real_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_diffweight_2_real_iter_200000_dict.cPickle']
            # #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_2time_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_iter_200000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_8w_iter_160000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_8w_iter_340000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_23_segment_twoloss_double_fc300_8w_2_iter_260000_dict.cPickle']
            # #             # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_diffweight_iter_70000_dict.cPickle']
            # #             # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_diffweight_iter_100000_dict.cPickle']
            # #             # ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_double_fc300_diffweight_iter_160000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc350_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc250_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_diffweight_2_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_diffweight_1.8_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_diffweight_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_diffweight_1.2_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_fc300_diffweight_0.75_iter_100000_dict.cPickle']
            #
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss_iter_100000_dict.cPickle']
            #             ,['CNN_all_event_1009/features/', '_' + self.validation_name + '_sigmoid9_10_segment_twoloss2_iter_120000_dict.cPickle']

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
                # print score
                #print score[i]
                try:
                    temp_score += score[i][0][event_index-1]
                except:
                    # print score[i]
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
        # model_names_to_combine = [
        #                 ['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_fromnoevent_iter_100000.cPickle']
        #                 ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000.cPickle']
        # #                 ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_fromnoevent_3_iter_100000.cPickle']
        #                ]

        model_names_to_combine = [['CNN_all_event_1009/features/','_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_2time_iter_100000.cPickle'],
                                  ['CNN_all_event_1009/features/','_test_sigmoid9_23_segment_twoloss_fc300_diffweight_real_2time_iter_100000.cPickle'],
                                  ['CNN_all_event_1009/features/','_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000.cPickle']
                                  # ['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_fc_iter_100000.cPickle'],
                                  # ['CNN_all_event_old/features/', '_test_sigmoid9_10_segment_3time_iter_100000.cPickle']
                       ]
        combine_name = '_combined_new'
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
            # self.combine_cues_ranking_euclidean(event_name,[event_name.join(i) for i in model_names_to_combine], dict_name2[event_name], '_' + validation_name+combine_name, validation_name)
            self.combine_cues(event_name,[event_name.join(i) for i in model_names_to_combine], dict_name2[event_name], '_' + validation_name+combine_name, validation_name)
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


def evaluation_event_recognition_peralbum_weighted_groundtruth(event_name, threshold,net_name, oversample = True):
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
            importance_score = i[2]
            prediction_ = np.array(i[1])
            # print prediction_thisevent, importance_score
            importance_score = (importance_score + 1) / 4
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


def evaluation_event_recognition_peralbum_weighted_top2(event_name, threshold,net_name, oversample = True):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')

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
    prediction_this_type_second = []
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
            prediction_thisevent += prediction_
            # prediction_thisevent += importance_score * prediction_
        prediction_this_type.append(np.argmax(prediction_thisevent))
        prediction_this_type_second.append(np.argsort(prediction_thisevent)[-2])
        album_names.append(album_name)

    temp = [i == dict_name2[event_name] - 1 for i in prediction_this_type]
    temp2 = [i == dict_name2[event_name] - 1 for i in prediction_this_type_second]
    accuracy_this = (float(np.sum(temp))+ float(np.sum(temp2))) / len(temp)

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
    # print str_to_print
    # print event_name, accuracy_this
    return accuracy_this, len(temp)


def evaluation_event_recognition_peralbum_weighted_corrected(event_name, threshold,net_name, oversample = True):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')
    prediction = cPickle.load(f)
    f.close()

    # f = open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle' ,'r')
    f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/' + event_name + '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_dict.cPickle' ,'r')
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
            # if i[1] == img_id:
                importance_this = i[2]
                ground_pred_comb_dict[event_id].append([img_id, predict_score, importance_this])
        # print ground_this_event


    prediction_this_type = []
    album_names = []
    ground_truth_list = []
    for album_name in ground_pred_comb_dict:
        prediction_thisevent = np.zeros((23,))
        if album_name in correct_list:
            # print album_name, dict_name2[event_name] - 1, dict_name2[correct_list[album_name]] - 1
            ground_truth_list.append(dict_name2[correct_list[album_name]] - 1)
        else:
            ground_truth_list.append(dict_name2[event_name] - 1)

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

    print prediction_this_type
    temp = [i == j for i,j in zip(prediction_this_type, ground_truth_list)]
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


def evaluation_event_recognition_peralbum_weighted_corrected_allevent(threshold,net_name, oversample = True):
    confusion_matrix = np.zeros((23, 23))
    accuracy_all = []
    event_len = []
    for event_name in dict_name2:
        f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')
        prediction = cPickle.load(f)
        f.close()

        # f = open('/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle' ,'r')
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/' + event_name + '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_dict.cPickle' ,'r')
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
                # if i[1] == img_id:
                    importance_this = i[2]
                    ground_pred_comb_dict[event_id].append([img_id, predict_score, importance_this])
            # print ground_this_event


        prediction_this_type = []
        album_names = []
        ground_truth_list = []
        event_len.append(len(ground_pred_comb_dict))
        event_recognition = dict()
        for album_name in ground_pred_comb_dict:
            prediction_thisevent = np.zeros((23,))
            if album_name in correct_list:
                # print album_name, dict_name2[event_name] - 1, dict_name2[correct_list[album_name]] - 1
                ground_truth_list.append(dict_name2[correct_list[album_name]] - 1)
            else:
                ground_truth_list.append(dict_name2[event_name] - 1)

            for i in ground_pred_comb_dict[album_name]:
                importance_score = i[2]
                prediction_ = np.array(i[1])
                # print prediction_thisevent, importance_score
                # importance_score = (importance_score + 1) / 4
                if importance_score < threshold:
                    # print importance_score
                    continue
                prediction_thisevent += importance_score**0.9 * prediction_
            prediction_this_type.append(np.argmax(prediction_thisevent))
            album_names.append(album_name)
            event_recognition[album_name] = prediction_thisevent

        f = open(root + 'CNN_all_event_1009/features/'+event_name + net_name + '_groundtruth_importance.cPickle', 'w')
        cPickle.dump(event_recognition, f)
        f.close()

        # print prediction_this_type
        temp = [i == j for i,j in zip(prediction_this_type, ground_truth_list)]
        for i,j in zip(prediction_this_type, ground_truth_list):
            confusion_matrix[j,i] += 1
        accuracy_all.append(np.sum(temp))

        # dict_name2_reverse = dict([(dict_name2[key] - 1, key) for key in dict_name2])
        # predicted_label = [0]*23
        # for i_this in xrange(23):
        #     predicted_label[i_this] = sum([i == i_this for i in prediction_this_type])
        # str_to_print = ''
        # for i in xrange(len(predicted_label)):
        #     if predicted_label[i] > 0 and i != dict_name2[event_name] - 1:
        #         albums_this = [album_names[ii] for ii in xrange(len(prediction_this_type)) if prediction_this_type[ii] == i]
        #         str_to_print += dict_name2_reverse[i]+':'+str(predicted_label[i])+ ' ' + str(albums_this) +', '
        #     if i == dict_name2[event_name] - 1:
        #         str_to_print += dict_name2_reverse[i]+':'+str(predicted_label[i])+', '
        # print str_to_print
        # print event_name, accuracy_this

    for i in xrange(23):
        # confusion_matrix[i,:] /= np.sum(confusion_matrix[i,:])
        print [int(j) for j in list(confusion_matrix[i,:])]
    print 'Overall accuracy: ', float(np.sum(accuracy_all)) / np.sum(event_len)


def evaluation_event_recognition_peralbum_weighted(event_name, threshold,net_name, oversample = True):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')

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


def evaluation_event_recognition_peralbum_noimportance(event_name,net_name, oversample = True):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')
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
                #
            # print np.argmax(prediction_)
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
    temp1 = np.zeros((23,), dtype=float)
    for i in prediction_this_type:
        temp1[i] += 1
    return accuracy_this, len(temp), temp1 / len(temp)


def evaluation_event_recognition_peralbum_noimportance_corrected(event_name,net_name, oversample = True):
    f = open("/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/" + event_name + net_name + '.cPickle','r')
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
    ground_truth_list = []
    for album_name in ground_pred_comb_dict:
        prediction_thisevent = np.zeros((23,))
        if album_name in correct_list:
            print album_name, dict_name2[event_name] - 1, dict_name2[correct_list[album_name]] - 1
            ground_truth_list.append(dict_name2[correct_list[album_name]] - 1)
        else:
            ground_truth_list.append(dict_name2[event_name] - 1)
        for i in ground_pred_comb_dict[album_name]:
            # importance_score = i[2]
            prediction_ = np.array(i[1])
            # print prediction_thisevent, importance_score
            # importance_score = (importance_score + 1) / 4
            # if importance_score < threshold:
                # print importance_score
                #
            # print np.argmax(prediction_)
            prediction_thisevent +=  prediction_
        prediction_this_type.append(np.argmax(prediction_thisevent))
        album_names.append(album_name)

    temp = [i == j for i,j in zip(prediction_this_type, ground_truth_list)]
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
    temp1 = np.zeros((23,), dtype=float)
    for i in prediction_this_type:
        temp1[i] += 1
    return accuracy_this, len(temp), temp1 / len(temp)


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


def combine_event_feature_traintest(event_name, importance_path='test_sigmoid9_23_segment_5time_iter_200000', event_path = 'test_predict_event_recognition_iter_100000'):
        # print "HIHIHI"
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/'+event_name + '_' + event_path+'.cPickle','r')
        event_recognition_feature = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
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
                # print prediction_
                prediction_thisevent += prediction_
            prediction_thisevent = (prediction_thisevent - np.min(prediction_thisevent)) / (np.max(prediction_thisevent) - np.min(prediction_thisevent))
            prediction_thisevent = np.power(prediction_thisevent, 2)
            # prediction_thisevent = [i>0.95for i in prediction_thisevent]
            # if np.argmax(prediction_thisevent) != dict_name2[event_name]-1:
            #     # if np.argsort(prediction_thisevent)[-2] != dict_name2[event_name]-1:
            #         prediction_thisevent = np.zeros((23,))
            #         prediction_thisevent[dict_name2[event_name]-1] = 1
            temp = np.argsort(prediction_thisevent)
            for ii in temp[:21]:
                prediction_thisevent[ii] = 0

            prediction_album[album_name] = prediction_thisevent

        # print prediction_album

        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+event_name + '_' + importance_path+'.cPickle','r')
        features = cPickle.load(f)
        f.close()
        # print features.shape

        combined_importance_all = []
        for img_name, importance_prediction in zip(all_event_ids, features):
            event_id = img_name.split('/')[0]
            new_importance = np.sum(prediction_album[event_id] * sigmoid(importance_prediction))
            combined_importance_all.append(new_importance)
        # print combined_importance_all
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+event_name + '_test_event_joint_segment_5time_top2_power2.cPickle','wb')
        # print '/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+event_name + '_test_event_joint_fc300_diffweight_top2_power2.cPickle'
        cPickle.dump(combined_importance_all, f)
        f.close()


def combine_event_feature_traintest_2round(event_name, importance_path='23_sigmoid9_10_segment_twoloss_fc300_iter_100000',#'test_sigmoid9_23_segment_twoloss_fc300_iter_100000',
                                           event_path = 'test_predict_event_recognition_iter_100000', threshold = 0.0):
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/'+event_name + '_' + event_path+'.cPickle','r')
        event_recognition_feature = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()

        max_importance = -np.Inf
        min_importance = np.Inf

        f = open('/raid/yufeihome/event_curation/CNN_all_event_1009/features/' + event_name + '_test_sigmoid9_10_segment_twoloss_fc300_iter_100000_dict.cPickle' ,'r')
        importance_predicted = cPickle.load(f)
        f.close()

        pred_comb_dict = defaultdict(list)
        for predict_score, img_id in zip(event_recognition_feature, all_event_ids):
            event_id = img_id.split('/')[0]
            ground_this_event = importance_predicted[event_id]
            for i in ground_this_event:
                if i[0] == img_id:
                    importance_this = i[2]
                    max_importance = max(max_importance, np.max(importance_this))
                    min_importance = min(min_importance, np.min(importance_this))
                    pred_comb_dict[event_id].append([img_id, predict_score, importance_this])
        print max_importance, min_importance

        prediction_album = dict()
        for album_name in pred_comb_dict:
            prediction_thisevent = np.zeros((23,))
            for i in pred_comb_dict[album_name]:
                prediction_ = np.array(i[1])
                importance_score = (i[2] - min_importance) / (max_importance - min_importance)

                if importance_score < threshold:
                    continue
                prediction_thisevent += importance_score * prediction_


                # print prediction_
            # print prediction_thisevent
            prediction_thisevent = (prediction_thisevent - np.min(prediction_thisevent)) / (np.max(prediction_thisevent) - np.min(prediction_thisevent))
            prediction_thisevent = np.power(prediction_thisevent, 2)
            # prediction_thisevent = [i>0.95for i in prediction_thisevent]
            if np.argmax(prediction_thisevent) != dict_name2[event_name]-1:
                # if np.argsort(prediction_thisevent)[-2] != dict_name2[event_name]-1:
                    prediction_thisevent = np.zeros((23,))
                    prediction_thisevent[dict_name2[event_name]-1] = 1
            temp = np.argsort(prediction_thisevent)
            for ii in temp[:21]:
                prediction_thisevent[ii] = 0

            prediction_album[album_name] = prediction_thisevent

        # print prediction_album

        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+event_name + '_' + importance_path+'.cPickle','r')
        features = cPickle.load(f)
        f.close()
        # print features.shape

        combined_importance_all = []
        for img_name, importance_prediction in zip(all_event_ids, features):
            event_id = img_name.split('/')[0]
            new_importance = np.sum(prediction_album[event_id] * (importance_prediction))
            combined_importance_all.append(new_importance)
        # print combined_importance_all
        f = open('/home/feiyu1990/local/event_curation/CNN_all_event_1009/features/'+event_name + '_test_event_joint_fc300_correct2_power2_round2.cPickle','wb')
        cPickle.dump(combined_importance_all, f)
        f.close()


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def em_combine_event_recognition_curation_corrected(threshold, threshold_m, poly = 1, importance_path='_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000',
                                          event_path = '_test_predict_event_recognition_expand_balanced_3_iter_100000',
                                          stop_criterion = 0.01, max_iter = 101):
    accuracy_events = defaultdict(list)
    event_lengths = dict()
    for event_name in dict_name2:
        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '.cPickle')
        importance_feature = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_1205/features/' + event_name + event_path + '.cPickle')
        recognition = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()

        # # #initial importance score prediction
        # event_recognition_ini = dict()
        # for i in all_img_ids:
        #     event_id = i.split('/')[0]
        #     if event_id not in event_recognition_ini:
        #         event_recognition_ini[event_id] = np.ones((23,))/23
        # importance_ini = m_step(importance_feature, event_recognition_ini, all_img_ids)
        # event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold)
        # importance_score = m_step(importance_feature, event_recognition, all_img_ids)


        # #initialization of \theta
        importance_ini = np.ones((len(recognition),))
        event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold, poly)
        event_lengths[event_name] = len(event_recognition)
        accuracy_this = []
        for event_id in event_recognition:
            if event_id in correct_list:
                # print event_id
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[correct_list[event_id]] - 1)
            else:
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_name] - 1)


        accuracy_events[event_name].append(np.sum(accuracy_this))

        importance_score = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)

        diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
        iter = 0
        while diff > stop_criterion: #* len(importance_score):
            iter += 1
            if iter >= max_iter:
                break
            # threshold_m = (float(iter) / max_iter) ** 0.5
            event_recognition = e_step(importance_score, recognition, all_img_ids, threshold, poly)

            accuracy_this = []
            for event_id in event_recognition:
                if event_id in correct_list:
                   accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[correct_list[event_id]] - 1)
                else:
                    accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_name] - 1)
            accuracy_events[event_name].append(np.sum(accuracy_this))
            importance_score_new = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
            diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
            importance_score = importance_score_new

            # print diff

        f = open(root + 'CNN_all_event_1009/features/'+event_name + event_path + '_em_'+str(max_iter)+'.cPickle', 'w')
        cPickle.dump(event_recognition, f)
        f.close()

        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '_em_'+str(max_iter)+'.cPickle', 'w')
        cPickle.dump(importance_score, f)
        f.close()
    print accuracy_events
    accuracy_all = float(np.sum([accuracy_events[event_name][-1] for event_name in accuracy_events])) \
                   / np.sum([event_lengths[event_name] for event_name in event_lengths])
    print accuracy_all



def combine_event_recognition_curation_corrected_cheating(threshold=0, importance_path='_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000',
                                          event_path = '_test_predict_event_recognition_expand_balanced_3_iter_100000'):
    # accuracy_events = defaultdict(list)
    # event_lengths = dict()
    for event_name in dict_name2:
        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '.cPickle')
        importance_feature = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_1205/features/' + event_name + event_path + '.cPickle')
        recognition = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()

        importance_ini = [i[dict_name2[event_name] - 1] for i in importance_feature]
        print len(importance_ini), len(recognition)
        # # #initial importance score prediction
        event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold)



        f = open(root + 'CNN_all_event_1009/features/'+event_name + event_path + '_cheating.cPickle', 'w')
        cPickle.dump(event_recognition, f)
        f.close()





def em_combine_event_recognition_curation_corrected_combine(threshold, threshold_m, poly = 1, importance_path='_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000',
                                          event_path = '_test_predict_event_recognition_expand_balanced_3_iter_100000',
                                          stop_criterion = 0.01, max_iter = 101):
    accuracy_events = defaultdict(list)
    event_lengths = dict()

    f = open('/home/feiyu1990/local/event_curation/lstm/data/test_lstm_prediction_img_dict.pkl')
    lstm_prediction = cPickle.load(f)
    f.close()

    for event_name in dict_name2:
        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '.cPickle')
        importance_feature = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_1205/features/' + event_name + event_path + '.cPickle')
        recognition = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()

        count = 0
        for img_id in all_img_ids:
            # print importance_feature[count]
            # print lstm_prediction[img_id]
            recognition[count] = lstm_prediction[img_id]
            # recognition[count] /= np.sum(recognition[count])
            count += 1

        # # #initial importance score prediction
        # event_recognition_ini = dict()
        # for i in all_img_ids:
        #     event_id = i.split('/')[0]
        #     if event_id not in event_recognition_ini:
        #         event_recognition_ini[event_id] = np.ones((23,))/23
        # importance_ini = m_step(importance_feature, event_recognition_ini, all_img_ids)
        # event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold)
        # importance_score = m_step(importance_feature, event_recognition, all_img_ids)


        # #initialization of \theta
        importance_ini = np.ones((len(recognition),))
        event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold, poly)
        event_lengths[event_name] = len(event_recognition)
        accuracy_this = []
        for event_id in event_recognition:
            if event_id in correct_list:
                # print event_id
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[correct_list[event_id]] - 1)
            else:
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_name] - 1)


        accuracy_events[event_name].append(np.sum(accuracy_this))

        importance_score = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)

        diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
        iter = 0
        while diff > stop_criterion: #* len(importance_score):
            iter += 1
            if iter >= max_iter:
                break
            # threshold_m = (float(iter) / max_iter) ** 0.5
            event_recognition = e_step(importance_score, recognition, all_img_ids, threshold, poly)

            accuracy_this = []
            for event_id in event_recognition:
                if event_id in correct_list:
                   accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[correct_list[event_id]] - 1)
                else:
                    accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_name] - 1)
            accuracy_events[event_name].append(np.sum(accuracy_this))
            importance_score_new = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
            diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
            importance_score = importance_score_new

            # print diff

        f = open(root + 'CNN_all_event_1009/features/'+event_name + event_path + '_lstm_img_em_'+str(max_iter)+'.cPickle', 'w')
        cPickle.dump(event_recognition, f)
        f.close()

        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '_lstm_img_em_'+str(max_iter)+'.cPickle', 'w')
        cPickle.dump(importance_score, f)
        f.close()
    print accuracy_events
    accuracy_all = float(np.sum([accuracy_events[event_name][-1] for event_name in accuracy_events])) \
                   / np.sum([event_lengths[event_name] for event_name in event_lengths])
    print accuracy_all


def em_combine_event_recognition_curation(threshold, threshold_m, poly = 1, importance_path='_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000',
                                          event_path = '_test_predict_event_recognition_expand_balanced_3_iter_100000',
                                          stop_criterion = 0.01, max_iter = 100):
    accuracy_events = defaultdict(list)
    event_lengths = dict()
    for event_name in dict_name2:
        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '.cPickle')
        importance_feature = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_1205/features/' + event_name + event_path + '.cPickle')
        recognition = cPickle.load(f)
        f.close()

        path = '/home/feiyu1990/local/event_curation/baseline_all_0509/' + event_name+ '/test_image_ids.cPickle'
        f = open(path, 'r')
        all_img_ids = cPickle.load(f)
        f.close()

        # # #initial importance score prediction
        # event_recognition_ini = dict()
        # for i in all_img_ids:
        #     event_id = i.split('/')[0]
        #     if event_id not in event_recognition_ini:
        #         event_recognition_ini[event_id] = np.ones((23,))/23
        # importance_ini = m_step(importance_feature, event_recognition_ini, all_img_ids)
        # event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold)
        # importance_score = m_step(importance_feature, event_recognition, all_img_ids)


        # #initialization of \theta
        importance_ini = np.ones((len(recognition),))
        event_recognition = e_step(importance_ini, recognition, all_img_ids, threshold, poly)
        event_lengths[event_name] = len(event_recognition)
        # threhsold_m = 0
        importance_score = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)

        diff = np.sum(np.abs(np.array(importance_score) - importance_ini))
        iter = 0
        while diff > stop_criterion: #* len(importance_score):
            iter += 1
            # threshold_m = (float(iter) / max_iter) ** 0.5
            event_recognition = e_step(importance_score, recognition, all_img_ids, threshold, poly)

            accuracy_this = []
            for event_id in event_recognition:
                accuracy_this.append(np.argmax(event_recognition[event_id]) == dict_name2[event_name] - 1)
            accuracy_events[event_name].append(np.sum(accuracy_this))

            importance_score_new = m_step(importance_feature, event_recognition, all_img_ids, threshold_m)
            diff = np.sum(np.abs(np.array(importance_score) - np.array(importance_score_new)))
            importance_score = importance_score_new
            if iter >= max_iter:
                break
            # print diff

        f = open(root + 'CNN_all_event_1009/features/' + event_path + '_em.cPickle', 'w')
        cPickle.dump(event_recognition, f)
        f.close()

        f = open(root + 'CNN_all_event_1009/features/' + event_name + importance_path + '_em.cPickle', 'w')
        cPickle.dump(importance_score, f)
        f.close()
    print accuracy_events
    accuracy_all = float(np.sum([accuracy_events[event_name][-1] for event_name in accuracy_events])) \
                   / np.sum([event_lengths[event_name] for event_name in event_lengths])
    print accuracy_all


def e_step(importance_feature, img_recognition, all_img_ids, threshold = 0, poly = 1):
    event_recognition = dict()
    last_event_id = ''

    for i,j,k in zip(importance_feature, img_recognition, all_img_ids):
        event_id = k.split('/')[0]
        if last_event_id != event_id:
            last_event_id = event_id
            event_recognition[event_id] = np.zeros((23, ))
        if i < threshold:
            continue
        event_recognition[event_id] += i **poly * j
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
                importance_out_all.extend(importance_out/np.max(importance_out))
            importance_out = []
        event_rec_this = event_recognition[event_id]
        event_rec_this = (event_rec_this - np.min(event_rec_this)) / (np.max(event_rec_this) - np.min(event_rec_this))
        if threshold_1 == 1:
            low_values_indices = event_rec_this < np.max(event_rec_this)
        else:
            low_values_indices = event_rec_this < threshold_1
        # low_values_indices = event_rec_this < sorted(event_rec_this)[-3]  # Where values are low
        event_rec_this[low_values_indices] = 0
        # event_rec_this = np.power(event_rec_this, 2)
        # event_rec_this = event_rec_this / np.sum(event_rec_this)
        importance_out.append(np.sum(event_rec_this * (i)))
    importance_out_all.extend(importance_out/np.max(importance_out))
    return importance_out_all


def lstm_recognition():
    prediction = np.load(root + 'lstm/data/test_lstm_prediction_oversample_50_alltraining.npy')
    with open(root + 'lstm/data/test_event_img_dict.pkl') as f:
        test_event_img_dict = cPickle.load(f)
    event_prediction_dict = dict()

    # with open(root + 'lstm/data/test_imdb.pkl') as f:
    #     lstm_test_feature, lstm_test_label = cPickle.load(f)

    count = 0
    for event_id in test_event_img_dict:
        event_prediction_dict[event_id] = prediction[count]
        count += 1
    confusion_matrix = np.zeros((23, 23), dtype=float)
    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle') as f:
            test_event_id = cPickle.load(f)
        for event in test_event_id:
            predict_ = np.argmax(event_prediction_dict[event])
            if event in correct_list:
                ground_ = dict_name2[correct_list[event]] - 1
            else:
                ground_ = dict_name2[event_type] - 1
            # print ground_, predict_
            confusion_matrix[ground_, predict_] += 1
    accuracy = float(np.trace(confusion_matrix)) / len(test_event_img_dict)
    for i in xrange(23):
        # confusion_matrix[i,:] /= np.sum(confusion_matrix[i,:])
        print [int(j) for j in list(confusion_matrix[i,:])]
    print 'Overall accuracy:', accuracy


def combine_lstm_cnn_result(poly):
    cnn_result_dict = dict()
    for event_name in dict_name2:
        with open(root + 'CNN_all_event_1009/features/' + event_name +
                          '_test_predict_event_recognition_expand_balanced_3_iter_100000_em_99.cPickle') as f:
            temp = cPickle.load(f)
        for i in temp:
            cnn_result_dict[i] = temp[i] ** poly
    with open(root + 'lstm/data/test_lstm_prediction_oversample_50_alltraining_prediction_dict.pkl') as f:
        lstm_result = cPickle.load(f)
    for event_id in lstm_result:
        cnn_result_dict[event_id] += lstm_result[event_id] ** poly

    with open(root + 'lstm/data/test_combine_lstm_em_prediction_dict.pkl', 'w') as f:
        cPickle.dump(cnn_result_dict, f)

    confusion_matrix = np.zeros((23, 23), dtype=float)
    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle') as f:
            test_event_id = cPickle.load(f)
        for event in test_event_id:
            predict_ = np.argmax(cnn_result_dict[event])
            if event in correct_list:
                ground_ = dict_name2[correct_list[event]] - 1
            else:
                ground_ = dict_name2[event_type] - 1
            # print ground_, predict_
            confusion_matrix[ground_, predict_] += 1
    accuracy = float(np.trace(confusion_matrix)) / len(cnn_result_dict)
    for i in xrange(23):
        # confusion_matrix[i,:] /= np.sum(confusion_matrix[i,:])
        print([int(j) for j in list(confusion_matrix[i, :])])
    print('Overall accuracy:', accuracy)


def create_confusion_matrix():
    cnn_result_dict = dict()
    for event_name in dict_name2:
        with open(root + 'CNN_all_event_1009/features/' + event_name +
                          # '_test_predict_event_recognition_expand_balanced_3_iter_100000_em_99.cPickle') as f:
                          # '_test_predict_event_recognition_expand_balanced_3_iter_100000_lstm_img_combine_em_9.cPickle') as f:
                          '_test_recognition_lstm_prediction_em_combine_dict.pkl') as f:
            temp = cPickle.load(f)
        for i in temp:
            cnn_result_dict[i] = temp[i]
        # with open(root + 'CNN_all_event_1009/features/' + event_name +
        #                   '_test_predict_event_recognition_expand_balanced_3_iter_100000_lstm_img_combine_em_9.cPickle') as f:
        #     temp = cPickle.load(f)
        # for i in temp:
        #     cnn_result_dict[i] *= temp[i]

    confusion_matrix = np.zeros((23, 23), dtype=float)
    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle') as f:
            test_event_id = cPickle.load(f)
        for event in test_event_id:
            predict_ = np.argmax(cnn_result_dict[event])
            if event in correct_list:
                ground_ = dict_name2[correct_list[event]] - 1
            else:
                ground_ = dict_name2[event_type] - 1
            # print ground_, predict_
            confusion_matrix[ground_, predict_] += 1
    accuracy = float(np.trace(confusion_matrix)) / len(cnn_result_dict)
    for i in xrange(23):
        # confusion_matrix[i,:] /= np.sum(confusion_matrix[i,:])
        print([int(j) for j in list(confusion_matrix[i, :])])
    print('Overall accuracy:', accuracy)

def cross_entropy_loss():
    pass

def create_confusion_matrix_multi():
    cnn_result_dict = dict()
    for event_name in dict_name2:
        with open(root + 'CNN_all_event_1009/features/' + event_name +
                          # '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_em_99.cPickle') as f:
                          # '_test_predict_event_recognition_expand_balanced_3_iter_100000_em_99.cPickle') as f:
                          # '_test_predict_event_recognition_expand_balanced_3_iter_100000_lstm_img_combine_em_9.cPickle') as f:
                          '_test_recognition_lstm_prediction_em_combine_dict.pkl') as f:
                          # '_test_predict_event_recognition_expand_balanced_3_iter_100000_cheating.cPickle') as f:
            temp = cPickle.load(f)
        for i in temp:
            cnn_result_dict[i] = temp[i]

    confusion_matrix = np.zeros((23, 23), dtype=float)
    count = 0
    for event_type in dict_name2:
        # with open(root + 'baseline_all_correction_multi/' + event_type + '/test_event_id.cPickle') as f:
        with open(root + 'baseline_all_correction_multi/' + event_type + '/test_event_id.cPickle') as f:
            test_event_id = cPickle.load(f)
        for event in test_event_id:
            count += 1
            predict_ = np.argmax(cnn_result_dict[event])
            # if event in correct_list:
            #     ground_ = dict_name2[correct_list[event]] - 1
            # else:
            ground_ = dict_name2[event_type] - 1
            # print ground_, predict_
            confusion_matrix[ground_, predict_] += 1

    event_type = 'multi_label'
    with open(root + 'baseline_all_correction_multi/' + event_type + '/test_event_id.cPickle') as f:
        test_event_id = cPickle.load(f)
    for event, event_type in test_event_id:
        count += 1
        predict_ = np.argmax(cnn_result_dict[event])
        event_type_n = [dict_name2[i[0]] - 1 for i in event_type]
        if predict_ in event_type_n:
            confusion_matrix[predict_, predict_] += 1
        else:
            print event, event_type, dict_name2_reverse[predict_ + 1]
            ground_ = [dict_name2[event_type[0][0]] - 1]
            confusion_matrix[ground_, predict_] += 1

    accuracy = float(np.trace(confusion_matrix)) / count
    for i in xrange(23):
        # confusion_matrix[i,:] /= np.sum(confusion_matrix[i,:])
        print([int(j) for j in list(confusion_matrix[i, :])])
    print('Overall accuracy:', accuracy)
    print count

if __name__ == '__main__':
    itera = '_10w'
    combine_face_model = '_combined_10_fromnoevent_2_10w.cPickle'
    noqual_name = '_noqual'
    TIE = False
    # extract_feature_10_recognition_traintest_fc7('/home/feiyu1990/local/event_curation/lstm/new_training_img_list.txt',
    #                                              'CNN_all_event_1205',
    #                                              'deploy_fc7.prototxt',
    #                                              'event_recognition_expand_balanced_3_iter_100000',
    #                                              '/home/feiyu1990/local/event_curation/lstm/new_training_img_feature.npy')
    # TIE = False
    # for event_name in dict_name2:
    #     combine_event_feature_traintest_2round(event_name, threshold = 0.2)
    #     combine_event_feature_traintest(event_name)
    #
    for event_name in dict_name2:
    #     a = extract_features('CNN_all_event_1205', event_name, 'multilabel_event_recognition_expand_balanced_3_iter_40000','deploy_fc7.prototxt','test', None, 227)
    #     a.extract_feature_10_traintest()
        # a = extract_features('CNN_all_event_1205', event_name, 'multilabel_event_recognition_expand_balanced_3_iter_40000','deploy.prototxt','test', None, (256, 256))
        # a.extract_feature_10_recognition_traintest()
    #     a = extract_features('CNN_all_event_1205', event_name, 'event_recognition_expand_3_equal_iter_10000','deploy.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_recognition_traintest()
    #     a = extract_features('CNN_all_event_1205', event_name, 'event_recognition_corrected_iter_100000','deploy.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_recognition_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_noevent_twoloss_fc300_iter_100000','python_deploy_siamese_noevent_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_diffweight_iter_200000_corrected','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc500_diffweight_2_real_iter_200000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_double_fc300_diffweight_2_real_iter_200000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_double_fc300_diffweight_real_iter_200000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_double_fc300_diffweight_real_iter_100000','python_deploy_siamese_fc300_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_double_fc300_diffweight_2_real_iter_100000','python_deploy_siamese_fc300_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
        # a = extract_features('CNN_all_event_1009', event_name, 'segment_rmsprop0.1_iter_100000','python_deploy_siamese_new.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()
        # a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc300_diffweight_2time_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()
        # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_twoloss_fc300_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()
        # a = extract_features('CNN_all_event_1009', event_name, 'segment_fromnoevent_iter_100000_corrected','python_deploy_siamese_old.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_adadelta_iter_100000','python_deploy_siamese_new.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc300_8w_0.5_iter_190000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc300_8w_iter_190000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_twoloss_fc300_4time_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_1009', event_name, 'segment_fc_iter_190000','python_deploy_siamese_fc300.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()

        a = extract_features('CNN_all_event_vgg', event_name, 'VGG_noevent_0.5_iter_100000.caffemodel','VGG_deploy_noevent.prototxt','test', None, (256, 256))
        a.extract_feature_10_23_traintest()
        a = extract_features('CNN_all_event_vgg', event_name, 'VGG_segment_0.5_2time_iter_100000.caffemodel','VGG_deploy_segment.prototxt','test', None, (256, 256))
        a.extract_feature_10_23_traintest()

    #
    # a = evaluation('CNN_all_event_1009','combine', 'test',0)
    # a = create_event_recognition('CNN_all_event_1205')

    # accuracy = 0;len_all = 0;threshold = -2
    # confusion_matrix = np.zeros((23,23))
    # for event_name in dict_name2:
    # #     # a,b = evaluation_event_recognition_peralbum_weighted(event_name, threshold, '_test_predict_event_recognition_oversample_iter_80000')
    # #     a,b = evaluation_event_recognition_peralbum_weighted(event_name, threshold, '_test_predict_event_recognition_importance_iter_60000', oversample = False)
    #     a,b,c= evaluation_event_recognition_peralbum_noimportance(event_name, '_test_predict_event_recognition_expand_balanced_3_iter_100000', oversample = False)
    #     # a,b= evaluation_event_recognition_peralbum_weighted_top2(event_name, threshold, '_test_predict_event_recognition_expand_balanced_3_iter_100000', oversample = False)
    # #     # a,b,c= evaluation_event_recognition_peralbum_noimportance(event_name, '_test_predict_event_recognition_iter_100000', oversample = False)
    #     accuracy += a * b
    #     len_all += b
    #     # confusion_matrix[dict_name2[event_name] - 1,:] = c
    # print 'Threshold:',threshold, 'Overall accuracy: ', float(accuracy) / len_all
    # # print confusion_matrix


    # for threshold in xrange(1):
    #     accuracy = 0;len_all = 0
    #     for event_name in dict_name2:
    #         # a,b = evaluation_event_recognition_peralbum(event_name, float(threshold)/10)
    #         a,b = evaluation_event_recognition_peralbum_weighted(event_name, float(threshold)/10,'_test_predict_event_recognition_expand_balanced_3_iter_100000', oversample = False)
    #         accuracy += a * b
    #         len_all += b
    #     print 'Threshold:',threshold, 'Overall accuracy: ', float(accuracy) / len_all
    # #
    # for threshold in xrange(1):
    #     accuracy = 0;len_all = 0
    #     for event_name in dict_name2:
    #         # a,b = evaluation_event_recognition_peralbum(event_name, float(threshold)/10)
    #         a,b = evaluation_event_recognition_peralbum_weighted_corrected(event_name, float(threshold)/10,'_test_predict_event_recognition_expand_balanced_3_iter_100000', oversample = False)
    #         accuracy += a * b
    #         len_all += b
    #     print 'Threshold:',threshold, 'Overall accuracy: ', float(accuracy) / len_all

    # threshold = 0
    # evaluation_event_recognition_peralbum_weighted_corrected_allevent(float(threshold)/10,'_test_predict_event_recognition_expand_balanced_3_corrected_iter_100000', oversample = False)

    # for poly in xrange(1, 20):
    # for threshold_m in xrange(0, 11):
    # # # threshold = 0.5
    # poly = 9
    # threshold_m = 10
    # em_combine_event_recognition_curation_corrected(0, float(threshold_m) / 10, float(poly)/10, max_iter=100)#, event_path='_test_predict_event_recognition_expand_balanced_3_iter_100000')
    # accuracy = 0;len_all = 0
    # for event_name in dict_name2:
    #         a, b = em_combine_event_recognition_curation_corrected(event_name, 0, float(threshold_m) / 10, float(poly)/10)
    #         # a, b = em_combine_event_recognition_curation_corrected(event_name, 0, threshold_m)
    #         accuracy += b
    #         len_all += a
    # print 'Overall accuracy: ', float(accuracy) / len_all

    # for poly in xrange(1, 20):
    # for threshold_m in xrange(0, 11):
    # # threshold = 0.5

    # poly = 9
    # threshold_m = 10
    # em_combine_event_recognition_curation_corrected(0, float(threshold_m) / 10, float(poly)/10, max_iter=99)
                                                        # importance_path='_test_sigmoid9_23_segment_twoloss_fc300_diffweight_real_2time_iter_100000')#,
                                                        # event_path='_test_predict_event_recognition_expand_balanced_3_corrected_iter_100000')
    #     # accuracy = 0;len_all = 0


    # a = evaluation('CNN_all_event_1009','worker', 'test',0)
    # for event_name in dict_name2:
    #     a.create_predict_dict_from_cpickle_multevent('test', event_name, '/home/feiyu1990/local/event_curation/CNN_all_event_1205/features/'+event_name+'_test_predict_event_recognition_expand_balanced_3_corrected_iter_100000', -np.Inf, multi_event=False)
    #
    # lstm_recognition()
    # for poly in xrange(1, 15):
    # combine_lstm_cnn_result(1)

    # combine_event_recognition_curation_corrected_cheating()
    # create_confusion_matrix_multi()
    # poly = 9
    # threshold_m = 10
    # em_combine_event_recognition_curation_corrected_combine(0, float(threshold_m) / 10, float(poly)/10, max_iter=1)


    # with open(root + 'lstm/data/test_lst_prediction_dict.pkl') as f:
    #     lstm_result = cPickle.load(f)
    # for event_name in dict_name2:
    #     cnn_result_dict = dict()
    #     with open(root + 'CNN_all_event_1009/features/' + event_name +
    #                       '_test_predict_event_recognition_expand_balanced_3_iter_100000_em_99.cPickle') as f:
    #         temp = cPickle.load(f)
    #     for i in temp:
    #         cnn_result_dict[i] = (temp[i] ** 2) + (lstm_result[i] ** 2)
    #     with open(root + 'CNN_all_event_1009/features/' + event_name + '_test_recognition_lstm_prediction_em_combine_dict.pkl', 'w') as f:
    #         cPickle.dump(cnn_result_dict, f)
#
    # extract_feature_10_recognition_traintest_multilabel('CNN_all_event_1205', 'deploy.prototxt', 'multilabel_event_recognition_expand_balanced_3_iter_40000', (256, 256))


        # a = extract_features('CNN_all_event_1205', event_name, 'multilabel_event_recognition_expand_balanced_3_iter_40000','deploy.prototxt','test', None, (256, 256))


