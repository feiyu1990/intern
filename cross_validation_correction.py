
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
# import caffe
import re
import operator
import scipy.stats
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, chi2
from collections import Counter, defaultdict

# combine_face_model = '_combined_10_fromnoevent.cPickle'
# combine_face_model = '_sigmoid9_10_segment_fromnoevent_2_iter_100000.cPickle'
global_permutation_time = 1
ground_truth_dict_name = 'new_multiple_result_2round_removedup_vote.pkl'
baseline_name = 'baseline_all_correction_removedup_vote/'
# baseline_name = 'baseline_all_0509/'


dict_name = {'Theme park':'ThemePark', 'Urban/City trip':'UrbanTrip', 'Beach trip':'BeachTrip', 'Nature trip':'NatureTrip',
             'Zoo/Aquarium/Botanic garden':'Zoo','Cruise trip':'Cruise','Show (air show/auto show/music show/fashion show/concert/parade etc.)':'Show',
            'Sports game':'Sports','Personal sports':'PersonalSports','Personal art activities':'PersonalArtActivity',
            'Personal music activities':'PersonalMusicActivity','Religious activities':'ReligiousActivity',
            'Group activities (party etc.)':'GroupActivity','Casual family/friends gathering':'CasualFamilyGather',
            'Business activity (conference/meeting/presentation etc.)':'BusinessActivity','Independence Day':'Independence',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture/Art':'Architecture'}

dict_name_reverse = dict([(dict_name[key], key) for key in dict_name])

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
# root = '/mnt/ilcompf3d1/user/yuwang/event_curation/'


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


def merge_two_csv(csv_1, csv_2, out_csv):

    input_path = csv_2
    line_count = 0
    head_meta_2 = []
    data_meta_2 = []
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta_2 = meta
            else:
                data_meta_2.append(meta)
            line_count += 1
    head_meta_2 = [[i] for i in head_meta_2]
    for data_meta in data_meta_2:
        for i, data in zip(range(len(data_meta)), data_meta):
            head_meta_2[i].append(data)

    meta_dict_2 = dict([(i[0], i[1:]) for i in head_meta_2])

    input_event_ids = meta_dict_2['Input.event_id']
    print len(input_event_ids)

    input_path = csv_1
    line_count = 0
    head_meta_1 = []
    data_meta_1 = []
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta_1 = meta
            line_count += 1

    index_event_id = -np.Inf
    index_event_type = -np.Inf
    i = 0
    for field in head_meta_1:
        if field.startswith('Input.event_id'):
            index_event_id = i
        if field.startswith('Input.event_type'):
            index_event_type = i
        i += 1

    line_count = 0
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count != 0:
                if meta[index_event_id] in input_event_ids:
                    print meta
                else:
                    if meta[index_event_id] in correct_list:
                        meta[index_event_type] = dict_name_reverse[correct_list[meta[index_event_id]]]
                    data_meta_1.append(meta)
            line_count += 1
    head_meta_1 = [[i] for i in head_meta_1]

    for data_meta in data_meta_1:
        for i, data in zip(range(len(data_meta)), data_meta):
            head_meta_1[i].append(data)

    meta_dict_1 = dict([(i[0], i[1:]) for i in head_meta_1])

    merged_meta_dict = dict()
    for head in meta_dict_1:
        if head in meta_dict_2:
            merged_meta_dict[head] = meta_dict_1[head] + meta_dict_2[head]
        else:
            merged_meta_dict[head] = meta_dict_1[head] + ['NA']*len(meta_dict_2[meta_dict_2.keys()[0]])

    print 1 + len(merged_meta_dict[merged_meta_dict.keys()[0]])
    merged_meta = [[] for i in range((1 + len(merged_meta_dict[merged_meta_dict.keys()[0]])))]
    print merged_meta
    for key in head_meta_1:
        merged_meta[0].append(key[0])
    for key in head_meta_1:
        temp = merged_meta_dict[key[0]]
        i = 1
        for data in temp:
            merged_meta[i].append(data)
            i += 1

    with open(out_csv, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in merged_meta:
            writer.writerow(i)



class create_cross_validation_corrected:
    def __init__(self):
        # self.training_test_correction()
        # self.curation_csv_correction()
        # self.correct_amt_result()
        # self.create_new_result()
        # self.create_url_dict()
        for name in dict_name2:
            self.create_csv_traintest(name)
    @staticmethod
    def training_test_correction():
        events_training = defaultdict(list)
        events_imgs_training = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_event_id.cPickle')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs_training[event_this].append(img)
            for event_this in event_ids:
                if event_this in correct_list:
                    events_training[correct_list[event_this]].append(event_this)
                else:
                    events_training[event_name].append(event_this)

        events_test = defaultdict(list)
        events_imgs_test = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/test_event_id.cPickle')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs_test[event_this].append(img)
            for event_this in event_ids:
                if event_this in correct_list:
                    events_test[correct_list[event_this]].append(event_this)
                else:
                    events_test[event_name].append(event_this)


        for event_name in events_training:
            print event_name, len(events_training[event_name]) + len(events_test[event_name])

        for event_name in dict_name2:
            if not os.path.exists(root + 'baseline_all_correction/' + event_name):
                os.mkdir(root + 'baseline_all_correction/' + event_name)
            f = open(root + 'baseline_all_correction/' + event_name + '/training_event_id.cPickle', 'w')
            cPickle.dump(events_training[event_name], f)
            f.close()
            f = open(root + 'baseline_all_correction/' + event_name + '/test_event_id.cPickle', 'w')
            cPickle.dump(events_test[event_name], f)
            f.close()

            f = open(root + 'baseline_all_correction/' + event_name + '/training_image_ids.cPickle','w')
            img_ids = []
            for event_id in events_training[event_name]:
                img_ids.extend(events_imgs_training[event_id])
            cPickle.dump(img_ids, f)
            f.close()
            f = open(root + 'baseline_all_correction/' + event_name + '/test_image_ids.cPickle','w')
            img_ids = []
            for event_id in events_test[event_name]:
                img_ids.extend(events_imgs_test[event_id])
            cPickle.dump(img_ids, f)
            f.close()

        training_paths_dict = defaultdict(list)
        for event_name in dict_name2:
            in_path = root + 'baseline_all_0509/' + event_name + '/guru_training_path.txt'
            with open(in_path, 'r') as data:
                for line in data:
                    img_id = ('/').join(line.split('.')[0].split('/')[-2:])
                    training_paths_dict[img_id] = event_name

        for event_name in dict_name2:
            f = open(root + 'baseline_all_correction/' + event_name + '/training_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            out_path = root + 'baseline_all_correction/' + event_name + '/guru_training_path.txt'
            f = open(out_path, 'w')
            for img_id in img_ids:
                img_id_this = img_id.split('_')[1]
                f.write('/home/feiyu1990/local/event_curation/curation_images/' + training_paths_dict[img_id_this] + '/' + img_id_this + '.jpg 0\n')
            f.close()

        test_paths_dict = defaultdict(list)
        for event_name in dict_name2:
            in_path = root + 'baseline_all_0509/' + event_name + '/guru_test_path.txt'
            with open(in_path, 'r') as data:
                for line in data:
                    img_id = ('/').join(line.split('.')[0].split('/')[-2:])
                    test_paths_dict[img_id] = event_name

        for event_name in dict_name2:
            f = open(root + 'baseline_all_correction/' + event_name + '/test_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            out_path = root + 'baseline_all_correction/' + event_name + '/guru_test_path.txt'
            f = open(out_path, 'w')
            for img_id in img_ids:
                img_id_this = img_id.split('_')[1]
                f.write('/home/feiyu1990/local/event_curation/curation_images/' + test_paths_dict[img_id_this] + '/' + img_id_this + '.jpg 0\n')
            f.close()
    @staticmethod
    def curation_csv_correction():
        events_imgs = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs[event_this].append(img)
            f = open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs[event_this].append(img)


        input_path = root + 'baseline_all_correction/curation_result.csv'
        line_count = 0
        head_meta = []
        HITs = {}
        with open(input_path, 'r') as data:
            reader = csv.reader(data)
            # reader = csv.reader(data, dialect=csv.excel_tab)
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
        index_num_image = -np.Inf
        index_event_id = -np.Inf
        index_distraction = -np.Inf
        for field in head_meta:
            if field.startswith('Input.image'):
                image_input_index[int(field[11:])] = i
            if field.startswith('Answer.image'):
                image_output_index[int(field[12:])] = i
            if field.startswith('Input.distraction_num'):
                index_distraction = i
            if field.startswith('Input.num_image'):
                index_num_image = i
            if field.startswith('Input.event_id'):
                index_event_id = i
            i += 1
        # print head_meta
        # print index_num_image, index_event_id, index_distraction

        input_and_answers = {}

        for HITId in HITs:
            this_hit = HITs[HITId]
            print this_hit[0][index_num_image]
            num_images = int(this_hit[0][index_num_image])
            distract_image = this_hit[0][index_distraction]
            event_id = this_hit[0][index_event_id]
            input_and_answers[event_id] = []
            print distract_image
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
                input_and_answers[event_id].append((image_url, events_imgs[event_id][ii], score))
                ii += 1
        f = open(root + 'baseline_all_correction/correction_result_v1.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
    @staticmethod
    def correct_amt_result():
        f = open(root + 'baseline_all_correction/correction_result_v1.cPickle','r')
        input_and_answers = cPickle.load(f)
        f.close()
        remove_list = []
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/'+event_name+'/vgg_training_similar_list.cPickle','r')
            remove_list.extend(cPickle.load(f))
            f.close()
            f = open(root + 'baseline_all_0509/'+event_name+'/vgg_test_similar_list.cPickle','r')
            remove_list.extend(cPickle.load(f))
            f.close()
        print len(remove_list)

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

        f = open(root + 'baseline_all_correction/correction_result_v2.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
        training_scores_dict = {}
        for event in input_and_answers:
            for img in input_and_answers[event]:
                training_scores_dict[img[1]] = img[2]
        f = open(root + 'baseline_all_correction/correction_result_dict_v2.cPickle', 'wb')
        cPickle.dump(training_scores_dict,f)
        f.close()
    @staticmethod
    def create_new_result():
        all_event_and_score = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_v2.cPickle','r')
            temp1 = cPickle.load(f)
            all_event_and_score.update(temp1)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle','r')
            temp2 = cPickle.load(f)
            all_event_and_score.update(temp2)
            f.close()
            print event_name, len(temp1), len(temp2)
            print len(all_event_and_score)

        f = open(root + 'baseline_all_correction/correction_result_v2.cPickle', 'r')
        training_scores_correction = cPickle.load(f)
        f.close()

        for event_id in training_scores_correction:
            print event_id
            all_event_and_score[event_id] = training_scores_correction[event_id]

        print len(all_event_and_score)
        for event_name in dict_name2:
            event_and_score_this = dict()
            f = open(root + 'baseline_all_correction/' + event_name + '/training_event_id.cPickle', 'r')
            events_training = cPickle.load(f)
            f.close()
            for event in events_training:
                event_and_score_this[event] = all_event_and_score[event]
            f = open(root + 'baseline_all_correction/' + event_name + '/vgg_training_result_v2.cPickle', 'w')
            cPickle.dump(event_and_score_this, f)
            f.close()

            training_scores_dict = {}
            for event in event_and_score_this:
                for img in event_and_score_this[event]:
                    training_scores_dict[img[1]] = img[2]
            f = open(root + 'baseline_all_correction/'+event_name+'/vgg_training_result_dict_v2.cPickle', 'wb')
            cPickle.dump(training_scores_dict,f)
            f.close()


            event_and_score_this = dict()
            f = open(root + 'baseline_all_correction/' + event_name + '/test_event_id.cPickle', 'r')
            events_test = cPickle.load(f)
            f.close()
            for event in events_test:
                event_and_score_this[event] = all_event_and_score[event]
            f = open(root + 'baseline_all_correction/' + event_name + '/vgg_test_result_v2.cPickle', 'w')
            cPickle.dump(event_and_score_this, f)
            f.close()

            test_scores_dict = {}
            for event in event_and_score_this:
                for img in event_and_score_this[event]:
                    test_scores_dict[img[1]] = img[2]
            f = open(root + 'baseline_all_correction/'+event_name+'/vgg_test_result_dict_v2.cPickle', 'wb')
            cPickle.dump(test_scores_dict,f)
            f.close()
    @staticmethod
    def create_url_dict():
        url_dict_all = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_ulr_dict.cPickle','r')
            temp1 = cPickle.load(f)
            url_dict_all.update(temp1)
            f.close()
        f = open(root + baseline_name + 'training_ulr_dict.cPickle','w')
        cPickle.dump(url_dict_all, f)
        f.close()
        url_dict_all = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/test_ulr_dict.cPickle','r')
            temp1 = cPickle.load(f)
            url_dict_all.update(temp1)
            f.close()
        f = open(root + baseline_name + 'test_ulr_dict.cPickle','w')
        cPickle.dump(url_dict_all, f)
        f.close()
    @staticmethod
    def create_csv_traintest(name):
        f = open(root + baseline_name +name+'/training_event_id.cPickle','r')
        event_ids = cPickle.load(f)
        f.close()
        f = open(root + baseline_name +name+'/training.csv','wb')
        writer = csv.writer(f)
        line_count = 0
        input_path = root + 'all_output/all_output_corrected.csv'
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

        f = open(root + baseline_name+name+'/test_event_id.cPickle','r')
        event_ids = cPickle.load(f)
        f.close()
        f = open(root + baseline_name+name+'/test.csv','wb')
        writer = csv.writer(f)
        line_count = 0
        # input_path = root + 'all_output/all_output_corrected.csv'
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

class create_cross_validation_corrected_2round_nomulti:
    def __init__(self):
        self.training_test_correction()
        self.curation_csv_correction()
        self.correct_amt_result()
        self.create_new_result()
        self.create_url_dict()
        for name in dict_name2.keys():
            self.create_csv_traintest(name)
    @staticmethod
    def training_test_correction():
        events_training = defaultdict(list)
        events_imgs_training = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_event_id.cPickle')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs_training[event_this].append(img)
            for event_this in event_ids:
                if event_this in correct_list:
                    events_training[correct_list[event_this]].append(event_this)
                else:
                    events_training[event_name].append(event_this)

        events_test = defaultdict(list)
        events_imgs_test = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/test_event_id.cPickle')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs_test[event_this].append(img)
            for event_this in event_ids:
                if event_this in correct_list:
                    events_test[correct_list[event_this]].append(event_this)
                else:
                    events_test[event_name].append(event_this)


        for event_name in events_training:
            print event_name, len(events_training[event_name]) + len(events_test[event_name])

        for event_name in dict_name2:
            if not os.path.exists(root + baseline_name + event_name):
                os.mkdir(root + baseline_name + event_name)
            f = open(root +baseline_name + event_name + '/training_event_id.cPickle', 'w')
            cPickle.dump(events_training[event_name], f)
            f.close()
            f = open(root + baseline_name + event_name + '/test_event_id.cPickle', 'w')
            cPickle.dump(events_test[event_name], f)
            f.close()

            f = open(root + baseline_name + event_name + '/training_image_ids.cPickle','w')
            img_ids = []
            for event_id in events_training[event_name]:
                img_ids.extend(events_imgs_training[event_id])
            cPickle.dump(img_ids, f)
            f.close()
            f = open(root + baseline_name + event_name + '/test_image_ids.cPickle','w')
            img_ids = []
            for event_id in events_test[event_name]:
                img_ids.extend(events_imgs_test[event_id])
            cPickle.dump(img_ids, f)
            f.close()

        training_paths_dict = defaultdict(list)
        for event_name in dict_name2:
            in_path = root + 'baseline_all_0509/' + event_name + '/guru_training_path.txt'
            with open(in_path, 'r') as data:
                for line in data:
                    img_id = ('/').join(line.split('.')[0].split('/')[-2:])
                    training_paths_dict[img_id] = event_name

        for event_name in dict_name2:
            f = open(root + baseline_name + event_name + '/training_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            out_path = root + baseline_name + event_name + '/guru_training_path.txt'
            f = open(out_path, 'w')
            for img_id in img_ids:
                img_id_this = img_id.split('_')[1]
                f.write('/home/feiyu1990/local/event_curation/curation_images/' + training_paths_dict[img_id_this] + '/' + img_id_this + '.jpg 0\n')
            f.close()

        test_paths_dict = defaultdict(list)
        for event_name in dict_name2:
            in_path = root + 'baseline_all_0509/' + event_name + '/guru_test_path.txt'
            with open(in_path, 'r') as data:
                for line in data:
                    img_id = ('/').join(line.split('.')[0].split('/')[-2:])
                    test_paths_dict[img_id] = event_name

        for event_name in dict_name2:
            f = open(root + baseline_name + event_name + '/test_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            out_path = root + baseline_name + event_name + '/guru_test_path.txt'
            f = open(out_path, 'w')
            for img_id in img_ids:
                img_id_this = img_id.split('_')[1]
                f.write('/home/feiyu1990/local/event_curation/curation_images/' + test_paths_dict[img_id_this] + '/' + img_id_this + '.jpg 0\n')
            f.close()
    @staticmethod
    def curation_csv_correction():
        events_imgs = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs[event_this].append(img)
            f = open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs[event_this].append(img)

        input_and_answers = {}
        csv_list = [
                    root + '0208_correction/all_input_and_result/result_curation_amt_1round.csv',
                    root + '0208_correction/all_input_and_result/result_curation_amt_2round.csv']
        for input_path in csv_list:
            line_count = 0
            head_meta = []
            HITs = {}
            with open(input_path, 'r') as data:
                reader = csv.reader(data)
                # reader = csv.reader(data, dialect=csv.excel_tab)
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
            index_num_image = -np.Inf
            index_event_id = -np.Inf
            index_distraction = -np.Inf
            index_worker_id = -np.Inf
            for field in head_meta:
                if field.startswith('Input.image'):
                    image_input_index[int(field[11:])] = i
                if field.startswith('Answer.image'):
                    image_output_index[int(field[12:])] = i
                if field.startswith('Input.distraction_num'):
                    index_distraction = i
                if field.startswith('Input.num_image'):
                    index_num_image = i
                if field.startswith('Input.event_id'):
                    index_event_id = i
                if field.startswith('WorkerId'):
                    index_worker_id = i
                i += 1

            for HITId in HITs:
                this_hit = HITs[HITId]
                num_images = int(this_hit[0][index_num_image])
                distract_image = this_hit[0][index_distraction]
                event_id = this_hit[0][index_event_id]
                input_and_answers[event_id] = []
                print event_id
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
                    input_and_answers[event_id].append((image_url, events_imgs[event_id][ii], score))
                    ii += 1

        f = open(root  +baseline_name+ '/correction_result_12round_v1.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
    @staticmethod
    def correct_amt_result():
        f = open(root + baseline_name +'/correction_result_12round_v1.cPickle','r')
        input_and_answers = cPickle.load(f)
        f.close()
        remove_list = []
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/'+event_name+'/vgg_training_similar_list.cPickle','r')
            remove_list.extend(cPickle.load(f))
            f.close()
            f = open(root + 'baseline_all_0509/'+event_name+'/vgg_test_similar_list.cPickle','r')
            remove_list.extend(cPickle.load(f))
            f.close()
        print len(remove_list)

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

        f = open(root + baseline_name +'/correction_result_12round_v2.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
        training_scores_dict = {}
        for event in input_and_answers:
            for img in input_and_answers[event]:
                training_scores_dict[img[1]] = img[2]
        f = open(root + baseline_name +'/correction_result_12round_dict_v2.cPickle', 'wb')
        cPickle.dump(training_scores_dict,f)
        f.close()
    @staticmethod
    def create_new_result():
        all_event_and_score = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_v2.cPickle','r')
            temp1 = cPickle.load(f)
            all_event_and_score.update(temp1)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle','r')
            temp2 = cPickle.load(f)
            all_event_and_score.update(temp2)
            f.close()
            print event_name, len(temp1), len(temp2)
            print len(all_event_and_score)

        f = open(root + baseline_name +'/correction_result_12round_v2.cPickle', 'r')
        training_scores_correction = cPickle.load(f)
        f.close()

        for event_id in training_scores_correction:
            print event_id
            all_event_and_score[event_id] = training_scores_correction[event_id]
        with open(root + baseline_name +'/correction_result_all_v2.pkl', 'w') as f:
            cPickle.dump(all_event_and_score, f)

        print len(all_event_and_score)
        for event_name in dict_name2:
            event_and_score_this = dict()
            f = open(root + baseline_name +'/' + event_name + '/training_event_id.cPickle', 'r')
            events_training = cPickle.load(f)
            f.close()
            for event in events_training:
                event_and_score_this[event] = all_event_and_score[event]
            f = open(root + baseline_name +'/' + event_name + '/vgg_training_result_v2.cPickle', 'w')
            cPickle.dump(event_and_score_this, f)
            f.close()

            training_scores_dict = {}
            for event in event_and_score_this:
                for img in event_and_score_this[event]:
                    training_scores_dict[img[1]] = img[2]
            f = open(root + baseline_name +'/'+event_name+'/vgg_training_result_dict_v2.cPickle', 'wb')
            cPickle.dump(training_scores_dict,f)
            f.close()
        for event_name in dict_name2.keys():
            event_and_score_this = dict()
            f = open(root + baseline_name +'/' + event_name + '/test_event_id.cPickle', 'r')
            events_test = cPickle.load(f)
            f.close()
            for event in events_test:
                try:
                    event_and_score_this[event] = all_event_and_score[event]
                except:
                    event_and_score_this[event[0]] = all_event_and_score[event[0]]
            f = open(root + baseline_name +'/' + event_name + '/vgg_test_result_v2.cPickle', 'w')
            cPickle.dump(event_and_score_this, f)
            f.close()

            test_scores_dict = {}
            for event in event_and_score_this:
                for img in event_and_score_this[event]:
                    test_scores_dict[img[1]] = img[2]
            f = open(root + baseline_name +'/'+event_name+'/vgg_test_result_dict_v2.cPickle', 'wb')
            cPickle.dump(test_scores_dict,f)
            f.close()
    @staticmethod
    def create_url_dict():
        url_dict_all = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_ulr_dict.cPickle','r')
            temp1 = cPickle.load(f)
            url_dict_all.update(temp1)
            f.close()
        f = open(root + baseline_name + 'training_ulr_dict.cPickle','w')
        cPickle.dump(url_dict_all, f)
        f.close()
        url_dict_all = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/test_ulr_dict.cPickle','r')
            temp1 = cPickle.load(f)
            url_dict_all.update(temp1)
            f.close()
        f = open(root + baseline_name + 'test_ulr_dict.cPickle','w')
        cPickle.dump(url_dict_all, f)
        f.close()
    @staticmethod
    def create_csv_traintest(name):
            input_path = root + 'all_output/all_output_corrected_2round.csv'
            f = open(root + baseline_name+name+'/test_event_id.cPickle','r')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + baseline_name+name+'/test.csv','wb')
            writer = csv.writer(f)
            line_count = 0
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

            f = open(root + baseline_name +name+'/training_event_id.cPickle','r')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + baseline_name +name+'/training.csv','wb')
            writer = csv.writer(f)
            line_count = 0
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


class create_cross_validation_corrected_2round:
    def __init__(self):
        self.training_test_correction()
        self.curation_csv_correction()
        self.correct_amt_result()
        self.create_new_result()
        self.create_url_dict()
        for name in dict_name2.keys() + ['multi_label']:
            self.create_csv_traintest(name)
    @staticmethod
    def training_test_correction():
        with open(root + '0208_correction/all_input_and_result/'+ground_truth_dict_name) as f:
            new_multiple_result = cPickle.load(f)

        events_training = defaultdict(list)
        events_imgs_training = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_event_id.cPickle')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                # multi_label = new_multiple_result[event_this]
                events_imgs_training[event_this].append(img)
            for event_this in event_ids:
                multi_label = new_multiple_result[event_this]
                for event in multi_label:
                    events_training[event[0]].append(event_this)


        events_test = defaultdict(list)
        events_imgs_test = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/test_event_id.cPickle')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                # multi_label = new_multiple_result[event_this]
                # if len(multi_label) == 1:
                #     events_imgs_test[multi_label[0][0]].append(img)
                # else:
                events_imgs_test[event_this].append(img)
            for event_this in event_ids:
                multi_label = new_multiple_result[event_this]
                if len(multi_label) == 1:
                    events_test[multi_label[0][0]].append(event_this)
                else:
                    events_test['multi_label'].append([event_this, multi_label])
        print events_imgs_test.keys()
        print events_imgs_training.keys()


        for event_name in events_training:
            print event_name, len(events_training[event_name]) + len(events_test[event_name])

        for event_name in events_training:
            if not os.path.exists(root + baseline_name + '/' + event_name):
                os.mkdir(root + baseline_name + '/' + event_name)
            f = open(root + baseline_name +'/' + event_name + '/training_event_id.cPickle', 'w')
            cPickle.dump(events_training[event_name], f)
            f.close()
            f = open(root + baseline_name +'/' + event_name + '/training_image_ids.cPickle','w')
            img_ids = []
            for event_id in events_training[event_name]:
                img_ids.extend(events_imgs_training[event_id])
            cPickle.dump(img_ids, f)
            f.close()

        for event_name in events_test:
            print event_name
            if not os.path.exists(root + baseline_name +'/' + event_name):
                os.mkdir(root + baseline_name +'/' + event_name)
            f = open(root + baseline_name +'/' + event_name + '/test_event_id.cPickle', 'w')
            cPickle.dump(events_test[event_name], f)
            f.close()
            f = open(root + baseline_name +'/' + event_name + '/test_image_ids.cPickle','w')
            img_ids = []
            for event_id in events_test[event_name]:
                try:
                    img_ids.extend(events_imgs_test[event_id])
                except:
                    # print event_id
                    img_ids.extend(events_imgs_test[event_id[0]])
                    # print events_imgs_test[event_id[0]]
            cPickle.dump(img_ids, f)
            f.close()

        training_paths_dict = defaultdict(list)
        for event_name in dict_name2:
            in_path = root + 'baseline_all_0509/' + event_name + '/guru_training_path.txt'
            with open(in_path, 'r') as data:
                for line in data:
                    img_id = ('/').join(line.split('.')[0].split('/')[-2:])
                    training_paths_dict[img_id] = event_name

        for event_name in dict_name2:
            f = open(root + baseline_name +'/' + event_name + '/training_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            out_path = root + baseline_name +'/' + event_name + '/guru_training_path.txt'
            f = open(out_path, 'w')
            for img_id in img_ids:
                img_id_this = img_id.split('_')[1]
                f.write('/home/feiyu1990/local/event_curation/curation_images/' + training_paths_dict[img_id_this] + '/' + img_id_this + '.jpg 0\n')
            f.close()

        test_paths_dict = defaultdict(list)
        for event_name in dict_name2:
            in_path = root + 'baseline_all_0509/' + event_name + '/guru_test_path.txt'
            with open(in_path, 'r') as data:
                for line in data:
                    img_id = ('/').join(line.split('.')[0].split('/')[-2:])
                    test_paths_dict[img_id] = event_name

        for event_name in events_test:
            f = open(root + baseline_name +'/' + event_name + '/test_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            out_path = root + baseline_name +'/' + event_name + '/guru_test_path.txt'
            f = open(out_path, 'w')
            for img_id in img_ids:
                img_id_this = img_id.split('_')[1]
                f.write('/home/feiyu1990/local/event_curation/curation_images/' + test_paths_dict[img_id_this] + '/' + img_id_this + '.jpg 0\n')
            f.close()
    @staticmethod
    def curation_csv_correction():
        events_imgs = defaultdict(list)
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs[event_this].append(img)
            f = open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
            img_ids = cPickle.load(f)
            f.close()
            for img in img_ids:
                event_this = img.split('/')[0]
                events_imgs[event_this].append(img)

        input_and_answers = {}
        csv_list = [
                    root + '0208_correction/all_input_and_result/result_curation_amt_1round.csv',
                    root + '0208_correction/all_input_and_result/result_curation_amt_2round.csv']
        for input_path in csv_list:
            line_count = 0
            head_meta = []
            HITs = {}
            with open(input_path, 'r') as data:
                reader = csv.reader(data)
                # reader = csv.reader(data, dialect=csv.excel_tab)
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
            index_num_image = -np.Inf
            index_event_id = -np.Inf
            index_distraction = -np.Inf
            index_worker_id = -np.Inf
            for field in head_meta:
                if field.startswith('Input.image'):
                    image_input_index[int(field[11:])] = i
                if field.startswith('Answer.image'):
                    image_output_index[int(field[12:])] = i
                if field.startswith('Input.distraction_num'):
                    index_distraction = i
                if field.startswith('Input.num_image'):
                    index_num_image = i
                if field.startswith('Input.event_id'):
                    index_event_id = i
                if field.startswith('WorkerId'):
                    index_worker_id = i
                i += 1

            for HITId in HITs:
                this_hit = HITs[HITId]
                num_images = int(this_hit[0][index_num_image])
                distract_image = this_hit[0][index_distraction]
                event_id = this_hit[0][index_event_id]
                input_and_answers[event_id] = []
                print event_id
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
                    input_and_answers[event_id].append((image_url, events_imgs[event_id][ii], score))
                    ii += 1

        f = open(root + baseline_name +'/correction_result_12round_v1.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
    @staticmethod
    def correct_amt_result():
        f = open(root + baseline_name +'/correction_result_12round_v1.cPickle','r')
        input_and_answers = cPickle.load(f)
        f.close()
        remove_list = []
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/'+event_name+'/vgg_training_similar_list.cPickle','r')
            remove_list.extend(cPickle.load(f))
            f.close()
            f = open(root + 'baseline_all_0509/'+event_name+'/vgg_test_similar_list.cPickle','r')
            remove_list.extend(cPickle.load(f))
            f.close()
        print len(remove_list)

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

        f = open(root + baseline_name +'/correction_result_12round_v2.cPickle','wb')
        cPickle.dump(input_and_answers, f)
        f.close()
        training_scores_dict = {}
        for event in input_and_answers:
            for img in input_and_answers[event]:
                training_scores_dict[img[1]] = img[2]
        f = open(root + baseline_name +'/correction_result_12round_dict_v2.cPickle', 'wb')
        cPickle.dump(training_scores_dict,f)
        f.close()
    @staticmethod
    def create_new_result():
        all_event_and_score = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_training_result_v2.cPickle','r')
            temp1 = cPickle.load(f)
            all_event_and_score.update(temp1)
            f.close()
            f = open(root + 'baseline_all_0509/' + event_name + '/vgg_test_result_v2.cPickle','r')
            temp2 = cPickle.load(f)
            all_event_and_score.update(temp2)
            f.close()
            print event_name, len(temp1), len(temp2)
            print len(all_event_and_score)

        f = open(root + baseline_name +'/correction_result_12round_v2.cPickle', 'r')
        training_scores_correction = cPickle.load(f)
        f.close()

        for event_id in training_scores_correction:
            print event_id
            all_event_and_score[event_id] = training_scores_correction[event_id]
        with open(root + baseline_name +'/correction_result_all_v2.pkl', 'w') as f:
            cPickle.dump(all_event_and_score, f)

        print len(all_event_and_score)
        for event_name in dict_name2:
            event_and_score_this = dict()
            f = open(root + baseline_name +'/' + event_name + '/training_event_id.cPickle', 'r')
            events_training = cPickle.load(f)
            f.close()
            for event in events_training:
                event_and_score_this[event] = all_event_and_score[event]
            f = open(root + baseline_name +'/' + event_name + '/vgg_training_result_v2.cPickle', 'w')
            cPickle.dump(event_and_score_this, f)
            f.close()

            training_scores_dict = {}
            for event in event_and_score_this:
                for img in event_and_score_this[event]:
                    training_scores_dict[img[1]] = img[2]
            f = open(root + baseline_name +'/'+event_name+'/vgg_training_result_dict_v2.cPickle', 'wb')
            cPickle.dump(training_scores_dict,f)
            f.close()
        for event_name in dict_name2.keys() + ['multi_label']:
            event_and_score_this = dict()
            f = open(root + baseline_name +'/' + event_name + '/test_event_id.cPickle', 'r')
            events_test = cPickle.load(f)
            f.close()
            for event in events_test:
                try:
                    event_and_score_this[event] = all_event_and_score[event]
                except:
                    event_and_score_this[event[0]] = all_event_and_score[event[0]]
            f = open(root + baseline_name +'/' + event_name + '/vgg_test_result_v2.cPickle', 'w')
            cPickle.dump(event_and_score_this, f)
            f.close()

            test_scores_dict = {}
            for event in event_and_score_this:
                for img in event_and_score_this[event]:
                    test_scores_dict[img[1]] = img[2]
            f = open(root + baseline_name +'/'+event_name+'/vgg_test_result_dict_v2.cPickle', 'wb')
            cPickle.dump(test_scores_dict,f)
            f.close()
    @staticmethod
    def create_url_dict():
        url_dict_all = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/training_ulr_dict.cPickle','r')
            temp1 = cPickle.load(f)
            url_dict_all.update(temp1)
            f.close()
        f = open(root + baseline_name + 'training_ulr_dict.cPickle','w')
        cPickle.dump(url_dict_all, f)
        f.close()
        url_dict_all = dict()
        for event_name in dict_name2:
            f = open(root + 'baseline_all_0509/' + event_name + '/test_ulr_dict.cPickle','r')
            temp1 = cPickle.load(f)
            url_dict_all.update(temp1)
            f.close()
        f = open(root + baseline_name + 'test_ulr_dict.cPickle','w')
        cPickle.dump(url_dict_all, f)
        f.close()
    @staticmethod
    def create_csv_traintest(name):
        if name == 'multi_label':
            input_path = root + 'all_output/all_output_corrected_2round.csv'
            f = open(root + baseline_name+name+'/test_event_id.cPickle','r')
            event_ids = [i[0] for i in cPickle.load(f)]
            f.close()
            f = open(root + baseline_name+name+'/test.csv','wb')
            writer = csv.writer(f)
            line_count = 0
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
        else:
            input_path = root + 'all_output/all_output_corrected_2round.csv'
            f = open(root + baseline_name+name+'/test_event_id.cPickle','r')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + baseline_name+name+'/test.csv','wb')
            writer = csv.writer(f)
            line_count = 0
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

            f = open(root + baseline_name +name+'/training_event_id.cPickle','r')
            event_ids = cPickle.load(f)
            f.close()
            f = open(root + baseline_name +name+'/training.csv','wb')
            writer = csv.writer(f)
            line_count = 0
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

class create_CNN_training_prototxts_corrected(object):
    def __init__(self, threshold, val_name, folder_name, oversample_n = 3, oversample_threshold = 0.3, oversample= False):
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
                self.guru_find_valid_examples_all_reallabel_traintest(oversample)
                self.merge_all_examples_traintest()
                # self.merge_all_examples_traintest_subcategory()
                self.create_label_txt_traintest()
    def guru_find_valid_examples_all_reallabel(self, val_id):

        for event_name in dict_name2:
            f = open(root + baseline_name + event_name + '/'+'validation_' + str(val_id)+'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_correction/' + event_name + '/'+'validation_' + str(val_id)+'/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_correction/' + event_name + '/vgg_training_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()


            in_path = root + 'baseline_all_correction/' + event_name + '/'+'validation_' + str(val_id)+'/guru_'+self.val_name+'_path.txt'
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
            f = open(root + 'baseline_all_correction/' + event_name +'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_correction/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + 'baseline_all_correction/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()

            in_path = root + 'baseline_all_correction/' + event_name + '/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

            img_path_dict = {}
            for (i,j) in zip(img_ids, img_paths):
                if j.split('.')[0].split('/')[-1] != i.split('/')[-1]:
                    print 'ERROR!'
                    return
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
            if self.val_name == 'test':
                img_pair = img_pair[:5000]
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


class create_CNN_training_prototxts_corrected_2round(object):
    def __init__(self, threshold, val_name, folder_name, oversample_n = 3, oversample_threshold = 0.3, oversample= False):
        self.threshold = threshold
        self.val_name = val_name
        self.folder_name = folder_name
        self.oversample_n = oversample_n
        self.oversample_threshold = oversample_threshold
        if self.threshold == 0:
            self.threshold_prefix = ''
        else:
            self.threshold_prefix = str(self.threshold) + '_'
        self.guru_find_valid_examples_all_reallabel_traintest(oversample)
        self.merge_all_examples_traintest()
        self.create_label_txt_traintest()


    def guru_find_valid_examples_all_reallabel_traintest(self, oversample):
        for event_name in dict_name2:
            f = open(root + baseline_name + '/' + event_name +'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + baseline_name + '/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + baseline_name + '/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()

            in_path = root + baseline_name + '/' + event_name + '/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

            img_path_dict = {}
            for (i,j) in zip(img_ids, img_paths):
                if j.split('.')[0].split('/')[-1] != i.split('/')[-1]:
                    print 'ERROR!'
                    return
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
            if self.val_name == 'test':
                img_pair = img_pair[:5000]
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


class create_CNN_training_prototxts_corrected_soft(object):

    def __init__(self, threshold, val_name, folder_name, oversample_n = 3, oversample_threshold = 0.3, oversample= False):
        self.threshold = threshold
        self.val_name = val_name
        self.folder_name = folder_name
        self.oversample_n = oversample_n
        self.oversample_threshold = oversample_threshold
        if self.threshold == 0:
            self.threshold_prefix = ''
        else:
            self.threshold_prefix = str(self.threshold) + '_'
        self.guru_find_valid_examples_all_reallabel_traintest(oversample)
        self.merge_all_examples_traintest()
        self.create_label_txt_traintest()


    def guru_find_valid_examples_all_reallabel_traintest(self, oversample):
        print baseline_name
        for event_name in dict_name2:
            f = open(root + baseline_name + '/' + event_name +'/vgg_'+self.val_name+'_result_v2.cPickle','r')
            ground_truth_training = cPickle.load(f)
            f.close()

            f = open(root + baseline_name + '/' + event_name + '/'+self.val_name+'_image_ids.cPickle','r')
            img_ids = cPickle.load(f)
            f.close()

            f = open(root + baseline_name + '/' + event_name + '/vgg_'+self.val_name+'_result_dict_v2.cPickle','r')
            ground_truth_training_dict = cPickle.load(f)
            f.close()

            in_path = root + baseline_name + '/' + event_name + '/guru_'+self.val_name+'_path.txt'
            img_paths = []
            with open(in_path,'r') as data:
                for line in data:
                    temp = line.split(' ')[0]
                    img_paths.append(temp)

            img_path_dict = {}
            for (i,j) in zip(img_ids, img_paths):
                if j.split('.')[0].split('/')[-1] != i.split('/')[-1]:
                    print 'ERROR!'
                    return
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
            if self.val_name == 'test':
                img_pair = img_pair[:5000]
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
            print count_all
    def merge_all_examples_traintest(self):
        with open(root + '0208_correction/all_input_and_result/'+ground_truth_dict_name) as f:
            event_type_dict = cPickle.load(f)

        img_event_dict = dict()
        for event_name in dict_name2:
            with open(root + baseline_name +event_name+ '/training_image_ids.cPickle') as f:
                img_event_list = cPickle.load(f)
            in_file_name = root + baseline_name +event_name+'/guru_training_path.txt'
            count = 0
            with open(in_file_name, 'r') as data:
                for line in data:
                    event_this = img_event_list[count].split('/')[0]
                    img_event_dict[line.split(' ')[0]] = event_this
                    count += 1

        img_type_dict = dict()
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    softmax_this = np.zeros((23, ))
                    event_this = img_event_dict[line.split(' ')[0]]
                    event_type_list = event_type_dict[event_this]
                    for event_ in event_type_list:
                        softmax_this[dict_name2[event_[0]] - 1] = event_[1]
                    img_type_dict[line.split(' ')[0]] = softmax_this

        imgs = []
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs.append([(meta + ' ', img_type_dict[meta.split(' ')[0]])])
        count = 0
        for event_name in dict_name2:
            in_file_name = root + self.folder_name+'/data/' +event_name+'_ranking_reallabel_'+self.val_name+'_p.txt'
            with open(in_file_name, 'r') as data:
                for line in data:
                    meta = line[:-1]
                    imgs[count].append((meta + ' ', img_type_dict[meta.split(' ')[0]]))
                    count += 1


        random.shuffle(imgs)
        if self.val_name == 'test':
            imgs = imgs[:5000]
        f1 = open(root + self.folder_name+'/data/'+ 'ranking_reallabel_'+self.val_name+'_all.txt', 'w')
        f2 = open(root + self.folder_name+'/data/'+'ranking_reallabel_'+self.val_name+'_all_p.txt', 'w')
        for i,j in imgs:
            f1.write(i[0] + '\n')
            f2.write(j[0] + '\n')
        f1.close()
        f2.close()

        event_labels = []
        for i,j in imgs:
            event_labels.append(j[1])
        event_labels = np.array(event_labels)
        f = h5py.File(root + self.folder_name+'/data/' +self.val_name+'_label.h5','w')
        f.create_dataset("event_label", data=event_labels)
        f.close()


    def create_label_txt_traintest(self):
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
    def extract_feature_10_23_traintest(self):
        imgs = []
        img_file = '/home/feiyu1990/local/event_curation/baseline_all_correction/'+self.event_name+'/guru_'+self.name+'_path.txt'
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

def create_expand_training_list(path):

    training_paths_dict = dict()
    for event_name in dict_name2:
        in_path = root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle'
        with open(in_path) as f:
            img_list = cPickle.load(f)
        file_list = []
        with open(root + 'baseline_all_0509/' + event_name + '/guru_training_path.txt') as f:
            for line in f:
                file_list.append(line.split(' ')[0])
        for img, img_path in zip(img_list, file_list):
            event_img = img_path.split(' ')[0]
            training_paths_dict['/'.join(event_img.split('/')[-2:])] = img
    with open(root + '0208_correction/all_input_and_result/'+ground_truth_dict_name) as f:
        event_type_dict = cPickle.load(f)

    img_path = '/home/feiyu1990/local/event_curation/'+path+'/data/ranking_reallabel_training_all.txt'
    img_path_p = '/home/feiyu1990/local/event_curation/'+path+'/data/ranking_reallabel_training_all_p.txt'
    f = h5py.File('/home/feiyu1990/local/event_curation/'+path+'/data/training_label.h5','r')
    label = f['event_label'][:]
    f.close()

    img_list = []; img_list_p = []
    with open(img_path) as data:
        for line in data:
            img_list.append(line)
    with open(img_path_p) as data:
        for line in data:
            img_list_p.append(line)

    img_list_new = []; img_list_new_p = []; label_new = []
    for img, img_p, label in zip(img_list, img_list_p, label):
        event_img = img.split(' ')[0]
        event_this = training_paths_dict['/'.join(event_img.split('/')[-2:])].split('/')[0]
        event_type_this = event_type_dict[event_this]
        if len(event_type_this) == 1:
            img_list_new.append(img);img_list_new_p.append(img_p);label_new.append(label)
            img_list_new.append(img);img_list_new_p.append(img_p);label_new.append(label)
        else:
            img_list_new.append(img);img_list_new_p.append(img_p);label_new.append(label)

    ind = range(len(img_list_new))
    random.shuffle(ind)
    img_list_new = [img_list_new[i] for i in ind]
    img_list_new_p = [img_list_new_p[i] for i in ind]
    label_new = [label_new[i] for i in ind]

    with open('/home/feiyu1990/local/event_curation/'+path+'/data/balance_ranking_reallabel_training_all.txt', 'w') as f:
        for line in img_list_new:
            f.write(line)
    with open('/home/feiyu1990/local/event_curation/'+path+'/data/balance_ranking_reallabel_training_all_p.txt', 'w') as f:
        for line in img_list_new_p:
            f.write(line)
    event_labels = np.array(label_new)

    f = h5py.File('/home/feiyu1990/local/event_curation/'+path+'/data/balance_training_label.h5','w')
    f.create_dataset("event_label", data=event_labels)
    f.close()


class evaluation:
    def __init__(self, net_path, type, validation_name, val_id, face_type = None):
        self.net_path = net_path
        self.validation_name = validation_name
        self.val_id = val_id
        if 'val' in self.validation_name:
            self.validation_path = '/validation_' + str(val_id) + '/'
        else:
            self.validation_path = '/'
        #self.evaluate_models = evaluate_models

        if type == 'worker_nonoverlap':
            f = open(root + baseline_name+'test_event_abandoned.pkl','r')
            self.abandoned_test = cPickle.load(f)
            f.close()
            self.evaluate_present_with_worker_nooverlap(True)
        if type == 'worker':
            self.evaluate_present_with_worker_nooverlap(False)
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
        path = root+baseline_name + event_name+validation_path +'_image_ids.cPickle'

        #print path
        f = open(path, 'r')
        all_event_ids = cPickle.load(f)
        f.close()
        f = open(mat_name + '.cPickle', 'r')
        predict_score = cPickle.load(f)
        f.close()

        f = open(root + baseline_name +validation_path +'_ulr_dict.cPickle', 'r')
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

    def evaluate_present_with_worker_nooverlap(self, abandon_overlap):
        print 'HERE!!!'
        # model_names = [['CNN_all_event/'+'/validation_'+str(self.val_id)+'/features/']]*len(self.evaluate_models)
        # for i in xrange(len(model_names)):
        #    model_names[i].append(self.evaluate_models[i])
        model_names = [

                       ['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_diffweight_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_diffweight_iter_200000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_fc500_diffweight_2_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_diffweight_2time_iter_200000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_fc500_diffweight_iter_100000_dict.cPickle']

                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_2time_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_2time_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_twoloss_fc300_diffweight_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_1_segment_noevent_iter_100000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_fromnoevent_iter_200000_dict.cPickle']
                       ,['CNN_all_event_corrected/features/', '_test_sigmoid9_23_segment_iter_100000_dict.cPickle']

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
        # model_names_to_combine = [
        #                 ['CNN_all_event_1009/features/', '_test_sigmoid9_10_segment_fromnoevent_iter_100000.cPickle']
        #                 ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_euclidean_nosigmoid_iter_10000.cPickle']
        # #                 ,['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_fromnoevent_3_iter_100000.cPickle']
        #                ]

        model_names_to_combine = [['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_twoloss_fc300_diffweight_iter_100000.cPickle'],
                                  ['CNN_all_event_1009/features/','_test_sigmoid9_10_segment_twoloss_fc300_iter_100000.cPickle']
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



if __name__ == '__main__':
    TIE = True
    aa = create_cross_validation_corrected_2round()
    # b = create_CNN_training_prototxts_corrected_2round(threshold = 0.0, val_name = 'training', folder_name='CNN_all_event_corrected_multi_removedup_vote/', oversample = False)
    # b = create_CNN_training_prototxts_corrected_soft(threshold = 0.0, val_name = 'training', folder_name='CNN_all_event_corrected_multi_removedup_vote_soft/', oversample = False)
    # b = create_CNN_training_prototxts_corrected_2round(threshold = 0.0, val_name = 'test', folder_name='CNN_all_event_corrected_multi_removedup_weighted/', oversample = False)
    # create_expand_training_list('CNN_all_event_corrected_multi_removedup_vote')



    # b = create_CNN_training_prototxts_corrected(threshold = 0.0, val_name = 'test', folder_name='CNN_all_event_corrected/', oversample = False)
    # for event_name in dict_name2:
        # a = extract_features('CNN_all_event_corrected', event_name, 'segment_fromnoevent_iter_200000','python_deploy_siamese_fromnoevent.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()
        # a = extract_features('CNN_all_event_corrected', event_name, 'segment_iter_100000','python_deploy_siamese_old.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()
    #     a = extract_features('CNN_all_event_corrected', event_name, 'segment_twoloss_diffweight_2time_iter_200000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
    #     a.extract_feature_10_23_traintest()

        # a = extract_features('CNN_all_event_corrected', event_name, 'segment_twoloss_fc500_diffweight_2_iter_100000','python_deploy_siamese_twoloss.prototxt','test', None, (256, 256))
        # a.extract_feature_10_23_traintest()


    # a = evaluation('CNN_all_event_corrected','worker', 'test',0)

    #
    # list_csv = [
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/0.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/1.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/2.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/3.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/4.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/5.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/6.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/7.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/8_pre.csv']
    # list_csv2 = [
    #     '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_rejected_redo.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_rejected_redo_2round.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_all_recognition_amt_urlcorrected_followup_first200.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
    #     '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_all_recognition_amt_urlcorrected_first200.csv',
    #
    # ]
    # # merge_two_csv(list_csv[0], list_csv[1], '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_recognition_combined.csv')
    # #
    # #
    # # for i in range(2, len(list_csv)):
    # #     merge_two_csv(list_csv[i],
    # #                   '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_recognition_combined.csv',
    # #                   '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_recognition_combined.csv')
    #
    # event_worker_dict = defaultdict(list)
    # for input_path in list_csv:
    #     print input_path
    #     head_meta = []
    #     HITs = []
    #     with open(input_path, 'r') as data:
    #         reader = csv.reader(data)
    #         line_count = 0
    #         for meta in reader:
    #             if line_count == 0:
    #                 head_meta = meta
    #             else:
    #                 HITs.append(meta)
    #             line_count += 1
    #     i = 0
    #     worker_indx = -1
    #     event_indx = np.ones((10,)) * -1
    #     for field in head_meta:
    #         if 'Input.event_id' in field:
    #             try:
    #                 this_event_indx = int(field.split('id_')[-1]) - 1
    #                 event_indx[this_event_indx] = i
    #             except:
    #                 this_event_indx = int(field.split('id')[-1]) - 1
    #                 event_indx[this_event_indx] = i
    #         if 'WorkerId' == field:
    #             worker_indx = i
    #         i += 1
    #     print event_indx
    #
    #     for i in HITs:
    #         for j in range(10):
    #             event_worker_dict[i[int(event_indx[j])]].append(i[worker_indx])
    #
    #
    # event_worker_dict2 = defaultdict(list)
    # for input_path in list_csv2:
    #     print input_path
    #     head_meta = []
    #     HITs = []
    #     with open(input_path, 'r') as data:
    #         reader = csv.reader(data)
    #         line_count = 0
    #         for meta in reader:
    #             if line_count == 0:
    #                 head_meta = meta
    #             else:
    #                 HITs.append(meta)
    #             line_count += 1
    #     i = 0
    #     worker_indx = -1
    #     event_indx = np.ones((10,)) * -1
    #     for field in head_meta:
    #         if 'Input.event_id' in field:
    #             try:
    #                 this_event_indx = int(field.split('id_')[-1]) - 1
    #                 event_indx[this_event_indx] = i
    #             except:
    #                 this_event_indx = int(field.split('id')[-1]) - 1
    #                 event_indx[this_event_indx] = i
    #         if 'WorkerId' == field:
    #             worker_indx = i
    #         i += 1
    #     print event_indx
    #
    #     for i in HITs:
    #         for j in range(10):
    #             event_worker_dict2[i[int(event_indx[j])]].append(i[worker_indx])
    #
    # event_need_me_to_do = [];event_need_3_worker = []
    # count = 0
    # for event_name in event_worker_dict2:
    #     worker_id = event_worker_dict[event_name]
    #     worker_id2 = event_worker_dict2[event_name]
    #     multiple_id = []
    #     for i in worker_id:
    #         if i in worker_id2:
    #             multiple_id.append(i)
    #     if len(worker_id) + len(set(worker_id2)) - len(multiple_id) < 12 and len(worker_id) > 0:
    #         print event_name, multiple_id,  worker_id, worker_id2
    #         event_need_me_to_do.append(event_name)
    #         count += 1
    #     if len(worker_id) == 0:
    #         event_need_3_worker.append(event_name)
    # print count
    # print len(event_need_3_worker), event_need_3_worker
    # print event_need_me_to_do
    # f = open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result.pkl')
    # event_type_dict = cPickle.load(f)
    # for event in event_need_3_worker:
    #     print event, event_type_dict[event]