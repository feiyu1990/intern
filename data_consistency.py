__author__ = 'wangyufei'

import csv
import os
import cPickle
import copy
import random
import numpy as np
import scipy.stats
import shutil
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata, chi2
from collections import Counter
#block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
block_workers = []
# root = '/home/feiyu1990/local/event_curation/'
root = '/Users/wangyufei/Documents/Study/intern_adobe/'
from PIL import Image
import sys
sys.path.append(root + 'codes/Krippendorff_alpha/')
import stats
from stats.Agreement import *
import scipy.stats as ss


permute_test_ws = {}
dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

dict_name = {'Theme park':'ThemePark', 'Urban/City trip':'UrbanTrip', 'Beach trip':'BeachTrip', 'Nature trip':'NatureTrip',
             'Zoo/Aquarium/Botanic garden':'Zoo','Cruise trip':'Cruise','Show (air show/auto show/music show/fashion show/concert/parade etc.)':'Show',
            'Sports game':'Sports','Personal sports':'PersonalSports','Personal art activities':'PersonalArtActivity',
            'Personal music activities':'PersonalMusicActivity','Religious activities':'ReligiousActivity',
            'Group activities (party etc.)':'GroupActivity','Casual family/friends gathering':'CasualFamilyGather',
            'Business activity (conference/meeting/presentation etc.)':'BusinessActivity','Independence Day':'Independence',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture/Art':'Architecture'}

def worker_refine():
    input_path = root + 'all_output/all_output.csv'
    out_path = root + 'all_output/all_output_cleaned.csv'
    out_file = open(out_path, 'wb')
    out_data = csv.writer(out_file)
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
                out_data.writerow(meta)
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
    for event_name in dict_name2:
        input_and_answers[event_name] = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_event_type = 29
    index_distraction = 31
    for HITId in HITs:
        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        try:
            event_type = dict_name[this_hit[0][index_event_type]]
        except:
            event_type = this_hit[0][index_event_type]
        if event_type == 'Independence':
            continue
        event_id = this_hit[0][index_event_id]
        this_event_dict = input_and_answers[event_type]
        this_event_dict[event_id] = {}
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = this_hit
        #for submission in this_hit:
            #if submission[index_worker_id] not in block_workers:
            #    this_hit_new.append(submission)
        #num_valid_submission = len(this_hit_new)
        for k in xrange(len(this_hit_new)):
            submission = this_hit_new[k]
            this_event_dict[event_id][k] = []
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    #print this_hit_new[0][image_input_index[i]]
                    continue
                score = 0
                image_index = image_input_index[i]
                score_index = image_output_index[i]
                image_url = this_hit_new[0][image_index]
            #for submission in this_hit_new:
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
                this_event_dict[event_id][k].append(score)

        this_dict = this_event_dict[event_id]
        len_rater = len(this_dict)
        correlations_this = []
        for i in this_dict:
                    except_this = [this_dict[j] for j in xrange(len_rater) if i != j]
                    rho, p = spearmanr(this_dict[i], np.sum(except_this, axis=0))
                    correlations_this.append(rho)
        if np.min(correlations_this) < 0.2:
                    this_dict.pop(np.argmin(correlations_this))
                    correlations_this = []
                    for i in this_dict:
                        except_this = [this_dict[j] for j in this_dict if i != j]
                        rho, p = spearmanr(this_dict[i], np.sum(except_this, axis=0))
                        correlations_this.append([i, rho])
                    correlations_this.sort(key=lambda x: x[1])
                    if correlations_this[0][1] < 0.2:
                        this_dict.pop(correlations_this[0][0])
        for i in this_dict:
            meta_this = this_hit[i]
            out_data.writerow(meta_this)
    out_file.close()
def evaluate_worker_agreement():
    input_path = root + 'all_output/all_output.csv'
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
    for event_name in dict_name2:
        input_and_answers[event_name] = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_event_type = 29
    index_distraction = 31
    for HITId in HITs:

        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        try:
            event_type = dict_name[this_hit[0][index_event_type]]
        except:
            event_type = this_hit[0][index_event_type]
        if event_type == 'Independence':
            continue
        event_id = this_hit[0][index_event_id]
        this_event_dict = input_and_answers[event_type]
        this_event_dict[event_id] = {}
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = []
        for submission in this_hit:
            if submission[index_worker_id] not in block_workers:
                this_hit_new.append(submission)
        num_valid_submission = len(this_hit_new)
        for k in xrange(len(this_hit_new)):
            submission = this_hit_new[k]
            this_event_dict[event_id][k] = []
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    #print this_hit_new[0][image_input_index[i]]
                    continue
                score = 0
                image_index = image_input_index[i]
                score_index = image_output_index[i]
                image_url = this_hit_new[0][image_index]
            #for submission in this_hit_new:
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
                this_event_dict[event_id][k].append(score)
            #this_event_dict[event_id].append((image_url, (image_url.split('/')[-1]).split('_')[0], score))
            #ii += 1

    #for event_name in input_and_answers:
    #    print event_name, len(input_and_answers[event_name])
    correlations_all = []
    count_all_lowp = 0
    count_all_lowp_pair = 0
    count_all_pair = 0
    for event_name in dict_name2:
            p_values = []
            count_lowp = 0
            count_pair = 0
            count_lowp_pair = 0
            correlations = []
            this_event_dict = input_and_answers[event_name]
            for event_id in this_event_dict:
                this_dict = this_event_dict[event_id]
                ranking_dict = []
                len_rater = len(this_dict)
                correlations_this = []
                #
                # # for i in this_dict:
                # #     for j in this_dict:
                # #         if i == j:
                # #             continue
                # #         except_this = [this_dict[k] for k in this_dict if k not in [i,j]]
                # #         this_ = [this_dict[k] for k in [i,j]]
                # #         rho, p = spearmanr(np.sum(this_,axis=0), np.sum(except_this, axis=0))
                # #         correlations_this.append(rho)
                # # correlations.extend(correlations_this)
                # for i in this_dict:
                #     # # for j in this_dict:
                #     # #     if i == j:
                #     # #         continue
                #     # #     rho, p = spearmanr(this_dict[i], this_dict[j])
                #     # #     correlations_this.append(rho)
                #     except_this = [this_dict[j] for j in this_dict if i != j]
                #     rho, p = spearmanr(this_dict[i], np.sum(except_this, axis=0))
                #     correlations_this.append(rho)
                # #correlations.extend(correlations_this)
                # #correlations_this.sort(reverse=True)
                # count_this = 0
                # if np.min(correlations_this) < 0.2 and len(this_dict) > 3:
                #     this_dict.pop(np.argmin(correlations_this))
                #     correlations_this = []
                #     # for i in this_dict:
                #     #     for j in this_dict:
                #     #         if i == j:
                #     #             continue
                #     #         except_this = [this_dict[k] for k in this_dict if k not in [i,j]]
                #     #         this_ = [this_dict[k] for k in [i,j]]
                #     #         rho, p = spearmanr(np.sum(this_,axis=0), np.sum(except_this, axis=0))
                #     #         correlations_this.append(rho)
                #     # correlations.extend(correlations_this)
                #     for i in this_dict:
                #         # for j in this_dict:
                #         #     if i == j:
                #         #         continue
                #         #     except_this = [this_dict[k] for k in this_dict if k not in [i,j]]
                #         #     this_ = [this_dict[k] for k in [i,j]]
                #         #     rho, p = spearmanr(np.sum(this_,axis=0), np.sum(except_this, axis=0))
                #         # #     correlations_this.append(rho)
                #         except_this = [this_dict[j] for j in this_dict if i != j]
                #         rho, p = spearmanr(this_dict[i], np.sum(except_this, axis=0))
                #         correlations_this.append([i, rho])
                #     correlations_this.sort(key=lambda x: x[1])
                #     if correlations_this[0][1] < 0.2 and len(this_dict) > 3:
                #     #if len(this_dict) > 3:
                #         count_this = 2
                #         this_dict.pop(correlations_this[0][0])
                #         # for i in this_dict:
                #         #     except_this = [this_dict[j] for j in this_dict if i != j]
                #         #     #print len(except_this)
                #         #     rho, p = spearmanr(this_dict[i], np.sum(except_this, axis=0))
                #         #     correlations.append(rho)
                #     #else:
                #         #correlations.extend([k[1] for k in correlations_this])

                # correlations_this = []
                # for i in this_dict:
                #     for j in this_dict:
                #             if i == j:
                #                 continue
                #             except_this = [this_dict[k] for k in this_dict if k not in [i,j]]
                #             this_ = [this_dict[k] for k in [i,j]]
                #             rho, p = spearmanr(np.sum(this_,axis=0), np.sum(except_this, axis=0))
                #             correlations_this.append(rho)
                # correlations.append(np.mean(correlations_this))
                all_ps = []
                all_keys = this_dict.keys()
                all_pairs = []
                for i in xrange(len(all_keys)):
                    for j in xrange(i + 1, len(all_keys)):
                        all_pairs.append([all_keys[i],all_keys[j]])
                for pair_this in all_pairs:
                    except_this = np.sum([this_dict[k] for k in this_dict if k not in pair_this], axis=0)
                    this_ = np.sum([this_dict[k] for k in pair_this], axis=0)
                    rho, p = spearmanr(this_, except_this)
                    all_ps.append(p)
                    correlations_this.append(rho)
                    if p < 0.05:
                        count_lowp_pair += 1
                    count_pair += 1
                #correlations.append(np.mean(correlations_this))
                #if scipy.stats.gmean(all_ps) < 0.05:
                    correlations.append(np.mean(correlations_this))

                #if np.mean(all_ps) < 0.05:
                #if min([i for i in all_ps if i != np.min(all_ps)]) < 0.05:
                #if np.sort(all_ps)[min(len(all_ps)-1,3)] < 0.05:
                #if np.min(all_ps) < 0.05:
                #    count_lowp += 1

                # else:
                #    correlations.extend(correlations_this)
                # if  np.mean(correlations_this) > 0.45:
                #    count_all += 1
                #    print event_id, np.mean(correlations_this)

                #p_values.append(np.sort(all_ps)[min(len(all_ps)-1,2)])
                p_values.append(scipy.stats.gmean(all_ps))
            p_values.sort()
            for i in xrange(len(p_values)):
                if p_values[i] < 0.05:# * i / len(p_values):
                    count_lowp += 1
            print event_name, np.mean(correlations), '('+str(count_lowp)+'/'+str(len(this_event_dict))+')', '('+str(count_lowp_pair)+'/'+str(count_pair)+')'
            count_all_lowp += count_lowp;count_all_lowp_pair += count_lowp_pair;count_all_pair += count_pair
            correlations_all.extend(correlations)
    print count_all_lowp, '('+str(count_all_lowp_pair)+'/'+str(count_all_pair)+')'
    print np.mean(correlations_all)
def evaluate_worker_agreement_topk(threshold = 10):
    input_path = root + 'all_output/all_output.csv'
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
    for event_name in dict_name2:
        input_and_answers[event_name] = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_event_type = 29
    index_distraction = 31
    for HITId in HITs:

        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        try:
            event_type = dict_name[this_hit[0][index_event_type]]
        except:
            event_type = this_hit[0][index_event_type]
        if event_type == 'Independence':
            continue
        event_id = this_hit[0][index_event_id]
        this_event_dict = input_and_answers[event_type]
        this_event_dict[event_id] = {}
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = []
        for submission in this_hit:
            if submission[index_worker_id] not in block_workers:
                this_hit_new.append(submission)
        num_valid_submission = len(this_hit_new)
        for k in xrange(len(this_hit_new)):
            submission = this_hit_new[k]
            this_event_dict[event_id][k] = []
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    continue
                score = 0
                image_index = image_input_index[i]
                score_index = image_output_index[i]
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
                this_event_dict[event_id][k].append(score)
    correlations_all = []
    count_all = 0
    for event_name in dict_name2:
            correlations = []
            this_event_dict = input_and_answers[event_name]
            for event_id in this_event_dict:
                this_dict = this_event_dict[event_id]
                threshold_n = float(threshold) / 100 * len(this_dict[0])
                all_keys = this_dict.keys()
                all_pairs = []
                correlations_this = []
                for i in xrange(len(all_keys)):
                    for j in xrange(i, len(all_keys)):
                        if i==j:
                            continue
                        all_pairs.append([all_keys[i],all_keys[j]])
                for pair_this in all_pairs:
                    except_this = np.sum([this_dict[k] for k in this_dict if k not in pair_this], axis=0)
                    this_ = np.sum([this_dict[k] for k in pair_this], axis=0)
                    except_this = [i+random.uniform(-0.02, 0.02)+4 for i in except_this]
                    this_ = [i+random.uniform(-0.02, 0.02)+4 for i in this_]
                    rank_this = scipy.stats.rankdata(this_)
                    rank_except_this = scipy.stats.rankdata(except_this)
                    this_threshold = [i*float(j<=threshold_n) for i,j in zip(this_, rank_this)]
                    except_this_threshold = [i*float(j<=threshold_n) for i,j in zip(except_this, rank_except_this)]
                    rho, p = spearmanr(this_threshold, except_this_threshold)
                    # rho, p = spearmanr(rank_this, rank_except_this)
                    correlations_this.append(rho)
                    #print rho, p
                correlations.append(np.mean(correlations_this))
            print event_name, np.mean(correlations)
            correlations_all.extend(correlations)
    print count_all
    print np.mean(correlations_all)
def evaluate_worker_agreement_refined(method = 'spearman'):
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/amt/CUFED/curation-results/all_output.csv'
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
    for event_name in dict_name2:
        input_and_answers[event_name] = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_event_type = 29
    index_distraction = 31
    for HITId in HITs:

        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        try:
            event_type = dict_name[this_hit[0][index_event_type]]
        except:
            event_type = this_hit[0][index_event_type]
        if event_type == 'Independence':
            continue
        event_id = this_hit[0][index_event_id]
        this_event_dict = input_and_answers[event_type]
        this_event_dict[event_id] = {}
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = []
        for submission in this_hit:
            if submission[index_worker_id] not in block_workers:
                this_hit_new.append(submission)
        for k in xrange(len(this_hit_new)):
            submission = this_hit_new[k]
            this_event_dict[event_id][k] = []
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    continue
                score = 0
                score_index = image_output_index[i]
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
                this_event_dict[event_id][k].append(score)

    correlations_all = []
    correlations_all_w = []
    p_all_w = []
    count_all_lowp = 0
    all_event_co = []
    for event_name in dict_name2:
            p_values = []
            p_values_w = []
            count_lowp = 0
            count_lowp_w = 0
            correlations = []
            correlation_ws = []
            correlation_alpha = []
            this_event_dict = input_and_answers[event_name]
            for event_id in this_event_dict:
                this_dict = this_event_dict[event_id]
                correlations_this = []
                all_ps = []
                all_keys = this_dict.keys()
                all_pairs = []
                w, p_new = kendall_w_chip(this_dict)
                # w, p_new = kendall_w_p(copy.deepcopy(this_dict))
                p_values_w.append(p_new)
                correlation_ws.append(w)
                print event_id,w
                # # alpha = CronbachAlpha([this_dict[i] for i in this_dict])
                # # correlation_alpha.append(alpha)
                for i in xrange(len(all_keys)):
                    for j in xrange(i + 1, len(all_keys)):
                        all_pairs.append([all_keys[i],all_keys[j]])
                for pair_this in all_pairs:
                        except_this = np.sum([this_dict[k] for k in this_dict if k not in pair_this], axis=0)
                        this_ = np.sum([this_dict[k] for k in pair_this], axis=0)
                        if method == 'spearman':
                            rho, p = spearmanr(this_, except_this)
                            #rho_ = 1 - Bio.Cluster.distancematrix((this_,except_this), dist="s")[1][0]
                            # rho, p = spearmanr(rankdata(this_), rankdata(except_this))
                            #if abs(rho_ - rho) > 0.001:
                            #    print 'DIFFERENT!'
                        if method == 'tau':
                            rho, p = kendalltau(rankdata(this_), rankdata(except_this))
                            #rho, p = kendalltau(this_, except_this)

                        all_ps.append(p)
                        correlations_this.append(rho)
                        correlations.append(np.mean(correlations_this))
                all_event_co.append((event_id, np.mean(correlations_this), event_name, len(this_)))
                p_values.append(np.sort(all_ps)[min(len(all_ps)-1,0)])
                # p_values.append(scipy.stats.gmean(all_ps))

            p_values.sort()
            # for i in xrange(len(p_values)):
            #    if p_values[i] < 0.05 * i / len(p_values):
            #        count_lowp += 1
            # p_values_w.sort()
            # for i in xrange(len(p_values_w)-1, -1, -1):
            #     if p_values_w[i] < 0.05 * i / len(p_values_w):
            #         break
            count_lowp_w += i + 1
            # for i in xrange(len(p_values_w)):
            #    if p_values_w[i] < 0.05 * i / len(p_values_w):
            #        count_lowp_w += 1
            # print event_name, np.mean(correlations), np.mean(correlation_ws), '('+str(count_lowp)+'/'+str(len(this_event_dict))+')', '('+str(count_lowp_w)+'/'+str(len(this_event_dict))+')', '('+str(float(count_lowp_w)/len(this_event_dict))+')'
            count_all_lowp += count_lowp_w
            correlations_all.extend(correlations)
            correlations_all_w.extend(correlation_ws)
    print count_all_lowp
    print np.mean(correlations_all), np.mean(correlations_all_w)
    all_event_co.sort(key = lambda x: x[1], reverse=True)
    print all_event_co

def evaluate_worker_agreement_alpha():
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/amt/CUFED/curation-results/all_output.csv'
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
    for event_name in dict_name2:
        input_and_answers[event_name] = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_event_type = 29
    index_distraction = 31
    for HITId in HITs:

        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        try:
            event_type = dict_name[this_hit[0][index_event_type]]
        except:
            event_type = this_hit[0][index_event_type]
        if event_type == 'Independence':
            continue
        event_id = this_hit[0][index_event_id]
        this_event_dict = input_and_answers[event_type]
        this_event_dict[event_id] = {}
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = []
        for submission in this_hit:
            if submission[index_worker_id] not in block_workers:
                this_hit_new.append(submission)
        for k in xrange(len(this_hit_new)):
            submission = this_hit_new[k]
            this_event_dict[event_id][k] = []
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    continue
                score = 0
                score_index = image_output_index[i]
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
                this_event_dict[event_id][k].append(score)
    alphas_all = []
    for event_name in dict_name2:
        alphas = []
        this_event_dict = input_and_answers[event_name]
        for event_id in this_event_dict:
            this_dict = this_event_dict[event_id]
            all_keys = this_dict.keys()
            all_pairs = []
            for i in xrange(len(all_keys)):
                    for j in xrange(i + 1, len(all_keys)):
                        all_pairs.append([all_keys[i],all_keys[j]])
            for pair_this in all_pairs:
                except_this = np.sum([this_dict[k] for k in this_dict if k not in pair_this], axis=0)
                this_ = np.sum([this_dict[k] for k in pair_this], axis=0)

                except_this = ss.rankdata(except_this)
                this_ = ss.rankdata(this_)
                rank_dict = [dict([(ii+1,int(jj)) for ii,jj in enumerate(except_this)]), dict([(ii+1,int(jj)) for ii,jj in enumerate(this_)])]
                agreement = Agreement(rank_dict)
                alpha = agreement.krippendorffAlpha(Agreement.differenceOrdinal)
                alphas.append(alpha)
        print event_name, np.mean(alphas)
        alphas_all.extend(alphas)
    print np.mean(alphas_all)

def kendall_w_chip(list_all):
    rank_all = []
    corrections = 0
    for i in list_all:
        rank_all.append(rankdata(list_all[i]))
        tie_count = Counter(rankdata(list_all[i]))
        corrections += np.sum(tie_count[i]**3 - tie_count[i] for i in tie_count)
    rank_ = np.sum(rank_all, axis = 0)
    rank_bar = np.mean(rank_)
    S = np.sum([(i - rank_bar)**2 for i in rank_])
    S_prime = np.sum([i**2 for i in rank_])
    m = len(list_all); n = len(rank_)
    W = (12 * S_prime - 3*m**2*n*(n+1)**2) / (m**2 * (n**3 - n) - m*corrections)

    chi2_ref = m*(n-1)*W
    df = n - 1
    p = 1 - chi2.cdf(chi2_ref, df)
    return W, p

def kendall_w(list_all):
    rank_all = []
    corrections = 0
    for i in list_all:
        rank_all.append(rankdata(list_all[i]))
        tie_count = Counter(rankdata(list_all[i]))
        corrections += np.sum(tie_count[i]**3 - tie_count[i] for i in tie_count)
    rank_ = np.sum(rank_all, axis = 0)
    rank_bar = np.mean(rank_)
    S = np.sum([(i - rank_bar)**2 for i in rank_])
    S_prime = np.sum([i**2 for i in rank_])
    m = len(list_all); n = len(rank_)
    W = (12 * S_prime - 3*m**2*n*(n+1)**2) / (m**2 * (n**3 - n) - m*corrections)
    return W

def kendall_w_p(list_all):
    W = kendall_w(list_all)
    #k = len(list_all); n = len(list_all[0])
    ws = permutation_test(list_all)
    p = float(sum([i > W for i in ws])) / len(ws)
    #p = float(sum([i > W for i in permute_test_ws[(n,k)]])) / len(permute_test_ws[(n,k)])
    return W, p

def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))

def permutation_test(list_all, times = 10000):
    ws = []
    k = len(list_all)
    for i in xrange(times):
        for j in list_all:
            random.shuffle(list_all[j])
        ws.append(kendall_w(list_all))
    #print ws
    return ws
    #return float(np.sum([i > w_real for i in ws])) / times

'''
def wrong_permutation_test(n, k = 5, times = 10000):
    possible_ratings = [-2,0,1,2]
    ws = []
    for i in xrange(times):
        this_dict = {}
        for j in xrange(k):
            while True:
                temp = [random.choice(possible_ratings) for q in xrange(n)]
                temp_count = Counter(temp)
                if not 2 in temp_count or temp_count[2] < 0.05 * n or temp_count[2] >= 0.3*n:
                    continue
                if not 1 in temp_count or temp_count[1] < 0.1 * n or temp_count[1] >= 0.5*n:
                    continue
                this_dict[j] = temp
                break
        #print this_dict
        #print kendall_w(this_dict)
        ws.append(kendall_w(this_dict))
    return ws
    #return float(np.sum([i > w_real for i in ws])) / times
def wrong_permute_all_needed():
    input_path = root + 'all_output/all_output.csv'
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
    for event_name in dict_name2:
        input_and_answers[event_name] = {}
    index_worker_id = 15
    index_num_image = 27
    index_event_id = 28
    index_event_type = 29
    index_distraction = 31
    for HITId in HITs:

        this_hit = HITs[HITId]
        num_images = int(this_hit[0][index_num_image])
        distract_image = this_hit[0][index_distraction]
        try:
            event_type = dict_name[this_hit[0][index_event_type]]
        except:
            event_type = this_hit[0][index_event_type]
        if event_type == 'Independence':
            continue
        event_id = this_hit[0][index_event_id]
        this_event_dict = input_and_answers[event_type]
        this_event_dict[event_id] = {}
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        this_hit_new = []
        for submission in this_hit:
            if submission[index_worker_id] not in block_workers:
                this_hit_new.append(submission)
        for k in xrange(len(this_hit_new)):
            submission = this_hit_new[k]
            this_event_dict[event_id][k] = []
            for i in xrange(1, 1+num_images):
                if i==distract1 or i==distract2:
                    continue
                score = 0
                score_index = image_output_index[i]
                vote = submission[score_index]
                if vote == 'selected':
                    score += 2
                elif vote == 'selected_sw':
                    score += 1
                elif vote == 'selected_irrelevant':
                    score -= 2
                this_event_dict[event_id][k].append(score)

    count_img_number = set()
    for event_name in dict_name2:
        this_event_dict = input_and_answers[event_name]
        for event_id in this_event_dict:
            this_dict = this_event_dict[event_id]
            n_this = len(this_dict[0])
            if (n_this, len(this_event_dict)) not in count_img_number:
                count_img_number.add((n_this, len(this_dict)))

    for i in count_img_number:
        print i
        permute_test_ws[i] = permutation_test(i[0], i[1])
        print permute_test_ws[i]
'''

def evaluation_prediction():
    correlation_all_w = []
    correlation_all_tau = []
    correlation_all_rho = []
    len_ = 0
    for event_name in dict_name2:
        f = open(root + 'baseline_all_0509/' + event_name+ '/vgg_test_result_v2.cPickle','r')
        # f = open(root + 'baseline_all_noblock/' + event_name+ '/vgg_test_result_v2.cPickle','r')
        ground_truth = cPickle.load(f)
        f.close()
        f = open(root + 'CNN_all_event_1009/features/' + event_name+ '_test_combined_10_combine_dict.cPickle','r')
        prediction = cPickle.load(f)
        f.close()
        correlation_rho = []
        correlation_tau = []
        correlation_w = []
        for event_id in ground_truth:
            g = [i[2] for i in ground_truth[event_id]]
            p = [i[2] for i in prediction[event_id]]
            temp_rho,temp1 = spearmanr(g, p)
            temp_w = kendall_w({1:g,2:p})
            temp_tau, temp1 = kendalltau(g, p)
            correlation_rho.append(temp_rho)
            correlation_w.append(temp_w)
            correlation_tau.append(temp_tau)
        len_ += len(ground_truth)
        print event_name, ', rho:', np.mean(correlation_rho), ', kendall\'s tau:', np.mean(correlation_tau), ', kendall\'s W:', np.mean(correlation_w)
        correlation_all_rho.append(np.mean(correlation_rho) * len(ground_truth))
        correlation_all_tau.append(np.mean(correlation_tau) * len(ground_truth))
        correlation_all_w.append(np.mean(correlation_w) * len(ground_truth))
    print 'rho:', np.sum(correlation_all_rho) / len_, ', kendall\'s tau:', np.sum(correlation_all_tau) / len_, ', kendall\'s W:', np.sum(correlation_all_w) / len_

def present_album():
    path = '/Users/wangyufei/Documents/Study/intern_adobe/supplementary/54634670@N03/'
    files = [f for f in os.listdir(path) if f.endswith('jpg')]
    t = int(np.floor(len(files) / 10))
    # im_canvas = Image.new("RGB", (2000, 2000), "black")
    im_canvas = np.zeros((t*300,3000,3))
    i = 0
    for img_file in files:
        img = Image.open(path + img_file)
        img = img.resize((300,300))
        start_width = (i/10)*300
        start_height = (i%10)*300
        im_canvas[start_width:start_width+300, start_height:start_height+300,:] = np.array(img)
        i+=1
        if i >= 10*t:
            break
    im_canvas = np.uint8(im_canvas)
    im_canvas_img = Image.fromarray(im_canvas)
    im_canvas_img.save(path + "../graduation_canvas.jpg", "JPEG")

if __name__ == '__main__':
    # present_album()
    # permute_all_needed()
    # evaluate_worker_agreement_refined('spearman')
    evaluate_worker_agreement_alpha()
    # evaluation_prediction()
    # in_path = root + 'aesthetic/original_imgs.txt'
    # with open(in_path, 'r') as data:
    #     for line in data:
    #         line = line[:-1]
    #         meta = line.split('/')[-2:]
    #         meta[0] = meta[0].split('_')[1]
    #         path = meta[0] + '/' + meta[1]
    #         old_root = '/Volumes/Vivian_backup/intern_adobe/from_server_tidy/datasets/download_event_recognition/'
    #         new_root ='/Volumes/Vivian_backup/intern_adobe/from_server_tidy/datasets/download_event_recognition_new/'
    #         if not os.path.exists(new_root + meta[0]):
    #             os.mkdir(new_root + meta[0])
    #         shutil.copyfile(old_root + path, new_root+path)
    # in_path = root + 'face_expression/face_imgs.txt'
    # with open(in_path, 'r') as data:
    #     for line in data:
    #         line = line[:-1]
    #         meta = line.split('/')
    #         root_this = '/Volumes/Vivian_backup/intern_adobe/from_server_tidy/event_curation/face_recognition/face_features/'
    #         new_root = '/Volumes/Vivian_backup/intern_adobe/from_server_tidy/event_curation/face_recognition/face_features_new/'
    #         path_this = new_root + meta[-3]
    #         if not os.path.exists(path_this):
    #             os.mkdir(path_this)
    #         if not os.path.exists(path_this + '/' + meta[-2]):
    #             os.mkdir(path_this + '/' + meta[-2])
    #         suffix = '/'.join(meta[-3:])
    #         shutil.copyfile(root_this + suffix, new_root + suffix)
