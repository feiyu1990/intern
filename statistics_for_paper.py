__author__ = 'wangyufei'
import cPickle
import operator
import random
import os
import numpy as np
from collections import Counter
import csv
import sys
root = '/Users/wangyufei/Documents/Study/intern_adobe/'
block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']

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
    count_all = 0
    agreement_event_id = []
    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        # if len(results) > 1:
        #     count+=1
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            count_all+= 1
            count_temp = 0
            # for temp in result_this.keys():
            #     if 'NOTEVENT_' in temp:
            #         count_temp += 1
            # if count_temp > len_vote/2:
            #     continue
            count+=1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            # if max_result[1] >= 2 and max_result[1] > len_vote/2:
            if max_result[1] >= 2 and max_result[1] > len_vote/2:
                agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
    print "number of events with more than 1 vote:", count
    print "number of events with more than 2 same votes:",len(agreement_event_id)
    return count_all, count, len(agreement_event_id)
def find_inconsistent_result():
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/'
    names = ['0','1','2','3','4','5','6','7']
    count = 0
    count_all = 0
    disagreement_dict = {}
    agreement_event_id = []
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

        for HIT in HIT_result_dic:
            results = HIT_result_dic[HIT]
            # if len(results) > 1:
            #     count+=1
            len_vote = len(results)
            for i in xrange(len(results[0])):
                result_this = {}
                for result in results:
                    if result[i][2] in result_this:
                        result_this[result[i][2]] += 1
                    else:
                        result_this[result[i][2]] = 1
                count_all+= 1
                count_temp = 0
                for temp in result_this.keys():
                    if 'NOTEVENT_' in temp:
                        count_temp += 1
                if count_temp > len_vote/2:
                    continue
                count+=1
                max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
                if max_result[1] == 2 :#and max_result[1] > len_vote/2:
                # if max_result[1] >= 3 and max_result[1] > len_vote/2:
                #     agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
                # else:
                #     if len(result_this.keys()) > 3:
                #         continue
                    # if len(result_this.keys()) == 3:
                    #     temp = (result_this.keys()[0], result_this.keys()[1], result_this.keys()[2])
                    # else:
                    if result_this.keys()[0][0] > result_this.keys()[1][0]:
                        temp = (result_this.keys()[0], result_this.keys()[1])
                    else:
                        temp = (result_this.keys()[1], result_this.keys()[0])
                    if temp not in disagreement_dict:
                        disagreement_dict[temp] = 1
                    else:
                        disagreement_dict[temp] += 1
    print "number of events with more than 1 vote:", count
    print "number of events with more than 2 same votes:",len(agreement_event_id)
    disagreement_dict =  sorted(disagreement_dict.items(), key=operator.itemgetter(1), reverse=True)
    for i,j in disagreement_dict:
        print i,j
    print np.sum([i[1] for i in disagreement_dict])
    #return count_all, count, len(agreement_event_id)
def img_count():
        img_count_dict = {}
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
        index_worker_id = 15
        index_num_image = 27
        index_event_id = 28
        index_distraction = 31
        for HITId in HITs:
            this_hit = HITs[HITId]
            num_images = int(this_hit[0][index_num_image])
            if num_images not in img_count_dict:
                img_count_dict[num_images] = 1
            else:
                img_count_dict[num_images] += 1
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
                # input_and_answers[event_id].append((image_url, event_ids[event_id][ii], score))
                ii += 1
                #print (image_url, score)
        # f = open(root + 'baseline_all_0509/' + name + '/' + type + '_result_v1.cPickle','wb')
        # cPickle.dump(input_and_answers, f)
        # f.close()
        #print img_count_dict
        x,y = zip(*[(i,img_count_dict[i]) for i in img_count_dict])
        average_count = 0; count = 0
        for i,j in zip(x,y):
            average_count += i*j; count += j
        print float(average_count) / count
        print x
        print y
def count_num_event():
    n_all = 0
    root = '/Volumes/Vivian_backup/intern_adobe/from_server/dataset/events_clean_tags/'
    files = [os.path.join(root,o) for o in os.listdir(root) if os.path.isfile(os.path.join(root,o))]
    for i in files:
        with open(i, 'r') as data:
            for line in data:
                # line = tail(open(i,'r'), 1)
                temp = line.split('\t')
                n = int(line.split('\t')[1])
        n_all += n + 1
    print n_all
def tail( f, lines=20 ):
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = [] # blocks of size BLOCK_SIZE, in reverse order starting
                # from the end of the file
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            # read the last block we haven't yet read
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count('\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = ''.join(reversed(blocks))
    return '\n'.join(all_read_text.splitlines()[-total_lines_wanted:])
def event_count_all():
        event_count = {}
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

        index_event_id = 28
        index_distraction = 31
        for HITId in HITs:
            this_hit = HITs[HITId]
            event_type = this_hit[0][29]
            if event_type in dict_name:
                event_type = dict_name[event_type]
            if event_type in event_count:
                event_count[event_type] += 1
            else:
                event_count[event_type] = 1
        print event_count
def average_completion_time():
        times = 0
        count = 0
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

        index_event_id = 28
        index_num_image = 27
        index_distraction = 31
        for HITId in HITs:
            this_hit = HITs[HITId]
            for i in this_hit:
                time = i[23]
                times += float(time) / float(i[index_num_image])
                count += 1
        print float(times) / count
def worker_count():
        worker_ids = {}
        count_all = 0
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
        for HITId in HITs:
            this_hit = HITs[HITId]
            for i in this_hit:
                worker_id = i[15]
                if worker_id not in worker_ids:
                    worker_ids[worker_id] = 1
                else:
                    worker_ids[worker_id] += 1
                count_all += 1
        print len(worker_ids)
        worker_ids =  sorted(worker_ids.items(), key=operator.itemgetter(1), reverse=True)
        count_90 = 0.9 * count_all
        count = 0; count_worker = 0
        for i,j in worker_ids:
            count += j; count_worker += 1
            if count >= count_90:
                print count_worker
                break
def block_worker_count():
        worker_ids = {block_workers[0]:0, block_workers[1]:0}
        event_worker_count = {3:0,4:0,5:0}
        count_all = 0
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

        for HITId in HITs:
            this_hit = HITs[HITId]
            worker_counts = 0
            for i in this_hit:
                worker_id = i[15]
                if worker_id in block_workers:
                    worker_ids[worker_id] += 1
                    worker_counts += 1
                event_worker_count[5-worker_counts] += 1

                count_all += 1
        # print len(worker_ids)
        # worker_ids =  sorted(worker_ids.items(), key=operator.itemgetter(1), reverse=True)
        # count_90 = 0.9 * count_all
        # count = 0; count_worker = 0
        print event_worker_count, worker_ids
if __name__ == '__main__':
    # find_inconsistent_result()
    a = 0; b = 0; c = 0
    for i in xrange(8):
        temp1, temp2, temp3 = find_consistent_result(str(i))
        a += temp1
        b += temp2
        c += temp3
    temp1, temp2, temp3 = find_consistent_result('8_pre')
    a += temp1
    b += temp2
    c += temp3
    print a, b, c
    # read_amt_result()
    # count_num_event()
    # event_count_all()
    # average_completion_time()
    # block_worker_count()