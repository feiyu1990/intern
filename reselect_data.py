__author__ = 'wangyufei'
import cPickle
import operator
import random
import os
import numpy as np
from collections import Counter
import csv

root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/'
root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'
root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv_resample_2/'

def find_labeled_event(name):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/input/'
    input_path = root + name+'.csv'
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
    id_index = []
    for i in xrange(len(head_meta)):
        if 'event_id' in head_meta[i]:
            id_index.append(i)
    event_ids = []
    for meta in metas:
        for i in id_index:
            event_ids.append(meta[i])
    return event_ids

threshold = 250
block_tags = ['travel', 'live','nature','concert','architecture','urban','trip','rock']
want_tags = ['graduation','birthday']
def select_4510_events_valid(event_labeled, c_min=30, c_max=200):
    event_prev_id = ''
    read_path = root + 'clean_imgs.txt'
    event_id = []
    all_event_id = []
    count = 0

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'

    f = open('/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/clean_tags_event_count.cPickle', 'rb')
    tags_count = cPickle.load(f)
    f.close()

    f = open(root1+'clean_length_event.cPickle','r')
    event_length = cPickle.load(f)
    f.close()
    event_length = dict((y, x) for x, y in event_length)
    dict_tag_count = dict((x, y) for x, y in tags_count)
    event_count = 0
    event_length_valid = []
    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            count += 1
            tag_list = tags.split(',')
            event_this_id = event_this + '_' + usr_this
            not_block = True
            if event_this_id != event_prev_id:
                event_prev_id = event_this_id
                all_event_id.append(event_this_id)
                length_this = event_length[event_this_id]
                rate = []

                for tag_individual in tag_list:
                    rate.append(dict_tag_count[tag_individual])
                    if tag_individual in block_tags:
                        not_block = False
                        break

                if (event_this_id not in event_labeled) and (not_block and max(rate) >= threshold):
                    if length_this <= c_max and length_this >= c_min:
                        event_id.append(event_this_id)
                        event_length_valid.append(length_this)
                        #print length_this, event_this_id
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv_resample_2/'

    #print event_id
    #print len(event_id),len(event_length_valid), len(event_length), len(all_event_id)
    indice = sorted(random.sample(xrange(len(event_id)), 4510))
    sample_event = [event_id[i] for i in indice]
    sample_length = [event_length_valid[i] for i in indice]

    sample_ = zip(sample_length, sample_event)
    length_sort = sorted(sample_,reverse=True)
    f = open(root1+'resample_precal_sample_length_event.txt','w')
    for length, id in length_sort:
        f.write(str(length)+' '+id+'\n')
    f.close()


    if not os.path.exists(root1):
        os.mkdir(root1)
    f = open(root1+'resample_event_id.cPickle','w')
    cPickle.dump(sample_event, f)
    f.close()
    f = open(root1+'resample_event_id.txt','w')
    for i in sample_event:
        f.write(i+'\n')
    f.close()

    f = open(root1+'re_valid_event_id.cPickle','w')
    cPickle.dump(event_id, f)
    f.close()
def select_more_graduation(event_labeled, tags=["graduation",'halloween','birthday','cruise','christmas'], c_min = 30, c_max=200):
    event_prev_id = ''
    read_path = root + 'clean_imgs.txt'
    all_event_id = []
    count = 0
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    f = open(root1+'clean_length_event.cPickle','r')
    event_length = cPickle.load(f)
    f.close()
    event_length = dict((y, x) for x, y in event_length)

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv_resample_2/'
    f = open(root1+'resample_event_id.cPickle','r')
    pre_event = cPickle.load(f)
    f.close()

    log_file = root1+'log.txt'
    f = open(log_file,'a')
    for tag in tags:
        f.write(tag)
        event_id = []
        with open(read_path, 'r') as data:
            for line in data:
                line = line.split('\n')[0]
                meta = line.split('\t')
                tags = meta[0]
                event_this = meta[1]
                usr_this = meta[3]
                count += 1
                tag_list = tags.split(',')
                event_this_id = event_this + '_' + usr_this
                if event_this_id != event_prev_id:
                    event_prev_id = event_this_id
                    all_event_id.append(event_this_id)
                    length_this = event_length[event_this_id]
                    if event_this_id not in pre_event and (event_this_id not in event_labeled) and tag in tag_list:
                        if length_this >= c_min and length_this <= c_max:
                            event_id.append(event_this_id)

        #print event_id
        f.write(str(len(event_id)))
        if len(event_id) >= 200:

            indice = sorted(random.sample(xrange(len(event_id)), 200))
            sample_event_tag = [event_id[i] for i in indice]
        else:
            sample_event_tag = event_id
        indice = sorted(random.sample(xrange(len(pre_event)), len(sample_event_tag)))
        sample_event = [pre_event[i] for i in xrange(len(pre_event)) if i not in indice]
        pre_event = sample_event_tag + sample_event
        f.write(str(len(set(pre_event))))


    f.close()


    f = open(root1+'resample_event_id.cPickle','w')
    cPickle.dump(pre_event, f)
    f.close()
    f = open(root1+'resample_event_id.txt','w')
    for i in pre_event:
        f.write(i+'\n')
    f.close()

def write_4510_resample_valid():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv_resample_2/'
    f = open(root1+'resample_event_id.cPickle','rb')
    sample_event = cPickle.load(f)
    f.close()

    read_path = root + 'clean_imgs.txt'
    write_path = root1 + 'resample_4510_imgs_0726.txt'
    count = 0
    i = 0
    event_prev_id = ''
    log_path = root1 + 'log.txt'
    f = open(write_path, 'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            #if (i % 1000) == 0:
            #    with open(log_path, 'a') as f1:
            #        f1.write(str(i)+'\n')
            if event_this_id in sample_event:
                f.write(line)
    f.close()
def tags_with_count():
    event_prev = -1
    user_prev = -1
    seen = {}
    root1='/Users/wangyufei/Documents/Study/intern_adobe/amt/clean_input_and_label/1_round/analysis/'
    read_path = root1 + '20000_all_events.txt'

    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            if usr_this != user_prev or event_prev != event_this:
                event_prev = event_this
                user_prev = usr_this
                tag_list = tags.split(',')
                for tag_individual in tag_list:
                    if tag_individual not in seen:
                        seen[tag_individual] = 1
                    else:
                        seen[tag_individual] += 1

    seen_sorted = sorted(seen.items(), key=operator.itemgetter(1), reverse = True)
    out_file_path = root1 + '20000_all_events_tags_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key, value in seen_sorted:
        try:
            out_f.write(key+'\t'+str(value)+'\n')
        except:
            print "write tag failed:", key, value
            #del seen[key]
    out_f.close()
def reselect_write_csv_10():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv_resample_2/'
    path=root1 + 'resample_4510_imgs_0726.txt'
    meta_path = path
    event_start = []
    event_id = []
    tags = []
    count = 0
    last_event = -1
    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                event_id.append(this_event)
                tags.append(meta[0])
                event_start.append(count)
                last_event = this_event
            count += 1
    event_start.append(count)
    event_length = []
    for i in xrange(1,len(event_start)):
        event_length.append(event_start[i]-event_start[i-1])
    print count
    print len(event_start)
    print len(event_length)
    print max(event_length)
    length_sort = sorted(event_length)
    f = open(root1+'log_length_event.txt','w')
    for i in length_sort:
        f.write(str(i)+'\n')
    f.close()

    out_path = root1 + 'reselect_4510.csv'
    f = open(out_path, 'w')
    max_size = max(event_length)
    f.write('num_image,event_id,tags')
    for i in xrange(max_size):
            f.write(',')
            f.write('image'+str(i+1))
    f.write('\n')

    count = 0
    event_count = 0
    event_prev = ''
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if event_this != event_prev:
                    if count > 0:
                        for kk in xrange(max_size-event_length[count - 1]):
                            f.write(',NA')
                        if event_count % 1 == 0:
                            f.write('\n')
                        else:
                            f.write(',')
                    f.write(str(event_length[count]))
                    count += 1
                    f.write(','+event_this)
                    tag_list = meta[0].split(',')
                    tag_this = ':'.join(tag_list)
                    f.write(','+tag_this)
                    event_prev = event_this
                    event_count += 1
                url = meta[16]
                f.write(','+url)
    for kk in xrange(max_size-event_length[count - 1]):
        f.write(',NA')
    f.write('\n')
    f.close()


    in_path = root1 + 'reselect_4510.csv'
    lines = []
    with open(in_path, 'r') as data:
        for line in data:
            lines.append(line)
    start_line = lines[0]
    lines = lines[1:]

    random.shuffle(lines)

    out_path = root1 + 'shuffle_reselect_4510.csv'
    f = open(out_path,'w')
    f.write(start_line)
    for line in lines:
        f.write(line)
    f.close()

    out_path = root1 + '10_shuffle_reselect_4510.csv'
    f = open(out_path, 'w')
    str_to_write = ''
    for k in xrange(10):
        str_to_write += ',num_image'+str(k+1)+',event_id'+str(k+1)+',tags'+str(k+1)
        for i in xrange(max_size):
                str_to_write += ',image'+str(k+1)+'_'+str(i+1)
    f.write(str_to_write[1:])
    f.write('\n')

    count = 1
    for l in lines:
        if count % 10 == 0:
            f.write(l)
        else:
            f.write(l[:-1]+',')
        count+=1
    f.close()
def create_csv_graduation_10(chunk_size = 250):
    #chunks = chunk(4000, 50)
    #print chunks

    #root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    root1 = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/4510_results/'
    in_path = root1 + '10_shuffle_reselect_4510.csv'
    if not os.path.exists(root1 + '10_shuffle_reselect_4510/'):
        os.mkdir(root1 + '10_shuffle_reselect_4510/')
    with open(in_path, 'r') as data:
        first_line = data.readline()
    j = -1
    out_path= root1 + '10_shuffle_reselect_4510/sample_4510_'+str(0)+'.csv'
    f = open(out_path, 'w')
    with open(in_path, 'r') as data:
        for line in data:
            if j==-1:
                j+=1
                continue
            #if j == 1150:
            #    break
            if j % chunk_size == 0:
                f.close()
                out_path= root1 + '10_shuffle_reselect_4510/sample_4510_'+str(j/chunk_size)+'.csv'
                f = open(out_path, 'w')
                f.write(first_line)
            f.write(line)
            j+=1
    f.close()

def check_duplicate():

    a = find_labeled_event('0')
    b = find_labeled_event('1')
    c = find_labeled_event('2')
    d = find_labeled_event('3')
    e = find_labeled_event('4')
    f = find_labeled_event('5')
    g = find_labeled_event('6')
    h = find_labeled_event('7')
    i = find_labeled_event('pre')
    event_ids = a+b+c+d+e+f+g
    event_ids_1 = set(event_ids)
    event_ids_2 = set(event_ids+h+i)
    f = open('all_event_id.cPickle','w')
    cPickle.dump(event_ids_2,f)
    f.close()
    pass
def recreate_all_events():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_recognition/'
    f = open(root1+'auxiliary_1000GB_event_id.cPickle','rb')
    sample_event = cPickle.load(f)
    f.close()

    read_path = root + 'clean_imgs.txt'
    write_path = root1 + 'auxiliary_1000GB_events.txt'
    count = 0
    i = 0
    event_prev_id = ''
    #log_path = root1 + 'log.txt'
    f = open(write_path, 'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            #if (i % 1000) == 0:
            #    with open(log_path, 'a') as f1:
            #        f1.write(str(i)+'\n')
            if event_this_id in sample_event:
                f.write(line)
    f.close()
def select_type_events_valid(c_min=30, c_max=100):
    event_prev_id = ''
    read_path = root + 'clean_imgs.txt'
    event_id = []
    all_event_id = []
    count = 0

    f = open('/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_recognition/all_event_id.cPickle','rb')
    event_labeled = cPickle.load(f)
    f.close()

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    f = open('/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/clean_tags_event_count.cPickle', 'rb')
    tags_count = cPickle.load(f)
    f.close()


    f = open(root1+'clean_length_event.cPickle','r')
    event_length = cPickle.load(f)
    f.close()
    event_length = dict((y, x) for x, y in event_length)
    dict_tag_count = dict((x, y) for x, y in tags_count)
    event_count = 0
    event_length_valid = []
    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            count += 1
            tag_list = tags.split(',')
            event_this_id = event_this + '_' + usr_this
            not_block = False
            if event_this_id != event_prev_id:
                event_prev_id = event_this_id
                all_event_id.append(event_this_id)
                length_this = event_length[event_this_id]
                rate = []

                for tag_individual in tag_list:
                    rate.append(dict_tag_count[tag_individual])
                    if tag_individual in want_tags:
                        not_block = True
                        break

                if (event_this_id not in event_labeled) and (not_block and max(rate) >= threshold):
                    if length_this <= c_max and length_this >= c_min:
                        event_id.append(event_this_id)
                        event_length_valid.append(length_this)
                        #print length_this, event_this_id
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_recognition/'
    #indice = sorted(random.sample(xrange(len(event_id)), 1000))
    sample_event = event_id
    sample_length = event_length_valid

    sample_ = zip(sample_length, sample_event)
    length_sort = sorted(sample_,reverse=True)
    #f = open(root1+'resample_precal_sample_length_event.txt','w')
    #for length, id in length_sort:
    #    f.write(str(length)+' '+id+'\n')
    #f.close()

    if not os.path.exists(root1):
        os.mkdir(root1)
    f = open(root1+'auxiliary_1000GB_event_id.cPickle','w')
    cPickle.dump(sample_event, f)
    f.close()
    f = open(root1+'auxiliary_1000GB_event_id.txt','w')
    for i in sample_event:
        f.write(i+'\n')
    f.close()


line_break = 40
def count_number():
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/'
    list_path = root+'resample_12000_imgs_0726.txt'
    count = 0
    last_event = ''
    over_all_count = 0
    event_number = 0
    counts = []
    count_this = 0
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tag_common = meta[0]
            this_event = meta[1]+'_'+meta[3]
            meta = meta[1:]
            if this_event != last_event:
                counts.append(count_this)
                last_event = this_event
                count_this =0
            count_this += 1
    print min(counts[1:]), max(counts[1:])

def images_for_resample(col_num=5, event_type='halloween', min_occurrence=5000,min_tag=3, write_tags = True):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/4510_results/'

    list_path = root+'resample_4510_imgs_0726.txt'
    out_path = root+'reselect_halloween.html'
    f = open(out_path, 'w')
    count = 0
    last_event = -1
    over_all_count = 0
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = -1
    #event_numbers = random.sample(xrange(12000),25)
    real_last_event = ''
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tag_common = meta[0]
            this_event = meta[1]+'_'+meta[3]
            meta = meta[1:]

            if real_last_event != this_event:
                event_number += 1
                real_last_event = this_event
            if this_event != last_event:
                if event_type not in tag_common:
                    continue
                #if event_number not in event_numbers:
                #    continue
                print count
                count = 0

                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + this_event + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common + '</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                #f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
                last_event = this_event

            path = meta[15]
            tags = meta[9]
            date = meta[4]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            if len(tags)>line_break:
                position = []
                for i in xrange(1, 1+len(tags)/line_break):
                    position.append(line_break*i+(i-1)*6)
                for pos in position:
                    tags = tags[:pos]+'<br />'+ tags[pos:]
            #f.write('\t\t\t<br /><b>'+str(over_all_count)+' '+date+'</b><br /> '+tags+'\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.close()
    print count
if __name__ == '__main__':
    #create_csv_graduation_10()
    #a = find_labeled_event('10_more_graduation_20000_except1_0')
    #b = find_labeled_event('10_more_graduation_20000_except1_1')
    #c = find_labeled_event('10_more_graduation_20000_except1_2')
    #d = find_labeled_event('sample_20000_0')
    #e = find_labeled_event('10_shuffle_reselect_12000/sample_12000_0')
    #f = find_labeled_event('10_shuffle_reselect_12000/sample_12000_1')
    #g = find_labeled_event('10_shuffle_reselect_12000/sample_12000_2')
    #event_ids = a+b+c+d+e+f+g
    #event_ids_set = set(event_ids)
    #root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/large_results/'
    #f = open(root+'labeled_event_id.cPickle','w')
    #cPickle.dump(event_ids_set, f)
    '''
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv_resample_2/'
    f = open(root1+'labeled_event_id.cPickle','r')
    event_ids = cPickle.load(f)
    select_4510_events_valid(event_ids)
    select_more_graduation(event_ids)
    write_4510_resample_valid()
    tags_with_count_reselect()
    reselect_write_csv_10()
    '''
    #images_for_resample()
    #create_csv_graduation_10()
    #check_duplicate()
    #select_type_events_valid()
    #recreate_all_events()
    tags_with_count()
