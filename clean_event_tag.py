__author__ = 'wangyufei'

import sys
import os
from collections import Counter

load_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/events/'
save_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/event_clean_tags/'
save_root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/event_abandoned/'

def clean_even_tag(usr_id, path=load_root, save_path = save_root, save_path_1 = save_root1):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_1):
        os.mkdir(save_path_1)
    usr_path = path+usr_id+'.txt'
    tags = []
    lines = []
    event_start = []
    event_pre = -1
    count = 0
    with open(usr_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[0]
            if event_this != event_pre:
                event_pre = event_this
                event_start.append(count)
            tag = meta[9]
            tag_list = tag.split(',')
            tags.append(tag_list)
            lines.append(line)
            count += 1

    out_file = open(save_path + usr_id + '.txt', 'w')
    out_file_1 = open(save_path_1 + usr_id + '.txt','w')
    #new_event_id = 0
    for i in xrange(len(event_start)):
        start = event_start[i]
        if i == len(event_start) - 1:
            end = len(tags)
        else:
            end = event_start[i + 1]
        tags_this = tags[start:end]
        len_event = end - start
        end_ori = end; start_ori = start
        #print i, set().union(*tags_this)
        len_all = []

        #METHOD 1

        #STEP 1: Eliminate the bad beginning/end
        tags_all = [tag for per_image in tags_this for tag in per_image]
        count = Counter(tags_all)
        okay_to_stop = False
        j = start
        while (not okay_to_stop) and j < end:
            tags_temp = tags[j]
            counts = [count[tag_now] for tag_now in tags_temp]
            max_counts = max(counts)
            if max_counts < len_event * 1/5:
                start += 1
            else:
                okay_to_stop = True
            j += 1
        j = end - 1
        okay_to_stop = False
        while (not okay_to_stop) and j > start:
            tags_temp = tags[j]
            counts = [count[tag_now] for tag_now in tags_temp]
            max_counts = max(counts)
            if max_counts < len_event * 1/5:
                end -= 1
            else:
                okay_to_stop = True
            j -= 1
        if end - start < 30:
            print 'This is not an event(too short)!'
            continue

        #STEP 2: COUNT THE OCCURRENCE OF EACH TAG
        tags_this_new = tags[start:end]
        len_event = end - start
        tags_all = [tag for per_image in tags_this_new for tag in per_image]
        count = Counter(tags_all)
        count_value = count.values()
        count_value.sort(reverse=True)
        min_num = min(2, max(1, len(count_value)/3))
        #print count.most_common(min_num)
        if count_value[min_num - 1] < len_event*2/3:
            print 'Start: %d, End: %d, length: %d' % (start, end, end-start)
            print count
            print 'This is not an event!'
            for jj in xrange(start_ori, end_ori):
                out_file_1.write(lines[jj])
            continue
        print count
        count = [(k, v) for k, v in count.iteritems()]
        #print count
        count_selected = [(u,v) for u,v in count if v >= len_event * 2/3]
        count_selected.sort(key=lambda x:x[1], reverse=True)
        #print count_selected
        for jj in xrange(start, end):
            lines_new = ''
            for tag, count in count_selected:
                lines_new += tag + ','
            lines_new = lines_new[:-1]
            out_file.write(lines_new + '\t' + lines[jj])
    out_file.close()





if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    usr_id = args[1]
    #split_event(usr_id)
    clean_even_tag(usr_id)