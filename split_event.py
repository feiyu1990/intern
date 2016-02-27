__author__ = 'wangyufei'
'''
split_event(usr_id, min_picture_per_event=30, max_time_interval=3h) -> split events of one user. Save to /events/
'''
import sys
import os
from datetime import datetime


load_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/all_users_time/'
save_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/events/'

def split_event(usr_id, min_picture_event = 30, time_diff = 3600*3, path=load_path, save_path =save_root):
    usr_path = path+usr_id+'.txt'
    dates = []
    #tags = []
    with open(usr_path, 'r') as data:
        for line in data:
                    meta = line.split('\t')
                    date_meta = meta[3]
                    temp = date_meta.split(' ')
                    date_info = temp[0]
                    time_info = temp[1]
                    date_info = date_info.split('-')
                    time_info = time_info.split(':')
                    time_this = datetime(int(date_info[0]),int(date_info[1]),int(date_info[2]),
                                         int(time_info[0]), int(float(time_info[1])), 0)
                    dates.append(time_this)
                    #tags.append(meta[8])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    date_prev = dates[0]
    event_start = [0]
    for i in xrange(1,len(dates)):
        date_this = dates[i]
        diff = date_this - date_prev
        date_prev = date_this
        #print diff.days, diff.seconds, diff
        if diff.days > 0 or diff.seconds > time_diff:
            event_start.append(i)
    events = []
    size = []
    start = event_start[0]
    for i in xrange(1, len(event_start)):
        end = event_start[i]
        if end-start >= min_picture_event:
            events.append((start, end))
            size.append(end-start)
        start = end
    end = len(dates)
    if end-start >= min_picture_event:
        events.append((start, end))
        size.append(end-start)

    print 'number of events:', len(events), ', number of images:', sum(size), 'number of all images: ', len(dates)
    print events

    save_p = save_path + usr_id + '.txt'
    i = 0
    count = 0
    out_file = open(save_p, 'w')
    with open(usr_path, 'r') as data:
            for line in data:
                if events[i][0] <= count and events[i][1] > count:
                    new_line = str(i)+'\t'+line
                    out_file.write(new_line)
                if events[i][1] == count:
                    i += 1
                    if i == len(events):
                        break
                    if events[i][0] <= count and events[i][1] > count:
                        new_line = str(i)+'\t'+line
                        out_file.write(new_line)
                count += 1
    print i
    out_file.close()



if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    usr_id = args[1]
    split_event(usr_id)
