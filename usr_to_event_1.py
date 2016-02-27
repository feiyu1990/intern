__author__ = 'wangyufei'
'''
SAMPLE EXPERIMENT. See load_root
sort_usr_with_time() -> sort one user in a directory (with .txt) based on timestamp. Save into /all_users_time/
'''
import os
import sys
from datetime import datetime


root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
ori_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/all_users/'
time_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/all_users_time/'
clean_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/all_users_nodup/'
event_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/events/'


def remove_duplicate(usr_id, min_occurrence=5000, min_tag=3):
    print 'removing duplicate...'
    usr_list = [f for f in os.listdir(time_path) if (f.endswith('_'+usr_id+'.txt') and f.startswith('0706'))]
    for usr in usr_list:
        print usr, '...'
        load_root = time_path + usr
        save_root = clean_path + usr
        prev_id = ''
        write_file = open(save_root, 'w')
        duplicated = 0
        with open(load_root, 'r') as data:
            for line in data:
                meta = line.split('\t')
                this_id = meta[0]
                if prev_id != this_id:
                    write_file.write(line)
                    prev_id = this_id
                else:
                    duplicated += 1
                    #print line
        print "find duplicate:", duplicated
        write_file.close()
def split_event(usr_id, min_picture_event = 30, time_diff = 3600*3, path=clean_path, save_path =event_path):
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

def sort_usr_with_time(usr_list, i, min_occurrence=5000, min_tag=3):
    invalid_line = 0
    print "sorting..."
    #print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    #load_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+ '/all_users/'
    load_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/0706/all_users_'+ i +'/'
    save_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+ 'all_users_time/0706'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    #for usr in os.listdir(load_root):
    for usr in [usr_list]:
            print usr, '...'
            list_path = load_root + usr + '.txt'
            print list_path
            images = []
            with open(list_path, 'r') as data:
                for line in data:
                    meta = line.split('\t')
                    if len(meta) != 23:
                        invalid_line += 1
                        continue
                    date = meta[3]
                    temp = date.split(' ')
                    date_info = temp[0]
                    try:
                        time_info = temp[1]
                    except:
                        print line
                    temp = date_info.split('-')
                    m = temp[1]
                    d = temp[2]
                    y = temp[0]
                    h = time_info.split(':')[0]
                    minute = time_info.split(':')[1]
                    second = time_info.split(':')[2]
                    time_this = float(y+m+d+h+minute+second)
                    images.append((time_this, line))
            save_path = save_root + i +'_' +usr + '.txt'
            print save_path
            images_sorted = sorted(images, key=lambda x:x[0])

            with open(save_path, 'w') as out_file:
                for time, line in images_sorted:
                    out_file.write(line)
            print 'User:',usr,' invalid lines:', invalid_line

#12420978@N07 tested
if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    usr_id = (args[1])
    #usr_name = (args[1])
    #sort_usr_with_time(usr_name)
    #i = (args[1])
    for j in xrange(10):
        i = str(j)
        sort_usr_with_time(usr_id,i)
    remove_duplicate(usr_id)
