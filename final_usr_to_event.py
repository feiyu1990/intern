__author__ = 'wangyufei'
'''
SAMPLE EXPERIMENT. See load_root
sort_usr_with_time() -> sort one user in a directory (with .txt) based on timestamp. Save into /all_users_time/
'''
import os
import sys
from datetime import datetime
from collections import Counter
import PIL
from PIL import Image
import urllib
import urllib
from itertools import izip


root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
#ori_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/all_users/'
ori_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/all_users/'
time_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/all_users_time/'
clean_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/all_users_nodup/'
event_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/events/'
tag_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/events_clean_tags/'
abandon_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/events_abandoned/'

if not os.path.exists(clean_path):
    os.mkdir(clean_path)
if not os.path.exists(event_path):
    os.mkdir(event_path)
if not os.path.exists(time_path):
    os.mkdir(time_path)
if not os.path.exists(tag_path):
    os.mkdir(tag_path)
if not os.path.exists(abandon_path):
    os.mkdir(abandon_path)
'''
def sort_usr_with_time(usr_list, min_occurrence=5000, min_tag=3):
    invalid_line = 0
    print 'Sorting user by time...'
    load_root = ori_path
    save_root = time_path
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for usr in [usr_list]:
        print usr, '...'
        if usr.endswith("txt"):
            list_path = load_root + usr
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
            save_path = save_root + usr
            images_sorted = sorted(images, key=lambda x:x[0])

            with open(save_path, 'w') as out_file:
                for time, line in images_sorted:
                    out_file.write(line)
        print 'User:',usr,' invalid lines:', invalid_line
'''

def concat_files(usr_name):
    filenames = [root+'clean_tags_5000_3/all_users_'+str(i)+'/'+usr_name+'.txt' for i in [0,1,2,3,4,5,10,11,12,13,14,15,16,17]]
    output_file = root+'clean_tags_5000_3_0710/all_users/'+usr_name+'.txt'
    with open(output_file, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
def sort_usr_with_time(usr_name, min_occurrence=5000, min_tag=3):
    invalid_line = 0
    load_root = ori_path
    save_root = time_path

    if not os.path.exists(save_root):
        os.mkdir(save_root)
    list_path = load_root + usr_name + '.txt'
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
                invalid_line += 1
                continue


            temp = date_info.split('-')
            m = temp[1]
            d = temp[2]
            y = temp[0]
            h = time_info.split(':')[0]
            minute = time_info.split(':')[1]
            second = time_info.split(':')[2]
            time_this = float(y+m+d+h+minute+second)
            images.append((time_this, line))


        save_path = save_root + usr_name + '.txt'
        images_sorted = sorted(images, key=lambda x:x[0])

        with open(save_path, 'w') as out_file:
            for time, line in images_sorted:
                out_file.write(line)
    return invalid_line
def remove_duplicate(usr_id, min_occurrence=5000, min_tag=3):
    load_root = time_path + usr_id + '.txt'
    save_root = clean_path + usr_id + '.txt'
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
    print "find duplicate:", duplicated
    write_file.close()
    return duplicated
def split_event(usr_id, min_picture_event = 30, time_diff = 3600*3, path=clean_path, save_path =event_path):
    usr_path = path+usr_id+'.txt'
    dates = []
    if not os.path.exists(usr_path):
        return
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
    if (len(dates) == 0):
        return
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
    if len(events) == 0:
        return
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
    out_file.close()
def clean_event_tag(usr_id, path=event_path, save_path = tag_path, save_path_1 = abandon_path):
    usr_path = path+usr_id+'.txt'
    tags = []
    lines = []
    event_start = []
    event_pre = -1
    count = 0
    if not os.path.exists(usr_path):
        return 0

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
    if len(event_start) == 0:
        return 0
    out_file = open(save_path + usr_id + '.txt', 'w')
    out_file_1 = open(save_path_1 + usr_id + '.txt','w')
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
            #print 'This is not an event(too short)!'
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
            #print 'Start: %d, End: %d, length: %d' % (start, end, end-start)
            #print count
            #print 'This is not an event!'
            for jj in xrange(start_ori, end_ori):
                out_file_1.write(lines[jj])
            continue
        #print count
        count = [(k, v) for k, v in count.iteritems()]
        #print count
        count_selected = [(u,v) for u,v in count if v >= len_event * 2/3]
        count_selected.sort(key=lambda x:x[1], reverse=True)

        for jj in xrange(start, end):
            lines_new = ''
            for tag, count in count_selected:
                lines_new += tag + ','
            lines_new = lines_new[:-1]
            out_file.write(lines_new + '\t' + lines[jj])
    out_file.close()
    print 'Cleaned event number:', len(event_start)
    return len(event_start)
def count_number_event(file_path=root+'clean_tags_5000_3_0710/log/create_event.txt'):
    count = 0
    valid_user = 0
    with open(file_path, 'r') as data:
        for line in data:
            if line[0] == '>':
                continue
            data = line.split('\n')[0]
            n = int(data.split(':')[-1])
            count += n
            if n > 0:
                valid_user += 1
    print 'Number of events: %d, Valid user: %d' % (count, valid_user)
def create_usr_url(path = tag_path):
    log_file = root + 'clean_tags_5000_3/log/create_url_list.txt'
    this_root = root + 'clean_tags_5000_3/download_events/'
    url_path = this_root + 'urls.txt'
    id_path = this_root + 'ids.txt'
    image_path = this_root + 'imgs.txt'
    file_names = [f for f in os.listdir(path) if f.endswith('.txt')]
    f_url = open(url_path,'w')
    f_id = open(id_path,'w')
    f_all = open(image_path,'w')
    for file in file_names:
        with open(log_file,'a') as f:
            f.write(file+'\n')
        file_path = path+file
        if os.path.getsize(file_path) == 0:
            continue
        usr = file.split('.')[0]
        if not os.path.exists(this_root+usr):
            os.mkdir(this_root + usr)
        with open(file_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                id = usr + '/' + meta[2] + '\n'
                url = meta[16] + '\n'
                f_url.write(url)
                f_id.write(id)
                f_all.write(line)
    f_url.close()
    f_id.close()
    f_all.close()

def exp_create_usr_url(usr_id = '7436989@N05', path = tag_path):
    this_root = root + 'clean_tags_5000_3/download_events_experiment/'
    url_path = this_root + 'urls.txt'
    id_path = this_root + 'ids.txt'
    f_url = open(url_path,'w')
    f_id = open(id_path,'w')

    file_path = this_root+usr_id
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    load_path = path + usr_id+'.txt'
    with open(load_path, 'r') as data:
        for line in data:
                meta = line.split('\t')
                id = usr_id + '/' + meta[2] + '\n'
                url = meta[16] + '\n'
                f_url.write(url)
                f_id.write(id)
    f_url.close()
    f_id.close()

'''

#before this do bash: find ./ -iname '*.jpg' -type f -size -5k >> not_downloaded_file.txt
def create_redownload_file(usr_id = '7436989@N05', path = tag_path):
    original_root = root + 'clean_tags_5000_3/events_clean_tags_new/'
    this_root = root + 'clean_tags_5000_3/download_events_experiment/'
    redownload_path = this_root + 'not_downloaded_file.txt'
    redownload_url = this_root + 'need_2_download.txt'
    line_number = []
    with open(redownload_path, 'r') as data:
        for line in data:
            line_number.append(int(line.split('\n')[0]))
    print "lines need redownloading:", len(line_number)
    f = open(redownload_url, 'w')
    url_path = original_root +
    i = 0
    count = 0
    with open(url_path):
        for line in data:
            if count == line_number[i]:
                i += 1
                f.write(line)
            count += 1
    f.close()
def create_usr_all(path = tag_path):
    log_file = root + 'clean_tags_5000_3/log/create_url_list.txt'
    this_root = root + 'clean_tags_5000_3/download_events/'
    #url_path = this_root + 'urls.txt'
    id_path = this_root + 'ids_copy.txt'
    image_path = this_root + 'imgs.txt'
    file_names = [f for f in os.listdir(path) if f.endswith('.txt')]
    #f_url = open(url_path,'w')
    f_id = open(id_path,'w')
    f_all = open(image_path,'w')

    for file in file_names:
        file_path = path+file
        if os.path.getsize(file_path) == 0:
            continue
        usr = file.split('.')[0]
        with open(file_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                id = usr + '/' + meta[2] + '\n'
                f_id.write(id)
                f_all.write(line)
    f_all.close()
    f_id.close()
'''

def create_usr_more(path = tag_path):
    log_file = root + 'clean_tags_5000_3_0710/log/create_url_list_more.txt'
    this_root = root + 'clean_tags_5000_3_0710/download_events/'
    url_path = this_root + 'urls_more.txt'
    id_path = this_root + 'ids_more.txt'
    image_path = this_root + 'imgs_more.txt'
    url_path_all = this_root + 'urls.txt'
    id_path_all = this_root + 'ids.txt'
    image_path_all = this_root + 'imgs.txt'
    #id_path_o = this_root + 'ids.txt'
    file_names = [f for f in os.listdir(path) if f.endswith('.txt')]
    f_url = open(url_path,'w')
    f_id = open(id_path,'w')
    f_all = open(image_path,'w')
    f_url_all = open(url_path_all,'w')
    f_id_all = open(id_path_all,'w')
    f_all_all = open(image_path_all,'w')
    for file in file_names:
        with open(log_file,'a') as f:
            f.write(file+'\n')
        file_path = path+file
        if os.path.getsize(file_path) == 0:
            continue
        usr = file.split('.')[0]
        if not os.path.exists(this_root+usr):
            os.mkdir(this_root + usr)
        with open(file_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                id = usr + '/' + meta[2]
                url = meta[16] + '\n'
                if not os.path.exists(this_root+id+'.jpg'):
                    f_url.write(url)
                    f_id.write(id+'\n')
                    f_all.write(line)
                f_url_all.write(url)
                f_id_all.write(id+'\n')
                f_all_all.write(line)

    f_url.close()
    f_id.close()
    f_all.close()
    f_url_all.close()
    f_id_all.close()
    f_all_all.close()
def create_ids_more(path = tag_path):
    #log_file = root + 'clean_tags_5000_3_0710/log/create_id_list_more.txt'
    this_root = root + 'clean_tags_5000_3_0710/download_events/'
    #url_path = this_root + 'urls_more.txt'
    id_path = this_root + 'ids_more.txt'
    image_path = this_root + 'imgs_more.txt'
    f_id = open(id_path,'w')
    with open(image_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            usr = meta[3]
            id = usr + '/' + meta[2]
            f_id.write(id+'\n')
    f_id.close()
def download_images(number):
    directory="/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/"
    log_file = root + 'clean_tags_5000_3_0710/log/download_url_'+str(number*50000+1)+'.txt'

    log_file_fail = root + 'clean_tags_5000_3_0710/log/download_url_fail_'+str(number*50000+1)+'.txt'
    id_path = directory + 'ids_more_' + str(number*50000+1) + '.txt'
    url_path = directory + 'urls_more_' + str(number*50000+1) + '.txt'
    with open(id_path, 'r') as f1, open(url_path,'r') as f2:
        for id, url in izip(f1, f2):
            id = id.split('\n')[0]
            url = url.split('\n')[0]
            try:
                urllib.urlretrieve(url, directory+id+'.jpg')
            except:
                with open(log_file_fail, 'a') as f:
                    f.write(id+'\t'+url+'\n')
            with open(log_file, 'a') as f:
                f.write(id+'\t'+url+'\n')


if __name__ == '__main__':



    '''
    usr_path = root + 'clean_tags_5000_3_0710/usrs_with_count.txt'
    count = 0
    with open(usr_path, 'r') as data:
        for line in data:
            count += 1
            meta = line.split()
            usr_id = meta[0]
            concat_files(usr_id)
    '''
    '''

    usr_path = root + 'clean_tags_5000_3_0710/usrs_with_count.txt'
    log_path = root + 'clean_tags_5000_3_0710/log/create_event.txt'
    count = 0
    with open(usr_path, 'r') as data:
        for line in data:
            count += 1
            meta = line.split()
            usr_id = meta[0]
            print '>>>>>>>>>>>>'+usr_id+'<<<<<<<<<<<<<'
            with open(log_path, 'a') as f_write:
                f_write.write('>>>>>>>>>>>>'+usr_id+'<<<<<<<<<<<<<\n')
            invalid_line = sort_usr_with_time(usr_id)
            duplicate = remove_duplicate(usr_id)
            split_event(usr_id)
            n = clean_event_tag(usr_id)

            str_this = 'Invalid line:%d, Duplicate:%d, Valid event:%d\n' % (invalid_line, duplicate, n)
            with open(log_path, 'a') as f_write:
                f_write.write(str_this)
    '''
    #create_usr_more()
    #create_ids_more()

    #args = sys.argv
    #assert len(args) > 1
    #number = (args[1])
    #download_images(int(number))

    #check_download()
    #create_redownload_file()
    #exp_create_usr_url()

    count_number_event()