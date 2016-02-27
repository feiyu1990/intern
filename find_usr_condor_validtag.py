__author__ = 'wangyufei'
'''
This is for:
1.mkdir_usrs(min_images) -> generate empty user text file with at least 100 images
2.valid_all_usr_from_clean(i) -> user text files with at least 100 images
3.sort_usr_with_time() -> sort all user text files according to timestamp
'''

import sys
import os
import cPickle

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'

def mkdir_usrs(i, min_images=100):
    load_path = root+'clean_tags_5000_3/usrs_with_count.cpickle'
    f = open(load_path, 'r')
    usrs_count = cPickle.load(f)
    f.close()
    for key, value in usrs_count:
        #print usrs_count[key]
        #print key
        if value < min_images:
            continue
        save_root = root+'clean_tags_5000_3/all_users_'+str(i)+'/'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        f = open(save_root+key+'.txt', 'w')
        f.close()
def valid_all_usr_from_clean(i, min_occurrence=5000, min_tag=3):
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    print i
    save_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/all_users/'

    count_all = 0

    #for i in xrange(10):
    data_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/' + str(i)+'.txt'
    with open(data_path, 'r') as data:
            for line in data:
                count_all += 1
                if count_all % 1000 == 0:
                    print count_all,'...'
                meta = line.split('\t')
                usr_this = meta[1]
                save_path = save_root + usr_this + '.txt'
                if not os.path.isfile(save_path):
                    continue
                with open(save_path, 'a') as out_file:
                    out_file.write(line)
def valid_all_usr_from_clean_10(i, min_occurrence=5000, min_tag=3):
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    print i
    save_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/all_users_'+str(i)+'/'
    log_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/log/'+str(i)+'.txt'
    count_all = 0

    data_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/new_' + str(i)+'.txt'
    data = open(data_path, 'r')
    for line in data:
        count_all = count_all + 1
        if count_all % 1000 == 0:
            with open(log_root, 'a') as log_file:
                log_file.write(str(count_all) + '\n')
            print count_all, '...'
        meta = line.split('\t')
        usr_this = meta[1]
        save_path = save_root + usr_this + '.txt'
        if not os.path.isfile(save_path):
            continue
        with open(save_path, 'a') as out_file:
            out_file.write(line)
    data.close()


def sort_usr_with_time(min_occurrence=5000, min_tag=3):
    #print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    load_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+ '/all_users/'
    save_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+ '/all_users_time/'
    for usr in os.listdir(load_root):
        print usr, '...'
        if usr.endswith("txt"):
            list_path = load_root + usr + '.txt'
            images = []
            with open(list_path, 'r') as data:
                for line in data:
                    meta = line.split('\t')
                    date = meta[3]
                    temp = date.split(' ')
                    date_info = temp[0]
                    time_info = temp[1]
                    temp = date_info.split('-')
                    m = temp[1]
                    d = temp[2]
                    y = temp[0]
                    h = time_info.split(':')[0]
                    minute = time_info.split(':')[1]
                    second = time_info.split(':')[2]
                    time_this = float(y+m+d+h+minute+second)
                    images.append((time_this, line))
            save_path = save_root + usr + '.txt'
            images_sorted = sorted(images, key=lambda x:x[0])

            with open(save_path, 'w') as out_file:
                for time, line in images_sorted:
                    out_file.write(line)


if __name__ == '__main__':
    args = sys.argv
    assert len(args) >1
    i = int(args[1])
    valid_all_usr_from_clean_10(i)
    #for i in xrange(10,18):
    #    print i
    #    mkdir_usrs(i)
