__author__ = 'wangyufei'

import os
import sys
from datetime import datetime
from collections import Counter


root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/'

def sort_usr_with_time(usr_name, min_occurrence=5000, min_tag=3):
    invalid_line = 0
    load_root = root
    save_path = root+usr_name+'_time.txt'
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


        images_sorted = sorted(images, key=lambda x:x[0])

        with open(save_path, 'w') as out_file:
            for time, line in images_sorted:
                out_file.write(line)
    return invalid_line
def remove_duplicate(usr_name, min_occurrence=5000, min_tag=3):
    load_root  = root+usr_name+'_time.txt'

    save_root = root + usr_name + '_nodup.txt'
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

if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    usr_id = args[1]
    sort_usr_with_time(usr_id)
    remove_duplicate(usr_id)