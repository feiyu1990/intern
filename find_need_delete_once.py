__author__ = 'wangyufei'

import sys
import random

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'

def find_need_delete_once_more(id, number):
    path = root + 'clean_ids_'+number+'.txt'
    line_number = 1
    with open(path, 'r') as data:
        for line in data:
            if id == line.split('\n')[0]:
                print line_number
            line_number += 1
if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 2
    id = args[1]
    i = int(args[2])*140000+1
    find_need_delete_once_more(id, str(i))
