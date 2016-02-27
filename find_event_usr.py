__author__ = 'wangyufei'
import numpy as np
import sys
import random
import os
import optparse
from os.path import isfile, join
from os import listdir
import cPickle
import urllib
import random
import operator

min_occurrence = 5000
min_tag = 3


def find_usr(tag,i=0):
    list_path = '../datasets/all_data/clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/'+str(i)+'.txt'
    usr_dict = {}
    with open(list_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[8]
            tag_individuals = tags.split(',')
            has_key = False
            for tag_this in tag_individuals:
                if tag in tag_this:
                    has_key = True
            if not has_key:
                continue
            #if not tag in tag_individuals:
            #    continue
            usr = meta[1]
            if usr in usr_dict.keys():
                usr_dict[usr] += 1
            else:
                usr_dict[usr] = 1
    usr_dict_sorted = sorted(usr_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    print usr_dict_sorted[:100]


if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    event_name = (args[1])
    if len(args) > 2:
        i = args[2]
    else:
        i = 0
    find_usr(event_name, i)