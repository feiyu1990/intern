__author__ = 'wangyufei'
'''
tags_with_count() -> count occurrence of all tags from 100M data
clean_tags(min_occurrence=5000) -> filter tags with minimum occurrence
condor_clean_all_list(i) -> clean the list of 100M data using condor
clean_usrs_with_count() -> count # of valid images of each users and sort them by #
usrs_with_count() -> count # of images of each users (all 100M)
DEPRECATED sort_usrs() -> sort the users by # of images
DEPRECATED (TOO SLOW) condor_valid_all_list_from_clean(i, usr_id) -> generate valid image list from certain user, using condor
'''

import numpy as np
import sys
import random
import os
import optparse
from os.path import isfile, join
from os import listdir
import cPickle
import urllib
import operator

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
dataset_root = '/mnt/ilcompf2d0/project/yuwang/datasets/download_data/yfcc100m_dataset-'
def tags_with_count():
    count = 0

    seen = {}
    for i in xrange(10):
        all_tag_path = dataset_root+str(i)
        with open(all_tag_path, 'r') as data:
            for line in data:
                line = line.split('\n')[0]
                meta = line.split('\t')
                tags = meta[8]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(',')
                for tag_individual in tag_list:
                    if '%' in tag_individual:
                        #try:
                        #    tag_individual=urllib.unquote(tag_individual).decode('utf8')
                        #except:
                            continue
                    if tag_individual not in seen:
                        seen[tag_individual] = 1
                    else:
                        seen[tag_individual] += 1

                    count += 1
                    if count % 100000 == 0:
                        print count,'...'

    out_file_path = root+'tags_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key in seen.keys():
        value = seen[key]
        try:
            out_f.write(key+'\t'+str(value)+'\n')
        except:
            print "write tag failed:", key, value
            del seen[key]


    f = open(root+'tags_with_count.cpickle',"wb")
    cPickle.dump(seen,f)
    f.close()
def clean_tags(min_occurrence):
    print 'Cleaning tags...'
    f = open(root+'tags_with_count.cpickle',"rb")
    tags_with_occurance = cPickle.load(f)
    f.close()
    tags_list = []
    seen = {}

    save_path = root+'clean_tags_largerthan_'+str(min_occurrence)+'.txt'
    out_file = open(save_path,'w')
    for key, value in tags_with_occurance.iteritems():
        if value >=min_occurrence:
            tags_list.append((key, value))
    tags_sorted = sorted(tags_list, key=lambda x:x[1], reverse=True)
    for key, value in tags_sorted:
            if not key.isdigit():
                out_file.write(key+'\t'+str(value)+'\n')
                seen[key] = value
    out_file.close()

    f = open(root+'clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"wb")
    print 'Saving tags with occurrence larger than ', min_occurrence, ' to ', save_path
    cPickle.dump(seen,f)
    f.close()
def condor_clean_all_list(count, min_occurrence=5000, min_tag=3):
    count_all = 0
    f = open(root+'clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"r")
    tags_all = cPickle.load(f)
    f.close()
    save_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in xrange(count,count+1):
        save_path = save_root + str(i) + '.txt'
        out_file = open(save_path,'w')
        all_tag_path = dataset_root+str(i)
        with open(all_tag_path, 'r') as data:
            for line in data:
                count_all += 1
                if count_all % 1000 == 0:
                    print count_all,'...'
                line = line.split('\n')[0]
                meta = line.split('\t')
                tags = meta[8]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(',')
                if tag_list < min_tag:
                    continue
                count = 0
                str_tag = ''
                for tag_individual in tag_list:
                    if '%' in tag_individual:
                        continue
                    if tag_individual in tags_all.keys():
                        count += 1
                        str_tag += tag_individual + ','
                if count >= min_tag:
                    str_tag = str_tag[:-1]
                    meta_list = meta[:8]
                    meta_list.append(str_tag)
                    meta_list.extend(meta[9:])
                    str_out = '\t'.join(meta_list)
                    str_out += '\n'
                    out_file.write(str_out)
        out_file.close()
def condor_clean_all_tags(count, min_occurrence=5000, min_tag=3):
    count_all = 0
    f = open(root+'clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"r")
    tags_all = cPickle.load(f)
    f.close()
    save_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in xrange(count,count+1):
        save_path = save_root + str(i) + '_all_tags.txt'
        out_file = open(save_path,'w')
        all_tag_path = dataset_root+str(i)
        with open(all_tag_path, 'r') as data:
            for line in data:
                count_all += 1
                if count_all % 10000 == 0:
                    print count_all,'...'
                line = line.split('\n')[0]
                meta = line.split('\t')
                tags = meta[8]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(',')
                if tag_list < min_tag:
                    continue
                count = 0
                str_tag = ''
                for tag_individual in tag_list:
                    if '%' in tag_individual:
                        continue
                    if tag_individual in tags_all.keys():
                        count += 1
                        str_tag += tag_individual + ','
                if count >= min_tag:
                    str_tag = str_tag[:-1]
                    meta_list = meta[:8]
                    meta_list.append(str_tag)
                    meta_list.extend(meta[9:])
                    #str_out = '\t'.join(meta_list)
                    #str_out += '\n'
                    out_file.write(line+'\n')
        out_file.close()
def clean_usrs_with_count(min_occurrence=5000, min_tag=3):
    count = 0
    seen = {}
    for i in xrange(10):
        all_tag_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/' + str(i)+'.txt'
        with open(all_tag_path, 'r') as data:
            for line in data:
                line = line.split('\n')[0]
                meta = line.split('\t')
                usrs = meta[1]
                if usrs not in seen:
                    seen[usrs] = 1
                else:
                    seen[usrs] += 1
                count += 1
                if count % 100000 == 0:
                    print count,'...'

    seen_sorted = sorted(seen.items(), key=operator.itemgetter(1),reverse=True)
    out_file_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+'usrs_with_count.txt'
    f = open(root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+'usrs_with_count.cpickle',"wb")
    cPickle.dump(seen_sorted,f)
    f.close()
    out_f = open(out_file_path, 'w')
    for key, value in seen_sorted:
        out_f.write(key+'\t'+str(value)+'\n')
    out_f.close()
def usrs_with_count():
    count = 0
    seen = {}
    for i in xrange(10):
        all_tag_path = dataset_root+str(i)
        with open(all_tag_path, 'r') as data:
            for line in data:
                line = line.split('\n')[0]
                meta = line.split('\t')
                tags = meta[1]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(',')
                for tag_individual in tag_list:
                    if '%' in tag_individual:
                            continue
                    if tag_individual not in seen:
                        seen[tag_individual] = 1
                    else:
                        seen[tag_individual] += 1

                    count += 1
                    if count % 100000 == 0:
                        print count,'...'

    out_file_path = root+'usrs_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key in seen.keys():
        value = seen[key]
        out_f.write(key+'\t'+str(value)+'\n')



    f = open(root+'usrs_with_count.cpickle',"wb")
    cPickle.dump(seen,f)
    f.close()
def sort_usrs():
    print 'Sorting usrs...'
    f = open(root+'usrs_with_count.cpickle',"rb")
    tags_with_occurance = cPickle.load(f)
    f.close()
    save_path = root+'usrs_with_count.txt'
    out_file = open(save_path,'w')
    tags_sorted = sorted(tags_with_occurance, key=lambda x:x[1], reverse=True)
    for key, value in tags_sorted:
            #if not key.isdigit():
                out_file.write(key+'\t'+str(value)+'\n')
    out_file.close()


'''
def valid_all_list_from_all(usr_id, min_occurrence, min_tag):
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    f = open(root+'clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"rb")
    tags_with_occurrence = cPickle.load(f)
    f.close()
    count_all = 0
    save_path = root+'clean_usr_'+usr_id+'_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    out_file = open(save_path,'w')

    for i in xrange(10):
        print i
        data_path = dataset_root+str(i)
        with open(data_path, 'r') as data:
            for line in data:
                count_all = count_all + 1
                if count_all % 1000 == 0:
                    print count_all,'...'
                line = line.split('\n')[0]
                meta = line.split('\t')
                tags = meta[8]
                #url = meta[14]
                tag_list = tags.split(',')
                if tag_list < min_tag:
                    continue
                count = 0
                str_tag = ''
                for tag_individual in tag_list:
                    if '%' in tag_individual:
                        continue
                    if tag_individual in tags_with_occurrence.keys():
                        count += 1
                        str_tag += tag_individual + ','
                if count >= min_tag:
                    str_tag = str_tag[:-1]
                    meta_list = meta[:8]
                    meta_list.append(str_tag)
                    meta_list.extend(meta[9:])
                    str_out = '\t'.join(meta_list)
                    str_out += '\n'
                    out_file.write(str_out)
    out_file.close()
'''

def condor_valid_all_list_from_clean(i, usr_id, min_occurrence=5000, min_tag=3):
    print i
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'

    count_all = 0
    save_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/usr_'+usr_id+'_'+str(i)+'.txt'
    out_file = open(save_path,'w')

    data_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/' + str(i)+'.txt'
    with open(data_path, 'r') as data:
            for line in data:
                count_all += count_all
                if count_all % 1000 == 0:
                    print count_all,'...'
                meta = line.split('\t')
                usr_this = meta[1]
                if usr_this == usr_id:
                    out_file.write(line)
    out_file.close()

def count_occurrence(str_to_find, i):
    print i

    count_all = 0
    lines = []
    count = 0
    data_path = root+'clean_tags_5000_3/' + str(i)+'_all_tags.txt'
    with open(data_path, 'r') as data:
            for line in data:
                count_all += 1
                if count_all % 10000 == 0:
                    print count_all,'...'
                meta = line.split('\t')
                this_str = meta[0]
                if this_str == str_to_find:
                    count += 1
                    lines.append(line)
    print lines
    print count


if __name__ == '__main__':
    #min_occurrence = 50
    #min_tag = 3

    #tags_with_count()
    #clean_tags(5000)

    #valid_all_list_from_all('81688406@N00',5000,3)


    #args = sys.argv
    #assert len(args) > 1
    #i = int(args[1])
    ##usr_id = args[2]
    ##clean_all_list(i)
    #condor_clean_all_tags(i)

    ##clean_usrs_with_count()

    count_occurrence(str_to_find="7904513136",i=9)