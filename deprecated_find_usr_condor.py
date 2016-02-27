__author__ = 'wangyufei'
'''
This is for:
1.tags_with_count -> find occurrence of each possible tags.
2.clean_tags() -> clean tags with minimum occurrence (5000)
3.condor_clean_all_list(i) -> generate filtered lists with at least 3 valid tags using condor
4.clean_usrs_with_count() -> generate # of images of users using cleaned list

usrs_with_count() -> generate # of images of users using original list
sort_usrs() -> sort users according to # of images
DEPRECATED(TOO SLOW) condor_valid_all_list_from_clean(i, usr_id) -> images from one usr (cleaned) using condor
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


'''
def merge_tags():
    f = open('../datasets/all_data/tags_with_count_0.cpickle',"r")
    tag_ori = cPickle.load(f)
    f.close()
    for i in xrange(1,10):
        f = open('../datasets/all_data/tags_with_count_'+str(i)+'.cpickle',"r")
        tag_new = cPickle.load(f)
        f.close()
        for key, value in tag_new.iteritems():
            if key in tag_ori:
                tag_ori[key] += value
            else:
                tag_ori[key] = value

'''
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

    out_file_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+'usrs_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key in seen.keys():
        value = seen[key]
        out_f.write(key+'\t'+str(value)+'\n')

    f = open(root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+'usrs_with_count.cpickle',"wb")
    cPickle.dump(seen,f)
    f.close()
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
            if not key.isdigit():
                out_file.write(key+'\t'+str(value)+'\n')
    out_file.close()
def condor_valid_all_list_from_clean(i, usr_id, min_occurrence=5000, min_tag=3):
    print i
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    save_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/usr_'+usr_id + '/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    count_all = 0
    save_path = save_root+'usr_'+str(i)+'.txt'
    out_file = open(save_path,'w')

    data_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/' + str(i)+'.txt'
    with open(data_path, 'r') as data:
            for line in data:
                count_all += 1
                if count_all % 1000 == 0:
                    print count_all,'...'
                meta = line.split('\t')
                usr_this = meta[1]
                if usr_this == usr_id:
                    out_file.write(line)
    out_file.close()


if __name__ == '__main__':

    args = sys.argv
    assert len(args) >2
    i = int(args[1])
    usr_id = args[2]
    condor_valid_all_list_from_clean(i, usr_id)


