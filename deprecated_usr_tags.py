__author__ = 'wangyufei'

'''
DEPRECATED!!! Just for sample data
tags_with_count() -> count tag occurrence
cleaned_usr_with_count() -> sort users according to # of images
clean_tags -> filter tags with minimum number of occurrence
valid_image_list() -> generate image list with valid tags (only id and tags)
valid_image_list_from_all() -> generate image list with valid tags (only id url tags)
valid_all_list_from_all() -> generate image list with valid tags
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


def tags_with_count():
    all_tag_path = '../datasets/usr_tags.txt'
    count = 0

    seen = {}
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
                    try:
                        tag_individual=urllib.unquote(tag_individual).decode('utf8')
                    except:
                        continue
                if tag_individual not in seen:
                    seen[tag_individual] = 1
                else:
                    seen[tag_individual] += 1

                count += 1
                if count % 10000 == 0:
                    print count,'...'

    out_file_path = '../datasets/tags_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key in seen.keys():
        value = seen[key]
        try:
            out_f.write(key+'\t'+str(value)+'\n')
        except:
            print "write tag failed:", key, value
            del seen[key]

    out_f.close()
    f = open('../datasets/tags_with_count.cpickle',"wb")
    cPickle.dump(seen,f)
    f.close()
def cleaned_usr_with_count(min_occurrence, min_tag):
    list_path = '../datasets/clean_all_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    usr_count_path = '../datasets/clean_usr_count_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    count = 0

    seen = {}
    with open(list_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            usr = meta[1]
            if usr not in seen:
                seen[usr] = 1
            else:
                seen[usr] += 1
            count += 1
            if count % 10000 == 0:
                print count,'...'

    out_f = open(usr_count_path, 'w')
    #seen = sorted(seen, key=lambda x:x.value, reverse=True)
    sorted_seen = sorted(seen.items(), key=operator.itemgetter(1), reverse=True)


    for key, value in sorted_seen:
        out_f.write(key+'\t'+str(value)+'\n')
    out_f.close()

    #f = open('../datasets/tags_with_count.cpickle',"wb")
    #cPickle.dump(seen,f)
    #f.close()
def clean_tags(min_occurrence):
    print 'Cleaning tags...'
    f = open('../datasets/tags_with_count.cpickle',"rb")
    tags_with_occurance = cPickle.load(f)
    f.close()
    tags_list = []
    seen = {}

    save_path = '../datasets/clean_tags_largerthan_'+str(min_occurrence)+'.txt'
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

    f = open('../datasets/clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"wb")
    print 'Saving tags with occurrence larger than ', min_occurrence, ' to ', save_path
    cPickle.dump(seen,f)
    f.close()

def valid_image_list(min_occurrence, min_tag):
    print 'Generating image lists with more than '+str(min_tag)+' cleaned tags...'
    f = open('../datasets/clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"rb")
    tags_with_occurrence = cPickle.load(f)
    f.close()

    data_path = '../datasets/usr_tags.txt'
    save_path = '../datasets/clean_usr_tags_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    out_file = open(save_path,'w')
    count_all = 0
    with open(data_path, 'r') as data:
        for line in data:
            count_all = count_all + 1
            if count_all % 1000 == 0:
                print count_all,'...'
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[1]
            id = meta[0]
            tag_list = tags.split(',')
            if tag_list < min_tag:
                continue
            count = 0
            str_tag = ''
            for tag_individual in tag_list:
                if tag_individual in tags_with_occurrence.keys():
                    count += 1
                    str_tag += tag_individual + ','
            if count >= min_tag:
                str_tag = str_tag[:-1]
                temp = (count_all-1) / 10000
                str_id = str(temp).zfill(5) + '/' + id + '.jpg'
                out_file.write(str_id + '\t' + str_tag + '\n')
    out_file.close()

def valid_image_list_from_all(min_occurrence, min_tag):
    print 'Generating image lists with more than '+str(min_tag)+' cleaned tags...'
    f = open('../datasets/clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"rb")
    tags_with_occurrence = cPickle.load(f)
    f.close()

    data_path = '../datasets/all_1m.txt'
    save_path = '../datasets/clean_usr_url_tags_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    out_file = open(save_path,'w')
    count_all = 0
    with open(data_path, 'r') as data:
        for line in data:
            count_all = count_all + 1
            if count_all % 1000 == 0:
                print count_all,'...'
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[8]
            id = meta[0]
            url = meta[14]
            tag_list = tags.split(',')
            if tag_list < min_tag:
                continue
            count = 0
            str_tag = ''
            for tag_individual in tag_list:
                if tag_individual in tags_with_occurrence.keys():
                    count += 1
                    str_tag += tag_individual + ','
            if count >= min_tag:
                str_tag = str_tag[:-1]
                temp = (count_all-1) / 10000
                str_id = str(temp).zfill(5) + '/' + id + '.jpg'
                out_file.write(str_id + '\t'+url+'\t'+ str_tag + '\n')
    out_file.close()

def valid_all_list_from_all(min_occurrence, min_tag):
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    f = open('../datasets/clean_tags_largerthan_'+str(min_occurrence)+'.cpickle',"rb")
    tags_with_occurrence = cPickle.load(f)
    f.close()

    data_path = '../datasets/all_1m.txt'
    save_path = '../datasets/clean_all_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    out_file = open(save_path,'w')
    count_all = 0
    with open(data_path, 'r') as data:
        for line in data:
            count_all = count_all + 1
            if count_all % 1000 == 0:
                print count_all,'...'
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[8]
            id = meta[0]
            #url = meta[14]
            tag_list = tags.split(',')
            if tag_list < min_tag:
                continue
            count = 0
            str_tag = ''
            for tag_individual in tag_list:
                if '%' in tag_individual:
                    try:
                        tag_individual=urllib.unquote(tag_individual).decode('utf8')
                    except:
                        continue
                if tag_individual in tags_with_occurrence.keys():
                    count += 1
                    str_tag += tag_individual + ','
            if count >= min_tag:
                str_tag = str_tag[:-1]
                temp = (count_all-1) / 10000
                str_id = str(temp).zfill(5) + '/' + id + '.jpg'
                meta_list = [str_id]
                meta_list.extend(meta[1:8])
                meta_list.append(str_tag)
                meta_list.extend(meta[9:])
                str_out = '\t'.join(meta_list)
                str_out += '\n'
                out_file.write(str_out)

    out_file.close()


if __name__ == '__main__':
    min_occurrence = 50
    min_tag = 3
    tags_with_count()
    clean_tags(min_occurrence)
    #if not os.path.isfile('../datasets/clean_tags_largerthan_'+str(min_occurrence)+'.cpickle'):
    #    clean_tags(min_occurrence)
    cleaned_usr_with_count(min_occurrence, min_tag)
    valid_all_list_from_all(min_occurrence, min_tag)