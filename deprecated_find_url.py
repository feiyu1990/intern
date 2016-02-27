__author__ = 'wangyufei'

'''
DEPRECATED
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

dataset_n = 0
file_path= '../datasets/download_data/dataset_0/yfcc100m_dataset-'+str(dataset_n)
#file_path= '../datasets/data_sample.txt'
output_path_url = '../datasets/download_data/dataset_'+str(dataset_n)+'/urls_rest'
output_path_id = '../datasets/download_data/dataset_'+str(dataset_n)+'/ids_rest'
download_path = '../datasets/flickr1M/'
'''not used
def download():
    #url = []
    #id = []
    unsuccessful = 0
    count = 0
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    with open(file_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            url = meta[14]
            id = meta[0]
            try:
                urllib.urlretrieve(url, download_path+id+'.jpg')
                count = count + 1
            except:
                unsuccessful = unsuccessful + 1
            if count % 1000 == 0:
                print count
            if count == 1000000:
                break

    print '# of Invalid url: %d' % unsuccessful

def extract_url():
    #url = []
    #id = []
    count = 0
    #if not os.path.exists(download_path):
    #    os.mkdir(download_path)
    i_last = 1
    out_url = open(output_path_url+str(i_last)+'.txt','w')
    out_id = open(output_path_id+str(i_last)+'.txt','w')
    with open(file_path, 'r') as data:
        for line in data:
            count = count + 1
            if count < 55000:
                continue
            i = (count - 55000) / 10000 + 1
            #i = count / 10000 + 1
            if i != i_last:
                out_url.close()
                out_id.close()
                out_url = open(output_path_url+str(i)+'.txt','w')
                out_id = open(output_path_id+str(i)+'.txt','w')
            meta = line.split('\t')
            url = meta[14]
            id = meta[0]
            out_url.write(url+'\n')
            out_id.write(id+'\n')
            if count == 1000000 + 55000:
                break
    out_url.close()
    out_id.close()
'''
def extract_url_one():
    #url = []
    #id = []
    count = 0
    #if not os.path.exists(download_path):
    #    os.mkdir(download_path)
    out_url = open(output_path_url+'.txt','w')
    out_id = open(output_path_id+'.txt','w')
    with open(file_path, 'r') as data:
        for line in data:
            count = count + 1
            if count <= 1000000:
                continue
            meta = line.split('\t')
            url = meta[14]
            id = meta[0]
            out_url.write(url+'\n')
            out_id.write(id+'\n')
            #if count == 1000000:
            #   break
    out_url.close()
    out_id.close()

def extract_anything_one():
    output_path = '../datasets/all_1m'
    count = 0
    out = open(output_path+'.txt','w')
    with open(file_path, 'r') as data:
        for line in data:
            count = count + 1
            meta = line.split('\t')
            id = meta[0]
            tag = meta[8]
            out.write(id+'\t'+tag+'\n')
            if count == 1000000:
                break
    out.close()



if __name__ == '__main__':
    extract_url_one()
