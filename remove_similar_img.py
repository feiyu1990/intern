__author__ = 'wangyufei'

'''
import os
import sys
from datetime import datetime
from collections import Counter
import scipy.io
import numpy as np
import operator
from sklearn.preprocessing import normalize
from numpy import linalg
'''
root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/download_events/'
'''
def create_full_path(path=root):
    load_path = path+'downloaded_imgs_50001.txt'
    out_path = path+'paths_50001.txt'
    #prefix = root
    #suffix = '.jpg'
    f = open(out_path, 'w')
    with open(load_path, 'r') as data:
        for line in data:
            #string = prefix + line.split('\n')[0] + suffix + ' 0\n'
            string = line.split('\n')[0]+' 0\n'
            f.write(string)
    f.close()

def find_similar(path=root):
    threshold = 0.8
    usr_id = '7436989@N05'
    img_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/download_events_experiment/' + usr_id + '/'
    this_path = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/fc7_7436989@N05.mat'
    mat_path = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/7436989@N05.txt'
    save_path = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/dup_pairs.txt'
    delete_path = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/delete_lines.txt'
    save_path_2 = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/dup_pairs_url.txt'
    mat = scipy.io.loadmat(this_path)
    feature = mat['feature']
    feature = np.transpose(feature)
    print feature.shape
    feature_normalize = normalize(feature, axis=1)
    print linalg.norm(feature_normalize[0,:])
    time = []
    ids = []
    urls = []
    with open(mat_path, 'r') as data:
        for line in data:
            id = line.split('\t')[2]
            url = line.split('\t')[16]
            date = line.split('\t')[5]
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
            time.append(time_this)
            urls.append(url)
            ids.append(id)

    print len(time)
    last_time = 0
    feature_stack = []
    id_this = []
    url_this = []
    feature_stack_before = []
    id_before = []
    url_before = []
    dup_pair = []
    dup_pair_url = []
    dup_score = []
    dup_index = []
    #a = feature_normalize[ids.index('7667238282')]
    #b = feature_normalize[ids.index('7668934920')]
    #c = np.zeros(4096,)
    #for i in xrange(4096):
    #    c[i] = a[i]*b[i]
    #print np.sum(c)

    for i in xrange(feature_normalize.shape[0]):
        this_time = time[i]
        feature_this = feature_normalize[i,:]
        if this_time != last_time:
            feature_stack_before = feature_stack
            id_before = id_this
            url_before=url_this
            feature_stack = []
            id_this = []
            url_this = []
            last_time = this_time

        for j in xrange(len(feature_stack)):
            value = np.dot(feature_stack[j], feature_this)
            if value > threshold:
                dup_pair.append((id_this[j], ids[i]))
                dup_pair_url.append((url_this[j], urls[i]))
                dup_score.append(value)
                dup_index.append(i)
        for j in xrange(len(feature_stack_before)):
            value = np.dot(feature_stack_before[j], feature_this)
            if value > threshold:
                dup_pair.append((id_before[j], ids[i]))
                dup_pair_url.append((url_before[j], urls[i]))
                dup_score.append(value)
                dup_index.append(i)
        feature_stack.append(feature_this)
        id_this.append(ids[i])
        url_this.append(urls[i])
    print sum([a for a in dup_score if a == 1.0])
    f = open(save_path, 'w')
    for score, pair in zip(dup_score, dup_pair):
        string = img_root+pair[0]+'.jpg\t'+img_root+pair[1]+'.jpg\t'+str(score)+'\n'
        f.write(string)
    f.close()

    f = open(save_path_2, 'w')
    for score, pair in zip(dup_score, dup_pair_url):
        string = pair[0]+'\t'+pair[1]+'\t'+str(score)+'\n'
        f.write(string)
    f.close()
    print np.sort(dup_score[500:])

    arg_sort = np.argsort(dup_index)
    dup_index_sorted = [dup_index[i] for i in arg_sort]
    dup_score_sorted = [dup_score[i] for i in arg_sort]
    dup_pair_sorted = [dup_pair[i] for i in arg_sort]
    dup_pair_url_sorted = [dup_pair_url[i] for i in arg_sort]

    remove_list = {}
    for i in xrange(len(dup_index_sorted)):
        if dup_pair_sorted[i][0] in remove_list:
            continue
        remove_list[dup_pair_sorted[i][1]] = dup_index_sorted[i]
    remove_list_sorted = sorted(remove_list.items(), key=operator.itemgetter(1))
    remove_line = [k[1] for k in remove_list_sorted ]
    #second pass
    j = 0
    prev = 0
    feature_prev = feature_normalize[0,:]
    remove_list_2 = []
    values = []
    for i in xrange(1, len(ids)):
        feature_this = feature_normalize[i,:]
        if len(remove_line) == j:
            pass
        elif remove_line[j] == i:
            j+=1
            continue
        value = np.dot(feature_prev, feature_this)

        values.append(value)
        if value > threshold:
            remove_list_2.append(i)
        else:
            feature_prev = feature_this


    print values
    print max(values)
    remove_line = remove_line+remove_list_2
    remove_list_sorted = sorted(remove_line)

    f = open(delete_path, 'w')
    for value in remove_list_sorted:
        f.write(str(value+1)+'\n')
    f.close()
'''
def rmdup_txt(this_count):
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_2_round/'
    meta_path = root + 'txt_for_CNN/more_graduation_imgs_0719_separate_'+str(this_count)+'.txt'
    out_path = root + 'txt_cleaned/more_graduation_imgs_0719_cleaned_'+str(this_count)+'.txt'
    delete_path = root+'txt_cleaned/delete_lines_'+str(this_count)+'.txt'
    delete_lines = []
    with open(delete_path,'r') as data:
        for line in data:
            delete_lines.append(int(line.split('\n')[0]))
    f = open(out_path,'w')
    count = 1
    i = 0
    with open(meta_path,'r') as data:
        for line in data:
            if i < len(delete_lines):
                if delete_lines[i] == count:
                    i += 1
                    count += 1
                    continue
            f.write(line)
            count += 1
    f.close()
if __name__ == '__main__':
    #find_similar()
    for i in xrange(123):
        this_count = i*10000+1
        rmdup_txt(this_count)
