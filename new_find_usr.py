__author__ = 'wangyufei'

import sys
import os
import cPickle

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
if not os.path.exists(root+'clean_tags_5000_3/0706/'):
    os.mkdir(root+'clean_tags_5000_3/0706/')
if not os.path.exists(root+'clean_tags_5000_3/0706/log'):
    os.mkdir(root+'clean_tags_5000_3/0706/log')
def mkdir_usrs(i, min_images=100):
    load_path = root+'clean_tags_5000_3/usrs_with_count.cpickle'
    f = open(load_path, 'r')
    usrs_count = cPickle.load(f)
    f.close()
    count_all = 0
    for key, value in usrs_count:
        count_all = count_all + 1
        if count_all % 1000 == 0:
            print count_all, '...'
        #print usrs_count[key]
        #print key
        if value < min_images:
            continue
        save_root = root+'clean_tags_5000_3/0706/all_users_'+str(i)+'/'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        f = open(save_root+key+'.txt', 'w')
        f.close()

def valid_all_usr_from_clean_10(i, min_occurrence=5000, min_tag=3):
    print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    print i
    save_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/0706/all_users_'+str(i)+'/'
    log_root = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/0706/log/'+str(i)+'.txt'
    count_all = 0

    data_path = root+'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/' + str(i)+'_all_tags.txt'
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



if __name__ == '__main__':
    args = sys.argv
    assert len(args) >1
    i = int(args[1])
    ##valid_all_usr_from_clean(i)
    valid_all_usr_from_clean_10(i)
    ##sort_usr_with_time()
    #for i in xrange(10):
    #mkdir_usrs(i)