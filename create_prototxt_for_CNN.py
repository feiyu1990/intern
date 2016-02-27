__author__ = 'wangyufei'

import sys
import h5py
root = '/home/feiyu1990/local/event_curation/'
#root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/'

def create_full_path(number, path=root):
    load_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_curation_CNN/all_images.txt'
    out_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_curation_CNN/full_paths.txt'
    prefix = root
    suffix = '.jpg'
    f = open(out_path, 'w')
    with open(load_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            string = prefix + meta[3]+'/'+meta[2] + suffix + ' 0\n'
            #string = line.split('\n')[0]+' 0\n'
            f.write(string)
    f.close()
def create_prototxt(number, path=root):
    out_path = '/mnt/ilcompf2d0/project/yuwang/CNN/models/JP/flickr_train_val_'+number+'.prototxt'
    load_path = '/mnt/ilcompf2d0/project/yuwang/CNN/models/JP/flickr_train_val_yw.prototxt'
    count = 0
    f = open(out_path,'w')
    with open(load_path,'r') as data:
        for line in data:
            count += 1
            if count == 8:
                f.write('source: \"/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/paths_'+number+'.txt\"\n')
            else:
                f.write(line)
    f.close()




if __name__ == '__main__':
    create_full_path(1)
    '''
    for i in xrange(93):
        j = i*140000+1
        number = str(j)
        create_full_path(number)
        create_prototxt(number)
    '''