__author__ = 'wangyufei'

import sys
import random

import PIL
from PIL import Image

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'

def extract_anything_one(path=root, number='1'):
    input_path = path+'need_delete_imgs_'+number+'.txt'
    output_path = path+'need_delete_ids_'+number+'.txt'
    out = open(output_path,'w')
    with open(input_path, 'r') as data:
        for line in data:
            meta = line.split(' ')
            id = meta[2]
            usr = meta[3]
            print id,usr
            out.write(usr+'/'+id+'\n')
    out.close()

def rm_invalid_txt(number='1', path=root):
    meta_path1 = path + 'imgs_'+number+'.txt'
    meta_path2 = path + 'ids_'+number+'.txt'
    meta_path3 = path + 'urls_'+number+'.txt'
    out_path1 = path + 'clean_imgs_'+number+'.txt'
    out_path2 = path + 'clean_ids_'+number+'.txt'
    out_path3 = path + 'clean_urls_'+number+'.txt'

    delete_path = path + 'need_delete_line_'+number+'.txt'
    delete_lines = []
    with open(delete_path,'r') as data:
        for line in data:
            delete_lines.append(int(line.split('\n')[0]))
    print len(delete_lines)
    print delete_lines
    f1 = open(out_path1,'w')
    count = 1
    i = 0
    with open(meta_path1,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f1.write(line)
                count += 1
    f1.close()
    print i

    f2 = open(out_path2,'w')
    count = 1
    i = 0
    with open(meta_path2,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f2.write(line)
                count += 1
    f2.close()
    f3 = open(out_path3,'w')
    count = 1
    i = 0
    with open(meta_path3,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f3.write(line)
                count += 1
    f3.close()
def rmdup_txt(number='1', path=root):
    meta_path1 = path + 'clean_imgs_'+number+'.txt'
    meta_path2 = path + 'clean_ids_'+number+'.txt'
    meta_path3 = path + 'clean_urls_'+number+'.txt'
    out_path1 = path + 'nosim_imgs_'+number+'.txt'
    out_path2 = path + 'nosim_ids_'+number+'.txt'
    out_path3 = path + 'nosim_urls_'+number+'.txt'

    delete_path = path + 'delete_lines_'+number+'.txt'
    delete_lines = []
    with open(delete_path,'r') as data:
        for line in data:
            delete_lines.append(int(line.split('\n')[0]))
    print len(delete_lines)
    print delete_lines
    f1 = open(out_path1,'w')
    count = 1
    i = 0
    with open(meta_path1,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f1.write(line)
                count += 1
    f1.close()
    print i

    f2 = open(out_path2,'w')
    count = 1
    i = 0
    with open(meta_path2,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f2.write(line)
                count += 1
    f2.close()

    f3 = open(out_path3,'w')
    count = 1
    i = 0
    with open(meta_path3,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f3.write(line)
                count += 1
    f3.close()
def rmdup_txt_temp(number='1', path='/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/download_events_experiment/'):
    meta_path1 = path + '7436989@N05.txt'
    meta_path2 = path + 'ids.txt'
    meta_path3 = path + 'urls.txt'
    out_path1 = path + 'nosim_imgs.txt'
    out_path2 = path + 'nosim_ids.txt'
    out_path3 = path + 'nosim_urls.txt'

    delete_path = path + 'delete_lines.txt'
    delete_lines = []
    with open(delete_path,'r') as data:
        for line in data:
            delete_lines.append(int(line.split('\n')[0]))
    print len(delete_lines)
    print delete_lines
    f1 = open(out_path1,'w')
    count = 1
    i = 0
    with open(meta_path1,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                    i += 1
                    count += 1
            else:
                f1.write(line)
                count += 1
    f1.close()
    print i

    f2 = open(out_path2,'w')
    count = 1
    i = 0
    with open(meta_path2,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                    i += 1
                    count += 1
            else:
                f2.write(line)
                count += 1
    f2.close()
    f3 = open(out_path3,'w')
    count = 1
    i = 0
    with open(meta_path3,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                    i += 1
                    count += 1
            else:
                f3.write(line)
                count += 1
    f3.close()
def write_csv_temp(number='1', path='/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3/download_events_experiment/'):
    meta_path = path + 'nosim_imgs.txt'
    event_start = []
    event_id = []
    tags = []
    count = 0
    last_event = -1
    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]
            if this_event != last_event:
                event_id.append(this_event)
                tags.append(meta[0])
                event_start.append(count)
                last_event = this_event
            count += 1
    event_start.append(count)
    event_length = []
    for i in xrange(1,len(event_start)):
        event_length.append(event_start[i]-event_start[i-1])
    max_event = 20
    event_id_sample = [i for i in xrange(max_event)]
    #event_id_sample = [ event_id[i] for i in sorted(random.sample(xrange(len(event_id)), max_event)) ]
    print event_id_sample

    out_path = path + 'amt.csv'
    f = open(out_path, 'w')
    max_size = max([event_length[i] for i in event_id_sample])
    f.write('num_image,tags')
    for i in xrange(max_size):
        f.write(',')
        f.write('image'+str(i+1))
    f.write('\n')
    ii = event_id_sample[0]
    i = 0
    count = 0
    tag_written = False
    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]
            url = meta[16]
            if int(this_event) not in event_id_sample:
                count += 1
            else:
                print this_event
                if not tag_written:
                    f.write(str(event_length[ii]))
                    tag_this = tags[ii].split(',')
                    tag_this=':'.join(tag_this)
                    f.write(','+tag_this)
                    tag_written = True
                f.write(','+url)
                if count == event_start[ii+1]-1:
                    for kk in xrange(max_size-event_length[ii]):
                        f.write(',NA')
                    f.write('\n')
                    i += 1
                    if i < len(event_id_sample):
                        ii = event_id_sample[i]
                    tag_written = False
                count += 1
    f.close()

def find_need_delete_once_more(id, number):
    path = root + 'clean_ids_'+number+'.txt'
    line_number = 1
    with open(path, 'r') as data:
        for line in data:
            if id == line.split('\n')[0]:
                print line_number
            line_number += 1

def check_download(number, path=root):
    this_root = root
    id_path = this_root + 'dep_clean_ids_'+number+'.txt'
    count = 1
    redownload_path = this_root + 'not_downloaded_line_'+number+'.txt'
    with open(id_path, 'r') as data:
        for line in data:
            meta = line.split('\n')[0]
            try:
                im = Image.open(this_root+meta+'.jpg')
            except IOError:
                print 'Not downloaded image:', meta
                print count
                with open(redownload_path,'a') as f:
                    f.write(str(count)+'\n')
            count += 1
def rm_invalid_txt_for_check_download(number, path=root):
    meta_path1 = path + 'dep_clean_imgs_'+number+'.txt'
    meta_path2 = path + 'dep_clean_ids_'+number+'.txt'
    meta_path3 = path + 'dep_clean_urls_'+number+'.txt'
    out_path1 = path + 'clean_imgs_'+number+'.txt'
    out_path2 = path + 'clean_ids_'+number+'.txt'
    out_path3 = path + 'clean_urls_'+number+'.txt'

    delete_path = path + 'not_downloaded_line_'+number+'.txt'
    delete_lines = []
    with open(delete_path,'r') as data:
        for line in data:
            delete_lines.append(int(line.split('\n')[0]))
    print len(delete_lines)
    print delete_lines
    f1 = open(out_path1,'w')
    count = 1
    i = 0
    with open(meta_path1,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f1.write(line)
                count += 1
    f1.close()
    print i

    f2 = open(out_path2,'w')
    count = 1
    i = 0
    with open(meta_path2,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f2.write(line)
                count += 1
    f2.close()
    f3 = open(out_path3,'w')
    count = 1
    i = 0
    with open(meta_path3,'r') as data:
        for line in data:
            if i < len(delete_lines) and delete_lines[i] == count:
                i += 1
                count += 1
            else:
                f3.write(line)
                count += 1
    f3.close()
def create_full_path(number, path=root):
    load_path = path+'clean_ids_'+number+'.txt'
    out_path = path+'paths_'+number+'.txt'
    prefix = root
    suffix = '.jpg'
    f = open(out_path, 'w')
    with open(load_path, 'r') as data:
        for line in data:
            string = prefix + line.split('\n')[0] + suffix + ' 0\n'
            f.write(string)
    f.close()


if __name__ == '__main__':
    #args = sys.argv
    #rm_invalid_txt(number)
    #rmdup_txt_temp(number)
    #write_csv_temp(number)

    #assert len(args) > 2
    #id = args[1]
    #i = int(args[2])*140000+1
    #number = args[1]
    #find_need_delete_once_more(id, str(i))


    '''check invalid downloads again'''
    #assert len(args) > 1
    #check_download(args[1])
    for i in [1]:
        number = str(i*140000+1)
        #rm_invalid_txt_for_check_download(number)
        create_full_path(number)