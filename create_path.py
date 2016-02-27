__author__ = 'wangyufei'

#import Image
import os

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/'

def create_path():
    save_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/'
    load_path = root+'event_recognition/20000_all_events.txt'

    out_path = save_root + 'paths_for_windows.txt'
    f = open(out_path, 'w')
    with open(load_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            path = 'C:\Users\yuwang\Documents\download_event_recognition\\'+meta[3]+'\\'+meta[2]+'.jpg\r\n'
            f.write(path)
    f.close()
def separate_path():
    load_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/'
    save_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/temp_for_windows/'
    out_path = save_root + 'paths_for_windows0.txt'
    f = open(out_path, 'w')
    count = 0
    with open(load_root + 'paths_for_windows.txt', 'r') as data:
        for line in data:
            count += 1
            if count % 10000 == 0:
                f.close()
                f = open(save_root + 'paths_for_windows' + str(count/10000) + '.txt','w')
            f.write(line)
def read_tag():
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'
    load_path = '../sample_output5.txt'
    load_path2 = root+'20000_all_events.txt'
    tags = []
    with open(load_path, 'r') as data:
        for line in data:
            meta = line[:-3].split('\t')
            tags.append(meta)
    urls = []
    count = 0
    with open(load_path2,'r') as data:
        for line in data:
            count += 1
            if count == 10000:
                break
            meta = line.split('\t')
            urls.append(meta[16])

    images_tags = zip(tags, urls)
    print images_tags[-10:]

def resave_image():
    save_root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/'
    load_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_curation_CNN/all_images.txt'
    count = 1
    with open(load_path, 'r') as data:
        for line in data:
            if count %10000 ==0:
                print count
            count+=1
            meta = line.split('\t')
            path = root+'download_events/'+meta[3]+'/'+meta[2]+'.jpg'
            new_path =  save_root + meta[3] + '/'
            new_file = new_path+meta[2]+'.jpg'
            if os.path.isfile(new_file):
                continue
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            print new_file
            os.rename(path, new_path+meta[2]+'.jpg')

if __name__ == '__main__':
    resave_image()




