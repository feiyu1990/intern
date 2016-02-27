__author__ = 'wangyufei'

root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_2_round/'

def create_full_path(path=root):
    load_path = path+'more_graduation_imgs_0719.txt'
    out_path = path+'paths_more_graduation_0719.txt'
    prefix = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'
    suffix = '.jpg'
    f = open(out_path, 'w')
    with open(load_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            usr = meta[3]
            id = meta[2]
            string = prefix + usr + '/' + id + suffix + ' 0\n'
            #string = line.split('\n')[0]+' 0\n'
            f.write(string)
    f.close()

def create_prototxt(number, path=root):
    out_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_2_round/CNN/models/flickr_train_val_'+str(number)+'.prototxt'
    load_path = '/mnt/ilcompf2d0/project/yuwang/CNN/models/JP/flickr_train_val_yw.prototxt'
    count = 0
    f = open(out_path,'w')
    with open(load_path,'r') as data:
        for line in data:
            count += 1
            if count == 8:
                f.write('source: \"/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_2_round/txt_for_CNN/paths_more_graduation_0719_'+str(number)+'.txt\"\n')
            else:
                f.write(line)
    f.close()

def merge_all_files():
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_2_round/txt_cleaned/'
    write_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_2_round/more_graduation_imgs_cleaned.txt'
    f = open(write_path, 'w')
    for i in xrange(123):
        count = i*10000+1
        read_path = root + 'more_graduation_imgs_0719_cleaned_'+str(count)+'.txt'
        print read_path
        with open(read_path, 'r') as data:
            for line in data:
                f.write(line)


if __name__ == '__main__':
    #create_full_path()
    #for i in xrange(13):
    #    j = i*100000+1
    #    create_prototxt(j)
    merge_all_files()