__author__ = 'wangyufei'

import os
root = '/mnt/ilcompf2d0/project/yuwang/datasets/flickr100/'

def rm_folder():
    rm_file = root + 'rm_file.txt'
    with open(rm_file,'r') as data:
        for line in data:
            line = line[:-1]
            print 'Deleting folder:', line
            os.rmdir(line)
def rm_line():
    #rm_file = root + 'rm_file.txt'
    #w_f = open(rm_file,'w')
    list_path = root + 'list/'
    new_list_path = root + 'list_new/'
    if not os.path.exists(new_list_path):
        os.mkdir(new_list_path)
    for i in xrange(100):
        file_path = list_path + str(i) + '.txt'
        new_file_path = new_list_path + str(i) + '.txt'
        write_file = open(new_file_path,'w')
        with open(file_path, 'r') as data:
            for line in data:
                name = line.split('/')[-1]
                name = name.split('.')[0]
                path = root + name
                if not os.path.exists(path):
                #    pass
                    write_file.write(line)
                else:
                    print name
                #    n_images = len([name_1 for name_1 in os.listdir(path)])
                #    if n_images < 9000:
                #        w_f.write(path)
                #        print path, name, n_images

if __name__ == '__main__':
    rm_line()

