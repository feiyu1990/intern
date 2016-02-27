__author__ = 'wangyufei'
'''
SAMPLE EXPERIMENT. See load_root
sort_usr_with_time() -> sort one user in a directory (with .txt) based on timestamp. Save into /all_users_time/
'''
import os
import sys


root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'

def sort_usr_with_time(usr_list, min_occurrence=5000, min_tag=3):
    invalid_line = 0
    #print 'Generating image lists(all information) with more than '+str(min_tag)+' cleaned tags...'
    #load_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+ '/all_users/'
    load_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'
    save_root = root + 'clean_tags_'+str(min_occurrence) +'_'+str(min_tag)+'/'+ 'all_users_time/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    #for usr in os.listdir(load_root):
    for usr in [usr_list]:
        print usr, '...'
        if usr.endswith("txt"):
            list_path = load_root + usr
            images = []
            with open(list_path, 'r') as data:
                for line in data:
                    meta = line.split('\t')
                    if len(meta) != 23:
                        invalid_line += 1
                        continue
                    date = meta[3]
                    temp = date.split(' ')
                    date_info = temp[0]
                    try:
                        time_info = temp[1]
                    except:
                        print line
                    temp = date_info.split('-')
                    m = temp[1]
                    d = temp[2]
                    y = temp[0]
                    h = time_info.split(':')[0]
                    minute = time_info.split(':')[1]
                    second = time_info.split(':')[2]
                    time_this = float(y+m+d+h+minute+second)
                    images.append((time_this, line))
            save_path = save_root + usr
            images_sorted = sorted(images, key=lambda x:x[0])

            with open(save_path, 'w') as out_file:
                for time, line in images_sorted:
                    out_file.write(line)
        print 'User:',usr,' invalid lines:', invalid_line

if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    usr_name = (args[1])
    sort_usr_with_time(usr_name)
