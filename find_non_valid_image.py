__author__ = 'wangyufei'


import os


min_occurrence = 5000
min_tag = 3
list_path = '../datasets/all_data/clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/all_users/'


def find_invalid():
    invalid_count = 0
    for usr in os.listdir(list_path):
        if not usr.endswith("txt"):
            continue
        path = list_path + usr
        with open(path, 'r') as data:
            for line in data:
                #line = line.split('\n')
                meta = line.split('\t')
                if len(meta) != 23:
                    invalid_count += 1
                    #print line
                    print invalid_count

if __name__ == '__main__':
    find_invalid()