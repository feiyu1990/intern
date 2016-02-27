__author__ = 'wangyufei'
root = '/mnt/ilcompf2d0/project/yuwang/'
my_root = '/Users/wangyufei/Documents/Study/intern_adobe/'
import csv
import cPickle
import os
dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

def create_aesthetic():
    save_name = '/Users/wangyufei/Documents/Study/intern_adobe/aesthetic/imgs.txt'
    f_w = open(save_name, 'w')
    for event_name in dict_name2:
        f = open('/Users/wangyufei/Documents/Study/intern_adobe/baseline_all/' + event_name + '/training_image_ids.cPickle','r')
        img_ids = cPickle.load(f)
        f.close()
        for i in img_ids:
            f_w.write(root + 'download_event_recognition/' + i + '.jpg\n')
        f = open('/Users/wangyufei/Documents/Study/intern_adobe/baseline_all/' + event_name + '/test_image_ids.cPickle','r')
        img_ids = cPickle.load(f)
        f.close()
        for i in img_ids:
            f_w.write(root + 'download_event_recognition/' + i + '.jpg\n')
    f_w.close()

def check_correct():
    in_path = my_root + 'all_output/all_output.csv'
    line_count = -1
    event_ids = []
    img_counts = 0
    with open(in_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            line_count += 1
            if line_count == 0:
                continue
            img_counts += int(meta[27])-2
    img_counts /= 5
    print img_counts

def create_face_path():
    in_path = '/Volumes/Vivian_backup/intern_adobe/from_server_tidy/event_curation/face_recognition/face_features/'
    face_folders = [o for o in os.listdir(in_path) if os.path.isdir(os.path.join(in_path,o)) and o != 'imgs_list']
    print len(face_folders)
    f_w = open(my_root + 'face_expression/imgs.txt','w')
    for folder in face_folders:
        if os.path.exists(in_path + folder + '/all-scores-faces-list-new'):
            #print folder
            string = '-new'
        else:
            string = ''
        count = 0
	with open(in_path + folder + '/all-scores-faces-list' + string, 'r') as data:
            for line in data:
		count += 1
                f_w.write(line)
	print folder, count
    f_w.close()

if __name__ == '__main__':
    #create_aesthetic()
    #check_correct()
    create_face_path()
