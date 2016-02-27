__author__ = 'wangyufei'
import os
import shutil


root = '/Volumes/Vivian_backup/intern_adobe/from_server_tidy/event_curation/face_recognition/face_features/'
lists =  [o for o in os.listdir(root) if os.path.isdir(root+o) and o != 'img_lists']

for folder in lists:
    if not os.path.exists('/Users/wangyufei/Documents/Study/intern_adobe/face_features/' + folder):
        os.mkdir('/Users/wangyufei/Documents/Study/intern_adobe/face_features/' + folder)
        os.mkdir('/Users/wangyufei/Documents/Study/intern_adobe/face_features/' + folder + '/all-scores-faces')
    src = root + folder
    dst = '/Users/wangyufei/Documents/Study/intern_adobe/face_features/' + folder
    shutil.copyfile(src +  '/all-scores-faces-list' , dst+ '/all-scores-faces-list')
    shutil.copyfile(src +  '/all-scores.txt' , dst+ '/all-scores.txt')
    shutil.copyfile(src +  '/_20_group.cPickle' , dst+ '/_20_group.cPickle')
