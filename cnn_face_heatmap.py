__author__ = 'wangyufei'

import cPickle
from operator import itemgetter
import os
from PIL import Image
import numpy as np
# import heatmap
import matplotlib as mpl
import matplotlib.cm as cm
import shutil

# root = 'C:/Users/yuwang/Documents/'
root = '/home/feiyu1990/local/event_curation/'
# root = '/Users/wangyufei/Documents/Study/intern_adobe/'
# root = '/Volumes/Vivian_backup/intern_adobe/from_server_tidy/'
dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

def create_gaussian(array, x, y, size1, size2, multi = 1):
    width, height = array.shape
    size = min(size1, size2)
    sigma = size/2
    size = size * 2
    for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
        for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
            temp = float((size - i)**2 + (size - j) ** 2)
            array[x + i - size,y + j - size] += multi * np.exp(-temp/2/(sigma**2))
            #print multi * np.exp(-temp/2/(sigma**2))
    #print array[max(0, size - x):min(width+size, 2*size+1),max(0, size - y):min(height + size, 2*size+1)]
    #print np.max(array)
    return array
def create_gaussian_color(important_indicator, array, x, y, size1, size2, multi = 1):
    width, height, temp = array.shape
    size = min(size1, size2)
    sigma = size/2
    size = size * 2
    if important_indicator == 0:
        #multi = 0.5
        for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
            for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
                temp = float((size - i)**2 + (size - j) ** 2)
                array[x + i - size,y + j - size, :] += multi * np.exp(-temp/2/(sigma**2))
    elif important_indicator == 1:
        for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
            for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
                temp = float((size - i)**2 + (size - j) ** 2)
                array[x + i - size,y + j - size, 0] += multi * np.exp(-temp/2/(sigma**2))
    elif important_indicator == 2:
        for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
            for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
                temp = float((size - i)**2 + (size - j) ** 2)
                array[x + i - size,y + j - size, 1] += multi * np.exp(-temp/2/(sigma**2))

    return array
def create_gaussian_importance(important_indicator, aesthetic_score, array, x, y, size1, size2, multi = 1):
    width, height, temp = array.shape
    size = min(size1, size2)
    sigma = size/2
    size = size * 2
    multi = float(multi) * aesthetic_score
    # print multi
    if important_indicator == 0:
        multi = 0.5 * multi
        for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
            for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
                temp = float((size - i)**2 + (size - j) ** 2)
                array[x + i - size,y + j - size, :] += multi * np.exp(-temp/2/(sigma**2))
    elif important_indicator == 1:
        for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
            for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
                temp = float((size - i)**2 + (size - j) ** 2)
                array[x + i - size,y + j - size, :] += multi * np.exp(-temp/2/(sigma**2))
    elif important_indicator == 2:
        for i in xrange(max(0, size - x), min(width+size-x, 2*size+1)):
            for j in xrange(max(0, size - y), min(height + size-y, 2*size+1)):
                temp = float((size - i)**2 + (size - j) ** 2)
                array[x + i - size,y + j - size, :] += multi * np.exp(-temp/2/(sigma**2))

    return array

def prepare_guru_dataset():
    '''
    path = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/wedding_CNN_net/data/guru_ranking_reallabel_training.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/face_heatmap/data/guru_ranking_reallabel_training.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            meta = line.split('/')
            meta[5] = 'person_heatmap_images_aesthetic_cropped'
            f.write(('/').join(meta))
    f.close()

    path = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/wedding_CNN_net/data/guru_ranking_reallabel_training_p.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/face_heatmap/data/guru_ranking_reallabel_training_p.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            meta = line.split('/')
            meta[5] = 'person_heatmap_images_aesthetic_cropped'
            f.write(('/').join(meta))
    f.close()
    '''
    path = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/wedding_test_path.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/face_heatmap/data/wedding_test_path_faceheatmap.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            meta = line.split('/')
            meta[5] = 'person_heatmap_images_aesthetic_cropped'
            f.write(('/').join(meta))
    f.close()

def find_face_positions():
    #create_knn_cPickle(path = root + 'baseline_wedding_test/' + model_name + '_wedding_knn_all.txt')
    root1 = root + 'face_recognition/'
    lists = [o for o in os.listdir(root1) if os.path.isdir(root1+o)]
    img_face_and_sizes = {}
    download_img_root = root + 'download_event_recognition/'
    for name in lists:
        folder_path = root1 + name
        event_name = name.split('-')[0]
        lines = []
        print folder_path + '/all-scores.txt'
        with open(folder_path + '/all-scores.txt','r') as data:
            for line in data:
                lines.append(line)
        i = 0
        while i < len(lines):
            temp_sizes = []
            line = lines[i]
            if line[0] != '/':
                print 'ERROR!'
            img_name = line.split('/')[-1]
            img_name = img_name.split()[0]
            img_path = download_img_root + event_name.split('_')[1] + '/' + img_name
            im = Image.open(img_path)
            width, height = im.size
            i += 1
            line = lines[i]
            num_img = int(line.split()[0])
            img_mask = np.zeros((height, width))
            #img_mask = np.zeros((width, height))

            for j in xrange(num_img):
                i += 1
                line = lines[i]
                x,y,size1, size2, not_used = line.split(' ')
                x = int(x); y=int(y); size1=int(size1); size2=int(size2)
                #print size1 * size2 / (width * height)
                if size1 * size2 / (width * height) < 1/40:
                    continue
                img_mask = create_gaussian(img_mask, int(y + size2/2), int(x + size1/2), size1, size2)
                #img_mask = create_gaussian(img_mask,int(x + size1/2),  int(y + size2/2), size1, size2)
            im = Image.fromarray(img_mask*256)
            if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/'):
                os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/')
            out_path = root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/' + img_name
            im = im.convert('L')
            im.save(out_path)

            new_width = min(width, height)
            left = (width - new_width)/2
            top = (height - new_width)/2
            right = (width + new_width)/2
            bottom = (height + new_width)/2
            img = im.crop((left, top, right, bottom))
            if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/'):
                os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/')
            out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/' + img_name
            img.save(out_path)


            i += 1
def create_group_groundtruth(name):
    path = root + 'face_recognition_CNN/' + '-'.join(name.split('@')) + '/group_groundtruth.cPickle'
    f = open(path, 'wb')
    inpath = root + 'face_recognition/' + name+ '.txt'
    group = {1:[], 2:[]}
    with open(inpath, 'r') as data:
        count = 0
        for line in data:
            if line[-1] == '\r':
                line = line[:-1]
            meta = line.split(' ')
            for i in meta:
                if len(i) > 4 and i[-4:] == '.jpg':
                    group[count + 1].append(i)
            count += 1
            if count > 2:
                print 'ERROR!'
    cPickle.dump(group, f)
    f.close()

def find_face_positions_importance():
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/')
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/')
    # root1 = root + 'event_curation/face_recognition/'
    # lists = [o for o in os.listdir(root1) if os.path.isdir(root1+o)]
    event_type_dict = {}
    for event_type in dict_name2:
        in_path = root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle'
        f = open(in_path)
        temp = cPickle.load(f)
        in_path = root + 'baseline_all_0509/' + event_type + '/training_event_id.cPickle'
        f = open(in_path)
        temp.extend(cPickle.load(f))
        for i in temp:
            event_type_dict[i] = event_type

    in_path = root + 'face_scores.txt'
    img_name_prev = ''
    line_prev = ''
    aesthetic_scores = []
    with open(in_path, 'r') as data:
            for line_this in data:
                img_name = line_this.split('/')[3].split('-')[0]
                if img_name_prev != img_name:
                    print aesthetic_scores
                    print line_prev
                    if img_name_prev == '':
                        aesthetic_scores = [float(line_this.split(' ')[1])]
                        img_name_prev = img_name
                        line_prev = line_this
                        continue
                    event_name_prev = line_prev.split('/')[1].split('-')[0]
                    try:
                        event_type_prev = event_type_dict[event_name_prev]
                    except:
                        continue

                    f = open(root + 'face_features/' + event_name_prev + '-dir/_20_group.cPickle','r')
                    groups, important_ = cPickle.load(f)
                    f.close()

                    if len(important_) > 0:
                        important_1 = groups[important_[0][0]]
                    if len(important_) > 1:
                        important_2 = groups[important_[1][0]]
                    lines = []
                    with open(root + 'face_features/' + event_name_prev + '-dir/all-scores.txt','r') as data:
                        for line_ in data:
                            lines.append(line_[:-1])
                    lines_dict = {}
                    i = 0
                    while i < len(lines):
                        temp = lines[i].split('/')[-1]
                        lines_dict[temp] = []
                        i += 1;len_ = int(lines[i])
                        for j in xrange(len_):
                            i += 1
                            lines_dict[temp].append(lines[i])
                        i+=1
                    img_path = root + 'curation_images/'+event_type_prev+'/' + event_name_prev.split('_')[1] + '/'+img_name_prev+'.jpg'
                    im = Image.open(img_path)
                    img_this = img_path.split('/')[-1]
                    width, height = im.size
                    img_mask = np.zeros((height, width, 3))

                    for i in xrange(len(aesthetic_scores)):
                        img_path_face = root  + '-'.join(line_prev.split('-')[:-1]) + '-' + str(i) + '.jpg'
                        im = Image.open(img_path_face)
                        important_indicator = 0
                        if img_this in important_1:
                            print 'IMPORTANT 1!'
                            important_indicator = 1
                        elif img_this in important_2:
                            print 'IMPORTANT 2!'
                            important_indicator = 2
                        line = lines_dict[img_name_prev+'.jpg'][i]
                        x,y,size1, size2, not_used = line.split(' ')
                        x = int(x); y=int(y); size1=int(size1); size2=int(size2)
                        img_mask = create_gaussian_importance(important_indicator,aesthetic_scores[i]**0.1,  img_mask, int(y + size2/2), int(x + size1/2), size1, size2)


                    img_mask = img_mask * 255 / 2
                    img_mask = np.asarray(img_mask, 'uint8')
                    im = Image.fromarray(img_mask)
                    if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name_prev.split('_')[1] + '/'):
                        os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name_prev.split('_')[1] + '/')
                    out_path = root + 'person_heatmap_images_aesthetic_lowqual/' + event_name_prev.split('_')[1] + '/' + img_this
                    im.save(out_path)
                    new_width = min(width, height)
                    left = (width - new_width)/2
                    top = (height - new_width)/2
                    right = (width + new_width)/2
                    bottom = (height + new_width)/2
                    im_crop = im.crop((left, top, right, bottom))
                    if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name_prev.split('_')[1] + '/'):
                        os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name_prev.split('_')[1] + '/')
                    out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name_prev.split('_')[1] + '/' + img_this
                    im_crop.save(out_path)

                    img_name_prev = img_name
                    aesthetic_scores = [float(line_this.split(' ')[1])]
                else:
                    aesthetic_scores.append(float(line_this.split(' ')[1]))

                line_prev = line_this

    event_name_prev = line_prev.split('/')[1].split('-')[0]
    try:
        event_type_prev = event_type_dict[event_name_prev]
    except:
        return

    f = open(root + 'face_features/' + event_name_prev + '-dir/_20_group.cPickle','r')
    groups, important_ = cPickle.load(f)
    f.close()
    if len(important_) > 0:
        important_1 = groups[important_[0][0]]
    if len(important_) > 1:
        important_2 = groups[important_[1][0]]
    lines = []
    with open(root + 'face_features/' + event_name_prev + '-dir/all-scores.txt','r') as data:
        for line_ in data:
            lines.append(line_[:-1])
    lines_dict = {}
    i = 0
    while i < len(lines):
        temp = lines[i].split('/')[-1]
        lines_dict[temp] = []
        i += 1;len_ = int(lines[i])
        for j in xrange(len_):
            i += 1
            lines_dict[temp].append(lines[i])
        i+=1
    img_path = root + 'curation_images/'+event_type_prev+'/' + event_name_prev.split('_')[1] + '/'+img_name_prev+'.jpg'
    im = Image.open(img_path)
    img_this = img_path.split('/')[-1]
    width, height = im.size
    img_mask = np.zeros((height, width, 3))
    for i in xrange(len(aesthetic_scores)):
            img_path_face = root  + '-'.join(line_prev.split('-')[:-1]) + '-' + str(i) + '.jpg'
            im = Image.open(img_path_face)
            important_indicator = 0
            if img_this in important_1:
                print 'IMPORTANT 1!'
                important_indicator = 1
            elif img_this in important_2:
                print 'IMPORTANT 2!'
                important_indicator = 2
            line = lines_dict[img_name_prev+'.jpg'][i]
            x,y,size1, size2, not_used = line.split(' ')
            x = int(x); y=int(y); size1=int(size1); size2=int(size2)
            img_mask = create_gaussian_importance(important_indicator,aesthetic_scores[i],  img_mask, int(y + size2/2), int(x + size1/2), size1, size2)


    img_mask = img_mask * 255 / 2
    img_mask = np.asarray(img_mask, 'uint8')
    im = Image.fromarray(img_mask)
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name_prev.split('_')[1] + '/'):
                        os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name_prev.split('_')[1] + '/')
    out_path = root + 'person_heatmap_images_aesthetic_lowqual/' + event_name_prev.split('_')[1] + '/' + img_this
    im.save(out_path)
    new_width = min(width, height)
    left = (width - new_width)/2
    top = (height - new_width)/2
    right = (width + new_width)/2
    bottom = (height + new_width)/2
    im_crop = im.crop((left, top, right, bottom))
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name_prev.split('_')[1] + '/'):
                        os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name_prev.split('_')[1] + '/')
    out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name_prev.split('_')[1] + '/' + img_this
    im_crop.save(out_path)

def no_face_positions_importance():
    for event_type in dict_name2:
        in_path = root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle'
        f = open(in_path)
        img_ids = cPickle.load(f)
        in_path = root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle'
        f = open(in_path)
        img_ids.extend(cPickle.load(f))

        for img_this in img_ids:
            # print img_this
            face_heatmap_path = root + 'person_heatmap_images_aesthetic_lowqual/' + img_this.split('_')[1] + '.jpg'
            if not os.path.exists(face_heatmap_path):
                print face_heatmap_path
                # album_name = img_this.split('/')[0].split('_')[1]
                # out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + album_name+ '/' + img_this
                # print face_heatmap_path
                # ori_img_path = root + 'curation_images/'+event_type+'/' + img_this.split('_')[1] + '.jpg'
                # img_face_path = root + 'face_features/' + img_this.split('/')[0] + '-dir/all-scores-faces/'
                # if not os.path.exists(img_face_path):
                #     print img_face_path
                #     print 'ERROR!'
                #     return
                # # print img_face_path + img_this.split('/')[1].split('.')[0]+'-0.jpg'
                # if os.path.exists(img_face_path + img_this.split('/')[1].split('.')[0]+'-0.jpg'):
                #     print img_face_path + img_this.split('/')[1].split('.')[0]+'-0.jpg exsits!!'
                #     print 'ERROR!'
                # if os.path.exists(out_path):
                #     print out_path + ' exists!!'
                #     print 'ERROR!'
                #     return
                #
                # im = Image.open(ori_img_path)
                # width, height = im.size
                # img_mask = np.zeros((height, width, 3))
                # img_mask = np.asarray(img_mask, 'uint8')
                # im = Image.fromarray(img_mask)
                # if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/' + album_name + '/'):
                #     os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/' + album_name + '/')
                # out_path = root + 'person_heatmap_images_aesthetic_lowqual/' + album_name + '/' + img_this.split('/')[1]+'.jpg'
                # im.save(out_path)
                # new_width = min(width, height)
                # left = (width - new_width)/2
                # top = (height - new_width)/2
                # right = (width + new_width)/2
                # bottom = (height + new_width)/2
                # im_crop = im.crop((left, top, right, bottom))
                # out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + album_name+ '/' + img_this.split('/')[1]+'.jpg'
                # if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + album_name + '/'):
                #     os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + album_name + '/')
                # im_crop.save(out_path)


def find_face_positions_importance_dep():
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/')
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/')
    # root1 = root + 'event_curation/face_recognition/'
    # lists = [o for o in os.listdir(root1) if os.path.isdir(root1+o)]
    event_type_dict = {}
    for event_type in dict_name2:
        in_path = root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle'
        f = open(in_path)
        temp = cPickle.load(f)
        in_path = root + 'baseline_all_0509/' + event_type + '/training_event_id.cPickle'
        f = open(in_path)
        temp.extend(cPickle.load(f))
        for i in temp:
            event_type_dict[i] = event_type

    in_path = root + 'face_scores.txt'
    img_name_prev = ''
    lines = []
    with open(in_path, 'r') as data:
            for line_this in data:
                img_name = line_this.split('/')[3].split('-')[0]
                event_name = line_this.split('/')[1].split('-')[0]
                try:
                    event_type = event_type_dict[event_name]
                except:
                    continue
                if os.path.exists('/home/feiyu1990/local/event_curation/person_heatmap_images_aesthetic_lowqual/' + line_this.split('_')[2].split('-')[0] + '/' + img_name + '.jpg'):
                    print img_name,'already exists'
                    continue
                else:
                    print line_this
                if img_name_prev != img_name:
                    if img_name_prev != '':
                        img_mask = img_mask * 255 / 2
                        img_mask = np.asarray(img_mask, 'uint8')
                        im = Image.fromarray(img_mask)
                        if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/'):
                            os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/')
                        out_path = root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/' + img_this
                        im.save(out_path)

                        new_width = min(width, height)
                        left = (width - new_width)/2
                        top = (height - new_width)/2
                        right = (width + new_width)/2
                        bottom = (height + new_width)/2
                        im_crop = im.crop((left, top, right, bottom))
                        if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/'):
                            os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/')
                        out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/' + img_this
                        im_crop.save(out_path)

                    img_name_prev = img_name
                    f = open(root + 'face_features/' + event_name + '-dir/_20_group.cPickle','r')
                    groups, important_ = cPickle.load(f)
                    f.close()

                    if len(important_) > 0:
                        important_1 = groups[important_[0][0]]
                    if len(important_) > 1:
                        important_2 = groups[important_[1][0]]
                    lines = []
                    # print root + 'event_curation/face_recognition/face_features/' + event_name + '-dir/all-scores.txt'
                    with open(root + 'face_features/' + event_name + '-dir/all-scores.txt','r') as data:
                        for line_ in data:
                            lines.append(line_[:-1])
                    lines_dict = {}
                    i = 0
                    while i < len(lines):
                        temp = lines[i].split('/')[-1]
                        lines_dict[temp] = []
                        i += 1;len_ = int(lines[i])
                        for j in xrange(len_):
                            i += 1
                            lines_dict[temp].append(lines[i])
                        i+=1
                    img_path = root + 'curation_images/'+event_type+'/' + line_this.split('/')[1].split('_')[1].split('-')[0] + '/'+line_this.split('/')[3].split('-')[0]+'.jpg'
                    im = Image.open(img_path)
                    img_this = img_path.split('/')[-1]
                    width, height = im.size
                    img_mask = np.zeros((height, width, 3))

                aesthetic_score = float(line_this.split(' ')[1])
                img_path_face = root  + line_this.split(' ')[0]
                i = int(line_this.split('/')[3].split('-')[1].split('.')[0])
                temp_sizes = []
                im = Image.open(img_path_face)
                important_indicator = 0
                if img_this in important_1:
                    print 'IMPORTANT 1!'
                    important_indicator = 1
                elif img_this in important_2:
                    print 'IMPORTANT 2!'
                    important_indicator = 2


                line = lines_dict[line_this.split('/')[3].split('-')[0]+'.jpg'][i]
                x,y,size1, size2, not_used = line.split(' ')
                x = int(x); y=int(y); size1=int(size1); size2=int(size2)
                #print size1 * size2 / (width * height)
                # if size1 * size2 / (width * height) < 1/40:
                #     continue
                img_mask = create_gaussian_importance(important_indicator,aesthetic_score,  img_mask, int(y + size2/2), int(x + size1/2), size1, size2)
                #print important_indicator
                #img_mask = create_gaussian(img_mask,int(x + size1/2),  int(y + size2/2), size1, size2)
def find_face_positions_importance_old():
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/')
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/')
    # root1 = root + 'event_curation/face_recognition/'
    # lists = [o for o in os.listdir(root1) if os.path.isdir(root1+o)]
    img_face_and_sizes = {}


    download_img_root = root + 'datasets/download_event_recognition/'
    in_path = root + 'face_scores.txt'
    img_name_prev = ''
    lines = []
    with open(in_path, 'r') as data:
            for line_this in data:
                print line_this
                img_name = line_this.split('/')[3].split('-')[0]
                event_name = line_this.split('/')[1].split('-')[0]
                if img_name_prev != img_name:
                    if img_name_prev != '':
                        img_mask = img_mask * 255 / 2
                        img_mask = np.asarray(img_mask, 'uint8')
                        im = Image.fromarray(img_mask)
                        if not os.path.exists(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/'):
                            os.mkdir(root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/')
                        out_path = root + 'person_heatmap_images_aesthetic_lowqual/' + event_name.split('_')[1] + '/' + img_this
                        im.save(out_path)

                        new_width = min(width, height)
                        left = (width - new_width)/2
                        top = (height - new_width)/2
                        right = (width + new_width)/2
                        bottom = (height + new_width)/2
                        im_crop = im.crop((left, top, right, bottom))
                        if not os.path.exists(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/'):
                            os.mkdir(root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/')
                        out_path = root + 'person_heatmap_images_aesthetic_cropped_lowqual/' + event_name.split('_')[1] + '/' + img_this
                        im_crop.save(out_path)

                    img_name_prev = img_name
                    f = open(root + 'event_curation/face_recognition/face_features/' + event_name + '-dir/_20_group.cPickle','r')
                    groups, important_ = cPickle.load(f)
                    f.close()

                    if len(important_) > 0:
                        important_1 = groups[important_[0][0]]
                    if len(important_) > 1:
                        important_2 = groups[important_[1][0]]
                    lines = []
                    # print root + 'event_curation/face_recognition/face_features/' + event_name + '-dir/all-scores.txt'
                    with open(root + 'event_curation/face_recognition/face_features/' + event_name + '-dir/all-scores.txt','r') as data:
                        for line_ in data:
                            lines.append(line_[:-1])
                    lines_dict = {}
                    i = 0
                    while i < len(lines):
                        temp = lines[i].split('/')[-1]
                        lines_dict[temp] = []
                        i += 1;len_ = int(lines[i])
                        for j in xrange(len_):
                            i += 1
                            lines_dict[temp].append(lines[i])
                        i+=1
                    img_path = root + 'datasets/download_event_recognition/' + line_this.split('/')[1].split('_')[1].split('-')[0] + '/'+line_this.split('/')[3].split('-')[0]+'.jpg'
                    im = Image.open(img_path)
                    img_this = img_path.split('/')[-1]
                    width, height = im.size
                    img_mask = np.zeros((height, width, 3))

                aesthetic_score = float(line_this.split(' ')[1])
                img_path_face = root + 'event_curation/face_recognition/' + line_this.split(' ')[0]
                i = int(line_this.split('/')[3].split('-')[1].split('.')[0])
                temp_sizes = []
                im = Image.open(img_path_face)
                important_indicator = 0
                if img_this in important_1:
                    print 'IMPORTANT 1!'
                    important_indicator = 1
                elif img_this in important_2:
                    print 'IMPORTANT 2!'
                    important_indicator = 2


                line = lines_dict[line_this.split('/')[3].split('-')[0]+'.jpg'][i]
                x,y,size1, size2, not_used = line.split(' ')
                x = int(x); y=int(y); size1=int(size1); size2=int(size2)
                #print size1 * size2 / (width * height)
                # if size1 * size2 / (width * height) < 1/40:
                #     continue
                img_mask = create_gaussian_importance(important_indicator,aesthetic_score,  img_mask, int(y + size2/2), int(x + size1/2), size1, size2)
                #print important_indicator
                #img_mask = create_gaussian(img_mask,int(x + size1/2),  int(y + size2/2), size1, size2)



def find_face_positions_importance_groundtruth(color = '_color'):
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_groundtruth'+color+'/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_groundtruth'+color+'/')
    if not os.path.exists(root + 'person_heatmap_images_aesthetic_groundtruth_cropped'+color+'/'):
        os.mkdir(root + 'person_heatmap_images_aesthetic_groundtruth_cropped'+color+'/')
    root1 = root + 'event_curation/face_recognition/'
    lists = [o for o in os.listdir(root1) if os.path.isdir(root1+o) and o != 'img_lists']
    download_img_root = root + 'download_event_recognition/'
    for name in lists:
        name1 = name.split('-')[0]
        #create_group_groundtruth(name1)
        name1 = ('-').join(name1.split('@'))
        f = open(root + 'face_recognition_CNN/' + name1 + '/group_groundtruth.cPickle','r')
        groups = cPickle.load(f)
        f.close()
        important_1 = groups[1]
        important_2 = groups[2]

        folder_path = root1 + name
        event_name = name.split('-')[0]
        lines = []
        print folder_path + '/all-scores.txt'
        with open(folder_path + '/all-scores.txt','r') as data:
            for line in data:
                lines.append(line)
        i = 0
        while i < len(lines):
            temp_sizes = []
            line = lines[i]
            if line[0] != '/':
                print 'ERROR!'
            img_name = line.split('/')[-1]
            img_name = img_name.split()[0]
            img_path = download_img_root + event_name.split('_')[1] + '/' + img_name
            im = Image.open(img_path)
            width, height = im.size
            i += 1
            line = lines[i]
            num_img = int(line.split()[0])
            img_mask = np.zeros((height, width, 3))
            #img_mask = np.zeros((width, height))

            for j in xrange(num_img):
                img_this = img_name.split('.')[0] + '-' + str(j) + '.jpg'
                #print img_this
                #print important_1
                important_indicator = 0
                if img_this in important_1:
                    print 'IMPORTANT 1!'
                    important_indicator = 1
                elif img_this in important_2:
                    print 'IMPORTANT 2!'
                    important_indicator = 2


                i += 1
                line = lines[i]
                x,y,size1, size2, not_used = line.split(' ')
                x = int(x); y=int(y); size1=int(size1); size2=int(size2)
                #print size1 * size2 / (width * height)
                if size1 * size2 / (width * height) < 1/40:
                    continue
                if color == '':
                    img_mask = create_gaussian_importance(important_indicator, img_mask, int(y + size2/2), int(x + size1/2), size1, size2)
                else:
                    img_mask = create_gaussian_color(important_indicator, img_mask, int(y + size2/2), int(x + size1/2), size1, size2)
                #print important_indicator
                #img_mask = create_gaussian(img_mask,int(x + size1/2),  int(y + size2/2), size1, size2)
            img_mask = img_mask * 256 / 2
            img_mask = np.asarray(img_mask, 'uint8')
            im = Image.fromarray(img_mask)
            if not os.path.exists(root + 'person_heatmap_images_aesthetic_groundtruth'+color+'/' + event_name.split('_')[1] + '/'):
                os.mkdir(root + 'person_heatmap_images_aesthetic_groundtruth'+color+'/' + event_name.split('_')[1] + '/')
            out_path = root + 'person_heatmap_images_aesthetic_groundtruth'+color+'/' + event_name.split('_')[1] + '/' + img_name
            im.save(out_path)

            new_width = min(width, height)
            left = (width - new_width)/2
            top = (height - new_width)/2
            right = (width + new_width)/2
            bottom = (height + new_width)/2
            im.crop((left, top, right, bottom))
            if not os.path.exists(root + 'person_heatmap_images_aesthetic_groundtruth_cropped'+color+'/' + event_name.split('_')[1] + '/'):
                os.mkdir(root + 'person_heatmap_images_aesthetic_groundtruth_cropped'+color+'/' + event_name.split('_')[1] + '/')
            out_path = root + 'person_heatmap_images_aesthetic_groundtruth_cropped'+color+'/' + event_name.split('_')[1] + '/' + img_name
            im.save(out_path)


            i += 1

def crop_imgs():
    in_root = root + 'person_heatmap_images_aesthetic_allevent/'
    out_root = root + 'person_heatmap_images_aesthetic_cropped_allevent/'
    folders = [o for o in os.listdir(in_root) if os.path.isdir(os.path.join(in_root,o))]
    for folder in folders:
        os.mkdir(out_root + folder)
        files = [f for f in os.listdir(in_root + folder) if f.endswith('.jpg')]
        for file in files:
            out_file = out_root + folder + '/' + file
            in_file = in_root + folder + '/' + file
            im = Image.open(in_file)
            width, height = im.size
            print width, height
            new_width = min(width, height)
            left = (width - new_width)/2
            top = (height - new_width)/2
            right = (width + new_width)/2
            bottom = (height + new_width)/2
            img = im.crop((left, top, right, bottom))
            img.save(out_file)

def create_face_finetune():
    in_root = root + 'face_heatmap/training/importance_lenet_train_test.prototxt'
    lines = []
    with open(in_root, 'r') as data:
        for line in data:
            lines.append(line)
    for event_name in dict_name2:
        out_root = root + 'face_heatmap/training/importance_lenet_train_test_'+event_name+'.prototxt'
        f = open(out_root, 'w')
        for line in lines:
            if 'guru_ranking_reallabel_part' in line:
                line1 = line.replace('guru_ranking_reallabel_part', event_name)
            elif 'guru_ranking_reallabel' in line:
                line1 = line.replace('guru_ranking_reallabel', event_name)
            else:
                line1 = line
            f.write(line1)
        f.close()


def create_heatmap():
    in_path = root + 'face/face_new_80418937@N00/'
    out_path = root + 'face/heatmap_new_80418937@N00/'
    img_files = [f for f in os.listdir(in_path) if f.endswith('.jpg')]

    for i in img_files:
        im = Image.open(in_path + i)
        new_img = np.zeros((im.size[1], im.size[0], 3))
        img_array = np.array(im)
        img_array = img_array[:,:,0]
        for k in xrange(img_array.shape[0]):
            for j in xrange(img_array.shape[1]):
                # print float(img_array[k][j])/255
                new_img[k,j,:] =  cm.jet(float(img_array[k][j])/255 * 2)[:3]
        new_img = np.asarray(new_img * 255, 'uint8')
        im_new = Image.fromarray(new_img)
        im_new.save(out_path + i)


if __name__ == '__main__':
    #find_face_positions_importance_groundtruth('')
    #prepare_guru_dataset()
    #crop_imgs()
    # create_face_finetune()
    # create_heatmap()
    find_face_positions_importance()
    # no_face_positions_importance()