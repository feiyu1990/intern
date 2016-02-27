__author__ = 'wangyufei'

import cPickle
import os
import numpy as np
#from sklearn.cluster import spectral_clustering
from sklearn.cluster import SpectralClustering
import scipy.io as sio
import csv
from PIL import Image

html_root = 'C:/Users/yuwang/Documents/present_htmls/'
#html_root = '/Users/wangyufei/Documents/Study/intern_adobe/present_htmls/'
#root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/clean_input_and_label/3_event_curation/'
#root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'
#root_all = '/Users/wangyufei/Documents/'
root = 'C:/Users/yuwang/Documents/face_recognition/'
root_all = 'C:/Users/yuwang/Documents/'
'''create_dataset'''
def linux_create_path_per_event():
    #root = '/mnt/ilcompf2d0/project/yuwang/event_curation/face_recognition/event_list_rest/'
    in_path = root + 'baseline_wedding_test/wedding_training_id.cPickle'
    f = open(in_path, 'r')
    events_id_already = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_wedding_test/wedding_test_id.cPickle'
    f = open(in_path, 'r')
    events_id_already.extend(cPickle.load(f))
    f.close()

    events_id_rest = []
    in_path = root + 'all_output/all_output.csv'

    line_count = -1
    with open(in_path, 'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            line_count += 1
            if line_count == 0:
                continue
            event_id_this = meta[28]
            if event_id_this in events_id_already:
                continue
            if event_id_this not in events_id_rest:
                events_id_rest.append(event_id_this)



    load_path = root+'all_images_curation.txt'
    prefix = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition/'
    save_root = root + '../../../face_recognition/events_rest/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for event_id in events_id_rest:
        save_path = save_root + event_id + '.txt'
        f = open(save_path, 'w')
        with open(load_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                if meta[1] + '_' + meta[3] == event_id:
                    string = prefix + meta[3]+'/'+meta[2] + '.jpg\n'
                    f.write(string)
        f.close()

'''face recognition'''
def cluster_faces(name = '9_8913259@N03', img_list = 'all-scores-faces-list'):
    cnn_root = root_all + 'face_recognition_CNN/'+('-').join(name.split('@')) + '/'
    in_path = root + name + '-dir/all-scores-faces-list-new-pw.mat'
    temp = sio.loadmat(in_path)
    matrix = temp['matrix']
    #print matrix

    '''1st method: exponential'''
    #a_std = np.std(matrix)
    #beta = 1.0
    #affinity_matrix = np.exp(beta * matrix / a_std)

    '''2nd method: sigmoid'''
    #affinity_matrix = 1 / (1 + np.exp(-matrix))

    '''3rd method: normalize'''
    matrix_ori = matrix
    min_ = np.min(matrix)
    matrix = matrix - min_

    diag = np.diag(matrix)
    diag = diag[:, np.newaxis]
    normalize_matrix = np.dot(diag, np.transpose(diag))
    normalize_matrix = np.sqrt(normalize_matrix)
    affinity_matrix = np.divide(matrix, normalize_matrix)
    min_ = np.min(affinity_matrix); max_ = np.max(affinity_matrix)
    affinity_matrix =  (affinity_matrix - min_) / (max_ - min_)
    #print affinity_matrix
    #print np.min(affinity_matrix), np.max(affinity_matrix)

    f = SpectralClustering(affinity='precomputed', n_clusters=min(30, affinity_matrix.shape[0]/2), eigen_solver = 'arpack', n_neighbors=min(10, affinity_matrix.shape[0]))
    #b = f.fit(affinity_matrix)
    a = f.fit_predict(affinity_matrix)
    mean_similarities = {}
    groups = {}
    temp = zip(a, xrange(len(a)))
    for i in temp:
        if i[0] not in groups:
            groups[i[0]] = [i[1]]
        else:
            groups[i[0]].append(i[1])
    unique_person_id = []
    for kk in groups:
        min_similarity = np.Inf
        max_similarity = -np.Inf
        mean_similarity = 0
        #median_similarity = []
        this_group_ids = groups[kk]
        for j in xrange(len(this_group_ids)):
            for i in xrange(j+1, len(this_group_ids)):
                temp = matrix_ori[this_group_ids[i],this_group_ids[j]]
                if temp < min_similarity:
                    min_similarity = temp
                if temp > max_similarity:
                    max_similarity = temp
                #mean_similarity += np.log10(temp)
                mean_similarity += temp
                #median_similarity.append(temp)
        mean_similarity /= max(1, len(this_group_ids)*(len(this_group_ids) - 1) / 2)
        mean_similarities[kk] = mean_similarity
        #if len(median_similarity) >= 1:
        #    median_ = np.median(np.array(median_similarity))
        #else:
        #    median_ = 0
        #mean_similarities[kk] = median_
        #print len(this_group_ids), mean_similarity, max_similarity, min_similarity
        if mean_similarity > 0 and len(this_group_ids) > 1:
        #if median_ > 0 and len(this_group_ids) > 1:
            unique_person_id.append(kk)
    important_person = []
    for i in unique_person_id:
        important_person.append([i, len(groups[i])])
    important_person.sort(key = lambda x:x[1], reverse=True)
    in_path = root + name + '-dir/' + img_list
    imgs_list = []
    with open(in_path, 'r') as data:
        for line in data:
            line = line[:-1]
            imgs_list.append(line.split('/')[-1])

    temp = zip(a, imgs_list)
    face_groups = {}
    for i in temp:
        if i[0] not in face_groups:
            face_groups[i[0]] = [i[1]]
        else:
            face_groups[i[0]].append(i[1])
    create_retrieval_image(name, matrix)
    create_face_group_html(name, face_groups, important_person, mean_similarities)

    #f = open(cnn_root + '_20_group.cPickle','w')
    #cPickle.dump([face_groups, important_person], f)
    #.close()



def dump_low_mean_(matrix_ori, groups_already, groups_notyet):
    mean_similarities = {}
    for kk in groups_notyet:
        min_similarity = np.Inf
        max_similarity = -np.Inf
        mean_similarity = 0
        #median_similarity = []
        this_group_ids = groups_notyet[kk]
        for j in xrange(len(this_group_ids)):
            for i in xrange(j+1, len(this_group_ids)):
                temp = matrix_ori[this_group_ids[i],this_group_ids[j]]
                if temp < min_similarity:
                    min_similarity = temp
                if temp > max_similarity:
                    max_similarity = temp
                #mean_similarity += np.log10(temp)
                mean_similarity += temp
                #median_similarity.append(temp)
        mean_similarity /= max(1, len(this_group_ids)*(len(this_group_ids) - 1) / 2)
        mean_similarities[kk] = mean_similarity
    for i in mean_similarities:
        if mean_similarities[i] < 0:
            groups_already[i] = groups_notyet[i]
            groups_notyet.pop(i, None)
def cal_mean_distance(matrix_ori, group):
    min_similarity = np.Inf
    max_similarity = -np.Inf
    mean_similarity = 0
    this_group_ids = group
    for j in xrange(len(this_group_ids)):
            for i in xrange(j+1, len(this_group_ids)):
                temp = matrix_ori[this_group_ids[i],this_group_ids[j]]
                if temp < min_similarity:
                    min_similarity = temp
                if temp > max_similarity:
                    max_similarity = temp
                mean_similarity += temp
    mean_similarity /= max(1, len(this_group_ids)*(len(this_group_ids) - 1) / 2)
    return mean_similarity
def cal_median_distance(matrix, group_1, group_2):
    idx_grid = np.ix_(group_1, group_2)
    matrix_this = matrix[idx_grid]
    return np.median(matrix_this)
def cal_median_distance_ingroup(matrix, group_1):
    idx_grid = np.ix_(group_1, group_1)
    matrix_this = matrix[idx_grid]
    return np.median(matrix_this)
def cluster_faces_new(name = '9_68821308@N00', img_list = 'all-scores-faces-list'):
    #cnn_root = root_all + 'face_recognition_CNN/'+('-').join(name.split('@')) + '/'
    in_path = root + name + '-dir/all-scores-faces-list-new-pw.mat'
    try:
        temp = sio.loadmat(in_path)
    except:
        in_path = root + name + '-dir/all-scores-faces-list-pw.mat'
        temp = sio.loadmat(in_path)
    matrix = temp['matrix']
    
    if len(matrix) == 0:
        out_root = root + name + '-dir/'
        f = open(out_root + '_20_group.cPickle','wb')
        cPickle.dump([{}, []], f)
        f.close()
        return
    #print matrix

    '''1st method: exponential'''
    #a_std = np.std(matrix)
    #beta = 1.0
    #affinity_matrix = np.exp(beta * matrix / a_std)

    '''2nd method: sigmoid'''
    #affinity_matrix = 1 / (1 + np.exp(-matrix))

    '''3rd method: normalize'''

    if len(matrix) == 1:
        imgs_list = []
        with open(in_path, 'r') as data:
            for line in data:
                line = line[:-1]
                imgs_list.append(line.split('/')[-1])
        out_root = root + name + '-dir/'
        f = open(out_root + '_20_group.cPickle','wb')
        cPickle.dump([{img_list[0]:1}, []], f)
        f.close()
        return        
        
    matrix_ori = matrix
    min_ = np.min(matrix)
    matrix = matrix - min_
    diag = np.diag(matrix)
    diag = diag[:, np.newaxis]
    normalize_matrix = np.dot(diag, np.transpose(diag))
    normalize_matrix = np.sqrt(normalize_matrix)
    affinity_matrix = np.divide(matrix, normalize_matrix)
    min_ = np.min(affinity_matrix); max_ = np.max(affinity_matrix)
    affinity_matrix =  (affinity_matrix - min_) / (max_ - min_)
    f = SpectralClustering(affinity='precomputed', n_clusters=min(30, affinity_matrix.shape[0]/2), eigen_solver = 'arpack', n_neighbors=min(10, affinity_matrix.shape[0]))

    #b = f.fit(affinity_matrix)
    a = f.fit_predict(affinity_matrix)
    mean_similarities = {}
    groups_notyet = {}
    temp = zip(a, xrange(len(a)))
    for i in temp:
        if i[0] not in groups_notyet:
            groups_notyet[i[0]] = [i[1]]
        else:
            groups_notyet[i[0]].append(i[1])

    groups_already = {}
    dump_low_mean_(matrix_ori, groups_already, groups_notyet)
    while len(groups_notyet) > 1:
        keys = groups_notyet.keys()
        max_ = [-1,-1]
        max_median = -np.Inf
        for i in xrange(len(keys)):
            for j in xrange(i + 1, len(keys)):
                group_i = groups_notyet[keys[i]]
                group_j = groups_notyet[keys[j]]
                temp = cal_median_distance(matrix_ori, group_i, group_j)
                #print temp
                if temp > max_median:
                    max_median = temp
                    #print temp
                    max_ = [keys[i],keys[j]]
        #print
        #if max_[0] == -1:
        #    break
        #print matrix_ori[np.ix_(groups_notyet[max_[0]], groups_notyet[max_[1]])]
        temp = cal_median_distance_ingroup(matrix_ori, groups_notyet[max_[0]] + groups_notyet[max_[1]])
        #print temp
        #if cal_median_distance_ingroup(matrix_ori, groups_notyet[max_[0]] + groups_notyet[max_[1]]) > -10:
        if cal_median_distance_ingroup(matrix_ori, groups_notyet[max_[0]] + groups_notyet[max_[1]]) > -10:
            temp = groups_notyet[max_[0]] + groups_notyet[max_[1]]
            groups_notyet.pop(max_[0], None)
            groups_notyet.pop(max_[1], None)
            groups_notyet[max_[0]] = temp
        else:
            groups_already[max_[0]] = groups_notyet[max_[0]]
            groups_already[max_[1]] = groups_notyet[max_[1]]
            groups_notyet.pop(max_[0], None)
            groups_notyet.pop(max_[1], None)
        #print len(groups_notyet)
    for i in groups_notyet:
        groups_already[i] = groups_notyet[i]
    #print len(groups_already), groups_already


    unique_person_id = []
    for kk in groups_already:
        min_similarity = np.Inf
        max_similarity = -np.Inf
        mean_similarity = 0
        #median_similarity = []
        this_group_ids = groups_already[kk]
        for j in xrange(len(this_group_ids)):
            for i in xrange(j+1, len(this_group_ids)):
                temp = matrix_ori[this_group_ids[i],this_group_ids[j]]
                if temp < min_similarity:
                    min_similarity = temp
                if temp > max_similarity:
                    max_similarity = temp
                #mean_similarity += np.log10(temp)
                mean_similarity += temp
                #median_similarity.append(temp)
        mean_similarity /= max(1, len(this_group_ids)*(len(this_group_ids) - 1) / 2)
        mean_similarities[kk] = mean_similarity
        #if len(median_similarity) >= 1:
        #    median_ = np.median(np.array(median_similarity))
        #else:
        #    median_ = 0
        #mean_similarities[kk] = median_
        #print len(this_group_ids), mean_similarity, max_similarity, min_similarity
        if mean_similarity > 0 and len(this_group_ids) > 1:
        #if median_ > 0 and len(this_group_ids) > 1:
            unique_person_id.append(kk)
    important_person = []
    for i in unique_person_id:
        important_person.append([i, len(groups_already[i])])
    important_person.sort(key = lambda x:x[1], reverse=True)
    in_path = root + name + '-dir/' + img_list
    imgs_list = []
    with open(in_path, 'r') as data:
        for line in data:
            line = line[:-1]
            imgs_list.append(line.split('/')[-1])

    a = np.zeros(len(imgs_list))
    for i in groups_already:
        for j in groups_already[i]:
            a[j] = i
    temp = zip(a, imgs_list)
    face_groups = {}
    for i in temp:
        if i[0] not in face_groups:
            face_groups[i[0]] = [i[1]]
        else:
            face_groups[i[0]].append(i[1])
    create_retrieval_image(name, matrix)
    create_face_group_html(name, face_groups, important_person, mean_similarities)
    out_root = root + name + '-dir/'
    f = open(out_root + '_20_group.cPickle','wb')
    cPickle.dump([face_groups, important_person], f)
    f.close()

'''display results'''
def create_face_group_html(name, face_groups, important_person, mean_similarities):
    important_ids = [i[0] for i in important_person[:2]]
    if not os.path.exists(html_root + name):
        os.mkdir(html_root + name)
    out_path = html_root + name + '/group.html'
    n_col = 10
    f = open(out_path, 'w')
    f.write('<head>'+name+' group result </head>\n' )
    f.write('<center>')


    for i in face_groups:
        this_img_ids = face_groups[i]
        if i in important_ids:
            f.write('<table class="important" border="1" style="width:100%">\n')
        else:
            f.write('<table border="1" style="width:100%">\n')
        f.write('<tr><td colspan="10" <b>'+str(i)+' mean similarity:' +str(mean_similarities[i]) + '</b></td></tr>\n<tr>')
        col_count = 0
        for id in this_img_ids:
            f.write('<td align=\"center\" valign=\"center\"><img class="test" src=\"'+root+name+'-dir/all-scores-faces/'+id+'\" alt=Loading... width = "120" /></td>\n')
            col_count += 1
            if col_count % 10 == 0:
                f.write('</tr>\n<tr>')
        f.write('</tr>')


    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n .important{border-width:3px;border-color:#000000;}\n ')
    f.close()
def create_retrieval_image(name, similarity_matrix, max_display = 10):
    root1 = root +name
    imgs_list = []
    in_path = root1 + '-dir/all-scores-faces-list'
    with open(in_path, 'r') as data:
        for line in data:
            line = line.split()[0]
            imgs_list.append(line.split('/')[-1])

    img_retrieval = {}
    for i in xrange(len(similarity_matrix)):
        temp = similarity_matrix[i,:]
        rank = np.argsort(temp)[::-1]
        img_retrieval[i] = rank
    if not os.path.exists(html_root + name):
        os.mkdir(html_root + name)
    f = open(html_root  + name + '/retrieval.html','w')
    f.write('<head>'+name+' group result </head>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    for i in xrange(len(img_retrieval)):
            test_index = i
            this_test = img_retrieval[test_index]
            test_url = root1+'-dir/all-scores-faces/'+imgs_list[this_test[0]]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "120" /></td>\n')
            for j in xrange(1, min(len(img_retrieval), max_display+1)):
                score = similarity_matrix[i][this_test[j]]; test_url = root1+'-dir/all-scores-faces/'+imgs_list[this_test[j]]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+test_url+'\" alt=Loading... width = "120" /><br /><b>('+str(score)+')</b><br /></td>\n')
            f.write('</tr>\n')

    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n')
    f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
    f.close()


if __name__ == '__main__':
    #linux_create_path_per_event('test')
    lists = [o[:-4] for o in os.listdir(root) if os.path.isdir(root+o) if o != 'imgs_list']
    print lists
    for name in lists:
        print name
        cluster_faces_new(name=name)
    #name = '0_74464146-N00'
    #name = '0_10284819@N06'
    #name = '10_26432031@N04'
    #name = '0_10323355@N08'
    #name = '17_7614607@N05'
    #name = '1_88916285@N00'
    #name = '0_49745694@N00'
    #name = '9_68821308@N00'
        
    #create_face_file_list()

    #'''other event types'''
    #linux_create_path_per_event()
    #cluster_faces(name = '101_26582481@N08')
    