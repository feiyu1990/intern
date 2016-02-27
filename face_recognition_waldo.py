__author__ = 'wangyufei'

import numpy as np
from PIL import Image
import cPickle
import os
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import scipy.io as sio
from sklearn import linear_model
import re

root_all = '/Users/wangyufei/Documents/Study/intern_adobe/'
html_root = 'C:/Users/yuwang/Documents/present_htmls/'
#root_all = 'C:/Users/yuwang/Documents/'
model_name = 'vgg'
def from_txt_to_pickle(name):
    root = root_all + 'face_recognition_CNN/'+name + '/'
    in_path = root + 'waldo_retrieval_result.txt'
    file_list = []
    line_count = 0
    with open(root + 'waldolist.dat', 'r') as data:
        for line in data:
            line_count += 1
            if line_count <= 2:
                continue
            this_name = line.split()[0]
            this_name = this_name.split('\\')[-1]
            file_list.append(this_name)

    similarity_matrix = np.zeros((len(file_list), len(file_list)))
    line_number = 0
    with open(in_path, 'r') as data:
        for line in data:
            groups = re.findall('\((.*?)\)',line)
            for pair in groups:
                img_name, score = pair.split(', ')
                img_name = int(img_name)
                score = float(score)
                similarity_matrix[line_number, img_name] = score
            line_number += 1

    print similarity_matrix
    f = open(root + 'waldo_retrieval.cPickle','w')
    cPickle.dump(similarity_matrix, f)
    f.close()

    for i in xrange(len(file_list)):
        for j in xrange(i + 1, len(file_list)):
            temp = (similarity_matrix[i][j] + similarity_matrix[j][i]) / 2
            similarity_matrix[i][j] = temp
            similarity_matrix[j][i] = temp
    f = open(root + 'waldo_similarity.cPickle','w')
    cPickle.dump(similarity_matrix, f)
    f.close()

'''combined waldo & face recognition result'''
def combine_cnn_face(name, factor = 0.5):
    face_root = root_all + 'face_recognition/'+ '@'.join(name.split('-'))
    temp = sio.loadmat(face_root + '-dir/all-scores-faces-list-new-pw.mat')
    face_similarity_matrix = temp['matrix']

    face_cnn_root = root_all + 'face_recognition_CNN/'+name + '/'
    f = open(face_cnn_root  + 'waldo_similarity.cPickle','r')
    CNN_matrix = cPickle.load(f)
    f.close()


    imgs_list_CNN = []
    line_count = 0
    with open(face_cnn_root + 'waldolist.dat', 'r') as data:
        for line in data:
            line_count += 1
            if line_count <= 2:
                continue
            this_name = line.split()[0]
            this_name = this_name.split('\\')[-1]
            imgs_list_CNN.append(this_name)


    new_order = []
    in_path = face_root+'-dir/all-scores-faces-list-new'
    with open(in_path, 'r') as data:
        for line in data:
            line = line[:-1]
            name_ = line.split('/')[-1]
            new_order.append(imgs_list_CNN.index(name_))

    #temp = [imgs_list_CNN[i] for i in new_order]
    CNN_matrix_reordered = CNN_matrix[:, new_order][new_order]

    diag = np.diag(face_similarity_matrix)
    diag = diag[:, np.newaxis]
    normalize_matrix = np.dot(diag, np.transpose(diag))
    normalize_matrix = np.sqrt(normalize_matrix)
    affinity_matrix = np.divide(face_similarity_matrix, normalize_matrix)
    min_ = np.min(affinity_matrix); max_ = np.max(affinity_matrix)
    affinity_matrix =  (affinity_matrix - min_) / (max_ - min_)

    print affinity_matrix

    '''1st method: multiply x^factor'''
    combined_matrix = affinity_matrix * (CNN_matrix_reordered ** factor)

    '''2nd method: sum'''
    #combined_matrix = affinity_matrix + (CNN_matrix_reordered)

    '''add if strongly correlated'''
    #factor_matrix = np.ones(CNN_matrix_reordered.shape)
    #mask = CNN_matrix_reordered > 0.85
    #mask2 = np.sign(face_similarity_matrix)
    #mask = mask*mask2
    #factor_matrix = factor_matrix + 0.2*mask
    #combined_matrix = face_similarity_matrix * (factor_matrix)


    #for i in xrange(len(factor_matrix)):
    #    print factor_matrix[:,i]


    face_cnn_path =  face_cnn_root + 'waldo_normalized_combined.cPickle'
    f = open(face_cnn_path, 'w')
    cPickle.dump(combined_matrix, f)
    f.close()
def cluster_faces(name, img_list = 'all-scores-faces-list-new'):
    root = root_all + 'face_recognition/'+ '@'.join(name.split('-'))
    cnn_root = root_all + 'face_recognition_CNN/'+name + '/'

    f = open(cnn_root + 'waldo_normalized_combined.cPickle','r')
    combined_matrix = cPickle.load(f)
    f.close()

    diag = np.diag(combined_matrix)
    diag = diag[:, np.newaxis]
    normalize_matrix = np.dot(diag, np.transpose(diag))
    normalize_matrix = np.sqrt(normalize_matrix)
    affinity_matrix = np.divide(combined_matrix, normalize_matrix)
    min_ = np.min(affinity_matrix); max_ = np.max(affinity_matrix)
    affinity_matrix =  (affinity_matrix - min_) / (max_ - min_)


    f = SpectralClustering(affinity='precomputed', n_clusters=min(8, affinity_matrix.shape[0] - 1), eigen_solver = 'arpack', n_neighbors=min(5, affinity_matrix.shape[0]))
    a = f.fit_predict(affinity_matrix)

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
        this_group_ids = groups[kk]
        for j in xrange(len(this_group_ids)):
            for i in xrange(j+1, len(this_group_ids)):
                temp = combined_matrix[this_group_ids[i],this_group_ids[j]]
                if temp < min_similarity:
                    min_similarity = temp
                if temp > max_similarity:
                    max_similarity = temp
                mean_similarity += temp
        mean_similarity /= max(1, len(this_group_ids)*(len(this_group_ids) - 1) / 2)
        print len(this_group_ids), mean_similarity, max_similarity, min_similarity
        print mean_similarity
        if mean_similarity > 0.4 and len(this_group_ids) > 1:
            unique_person_id.append(kk)
    important_person = []
    for i in unique_person_id:
        important_person.append([i, len(groups[i])])
    important_person.sort(key = lambda x:x[1], reverse=True)
    in_path = root + '-dir/' + img_list
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

    create_face_group_html(name, face_groups, important_person)

    f = open(cnn_root + 'waldo_group_combined.cPickle','w')
    cPickle.dump([face_groups, important_person], f)
    f.close()


'''display results'''
def create_retrieval_image_waldo(name, max_display = 10):
    root_h = html_root + ('@').join(name.split('-')) + '/'
    root = root_all + 'face_recognition_CNN/'+name + '/'
    f = open(root + 'waldo_similarity.cPickle','r')
    similarity_matrix = cPickle.load(f)
    f.close()

    imgs_list = []
    line_count = 0
    with open(root + 'waldolist.dat', 'r') as data:
        for line in data:
            line_count += 1
            if line_count <= 2:
                continue
            this_name = line.split()[0]
            this_name = this_name.split('\\')[-1]
            imgs_list.append(this_name)



    img_retrieval = {}
    for i in xrange(len(similarity_matrix)):
        temp = similarity_matrix[i,:]
        rank = np.argsort(temp)[::-1]
        img_retrieval[i] = rank

    f = open(root_h+'waldo_retrieval.html','w')
    f.write('<head>'+name+' waldo retrieval result </head>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    for i in xrange(len(img_retrieval)):
            test_index = i
            this_test = img_retrieval[test_index]
            test_url = root+'/all-upper/'+imgs_list[this_test[0]]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "120" /></td>\n')
            for j in xrange(1, min(len(img_retrieval), max_display+1)):
                score = similarity_matrix[i][this_test[j]]; test_url = root+'/all-upper/'+imgs_list[this_test[j]]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+test_url+'\" alt=Loading... width = "120" /><br /><b>('+str(score)+')</b><br /></td>\n')
            f.write('</tr>\n')

    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n')
    f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
    f.close()
def create_retrieval_image(name, max_display = 10):
    root_h = html_root + ('@').join(name.split('-')) + '/'
    root = root_all + 'face_recognition_CNN/'+name + '/'
    face_root = root_all + 'face_recognition/'+ '@'.join(name.split('-'))
    f = open(root + 'waldo_normalized_combined.cPickle','r')
    similarity_matrix = cPickle.load(f)
    f.close()

    imgs_list = []
    in_path = face_root + '-dir/all-scores-faces-list'

    with open(in_path, 'r') as data:
        for line in data:
            line = line.split()[0]
            imgs_list.append(line.split('/')[-1])

    img_retrieval = {}
    for i in xrange(len(similarity_matrix)):
        temp = similarity_matrix[i,:]
        rank = np.argsort(temp)[::-1]
        img_retrieval[i] = rank

    f = open(root_h + 'waldo_combined_retrieval.html','w')
    f.write('<head>'+name+' group result </head>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    for i in xrange(len(img_retrieval)):
            test_index = i
            this_test = img_retrieval[test_index]
            test_url = root+'/all-upper/'+imgs_list[this_test[0]]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "120" /></td>\n')
            for j in xrange(1, min(max_display+1, len(img_retrieval))):
                score = similarity_matrix[i][this_test[j]]; test_url = root+'/all-upper/'+imgs_list[this_test[j]]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+test_url+'\" alt=Loading... width = "120" /><br /><b>('+str(score)+')</b><br /></td>\n')
            f.write('</tr>\n')

    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n')
    f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
    f.close()
def create_face_group_html(name, face_groups, important_person):
    root_h = html_root + ('@').join(name.split('-')) + '/'
    root = root_all + 'face_recognition_CNN/'+name + '/'
    important_ids = [i[0] for i in important_person[:2]]
    out_path = root_h + 'waldo_combined_group.html'
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
        f.write('<tr><td colspan="10" <b>'+str(i)+'</b></td></tr>\n<tr>')
        col_count = 0
        for id in this_img_ids:
            f.write('<td align=\"center\" valign=\"center\"><img class="test" src=\"'+root+'/all-upper/'+id+'\" alt=Loading... width = "120" /></td>\n')
            col_count += 1
            if col_count % n_col == 0:
                f.write('</tr>\n<tr>')
        f.write('</tr>')


    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n .important{border-width:3px;border-color:#000000;}\n ')
    f.close()


'''use validation set to train parameters'''
group_name = 'group.cPickle'
#group_name = 'waldo_group_combined.cPickle'
def combine_two_cue(name):
    #root2 = root_all + 'amt/clean_input_and_label/3_event_curation/baseline_wedding_test/training_validation/'
    root2 = root_all + 'baseline_wedding_test/training_validation/'
    f = open(root2 + 'val_validation_image_ids.cPickle','r')
    training_image_ids = cPickle.load(f)
    f.close()
    name_this = ('@').join(name.split('-'))
    img_names = []
    img_names_short = []
    for img in training_image_ids:
        if img.split('/')[0] == name_this:
            img_names.append(img)
            img_names_short.append(img.split('/')[1])


    cnn_root = root_all + 'face_recognition_CNN/'+name + '/'
    root = root_all + 'face_recognition/'+ '@'.join(name.split('-'))
    f = open(cnn_root + group_name,'r')
    face_groups, important_person = cPickle.load(f)
    f.close()
    img_sizes = {}
    lines = []
    with open(root + '-dir/all-scores.txt','r') as data:
        for line in data:
            lines.append(line)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[0] != '/':
            print 'ERROR!'
        img_name = line.split('/')[-1]
        img_name = img_name.split()[0]
        i += 1
        line = lines[i]
        num_img = int(line.split()[0])
        for j in xrange(num_img):
            i += 1
            line = lines[i]
            size = int(line.split(' ')[2])
            #size = size*size
            img_name_this = img_name.split('.')[0] + '-' + str(j)
            img_sizes[img_name_this] = size
        i += 1

    important_persons = [np.zeros((len(img_names),1)), np.zeros((len(img_names),1))]

    important_person_filtered = []
    for i in important_person:
        k = i[0]
        group_this = face_groups[k]
        average_size = 0
        for img in group_this:
            average_size += img_sizes[img.split('.')[0]]
        average_size /= len(group_this)
        #print average_size
        if average_size > 20:
            important_person_filtered.append(i)

    important_person_filtered.sort(key = lambda x:x[1], reverse=True)
    if len(important_person_filtered) > 1:
        important_id = [important_person_filtered[0][0], important_person_filtered[1][0]]
    elif len(important_person_filtered) == 1:
        important_id = [important_person_filtered[0][0]]
    else:
        important_id = []
    count = 0
    for i in important_id:
        this_group = face_groups[i]
        for img in this_group:
            img = img.split('.')[0]
            #print img_names.index(img.split('-')[0])
            important_persons[count][img_names_short.index(img.split('-')[0])] = 1
        count += 1

    important_images1 = np.add(important_persons[0], important_persons[1])
    important_images1 = important_images1 % 2
    important_images2 = np.add(important_persons[0], important_persons[1])
    important_images2 = important_images2 / 2
    #print important_images2
    important_images1 = important_images1.astype(int)
    important_images2 = important_images2.astype(int)

    n_vote = 10
    f = open(root_all + 'baseline_wedding_test/training_validation/val_'+model_name+'_predict_result_'+str(n_vote)+'_facereweighted.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()

    this_test_feature = test_prediction_event[name_this]
    #knn_features = np.asarray([i[2] for i in this_test_feature])
    knn_features = np.zeros((len(img_names),1))
    for i in this_test_feature:
        if i[0] in img_names:
            knn_features[img_names.index(i[0])] = i[2]
    features = np.concatenate((knn_features, important_images1, important_images2), axis=1)

    f = open(root2 + '../'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    all_ground_result = cPickle.load(f)
    f.close()

    score_ground_truth = []
    #print features.shape

    for i in img_names:
        score_ground_truth.append(all_ground_result[i])

    return features, score_ground_truth, img_names
def combine_two_cue_test(name):
    #root2 = root_all + 'amt/clean_input_and_label/3_event_curation/baseline_wedding_test/training_validation/'
    root2 = root_all + 'baseline_wedding_test/'
    f = open(root2 + 'wedding_test_image_ids.cPickle','r')
    training_image_ids = cPickle.load(f)
    f.close()
    name_this = ('@').join(name.split('-'))
    img_names = []
    img_names_short = []
    for img in training_image_ids:
        if img.split('/')[0] == name_this:
            img_names.append(img)
            img_names_short.append(img.split('/')[1])


    cnn_root = root_all + 'face_recognition_CNN/'+name + '/'
    root = root_all + 'face_recognition/'+ '@'.join(name.split('-'))
    f = open(cnn_root + group_name,'r')
    face_groups, important_person = cPickle.load(f)
    f.close()
    img_sizes = {}
    lines = []
    with open(root + '-dir/all-scores.txt','r') as data:
        for line in data:
            lines.append(line)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[0] != '/':
            print 'ERROR!'
        img_name = line.split('/')[-1]
        img_name = img_name.split()[0]
        i += 1
        line = lines[i]
        num_img = int(line.split()[0])
        for j in xrange(num_img):
            i += 1
            line = lines[i]
            size = int(line.split(' ')[2])
            #size = size*size
            img_name_this = img_name.split('.')[0] + '-' + str(j)
            img_sizes[img_name_this] = size
        i += 1

    important_persons = [np.zeros((len(img_names),1)), np.zeros((len(img_names),1))]

    important_person_filtered = []
    for i in important_person:
        k = i[0]
        group_this = face_groups[k]
        average_size = 0
        for img in group_this:
            average_size += img_sizes[img.split('.')[0]]
        average_size /= len(group_this)
        #print average_size
        if average_size > 20:
            important_person_filtered.append(i)

    important_person_filtered.sort(key = lambda x:x[1], reverse=True)
    if len(important_person_filtered) > 1:
        important_id = [important_person_filtered[0][0], important_person_filtered[1][0]]
    elif len(important_person_filtered) == 1:
        important_id = [important_person_filtered[0][0]]
    else:
        important_id = []
    count = 0
    for i in important_id:
        this_group = face_groups[i]
        for img in this_group:
            img = img.split('.')[0]
            #print img_names.index(img.split('-')[0])
            important_persons[count][img_names_short.index(img.split('-')[0])] = 1
        count += 1

    important_images1 = np.add(important_persons[0], important_persons[1])
    important_images1 = important_images1 % 2
    important_images2 = np.add(important_persons[0], important_persons[1])
    important_images2 = important_images2 / 2
    #print important_images2
    important_images1 = important_images1.astype(int)
    important_images2 = important_images2.astype(int)

    model_name = 'vgg'; n_vote = 10
    f = open(root_all + 'baseline_wedding_test/'+model_name+'_wedding_predict_result_'+str(n_vote)+'_facereweighted_old_dict.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()

    this_test_feature = test_prediction_event[name_this]
    knn_features = np.zeros((len(img_names),1))
    for i in this_test_feature:
        if i[0] in img_names:
            knn_features[img_names.index(i[0])] = i[2]
    features = np.concatenate((knn_features, important_images1, important_images2), axis=1)

    return features, img_names
def train_and_predict(combination_coef, n_vote = 10):
    root2 = root_all + 'baseline_wedding_test/training_validation/'
    f = open(root2 + 'validation_id.cPickle','r')
    ids = cPickle.load(f)
    f.close()

    features = np.zeros((0, 3))
    score_ground_truth = []
    img_names = []
    for name in ids:
        name = ('-').join(name.split('@'))
        a, b, c = combine_two_cue(name)
        features = np.concatenate((features, a), axis=0)
        score_ground_truth += b
        img_names += c
    print len(score_ground_truth)
    print features.shape
    model = linear_model.LinearRegression()
    model.fit(features, score_ground_truth)
    print model.coef_

    root2 = root_all + 'baseline_wedding_test/'
    f = open(root2 + 'wedding_test_id.cPickle','r')
    ids = cPickle.load(f)
    f.close()

    temp = model.coef_
    temp[1] = temp[1] * combination_coef
    temp[2] = temp[2] * combination_coef
    model.coef_ = temp
    print temp
    


    t_features = np.zeros((0, 3))
    t_img_names = []
    for name in ids:
        name = ('-').join(name.split('@'))
        a, b = combine_two_cue_test(name)
        t_features = np.concatenate((t_features, a), axis=0)
        t_img_names += b
    print t_features.shape
    score_predict = model.predict(t_features)
    print model.coef_


    f = open(root_all + 'baseline_wedding_test/wedding_test_ulr_dict.cPickle', 'r')
    url_dict = cPickle.load(f)
    f.close()

    test_prediction = []
    for i in xrange(len(score_predict)):
        test_prediction.append([t_img_names[i], url_dict[t_img_names[i]], score_predict[i]])



    f = open(root_all + 'baseline_wedding_test/'+model_name+'_wedding_training_result_dict_v2.cPickle','r')
    training_scores_dict = cPickle.load(f)
    f.close()
    f = open(root_all + 'baseline_wedding_test/'+model_name+'_wedding_knn_facereweighted_old_dict.cPickle', 'r')
    knn = cPickle.load(f)
    f.close()

    for i in knn:
        this_test_img = knn[i]
        this_test_id = this_test_img[0][0]
        if this_test_id in training_scores_dict:
            print 'ERROR!'


    test_prediction_event = {}
    for i in test_prediction:
        img_id = i
        event_id = img_id[0].split('/')[0]
        if event_id in test_prediction_event:
            test_prediction_event[event_id].append(i)
        else:
            test_prediction_event[event_id] = [i]

    f = open(root_all + 'baseline_wedding_test/face_'+model_name+'_wedding_predict_result_'+str(n_vote)+'_facereweighted_old_dict.cPickle','w')
    cPickle.dump(test_prediction_event, f)
    f.close()

if __name__ == '__main__':
    train_and_predict(0.5)

    #create_face_file_list()
    #create_face_list_all()

    #name = '0_74464146-N00'
    #name = '0_10284819@N06'
    #name = '10_26432031@N04'
    #name = '0_10323355@N08'
    #name = '17_7614607@N05'
    #name = '1_88916285@N00'
    #name = '0_49745694@N00'
    #name = '9_68821308@N00'
    
    '''
    lists = [o for o in os.listdir(root_all + 'face_recognition_CNN/') if os.path.isdir(root_all + 'face_recognition_CNN/'+o)]
    for name in lists:
        print name
        combine_cnn_face(name)
        create_retrieval_image_waldo(name)
        cluster_faces(name)
        create_retrieval_image(name)
    '''




