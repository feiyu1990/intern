__author__ = 'wangyufei'

import numpy as np
from PIL import Image
import cPickle
import os
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import scipy.io as sio

model_name = 'alexnet3k_'

def create_face_file_list():
    root = '/mnt/ilcompf2d0/project/yuwang/event_curation/face_recognition/face_features_wedding/'
    in_path = root + '../../clean_input_and_label/3_event_curation/baseline_wedding_test/wedding_training_id.cPickle'
    f = open(in_path, 'r')
    events_id = cPickle.load(f)
    f.close()
    in_path = root + '../../clean_input_and_label/3_event_curation/baseline_wedding_test/wedding_test_id.cPickle'
    f = open(in_path, 'r')
    events_id.extend(cPickle.load(f))
    f.close()
    print len(events_id)
    #events_id = ['9_8913259@N03']
    for event_id in events_id:
        print event_id
        save_root = root + '../face_CNN_wedding/' + event_id + '/'
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        if not os.path.exists(save_root + 'all-upper'):
            os.mkdir(save_root + 'all-upper')
        f = open(save_root + 'faces_list.txt', 'w')
        path = root + event_id + '-dir/all-scores.txt'
        with open(path, 'r') as data:
            lines = data.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                line = line[:-1]
                if line[0] == '/':
                    meta = line.split('/')
                    img_path = line
                    img_name = meta[-2] + '_' + meta[-1].split('.')[0]
                    im = Image.open(img_path)
                    i += 1
                    line = lines[i][:-1]
                    n_face = int(line)
                    for j in xrange(n_face):
                        i += 1
                        line = lines[i][:-1]
                        meta = line.split(' ')
                        x = int(meta[0]); y=int(meta[1]); width=int(meta[2])
                        new_x = x - int(0.5*width)
                        new_y = y - int(0.5*width)
                        new_width = width * 2
                        new_height = int(width * 3.5)
                        im_w, im_h = im.size
                        v_x = max(0,new_x)
                        v_y = max(0,new_y)
                        v_x2 = min(new_x+new_width, im_w)
                        v_y2 =  min(im_h, new_y+new_height)
                        cr = im.crop([v_x,v_y,v_x2,v_y2])
                        cr_new = cr.resize((256,256))
                        #cr_pad = Image.new("RGB", (new_width, new_height))
                        #cr_pad.paste(cr, (v_x - new_x, v_y - new_y))
                        cr_new.save(save_root + 'all-upper/' + img_name +'-' + str(j) +'.jpg', 'JPEG')
                        f.write('C:/Users/yuwang/Documents/face_recognition_CNN/'+event_id+'/all-upper/' + img_name +'-' + str(j) +'.jpg\r\n')
                else:
                    print line
                i += 1
        f.close()
def create_face_list_all():
    root = '/mnt/ilcompf2d0/project/yuwang/event_curation/face_recognition/face_features_wedding/'
    out_file = open(root + '../face_CNN_wedding/file_list.txt', 'w')
    path = root + '../face_CNN_wedding/'

    paths = [ 'C:/Users/yuwang/Documents/face_recognition_CNN/'+o+'/faces_list.txt'  for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]
    for i in paths:
        out_file.write(i + '\r\n')
    out_file.close()

def from_txt_to_pickle(name):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name+'/'
    in_path = root + model_name + 'faces_feature.txt'
    features = []
    with open(in_path, 'r') as data:
        for line in data:
            meta = line.split()
            if len(meta) != 4096:
                print 'Error! length not equal to 4096!'
                return
            feature = [float(i) for i in meta]
            features.append(feature)
    f = open(root + model_name + 'faces_feature.cPickle','w')
    cPickle.dump(features, f)
    f.close()
def create_similarity_matrix(name):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name+'/'
    f = open(root + model_name + 'faces_feature.cPickle','r')
    features = cPickle.load(f)
    f.close()

    features = np.asarray(features)
    feature_normalize = normalize(features, axis=1)
    print np.dot(feature_normalize[1,:], feature_normalize[1,:])
    similarity_matrix = np.dot(feature_normalize, np.transpose(feature_normalize))
    min_ = np.min(similarity_matrix); max_ = np.max(similarity_matrix)
    similarity_matrix =  (similarity_matrix - min_) / (max_ - min_)

    f = open(root + model_name + 'similarity_matrix.cPickle','w')
    cPickle.dump(similarity_matrix, f)
    f.close()
    similar_pairs = []
    for i in xrange(len(similarity_matrix)):
        for j in xrange(i + 1, len(similarity_matrix)):
            if similarity_matrix[i,j] > 0.95:
                similar_pairs.append((i,j))



def cluster_faces_CNN(name = '9_8913259@N03', img_list = 'faces_list.txt'):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name + '/'
    f = open(root + model_name + 'similarity_matrix.cPickle','r')
    affinity_matrix = cPickle.load(f)
    f.close()

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
                temp = affinity_matrix[this_group_ids[i],this_group_ids[j]]
                if temp < min_similarity:
                    min_similarity = temp
                if temp > max_similarity:
                    max_similarity = temp
                mean_similarity += temp
        mean_similarity /= max(1, len(this_group_ids)*(len(this_group_ids) - 1) / 2)
        print len(this_group_ids), mean_similarity, max_similarity, min_similarity
        if mean_similarity > 0.5:
            unique_person_id.append(kk)
    important_person = []
    for i in unique_person_id:
        important_person.append([i, len(groups[i])])
    important_person.sort(key = lambda x:x[1], reverse=True)
    in_path = root + img_list
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

    create_face_group_html_CNN(name, face_groups, important_person)
def create_face_group_html_CNN(name, face_groups, important_person):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name + '/'
    important_ids = [i[0] for i in important_person[:2]]
    out_path = root + model_name + 'CNN_group.html'
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
def create_retrieval_image_CNN(name, max_display = 10):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name + '/'
    f = open(root + model_name + 'similarity_matrix.cPickle','r')
    similarity_matrix = cPickle.load(f)
    f.close()

    imgs_list = []
    in_path = root + 'faces_list.txt'

    with open(in_path, 'r') as data:
        for line in data:
            line = line.split()[0]
            imgs_list.append(line.split('_')[-1])

    img_retrieval = {}
    for i in xrange(len(similarity_matrix)):
        temp = similarity_matrix[i,:]
        rank = np.argsort(temp)[::-1]
        img_retrieval[i] = rank

    f = open(root + name + '_' + model_name + 'CNN_retrieval.html','w')
    f.write('<head>'+name+' group result </head>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    for i in xrange(len(img_retrieval)):
            test_index = i
            this_test = img_retrieval[test_index]
            test_url = root+'/all-upper/'+imgs_list[this_test[0]]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "120" /></td>\n')
            for j in xrange(1, max_display+1):
                score = similarity_matrix[i][this_test[j]]; test_url = root+'/all-upper/'+imgs_list[this_test[j]]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+test_url+'\" alt=Loading... width = "120" /><br /><b>('+str(score)+')</b><br /></td>\n')
            f.write('</tr>\n')

    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n')
    f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
    f.close()



'''combined CNN & face recognition result'''
def combine_cnn_face_before(name, factor = 1):
    face_root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'+name
    temp = sio.loadmat(face_root + '-pw.mat')
    face_similarity_matrix = temp['matrix']
    min_ = np.min(face_similarity_matrix)
    face_similarity_matrix = face_similarity_matrix - min_

    face_cnn_root =  '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name + '/'
    f = open(face_cnn_root +  model_name + 'similarity_matrix.cPickle','r')
    CNN_matrix = cPickle.load(f)
    f.close()

    diag = np.diag(face_similarity_matrix)
    diag = diag[:, np.newaxis]
    normalize_matrix = np.dot(diag, np.transpose(diag))
    normalize_matrix = np.sqrt(normalize_matrix)
    affinity_matrix = np.divide(face_similarity_matrix, normalize_matrix)
    min_ = np.min(affinity_matrix); max_ = np.max(affinity_matrix)
    affinity_matrix =  (affinity_matrix - min_) / (max_ - min_)

    imgs_list_CNN = []
    in_path = face_cnn_root + 'faces_list.txt'
    with open(in_path, 'r') as data:
        for line in data:
            line = line.split()[0]
            imgs_list_CNN.append(line.split('_')[-1])

    new_order = []
    in_path = face_root+'-dir/all-scores-faces-list'
    with open(in_path, 'r') as data:
        for line in data:
            line = line[:-1]
            name_ = line.split('/')[-1]
            new_order.append(imgs_list_CNN.index(name_))

    temp = [imgs_list_CNN[i] for i in new_order]
    CNN_matrix_reordered = CNN_matrix[:, new_order][new_order]

    '''1st method: multiply x^factor'''
    #combined_matrix = affinity_matrix * (CNN_matrix_reordered ** factor)

    '''2nd method: sum'''
    #combined_matrix = affinity_matrix + (CNN_matrix_reordered)

    '''add if strongly correlated'''
    factor_matrix = np.ones(CNN_matrix_reordered.shape)
    factor_matrix = factor_matrix + 0.2*(CNN_matrix_reordered > 0.85)
    combined_matrix = affinity_matrix * (factor_matrix)



    min_ = np.min(combined_matrix); max_ = np.max(combined_matrix)
    combined_matrix =  (combined_matrix - min_) / (max_ - min_)
    print np.max(CNN_matrix_reordered), np.min(CNN_matrix_reordered)
    face_cnn_path =  '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'+ model_name + name + '_combined.cPickle'
    f = open(face_cnn_path, 'w')
    cPickle.dump(combined_matrix, f)
    f.close()
def combine_cnn_face_after(name, factor = 1):
    face_root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'+name
    temp = sio.loadmat(face_root + '-pw.mat')
    face_similarity_matrix = temp['matrix']

    face_cnn_root =  '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition_CNN/'+name + '/'
    f = open(face_cnn_root +  model_name + 'similarity_matrix.cPickle','r')
    CNN_matrix = cPickle.load(f)
    f.close()


    imgs_list_CNN = []
    in_path = face_cnn_root + 'faces_list.txt'
    with open(in_path, 'r') as data:
        for line in data:
            line = line.split()[0]
            imgs_list_CNN.append(line.split('_')[-1])

    new_order = []
    in_path = face_root+'-dir/all-scores-faces-list'
    with open(in_path, 'r') as data:
        for line in data:
            line = line[:-1]
            name_ = line.split('/')[-1]
            new_order.append(imgs_list_CNN.index(name_))

    temp = [imgs_list_CNN[i] for i in new_order]
    CNN_matrix_reordered = CNN_matrix[:, new_order][new_order]

    '''1st method: multiply x^factor'''
    #combined_matrix = affinity_matrix * (CNN_matrix_reordered ** factor)

    '''2nd method: sum'''
    #combined_matrix = affinity_matrix + (CNN_matrix_reordered)

    '''add if strongly correlated'''
    factor_matrix = np.ones(CNN_matrix_reordered.shape)
    mask = CNN_matrix_reordered > 0.85
    mask2 = np.sign(face_similarity_matrix)
    mask = mask*mask2
    factor_matrix = factor_matrix + 0.2*mask
    combined_matrix = face_similarity_matrix * (factor_matrix)
    #for i in xrange(len(factor_matrix)):
    #    print factor_matrix[:,i]

    face_cnn_path =  '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'+ model_name + name + '_combined.cPickle'
    f = open(face_cnn_path, 'w')
    cPickle.dump(combined_matrix, f)
    f.close()

    min_ = np.min(combined_matrix)
    combined_matrix = combined_matrix - min_

    diag = np.diag(combined_matrix)
    diag = diag[:, np.newaxis]
    normalize_matrix = np.dot(diag, np.transpose(diag))
    normalize_matrix = np.sqrt(normalize_matrix)
    affinity_matrix = np.divide(combined_matrix, normalize_matrix)
    min_ = np.min(affinity_matrix); max_ = np.max(affinity_matrix)
    affinity_matrix =  (affinity_matrix - min_) / (max_ - min_)

    face_cnn_path =  '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'+ model_name + name + '_normalized_combined.cPickle'
    f = open(face_cnn_path, 'w')
    cPickle.dump(affinity_matrix, f)
    f.close()


def create_retrieval_image(name, max_display = 10):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'
    f = open(root + model_name + name + '_combined.cPickle','r')
    similarity_matrix = cPickle.load(f)
    f.close()

    imgs_list = []
    in_path = root + name + '-dir/all-scores-faces-list'

    with open(in_path, 'r') as data:
        for line in data:
            line = line.split()[0]
            imgs_list.append(line.split('/')[-1])

    img_retrieval = {}
    for i in xrange(len(similarity_matrix)):
        temp = similarity_matrix[i,:]
        rank = np.argsort(temp)[::-1]
        img_retrieval[i] = rank

    f = open(root + model_name + name + '_combined_retrieval.html','w')
    f.write('<head>'+name+' group result </head>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    for i in xrange(len(img_retrieval)):
            test_index = i
            this_test = img_retrieval[test_index]
            test_url = root+name+'-dir/all-scores-faces/'+imgs_list[this_test[0]]
            f.write('<tr><td align=\"center\" valign=\"center\"><img class="test" src=\"'+test_url+'\" alt=Loading... width = "120" /></td>\n')
            for j in xrange(1, max_display+1):
                score = similarity_matrix[i][this_test[j]]; test_url = root+name+'-dir/all-scores-faces/'+imgs_list[this_test[j]]
                f.write('<td align=\"center\" valign=\"center\"><img src=\"'+test_url+'\" alt=Loading... width = "120" /><br /><b>('+str(score)+')</b><br /></td>\n')
            f.write('</tr>\n')

    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n')
    f.write('img.test{ border: 3px ridge #FF0000;}\n </style>')
    f.close()
def cluster_faces(name = '9_8913259@N03', img_list = 'all-scores-faces-list'):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'
    f = open(root + model_name + name + '_combined.cPickle','r')
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
        if mean_similarity > 10 and len(this_group_ids) > 1:
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

    create_face_group_html(name, face_groups, important_person)
def create_face_group_html(name, face_groups, important_person):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/face_recognition/'
    important_ids = [i[0] for i in important_person[:2]]
    out_path = root + model_name + name + '_combined_group.html'
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
            f.write('<td align=\"center\" valign=\"center\"><img class="test" src=\"'+root+name+'-dir/all-scores-faces/'+id+'\" alt=Loading... width = "120" /></td>\n')
            col_count += 1
            if col_count % n_col == 0:
                f.write('</tr>\n<tr>')
        f.write('</tr>')


    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n .important{border-width:3px;border-color:#000000;}\n ')
    f.close()



if __name__ == '__main__':
    #create_face_file_list()
    #create_face_list_all()

    name = '0_74464146-N00'
    #name = '0_10284819@N06'
    #name = '10_26432031@N04'
    #name = '0_10323355@N08'
    #name = '17_7614607@N05'
    #name = '1_88916285@N00'
    #name = '0_49745694@N00'
    #name = '9_68821308@N00'
    from_txt_to_pickle(name)
    create_similarity_matrix(name)

    create_retrieval_image_CNN(name)
    combine_cnn_face_after(name)
    create_retrieval_image(name)
    cluster_faces(name)
    cluster_faces_CNN(name)
