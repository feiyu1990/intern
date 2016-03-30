__author__ = 'wangyufei'
import cPickle
import os
import numpy as np
#from sklearn.cluster import spectral_clustering
from sklearn.cluster import SpectralClustering
import scipy.io as sio
import csv
from PIL import Image
from collections import Counter, defaultdict

correct_list = {'5_19479358@N00':'Museum', '38_59616483@N00':'Museum','136_95413346@N00':'Museum',
                    '0_27302158@N00':'CasualFamilyGather','7_55455788@N00':'Birthday',
                    '144_95413346@N00':'Halloween', '29_13125640@N07':'Christmas', '1_21856707@N00': 'GroupActivity',
                    '0_22928590@N00':'GroupActivity','3_7531619@N05':'Zoo',
                    '16_18108851@N00':'Show', '23_89182227@N00':'Show', '2_27883710@N08':'Sports',
                    '35_8743691@N02':'Wedding', '14_93241698@N00':'Museum', '9_34507951@N07':'BusinessActivity',
                    '32_35578067@N00':'Protest', '20_89138584@N00':'PersonalSports', '18_50938313@N00':'PersonalSports',
                    '376_86383385@N00':'PersonalSports','439_86383385@N00':'PersonalSports','545_86383385@N00':'PersonalSports',
                    '2_43198495@N05':'PersonalSports', '3_60652642@N00':'ReligiousActivity', '9_60053005@N00':'GroupActivity',

                        '56_74814994@N00':'BusinessActivity', '22_32994285@N00':'Sports', '15_66390637@N08':'Sports',
                         '3_54218473@N05':'Zoo', '4_53628484@N00':'Sports', '0_7706183@N06':'GroupActivity',
                         '4_15251430@N03':'Zoo', '63_52304204@N00':'Sports', '2_36319742@N05':'Architecture',
                         '2_12882543@N00':'Sports', '1_75003318@N00':'Sports', '1_88464035@N00':'GroupActivity',
                         '21_49503048699@N01':'CasualFamilyGather', '211_86383385@N00':'Sports',
                         '0_70073383@N00':'PersonalArtActivity'}




dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
             'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}
            
#html_root_test = 'C:/Users/yuwang/Documents/present_htmls_val/'
#face_root = 'C:/Users/yuwang/Documents/face_recognition/'
# root = '/Users/wangyufei/Documents/Study/intern_adobe/'
# html_root = root + '/present_htmls_new/'
root = '/home/feiyu1990/local/event_curation/'
html_root = root + '/present_htmls_rebuttal/'
html_root_test = root + 'present_htmls_rebuttal/'
# if not os.path.exists(html_root_test):
#     os.mkdir(html_root_test)
block_workers = ['A3U0FX9MF8C6RU','A2NJAHKIJQXPCI']
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
def cluster_faces_new(name, event_type, img_list = 'all-scores-faces-list'):
    #cnn_root = root_all + 'face_recognition_CNN/'+('-').join(name.split('@')) + '/'
    in_path = face_root + name + '-dir/all-scores-faces-list-new-pw.mat'
    try:
        temp = sio.loadmat(in_path)
    except:
        in_path = face_root + name + '-dir/all-scores-faces-list-pw.mat'
        temp = sio.loadmat(in_path)
    matrix = temp['matrix']

    if len(matrix) == 0:
        out_root = face_root + name + '-dir/'
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
        out_root = face_root + name + '-dir/'
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
    in_path = face_root + name + '-dir/' + img_list
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
    create_retrieval_image(name, event_type, matrix)
    create_face_group_html(name, event_type, face_groups, important_person, mean_similarities)
    out_root = face_root + name + '-dir/'
    f = open(face_root + '_20_group.cPickle','wb')
    cPickle.dump([face_groups, important_person], f)
    f.close()
def create_face_group_html(name, event_type, face_groups, important_person, mean_similarities):
    important_ids = [i[0] for i in important_person[:2]]
    if not os.path.exists(html_root +event_type):
        os.mkdir(html_root + event_type)
    out_path = html_root + event_type + '/' + name + '/group.html'
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
            f.write('<td align=\"center\" valign=\"center\"><img class="test" src=\"'+face_root+name+'-dir/all-scores-faces/'+id+'\" alt=Loading... width = "120" /></td>\n')
            col_count += 1
            if col_count % 10 == 0:
                f.write('</tr>\n<tr>')
        f.write('</tr>')


    f.write('</table>\n')
    f.write('<style type=\"text/css\">img { height:auto;width:\"120px\";}\n .important{border-width:3px;border-color:#000000;}\n ')
    f.close()
def create_retrieval_image(name, event_type, similarity_matrix, max_display = 10):
    root1 = face_root +name
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
    f = open(html_root  + event_type + '/' + name + '/retrieval.html','w')
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
def create_result_htmls(input_path = root + 'all_output/all_output.csv'):
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                if meta[15] in block_workers:
                    line_count+=1
                    continue
                elif meta[0] not in HITs:
                    HITs[meta[0]] = [meta]
                else:
                    HITs[meta[0]].append(meta)
            line_count+=1


    for HITId in HITs:

        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        event_id = this_hits[0][28]
        worker_id = this_hits[0][15]
        input_field = []
        output_field = []
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_type' in field:
                    name = value1
                    event_type = ''
                    for i in dict_name:
                        if dict_name[i] == name:
                            event_type = i
                    #print name
                    #name = ''.join(e for e in name if e.isalnum())
                    #name = name[:10]
                if 'num_image' in field:
                    print value1
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        #if event_id == '84_7273032@N04':
        #    print event_id
        root1 = html_root + '/' + event_type + '/' + event_id  + '/'
        if not os.path.exists(html_root + '/' + event_type):
            os.mkdir(html_root + '/' + event_type)
        if not os.path.exists(html_root + '/' + event_type+'/' + event_id ):
            os.mkdir(html_root + '/' + event_type+'/' + event_id )
        out_file = root1 +'present_groundtruth.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_predict.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)

        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]
                    len_selection = len(curr_hit)
                    if field == 'feedback':
                        pass
                    else:

                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if key == '' or key == 'NOTVALID':
                            continue
                        if field != 'difficulty':
                            line_stack[i] += '\ndocument.getElementById("'+field+'selected").value="'+str(float(score)/len(curr_hit))+'";\n'
                        if score > 7*len_selection/5:
                            number_1 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >3*len_selection/5 and score <= 7*len_selection/5:
                            number_2 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >= 0*len_selection/5 and score <=3*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score < 0*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
        print number_1, number_2

        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_htmls_new(input_path = root + 'all_output/all_output.csv'):
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                if meta[15] in block_workers:
                    line_count+=1
                    continue
                elif meta[0] not in HITs:
                    HITs[meta[0]] = [meta]
                else:
                    HITs[meta[0]].append(meta)
            line_count+=1


    for HITId in HITs:

        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        event_id = this_hits[0][128]
        worker_id = this_hits[0][15]
        input_field = []
        output_field = []
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_type' in field:
                    name = value1
                    event_type = ''
                    for i in dict_name:
                        if dict_name[i] == name:
                            event_type = i
                    #print name
                    #name = ''.join(e for e in name if e.isalnum())
                    #name = name[:10]
                if 'num_image' in field:
                    print value1
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        #if event_id == '84_7273032@N04':
        #    print event_id
        root1 = html_root + '/' + event_type + '/' + event_id  + '/'
        if not os.path.exists(html_root + '/' + event_type):
            os.mkdir(html_root + '/' + event_type)
        if not os.path.exists(html_root + '/' + event_type+'/' + event_id ):
            os.mkdir(html_root + '/' + event_type+'/' + event_id )
        out_file = root1 +'present_groundtruth.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_predict.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)

        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]
                    len_selection = len(curr_hit)
                    if field == 'feedback':
                        pass
                    else:

                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if key == '' or key == 'NOTVALID':
                            continue
                        if field != 'difficulty':
                            line_stack[i] += '\ndocument.getElementById("'+field+'selected").value="'+str(float(score)/len(curr_hit))+'";\n'
                        if score > 7*len_selection/5:
                            number_1 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >3*len_selection/5 and score <= 7*len_selection/5:
                            number_2 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >= 0*len_selection/5 and score <=3*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score < 0*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
        print number_1, number_2

        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()

def create_result_htmls_present(input_path = root + 'all_output/all_output.csv'):
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                if meta[15] in block_workers:
                    line_count+=1
                    continue
                elif meta[0] not in HITs:
                    HITs[meta[0]] = [meta]
                else:
                    HITs[meta[0]].append(meta)
            line_count+=1


    for HITId in HITs:

        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        event_id = this_hits[0][28]
        worker_id = this_hits[0][15]
        input_field = []
        output_field = []
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_type' in field:
                    name = value1
                    event_type = ''
                    for i in dict_name:
                        if dict_name[i] == name:
                            event_type = i
                    #print name
                    #name = ''.join(e for e in name if e.isalnum())
                    #name = name[:10]
                if 'num_image' in field:
                    print value1
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        #if event_id == '84_7273032@N04':
        #    print event_id
        root1 = html_root + '/' + event_type + '/' + event_id  + '/'
        if not os.path.exists(html_root + '/' + event_type):
            os.mkdir(html_root + '/' + event_type)
        if not os.path.exists(html_root + '/' + event_type+'/' + event_id ):
            os.mkdir(html_root + '/' + event_type+'/' + event_id )
        out_file = root1 +'present_groundtruth.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_predict.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)

        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]
                    len_selection = len(curr_hit)
                    if field == 'feedback':
                        pass
                    else:

                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if key == '' or key == 'NOTVALID':
                            continue
                        if field != 'difficulty':
                            line_stack[i] += '\ndocument.getElementById("'+field+'selected").value="'+str(float(score)/len(curr_hit))+'";\n'
                        if score > 7*len_selection/5:
                            number_1 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >3*len_selection/5 and score <= 7*len_selection/5:
                            number_2 += 1
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >= 0*len_selection/5 and score <=3*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score < 0*len_selection/5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
        print number_1, number_2

        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_htmls_rearranged():
    input_path = root + 'all_output/all_output.csv'
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                if meta[15] in block_workers:
                    line_count+=1
                    continue
                elif meta[0] not in HITs:
                    HITs[meta[0]] = [meta]
                else:
                    HITs[meta[0]].append(meta)
            line_count+=1

    for HITId in HITs:
        num_images = 0
        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        input_field = []
        output_field = []
        event_id = ''
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_id' in field:
                    event_id = value1
                if 'event_type' in field:
                    name = value1
                    #print name
                    event_type = ''
                    for i in dict_name:
                        if dict_name[i] == name:
                            event_type = i
                if 'num_image' in field:
                    
                    print value1
                    if value1 == 'Input.num_image':
                        print this_hits
                    num_images = int(value1)
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        #if not os.path.exists(root + 'baseline_wedding_test/test_htmls/'):
        #    os.mkdir(root + 'baseline_wedding_test/test_htmls/')
        out_file = html_root + '/' + event_type +  '/' + event_id + '/rearranged_groundtruth_nobox.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)
        scores = {}
        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]

                    if field == 'feedback':
                        pass
                    else:
                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            #line_stack[i] += '\ndocument.getElementById("'+field+key+'").value="'+str(count_type[key])+'";\n'
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if field != 'difficulty' and int(field[5:]) <= num_images:
                            scores[int(field[5:])] = score


        for i in xrange(len(line_stack)):
            if 'var images = ' in line_stack[i]:
                image_list = line_stack[i].split('","')
                image_list[0] = image_list[0].split('"')[1]
                image_list[-1] = image_list[0].split('"')[0]
        image_list = image_list[:num_images]
        scores_ordered = [scores[i+1] for i in xrange(num_images)]
        sorted_image_list = sorted(zip(image_list, scores_ordered),key=lambda x: x[1], reverse=True)

        this_line = 'var images = ['
        for k in sorted_image_list:
            this_line += '"'+k[0]+'",'
        this_line = this_line[:-1] + '];'
        for i in xrange(len(line_stack)):
            if 'var images = ' in line_stack[i]:
                line_stack[i] = this_line
            if '$(document).ready(function()' in line_stack[i]:
                for k in xrange(len(sorted_image_list)):
                    line_stack[i] += '\ndocument.getElementById("image'+str(k+1)+'selected").value="'+str(float(sorted_image_list[k][1])/20 + 0.5)+'";\n'
                    score = sorted_image_list[k][1]
                    len_selection = len(output_field[0]) - 1
                    if score > 7*len_selection/5:
                            number_1 += 1
                            line_stack[i]+='$("label[name=\'image'+str(k+1)+ '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    if score >3*len_selection/5 and score <= 7*len_selection/5:
                            number_2 += 1
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    if score >= 0*len_selection/5 and score <=3*len_selection/5:
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    if score < 0*len_selection/5:
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'

        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_htmls_rearranged_forsupp():
    input_path = root + 'all_output/all_output.csv'
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                if meta[15] in block_workers:
                    line_count+=1
                    continue
                elif meta[0] not in HITs:
                    HITs[meta[0]] = [meta]
                else:
                    HITs[meta[0]].append(meta)
            line_count+=1

    for HITId in HITs:
        num_images = 0
        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        input_field = []
        output_field = []
        event_id = ''
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_id' in field:
                    event_id = value1
                if 'event_type' in field:
                    name = value1
                    #print name
                    event_type = ''
                    for i in dict_name:
                        if dict_name[i] == name:
                            event_type = i
                if 'num_image' in field:

                    print value1
                    if value1 == 'Input.num_image':
                        print this_hits
                    num_images = int(value1)
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        #if not os.path.exists(root + 'baseline_wedding_test/test_htmls/'):
        #    os.mkdir(root + 'baseline_wedding_test/test_htmls/')
        out_file = html_root + '/' + event_type +  '/' + event_id + '/rearranged_groundtruth_nobox.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange_supp.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)
        scores = {}
        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback"' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'
                        for value in curr_hit[1:]:
                            line_stack[i] += value +'**'
                        line_stack[i] += '</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for curr_hit in output_field:
                    field = curr_hit[0]
                    curr_hit = curr_hit[1:]

                    if field == 'feedback':
                        pass
                    else:
                        score = 0
                        count_type = Counter(curr_hit)
                        #print count_type
                        for key in count_type:
                            if key == '':
                                continue
                            #line_stack[i] += '\ndocument.getElementById("'+field+key+'").value="'+str(count_type[key])+'";\n'
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
                        if field != 'difficulty' and int(field[5:]) <= num_images:
                            scores[int(field[5:])] = score


        for i in xrange(len(line_stack)):
            if 'var images = ' in line_stack[i]:
                image_list = line_stack[i].split('","')
                image_list[0] = image_list[0].split('"')[1]
                image_list[-1] = image_list[0].split('"')[0]
        image_list = image_list[:num_images]
        scores_ordered = [scores[i+1] for i in xrange(num_images)]
        sorted_image_list = sorted(zip(image_list, scores_ordered),key=lambda x: x[1], reverse=True)

        this_line = 'var images = ['
        for k in sorted_image_list:
            this_line += '"'+k[0]+'",'
        this_line = this_line[:-1] + '];'
        for i in xrange(len(line_stack)):
            if 'var images = ' in line_stack[i]:
                line_stack[i] = this_line
            if '$(document).ready(function()' in line_stack[i]:
                for k in xrange(len(sorted_image_list)):
                    line_stack[i] += '\ndocument.getElementById("image'+str(k+1)+'selected").value="'+str(float(sorted_image_list[k][1])/20 + 0.5)+'";\n'
                    score = sorted_image_list[k][1]
                    len_selection = len(output_field[0]) - 1
                    # if score > 7*len_selection/5:
                    #         number_1 += 1
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1)+ '\']").addClass("highlight");\n'
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    # if score >3*len_selection/5 and score <= 7*len_selection/5:
                    #         number_2 += 1
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("selected_sw");\n'
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    # if score >= 0*len_selection/5 and score <=3*len_selection/5:
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("neutral");\n'
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    # if score < 0*len_selection/5:
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").addClass("irrelevant");\n'
                    #         line_stack[i]+='$("label[name=\'image'+str(k+1) + '\']").removeClass("not_selected");\n'
                    #



        f = open(out_file, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def show_test_predict_img(model_path, model_name, event_type, groundtruth = False):
    f = open(root + model_path + model_name + '.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()

    if groundtruth:
        thresholds = [0, 1, 1.5]
    else:
        f = open(root + model_path + model_name[:-5] + '.cPickle','r')
        test_prediction = cPickle.load(f)
        f.close()
        test_prediction = [i[0][dict_name2[event_type]- 1] for i in test_prediction]
        try:
            test_prediction_sorted = sorted(test_prediction)
        except:
            print root + model_path + model_name[:-5]
            print test_prediction
        len_ = len(test_prediction_sorted)
        thresholds = [test_prediction_sorted[int(0.2*len_)], test_prediction_sorted[int(0.6*len_)],
                      test_prediction_sorted[int(0.8*len_)]]
                      
    root1 = html_root_test + '/' + event_type + '/'
    if not os.path.exists(root1):
        os.mkdir(root1)

    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        if groundtruth:
            for img in this_event:
                img_ids.append(img[1])
                img_urls.append(img[0])
                img_scores.append(img[2])
        else:
            for img in this_event:
                img_ids.append(img[0])
                img_urls.append(img[1])
                img_scores.append(img[2])

        html_path = root1 + event_id + '/' + model_name +'_predict' + '.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_predict.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'

                    if score > thresholds[2]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score > thresholds[1]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >= thresholds[0]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    else:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def show_test_predict_img_rearranged(model_path, model_name, event_type, groundtruth=False):
    f = open(root + model_path + model_name + '.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()

    if groundtruth:
        thresholds = [0, 1, 1.5]
    else:
        f = open(root + model_path + model_name[:-5] + '.cPickle','r')
        test_prediction = cPickle.load(f)
        f.close()
        try:
            test_prediction = [i[dict_name2[event_type]- 1] for i in test_prediction]
        except:
            pass
        # try:
        test_prediction_sorted = sorted(test_prediction)
        # except:
        #     print root + model_path + model_name[:-5]
            # print test_prediction
        len_ = len(test_prediction_sorted)
        thresholds = [test_prediction_sorted[int(0.2*len_)], test_prediction_sorted[int(0.6*len_)],
                      test_prediction_sorted[int(0.8*len_)]]



    root1 = html_root_test + '/' + event_type + '/'
    if not os.path.exists(root1):
        os.mkdir(root1)
    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        if groundtruth:
            for img in this_event:
                img_ids.append(img[1])
                img_urls.append(img[0])
                img_scores.append(img[2])
        else:
            for img in this_event:
                img_ids.append(img[0])
                img_urls.append(img[1])
                img_scores.append(img[2])
        temp = zip(img_ids, img_urls, img_scores)
        temp = sorted(temp, key=lambda x: x[2], reverse=True)
        img_ids = [i[0] for i in temp]
        img_urls = [i[1] for i in temp]
        img_scores = [i[2] for i in temp]
        html_path = root1 + event_id + '/' + model_name +'_predict_rearranged' + '.html'
        print html_path
        if not os.path.exists(root1 + event_id):
            os.mkdir(root1 + event_id)
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]#[0][0]
                    #score = 1/(1 + np.exp(-score))
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'
                    if score > thresholds[2]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score > thresholds[1]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif  score >= thresholds[0]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    else:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def show_test_predict_img_topk(model_path, model_name, event_type, k_percent, groundtruth = False):
    f = open(root + model_path + model_name + '.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()

    root1 = html_root_test + '/' + event_type + '/'
    if not os.path.exists(root1):
        os.mkdir(root1)
    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        for img in this_event:
            img_ids.append(img[0])
            img_urls.append(img[1])
            img_scores.append(img[2])
        temp = zip(img_ids, img_urls, img_scores)
        seq = [i[2] for i in temp]
        argsort_temp = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
        argsort_temp = argsort_temp[::-1]
        length_select = max(2, k_percent*len(img_ids)/100)
        temp = [temp[i] for i in argsort_temp[:length_select]]
        if groundtruth:
            img_ids = [i[1] for i in temp]
            img_urls = [i[0] for i in temp]
            img_scores = [i[2] for i in temp]
        else:
            img_ids = [i[0] for i in temp]
            img_urls = [i[1] for i in temp]
            img_scores = [i[2] for i in temp]
        score_threshold = img_scores[length_select/2]

        if not os.path.exists(root1+event_id):
            os.mkdir(root1+event_id)
        html_path = root1 + event_id + '/' + model_name +'_predict_rearranged_top'+str(k_percent) + '.html'
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]#[0][0]
                    #score = 1/(1 + np.exp(-score))
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'
                    if score > score_threshold:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    else:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    #elif  temp > 0.2:
                    #        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                    #        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    #else:
                    #        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                    #        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()

def create_predict_dict_from_cpickle_multevent(validation_name, event_name, mat_name, event_index = 17, multi_event = True):
    path = root+'baseline_all_noblock/' + event_name+ '/'+validation_name+'_image_ids.cPickle'

    f = open(path, 'r')
    all_event_ids = cPickle.load(f)
    f.close()
    f = open(mat_name + '.cPickle', 'r')
    predict_score = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_all_noblock/' + event_name+ '/'+validation_name+'_ulr_dict.cPickle', 'r')
    test_url_dict = cPickle.load(f)
    f.close()

    prediction_dict = {}
    for (name_, score) in zip(all_event_ids, predict_score):
        event_name = name_.split('/')[0]
        if event_name in prediction_dict:
            if multi_event:
                prediction_dict[event_name] += [[name_, test_url_dict[name_], score[0][event_index-1]]]
            else:
                prediction_dict[event_name] += [[name_, test_url_dict[name_], score]]
        else:
            if multi_event:
                prediction_dict[event_name] = [[name_, test_url_dict[name_], score[0][event_index-1]]]
            else:
                prediction_dict[event_name] = [[name_, test_url_dict[name_], score]]


    f = open(mat_name + '_dict.cPickle','wb')
    cPickle.dump(prediction_dict, f)
    f.close()

def show_test_predict_img_rearranged_facechange(validation_name, model_path, model_name, event_type, groundtruth=False):
    f = open(root + model_path + model_name + '.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()
    create_predict_dict_from_cpickle_multevent(validation_name, event_type, root + 'face_heatmap/features/'+event_type+'_training_sigmoidcropped_importance_allevent_iter_100000', dict_name2[event_type], multi_event=False)
    create_predict_dict_from_cpickle_multevent(validation_name, event_type, root + 'face_heatmap/features/'+event_type+'_'+validation_name+'_sigmoidcropped_importance_allevent_iter_100000', dict_name2[event_type], multi_event=False)
    f = open(root + 'face_heatmap/features/'+event_type+'_'+validation_name+'_sigmoidcropped_importance_allevent_iter_100000_dict.cPickle')
    face_result = cPickle.load(f)
    f.close()

    if groundtruth:
        thresholds = [0, 1, 1.5]
    else:
        f = open(root + model_path + model_name[:-5] + '.cPickle','r')
        test_prediction = cPickle.load(f)
        f.close()
        try:
            test_prediction = [i[0][dict_name2[event_type]- 1] for i in test_prediction]
        except:
            pass
        try:
            test_prediction_sorted = sorted(test_prediction)
        except:
            print root + model_path + model_name[:-5]
            print test_prediction
        len_ = len(test_prediction_sorted)
        thresholds = [test_prediction_sorted[int(0.2*len_)], test_prediction_sorted[int(0.6*len_)],
                      test_prediction_sorted[int(0.8*len_)]]

    f = open(root + 'face_heatmap/features/'+event_type+'_training_sigmoidcropped_importance_allevent_iter_100000_dict.cPickle','r')
    face_training = cPickle.load(f)
    f.close()
    f = open(root + 'face_heatmap/features/'+event_type+'_training_sigmoidcropped_importance_allevent_iter_100000.cPickle','r')
    face_training_score = cPickle.load(f)
    f.close()
    try:
        face_training_score = [i[0][dict_name2[event_type]-1] for i in face_training_score]
    except:
        pass


    sorted_face = np.sort(face_training_score)
    count_last = 0
    count_this = 0
    no_face_value = 0
    for i in xrange(1,len(sorted_face)):
        count_this += 1
        if sorted_face[i] != sorted_face[i-1]:
            if count_this > count_last:
                count_last = count_this
                no_face_value = sorted_face[i-1]
            count_this = 0
    print face_training_score
    print no_face_value


    root1 = html_root_test + '/' + event_type + '/'
    if not os.path.exists(root1):
        os.mkdir(root1)
    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        img_face_scores = []
        face_this = face_result[event_id]
        if groundtruth:
            for (img, img_face) in zip(this_event,face_this):
                img_ids.append(img[1])
                img_urls.append(img[0])
                img_scores.append(img[2])
                img_face_scores.append(img_face[2])
        else:
            for (img, img_face) in zip(this_event,face_this):
                img_ids.append(img[0])
                img_urls.append(img[1])
                img_scores.append(img[2])
                img_face_scores.append(img_face[2][0][dict_name2[event_type]-1])
        temp = zip(img_ids, img_urls, img_scores, img_face_scores)
        temp = sorted(temp, key=lambda x: x[3], reverse=True)
        img_ids = [i[0] for i in temp]
        img_urls = [i[1] for i in temp]
        img_scores = [i[2] for i in temp]
        try:
            img_face_scores = [i[3][0][dict_name2[event_type]-1] for i in temp]
        except:
            img_face_scores = [i[3] for i in temp]




        html_path = root1 + event_id + '/' + model_name +'_predict_rearranged' + '.html'
        if not os.path.exists(root1+event_id):
            os.mkdir(root1+event_id)
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]#[0][0]
                    #score = 1/(1 + np.exp(-score))

                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'
                    if img_face_scores[i] > no_face_value:
                        print img_face_scores
                        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    else:
                        line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("not_selected");\n'
                    '''
                    if score > thresholds[2]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score > thresholds[1]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif  score >= thresholds[0]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    else:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    '''

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()

def show_test_predict_img_topk_percent_simple(event_type, k_percent):
    print 'HERE!'
    model_paths = ['baseline_all_0509/'+event_type + '/', 'CNN_all_event_1009/features/', 'baseline_all_noblock/'+event_type+'/']
    model_names = ['vgg_test_result_v2', event_type + '_test_combined_10_fromnoevent_dict',  'test_random_dict']
    ground_truths = [True, False, False]
    root1 = html_root_test + '/' + event_type + '/'
    f = open(root + 'baseline_all_noblock/' + event_type + '/test_event_id.cPickle')
    event_ids = cPickle.load(f)
    f.close()
    test_prediction_events = []
    for i in xrange(len(model_paths)):
        f = open(root + model_paths[i] + model_names[i] + '.cPickle','r')
        test_prediction_events.append(cPickle.load(f))
        f.close()

    for event_id in event_ids:
            html_path = root1 + event_id + '/predict_rearranged_simple_top'+str(k_percent) + '.html'
            f_out = open(html_path,'w')
            f_out.write('<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js">'
                        '</script><script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>'
                        '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />')

            f_out.write('<section id="EventCuration" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">')
            f_out.write('<center>')
            #f_out.write('<div class="panel panel-primary"><div class="panel-heading"><strong>Event type:'+event_type+' Event id:' + event_id+'</strong> </div></div>')

            for iii in xrange(len(model_paths)):
                ground_truth = ground_truths[iii]

                this_event = test_prediction_events[iii][event_id]
                img_ids = []
                img_urls = []
                img_scores = []
                for img in this_event:
                    img_ids.append(img[0])
                    img_urls.append(img[1])
                    img_scores.append(img[2])
                temp = zip(img_ids, img_urls, img_scores)
                seq = [i[2] for i in temp]
                argsort_temp = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
                argsort_temp = argsort_temp[::-1]
                length_select = max(2, k_percent*len(img_ids)/100)
                argsort_temp = argsort_temp[:length_select]
                argsort_temp.sort()
                temp = [temp[i] for i in argsort_temp]

                if ground_truth:
                    img_ids = [i[1] for i in temp]
                    img_urls = [i[0] for i in temp]
                    img_scores = [i[2] for i in temp]
                else:
                    img_ids = [i[0] for i in temp]
                    img_urls = [i[1] for i in temp]
                    img_scores = [i[2] for i in temp]
                score_threshold = np.median(np.array(img_scores))
                #f_out.write('<center>')
                #f_out.write('<p>'+model_names[iii]+'</p>\n')
                f_out.write('<table border="1" style="width:500">\n')
                img_count = 0
                for i in xrange(length_select):
                    img_count += 1
                    if img_count % 10 == 0:
                        f_out.write('\t</tr><tr>\n')
                    f_out.write('\t\t<td align=\"center\" valign=\"center\">\n')
                    if img_scores[i] > score_threshold:
                        # f_out.write('\t\t\t<img class=\"selected\" src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                        f_out.write('\t\t\t<img src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                    else:
                        # f_out.write('\t\t\t<img class=\"selected_sw\" src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                        f_out.write('\t\t\t<img  src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                f_out.write('</tr></table></section>\n')
            f_out.write('<style type=\"text/css\">img {width:240px;height:240px;}'
                        '.selected_sw {border: 3px solid #9900FF !important;opacity:0.7 !important;cursor: pointer;}'
                        '.selected {border: 3px solid #FF0000 !important;opacity:1.0 !important;cursor: pointer;}'
                        'table, th, td { border: 3px solid white; font-size: 12px;}'
                        '</style>\n')
            f_out.close()

    print 'THERE!!'
def show_test_predict_img_topk_simple(event_type, k):
    print 'HERE!'
    model_paths = ['baseline_all_0509/'+event_type + '/', 'CNN_all_event_1009/features/', 'baseline_all_noblock/'+event_type+'/']
    model_names = ['vgg_test_result_v2', event_type + '_test_combined_10_fromnoevent_dict',  'test_random_dict']
    ground_truths = [True, False, False]
    root1 = html_root_test + '/' + event_type + '/'
    f = open(root + 'baseline_all_noblock/' + event_type + '/test_event_id.cPickle')
    event_ids = cPickle.load(f)
    f.close()
    test_prediction_events = []
    for i in xrange(len(model_paths)):
        f = open(root + model_paths[i] + model_names[i] + '.cPickle','r')
        test_prediction_events.append(cPickle.load(f))
        f.close()

    for event_id in event_ids:
            html_path = root1 + event_id + '/predict_rearranged_simple_top_number_'+str(k) + '.html'
            f_out = open(html_path,'w')
            f_out.write('<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js">'
                        '</script><script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>'
                        '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />')

            f_out.write('<section id="EventCuration" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">')
            f_out.write('<center>')
            f_out.write('<div class="panel panel-primary"><div class="panel-heading"><strong>Event type:'+event_type+' Event id:' + event_id+'</strong> </div></div>')

            for iii in xrange(len(model_paths)):
                ground_truth = ground_truths[iii]

                this_event = test_prediction_events[iii][event_id]
                img_ids = []
                img_urls = []
                img_scores = []
                for img in this_event:
                    img_ids.append(img[0])
                    img_urls.append(img[1])
                    img_scores.append(img[2])
                temp = zip(img_ids, img_urls, img_scores)
                seq = [i[2] for i in temp]
                argsort_temp = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
                argsort_temp = argsort_temp[::-1]
                length_select = max(2, k)
                argsort_temp = argsort_temp[:length_select]
                argsort_temp.sort()
                temp = [temp[i] for i in argsort_temp]

                if ground_truth:
                    img_ids = [i[1] for i in temp]
                    img_urls = [i[0] for i in temp]
                    img_scores = [i[2] for i in temp]
                else:
                    img_ids = [i[0] for i in temp]
                    img_urls = [i[1] for i in temp]
                    img_scores = [i[2] for i in temp]
                score_threshold = np.median(np.array(img_scores))
                f_out.write('<center>')
                f_out.write('<p>'+model_names[iii]+'</p>\n')
                f_out.write('<table border="1" style="width:500">\n')
                img_count = 0
                for i in xrange(length_select):
                    img_count += 1
                    if img_count % 10 == 0:
                        f_out.write('\t</tr><tr>\n')
                    f_out.write('\t\t<td align=\"center\" valign=\"center\">\n')
                    if img_scores[i] > score_threshold:
                        # f_out.write('\t\t\t<img class=\"selected\" src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                        f_out.write('\t\t\t<img src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                    else:
                        # f_out.write('\t\t\t<img class=\"selected_sw\" src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                        f_out.write('\t\t\t<img  src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                f_out.write('</tr></table></section>\n')
            f_out.write('<style type=\"text/css\">img {width:240px;height:auto;}'
                        '.selected_sw {border: 3px solid #9900FF !important;opacity:0.7 !important;cursor: pointer;}'
                        '.selected {border: 3px solid #FF0000 !important;opacity:1.0 !important;cursor: pointer;}'
                        'table, th, td { border: 3px solid white; font-size: 12px;}'
                        '</style>\n')
            f_out.close()

    print 'THERE!!'

def show_test_predict_img_topk_percent_forsupp(event_type, k_percent):
    event_to_present = ['7_47519867@N07','3_12515159@N07','155_27998473@N02', '10_42445822@N00',
                        '8_84905000@N00', '8_66724483@N00', '8_43257106@N07', '5_25093253@N05',
                        '1_42943151@N06', '49_30952578@N00', '13_44564547@N00','5_12482312@N00',
                        '3_82373898@N00', '1_82802179@N00', '1_60509459@N00', '1_51164183@N07',
                        '1_7614607@N05', '0_67618625@N00', '97_32323502@N00', '2_96212491@N00',
                        '95_66478195@N00', '51_66478195@N00', '50_12734746@N00', '18_97864553@N00',
                        '0_13401513@N08', '35_59755673@N04', '0_51517883@N00', '38_27988337@N00',
                        '0_70054695@N00', '1_65675500@N00', '32_43661283@N00', '11_99245765@N00',
                        '40_44082489@N00', '20_9674366@N08', '0_70644035@N00', '0_52685047@N00',
                        '124_8798099@N02', '85_43162195@N00', '41_22539273@N00', '7_69966484@N00',
                        '17_16036153@N04', '14_97402086@N00', '7_7349747@N02', '0_9137715@N05',
                        '1_30764740@N07', '0_97681995@N00', '0_57185608@N00', '0_40958113@N00',
                        '144_95413346@N00', '5_25369032@N00', '4_49561754@N00', '2_10351901@N00',
                        '2_30275727@N02', '0_12356498@N06', '1_56922885@N00', '14_53746192@N00',
                        '78_14451269@N00', '1_66263830@N00', '3_9025932@N08', '2_30745127@N07',
                        '24_97863854@N00', '4_76228749@N00', '1_28004076@N04', '0_28004076@N04',
                        '0_17491435@N00', '2_33237963@N00']


    model_paths = ['baseline_all_0509/'+event_type + '/', 'CNN_all_event_1009/features/', 'baseline_all_noblock/'+event_type+'/']
    model_names = ['vgg_test_result_v2', event_type + '_test_combined_10_fromnoevent_dict',  'test_random_dict']
    ground_truths = [True, False, False]
    root1 = html_root_test + '/' + event_type + '/'
    f = open(root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle')
    event_ids = cPickle.load(f)
    # print event_ids
    f.close()
    test_prediction_events = []
    for i in xrange(len(model_paths)):
        f = open(root + model_paths[i] + model_names[i] + '.cPickle','r')
        test_prediction_events.append(cPickle.load(f))
        f.close()

    for event_id in event_ids:
            if event_id not in event_to_present:
                continue
            print event_id
            len_img = len(test_prediction_events[0][event_id])
            temp = 20
            length_select = max(2, 20*len_img/100)
            if length_select > 10:
                length_select = max(2, 15*len_img/100)
                temp = 15
                if length_select > 10:
                    length_select = max(2, 10*len_img/100)
                    temp = 10

            html_path = root + 'supplementary/results/'+event_id + '_'+ str(length_select) +'_'+str(temp)+ '.html'
            f_out = open(html_path,'w')
            f_out.write('<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js">'
                        '</script><script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>'
                        '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />')

            f_out.write('<section id="EventCuration" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">')
            f_out.write('<center>')
            #f_out.write('<div class="panel panel-primary"><div class="panel-heading"><strong>Event type:'+event_type+' Event id:' + event_id+'</strong> </div></div>')

            for iii in xrange(len(model_paths)):
                ground_truth = ground_truths[iii]

                this_event = test_prediction_events[iii][event_id]
                img_ids = []
                img_urls = []
                img_scores = []
                for img in this_event:
                    img_ids.append(img[0])
                    img_urls.append(img[1])
                    img_scores.append(img[2])
                temp = zip(img_ids, img_urls, img_scores)
                seq = [i[2] for i in temp]
                argsort_temp = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
                argsort_temp = argsort_temp[::-1]


                argsort_temp = argsort_temp[:length_select]
                argsort_temp.sort()
                temp = [temp[i] for i in argsort_temp]

                if ground_truth:
                    img_ids = [i[1] for i in temp]
                    img_urls = [i[0] for i in temp]
                    img_scores = [i[2] for i in temp]
                else:
                    img_ids = [i[0] for i in temp]
                    img_urls = [i[1] for i in temp]
                    img_scores = [i[2] for i in temp]
                score_threshold = np.median(np.array(img_scores))
                #f_out.write('<center>')
                #f_out.write('<p>'+model_names[iii]+'</p>\n')
                f_out.write('<table border="1" style="width:500">\n')
                img_count = 0
                for i in xrange(length_select):
                    img_count += 1
                    # if img_count % 10 == 0:
                    #     f_out.write('\t</tr><tr>\n')
                    f_out.write('\t\t<td align=\"center\" valign=\"center\">\n')
                    if img_scores[i] > score_threshold:
                        # f_out.write('\t\t\t<img class=\"selected\" src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                        f_out.write('\t\t\t<img src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                    else:
                        # f_out.write('\t\t\t<img class=\"selected_sw\" src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                        f_out.write('\t\t\t<img  src=\"'+img_urls[i]+'\" alt=Loading... />\n')
                f_out.write('</tr></table></section>\n')
            f_out.write('<style type=\"text/css\">img {width:240px;height:240px;}'
                        '.selected_sw {border: 3px solid #9900FF !important;opacity:0.7 !important;cursor: pointer;}'
                        '.selected {border: 3px solid #FF0000 !important;opacity:1.0 !important;cursor: pointer;}'
                        'table, th, td { border: 3px solid white; font-size: 12px;}'
                        '</style>\n')
            f_out.close()

def pic_test_predict_img_topk_percent_forsupp(event_type):
    # event_to_present = ['7_47519867@N07','3_12515159@N07','155_27998473@N02', '10_42445822@N00',
    #                     '8_84905000@N00', '8_66724483@N00', '8_43257106@N07', '5_25093253@N05',
    #                     '1_42943151@N06', '49_30952578@N00', '13_44564547@N00','5_12482312@N00',
    #                     '3_82373898@N00', '1_82802179@N00', '1_60509459@N00', '1_51164183@N07',
    #                     '1_7614607@N05', '0_67618625@N00', '97_32323502@N00', '2_96212491@N00',
    #                     '95_66478195@N00', '51_66478195@N00', '50_12734746@N00', '18_97864553@N00',
    #                     '0_13401513@N08', '35_59755673@N04', '0_51517883@N00', '38_27988337@N00',
    #                     '0_70054695@N00', '1_65675500@N00', '32_43661283@N00', '11_99245765@N00',
    #                     '40_44082489@N00', '20_9674366@N08', '0_70644035@N00', '0_52685047@N00',
    #                     '124_8798099@N02', '85_43162195@N00', '41_22539273@N00', '7_69966484@N00',
    #                     '17_16036153@N04', '14_97402086@N00', '7_7349747@N02', '0_9137715@N05',
    #                     '1_30764740@N07', '0_97681995@N00', '0_57185608@N00', '0_40958113@N00',
    #                     '144_95413346@N00', '5_25369032@N00', '4_49561754@N00', '2_10351901@N00',
    #                     '2_30275727@N02', '0_12356498@N06', '1_56922885@N00', '14_53746192@N00',
    #                     '78_14451269@N00', '1_66263830@N00', '3_9025932@N08', '2_30745127@N07',
    #                     '24_97863854@N00', '4_76228749@N00', '1_28004076@N04', '0_28004076@N04',
    #                     '0_17491435@N00', '2_33237963@N00','26_21186435@N00']
    event_to_present = ['6_93995264@N00','0_31998658@N06']

    model_paths = ['baseline_all_0509/'+event_type + '/', 'CNN_all_event_1009/features/', 'baseline_all_noblock/'+event_type+'/']
    model_names = ['vgg_test_result_v2', event_type + '_test_combined_10_fromnoevent_dict',  'test_random_dict']
    ground_truths = [True, False, False]
    root1 = html_root_test + '/' + event_type + '/'
    f = open(root + 'baseline_all_0509/' + event_type + '/test_event_id.cPickle')
    event_ids = cPickle.load(f)
    # print event_ids
    f.close()
    test_prediction_events = []
    for i in xrange(len(model_paths)):
        f = open(root + model_paths[i] + model_names[i] + '.cPickle','r')
        test_prediction_events.append(cPickle.load(f))
        f.close()
    size = 2
    for event_id in event_ids:
            if event_id not in event_to_present:
                continue
            print event_id
            len_img = len(test_prediction_events[0][event_id])
            temp = 20
            length_select = max(2, 20*len_img/100)
            if length_select > 10:
                length_select = max(2, 15*len_img/100)
                temp = 15
                if length_select > 10:
                    length_select = max(2, 10*len_img/100)
                    temp = 10
            #length_select = 5
            out_path = root + 'supplementary/results/'+event_id + '_'+ str(length_select)+ '_'+str(temp)+'.jpg'

            im_canvas = np.ones((size*(240*3+50),length_select*250*size-10*size,3))*255

            for iii in xrange(len(model_paths)):
                ground_truth = ground_truths[iii]

                this_event = test_prediction_events[iii][event_id]
                img_ids = []
                img_urls = []
                img_scores = []
                for img in this_event:
                    img_ids.append(img[0])
                    img_urls.append(img[1])
                    img_scores.append(img[2])
                temp = zip(img_ids, img_urls, img_scores)
                seq = [i[2] for i in temp]
                argsort_temp = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
                argsort_temp = argsort_temp[::-1]

                argsort_temp = argsort_temp[:length_select]
                argsort_temp.sort()
                temp = [temp[i] for i in argsort_temp]

                if ground_truth:
                    img_ids = [i[1] for i in temp]
                    img_urls = [i[0] for i in temp]
                    img_scores = [i[2] for i in temp]
                else:
                    img_ids = [i[0] for i in temp]
                    img_urls = [i[1] for i in temp]
                    img_scores = [i[2] for i in temp]
                for i in xrange(length_select):
                    img_path = root + 'curation_images/' + event_type + '/' + event_id.split('_')[1] + '/' + img_ids[i].split('/')[1] +'.jpg'
                    # print img_path
                    img = Image.open(img_path)
                    img = img.resize((240*size,240*size))
                    start_width = i*250*size
                    start_height = iii * (240+25)*size
                    im_canvas[start_height:start_height+240*size,start_width:start_width+240*size, :] = np.array(img)

            im_canvas = np.uint8(im_canvas)
            im_canvas_img = Image.fromarray(im_canvas)
            im_canvas_img.save(out_path, "JPEG")

def show_test_predict_img_rearranged_recognition_corrected(model_path, model_name, event_model_name, event_type, groundtruth=False):
    f = open(root + model_path + model_name + '.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()

    f = open(root + model_path  + event_model_name + '.cPickle','r')
    event_prediction = cPickle.load(f)
    f.close()

    f = open(root + model_path + 'test_predict_event_recognition_expand_balanced_3_corrected_iter_100000_dict.cPickle')
    img_recognition = cPickle.load(f)
    f.close()


    if groundtruth:
        thresholds = [0, 1, 1.5]
    else:
        f = open(root + model_path + model_name[:-5] + '.cPickle','r')
        test_prediction = cPickle.load(f)
        f.close()
        try:
            test_prediction = [i[dict_name2[event_type]- 1] for i in test_prediction]
        except:
            pass
        try:
            test_prediction_sorted = sorted(test_prediction)
        except:
            print root + model_path + model_name[:-5]
            print test_prediction
        len_ = len(test_prediction_sorted)
        thresholds = [test_prediction_sorted[int(0.2*len_)], test_prediction_sorted[int(0.6*len_)],
                      test_prediction_sorted[int(0.8*len_)]]



    for event_id in test_prediction_event:
        print event_id
        this_event = test_prediction_event[event_id]
        this_event_recognition = img_recognition[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        img_recognitions = []
        if groundtruth:
            for img, img_recognition_ in zip(this_event, this_event_recognition):
                if img[0] != img_recognition_[0]:
                    print 'ERROR!'
                    return
                img_ids.append(img[1])
                img_urls.append(img[0])
                img_scores.append(img[2])
                img_recognitions.append(img_recognition_[2])
        else:
            for img, img_recognition_ in zip(this_event, this_event_recognition):
                if img[0] != img_recognition_[0]:
                    print 'ERROR!'
                    return
                img_ids.append(img[0])
                img_urls.append(img[1])
                img_scores.append(img[2])
                img_recognitions.append(img_recognition_[2])
        temp = zip(img_ids, img_urls, img_scores, img_recognitions)
        temp = sorted(temp, key=lambda x: x[2], reverse=True)
        img_ids = [i[0] for i in temp]
        img_urls = [i[1] for i in temp]
        img_scores = [i[2] for i in temp]
        img_recognitions = [i[3] for i in temp]

        dict_name_reverse = dict([(dict_name2[event_name], event_name) for event_name in dict_name2])
        if event_id in correct_list:
            root1 = html_root_test + '/' + correct_list[event_id] + '/'
        else:
            root1 = html_root_test + '/' + event_type + '/'
        if not os.path.exists(root1):
            os.mkdir(root1)

        html_path = root1 + dict_name_reverse[np.argmax(event_prediction[event_id]) + 1] + '_' + event_id + '_' + model_name +'_predict_rearranged' + '.html'
        print html_path
        # if not os.path.exists(root1 + event_id):
        #     os.mkdir(root1 + event_id)
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_scores)):
                    score = img_scores[i]#[0][0]
                    score = format(score, '.2f')
                    #score = 1/(1 + np.exp(-score))
                    temp = np.argsort(img_recognitions[i])
                    recognition_score = dict_name_reverse[temp[-1] + 1] + '_' + dict_name_reverse[temp[-2] + 1]
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+',' + recognition_score+'";\n'
                    if score > thresholds[2]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score > thresholds[1]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif  score >= thresholds[0]:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    else:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()
def show_test_predict_img_rearrange_allevent():
    print 'HERE!'
    model_paths = ['CNN_all_event_corrected_multi/features_validation/']
    model_names = ['THRESHOLDM_POLY2_LSTM_iter0_importance_cross_validation_combine_best_dict']
    root1 = html_root + '/'
    event_ids = []
    for event_type in dict_name2:
        f = open(root + 'baseline_all_noblock/' + event_type + '/test_event_id.cPickle')
        event_ids.extend(cPickle.load(f))
        f.close()
    test_prediction_events = []
    for i in xrange(len(model_paths)):
        f = open(root + model_paths[i] + model_names[i] + '.pkl','r')
        test_prediction_events.append(cPickle.load(f))
        f.close()
    for model_path, model_name, test_prediction_event in zip(model_paths, model_names, test_prediction_events):
        f = open(root + model_path + model_name[:-5] + '.pkl','r')
        test_prediction = cPickle.load(f)
        f.close()
        try:
            test_prediction_sorted = sorted(test_prediction)
        except:
            print root + model_path + model_name[:-5]
            print test_prediction
        len_ = len(test_prediction_sorted)
        thresholds = [test_prediction_sorted[int(0.2*len_)], test_prediction_sorted[int(0.6*len_)],
                      test_prediction_sorted[int(0.8*len_)]]
        if not os.path.exists(root1):
            os.mkdir(root1)
        for event_id in test_prediction_event:
            this_event = test_prediction_event[event_id]
            img_ids = []
            img_urls = []
            img_scores = []
            for img in this_event:
                    img_ids.append(img[0])
                    img_urls.append(img[1])
                    img_scores.append(img[2])
            temp = zip(img_ids, img_urls, img_scores)
            temp = sorted(temp, key=lambda x: x[2], reverse=True)
            img_ids = [i[0] for i in temp]
            img_urls = [i[1] for i in temp]
            img_scores = [i[2] for i in temp]
            html_path = root1 + event_id + '/' + model_name +'_predict_rearranged' + '.html'
            print html_path
            if not os.path.exists(root1 + event_id):
                os.mkdir(root1 + event_id)
            in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
            line_stack = []
            with open(in_file, 'r') as data:
                for line in data:
                    line_stack.append(line)

            for kk in xrange(len(line_stack)):
                if '${num_image}' in line_stack[kk]:
                    line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
                if 'var images = ["${image1}"' in line_stack[kk]:
                    new_line = 'var images = ['
                    for img_url in img_urls:
                        new_line += '"'+img_url+'",'
                    new_line = new_line[:-1]
                    new_line += '];\n'
                    line_stack[kk] = new_line

                if '$(document).ready(function()' in line_stack[kk]:
                    for i in xrange(len(img_scores)):
                        score = img_scores[i]#[0][0]
                        print score, thresholds
                        #score = 1/(1 + np.exp(-score))
                        line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+str(score)+'";\n'
                        # if score > thresholds[2]:
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                        # elif score > thresholds[1]:
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                        # elif  score >= thresholds[0]:
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                        # else:
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                        #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

            f = open(html_path, 'wb')
            for line in line_stack:
                f.write(line)
            f.close()

def show_test_predict_img_recognition_corrected(model_path, event_model_name, event_type, groundtruth=False):
    f = open(root + model_path[:-1] + event_model_name + '.pkl', 'r')
    event_prediction = cPickle.load(f)
    f.close()

    f = open(root + model_path + 'test_predict_event_recognition_expand_balanced_3_corrected_iter_100000_dict.cPickle')
    temp = cPickle.load(f)
    f.close()

    img_recognition = defaultdict(list)
    with open(root + 'lstm/data/test_lstm_prediction_img_dict.pkl') as f:
        all_recognition = cPickle.load(f)
    for event_id in temp:
        for img in temp[event_id]:
            img_id = img[0]
            img_recognition[event_id].append([img_id, img[1], all_recognition[img_id]])

    for event_id in img_recognition:
        print event_id
        this_event = img_recognition[event_id]
        img_ids = []
        img_urls = []
        img_recognitions = []
        if groundtruth:
            for img in this_event:
                img_ids.append(img[1])
                img_urls.append(img[0])
                img_recognitions.append(img[2])
        else:
            for img in this_event:
                img_ids.append(img[0])
                img_urls.append(img[1])
                img_recognitions.append(img[2])
        temp = zip(img_ids, img_urls, img_recognitions)
        # temp = sorted(temp, key=lambda x: x[2], reverse=True)
        img_ids = [i[0] for i in temp]
        img_urls = [i[1] for i in temp]
        img_recognitions = [i[2] for i in temp]

        dict_name_reverse = dict([(dict_name2[event_name], event_name) for event_name in dict_name2])
        if event_id in correct_list:
            root1 = html_root_test + '/' + correct_list[event_id] + '/'
        else:
            root1 = html_root_test + '/' + event_type + '/'
        if not os.path.exists(root1):
            os.mkdir(root1)

        html_path = root1 + dict_name_reverse[np.argmax(event_prediction[event_id]) + 1] + '_' + event_id + '_' + event_model_name + '.html'
        print html_path
        # if not os.path.exists(root1 + event_id):
        #     os.mkdir(root1 + event_id)
        in_file = root + 'all_output/Amazon Mechanical Turk_score_hori_rearrange.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for kk in xrange(len(line_stack)):
            if '${num_image}' in line_stack[kk]:
                line_stack[kk] = 'var n_images = '+str(len(img_ids))+';\n'
            if 'var images = ["${image1}"' in line_stack[kk]:
                new_line = 'var images = ['
                for img_url in img_urls:
                    new_line += '"'+img_url+'",'
                new_line = new_line[:-1]
                new_line += '];\n'
                line_stack[kk] = new_line

            if '$(document).ready(function()' in line_stack[kk]:
                for i in xrange(len(img_recognitions)):
                    score = img_recognitions[i]#[0][0]
                    # score = format(score, '.2f')
                    #score = 1/(1 + np.exp(-score))
                    temp = np.argsort(score)
                    recognition_score = dict_name_reverse[temp[-1] + 1] + str(score[temp[-1]]) + \
                                        '_' + dict_name_reverse[temp[-2] + 1] + str(score[temp[-2]])
                    line_stack[kk] += '\ndocument.getElementById("image'+str(i+1)+'selected").value="'+ recognition_score+'";\n'
                    # if score > thresholds[2]:
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    # elif score > thresholds[1]:
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    # elif  score >= thresholds[0]:
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    # else:
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                    #         line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()


if __name__ == '__main__':
    show_test_predict_img_rearrange_allevent()
    # create_result_htmls_rearranged_forsupp()
    # for event_type in dict_name:
    #     path = root + 'baseline_all_0509/' + event_type + '/'
    #     f = open(path + 'test_event_id.cPickle','r')
    #     ids = cPickle.load(f)
    #     f.close()
    #     show_test_predict_img_rearranged('CNN_all_event_corrected_multi/features_validation/'+event_type+'_',
    #                                'THRESHOLDM_POLY2_LSTM_iter0_importance_cross_validation_combine_best_dict',
    #                                event_type, False)
        # show_test_predict_img_rearranged_recognition_corrected('features/'+event_type+'_',
        #                            'test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_dict',
        #                            'test_predict_event_recognition_expand_balanced_3_iter_100000_groundtruth_importance',
        #                            event_type)
        # show_test_predict_img_rearranged_recognition_corrected('features/'+event_type+'_',
        #                            'test_sigmoid9_23_segment_twoloss_fc300_diffweight_2_real_3time_iter_100000_em_dict',
        #                            'test_predict_event_recognition_expand_balanced_3_corrected_iter_100000_em',
        #                            event_type)
        # show_test_predict_img_recognition_corrected('features/'+event_type+'_',
        #                            'recognition_lstm_prediction_dict',
        #                            event_type)
        # show_test_predict_img_recognition_corrected('features/'+event_type+'_',
        #                            'test_predict_event_recognition_expand_balanced_3_corrected_iter_100000_em',
        #                            event_type)
    '''
    # for event_type in dict_name2:
    #     validation_name = 'val_test'
    #     show_test_predict_img_rearranged_facechange(validation_name, 'CNN_all_event/features/',
    #                                                 event_type + '_'+validation_name+'_combined_10_dict',
    #                                                 event_type)
    #     validation_name = 'val_validation'
    #     show_test_predict_img_rearranged_facechange(validation_name, 'CNN_all_event/features/',
    #                                                 event_type + '_'+validation_name+'_combined_10_dict',
    #                                                 event_type)
    '''

    # create_result_htmls_new(root + '0208_correction/curation_results.csv')
    # for event_type in dict_name2:
    #     #show_test_predict_img_rearranged('to_guru/CNN_all_event/features/',
    #     #                           event_type + '_test_sigmoid9segment_iter_70000_dict',
    #     #                           event_type)
    #     # show_test_predict_img_rearranged('baseline_all_noblock/'+event_type+'/',
    #     #                           'vgg_training_result_v2',
    #     #                           event_type, True)
    #     # show_test_predict_img_rearranged('CNN_all_event_1009/features/',
    #     #                           event_type + '_test_combined_10_fromnoevent_dict',
    #     #                           event_type)
    #     #show_test_predict_img('to_guru/CNN_all_event/features/',
    #     #                           event_type + '_test_sigmoid9segment_iter_70000_dict',
    #     #                           event_type)
    #     #show_test_predict_img('baseline_all_noblock/'+event_type+'/',
    #     #                           'vgg_test_result_v2',
    #     #                           event_type, True)
    #     # show_test_predict_img('to_guru/CNN_all_event/features/',
    #     #                           event_type + '_combined_test',
    #     #                           event_type)
    #     print event_type
    #     pic_test_predict_img_topk_percent_forsupp(event_type)
    #
    #     # path = root + 'baseline_all_noblock/' + event_type + '/'
    #     # f = open(path + 'test_event_id.cPickle','r')
    #     # ids = cPickle.load(f)
    #     # f.close()
    #     # top_kpercent = 10
    #     # for id in ids:
    #         #cluster_faces_new(id, event_type)
    #         # show_test_predict_img_topk('baseline_all_noblock/'+event_type+'/',
    #         #                     'test_random_dict',
    #         #                       event_type, 10)
    #         # show_test_predict_img_topk('baseline_all_noblock/'+event_type+'/',
    #         #                        'vgg_test_result_v2',
    #         #                       event_type, 10, True)
    #         # show_test_predict_img_topk('baseline_all_noblock/'+event_type+'/',
    #         #                        'vgg_predict_result_10_dict',
    #         #                       event_type, 10)
    #         # show_test_predict_img_topk('CNN_all_event_1009/features/',
    #         #                        event_type+'_test_combined_10_new_dict',
    #         #                       event_type, 10)
    #         # show_test_predict_img_topk('CNN_all_event_old/features/',
    #         #                        event_type+'_test_sigmoid9segment_iter_70000_dict',
    #         #                       event_type, 10)
    #         # show_test_predict_img_topk('CNN_all_event_1009/features/',
    #         #                        event_type+'_test_sigmoid9segment_2round_iter_750000_dict',
    #         #                       event_type, 10)
    #         #show_test_predict_img('baseline_all_noblock/' + event_type + '/',
    #         #                      'vgg_predict_result_10_dict',
    #         #                      event_type)
    #         #show_test_predict_img_rearranged('baseline_all_noblock/' + event_type + '/',
    #         #                      'vgg_predict_result_10_dict',
    #         #                      event_type)
    #         #
    #         # show_test_predict_img('baseline_all_noblock/' + event_type + '/',
    #         #                      'vgg_test_result_v2',
    #         #                      event_type, groundtruth=True)