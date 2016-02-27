__author__ = 'wangyufei'

import cPickle
import random
import scipy.io as sio
import os
import numpy as np
import csv
root = '/Users/wangyufei/Documents/Study/intern_adobe/'
#root = 'C:/Users/yuwang/Documents/'

def find_valid_examples(threshold = [0.8, 0.55]):
    f = open(root + 'baseline_wedding_test/vgg_wedding_training_result_dict_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_training_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    negative_idx = []
    positive_idx = []
    count = 0
    for img_id in img_ids:
        training_score = ground_truth_training[img_id]
        #if training_score
        training_score_normalized = (training_score + 2)/4
        if training_score_normalized <= threshold[1]:
            negative_idx.append(count)
        if training_score_normalized >= threshold[0]:
            positive_idx.append(count)
        count += 1

    #temp = [i[1] for i in positive_idx]
    #print min(temp)
    #temp = [i[1] for i in negative_idx]
    #print max(temp)
    print len(positive_idx), len(negative_idx)

    in_path = root + 'baseline_wedding_test/linux_wedding_training_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])

    img_paths_positive = [img_paths[i] + ' 1\n' for i in positive_idx]
    img_paths_negative = [img_paths[i] + ' 0\n' for i in negative_idx]

    img_ids_short = [kk.split('_')[1] for kk in img_ids]
    for img in img_paths_positive:
        img_this = img.split('.')[0]
        img_this = '/'.join(img_this.split('/')[-2:])
        if ground_truth_training[img_ids[img_ids_short.index(img_this)]] < threshold[0]:
            print 'ERROR!'

    for img in img_paths_negative:
        img_this = img.split('.')[0]
        img_this = '/'.join(img_this.split('/')[-2:])
        if ground_truth_training[img_ids[img_ids_short.index(img_this)]] > threshold[1]:
            print 'ERROR!'

    img_paths_selected = img_paths_positive + img_paths_negative
    random.shuffle(img_paths_selected)

    out_path = root + 'classify_CNN_wedding/training_path_' + str(threshold[0]) + '_' + str(threshold[1]) + '.txt'
    f = open(out_path,'wb')
    for i in img_paths_selected:
        f.write(i)
    f.close()



    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_dict_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_wedding_test/linux_wedding_test_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])


    count = 0
    scores = []
    for img_id in img_ids:
        training_score = ground_truth_training[img_id]
        #if training_score
        training_score_normalized = (training_score + 2)/4
        if training_score_normalized <= (threshold[1] + threshold[0])/2:
            scores.append(0)
        if training_score_normalized > (threshold[1] + threshold[0])/2:
            scores.append(1)
        count += 1

    img_paths_with_label = [i + ' '+str(j)+'\n' for (i,j) in zip(img_paths, scores)]


    out_path = root + 'classify_CNN_wedding/test_path_' + str((threshold[0]+threshold[1])/2) + '.txt'
    f = open(out_path,'wb')
    for i in img_paths_with_label:
        f.write(i)
    f.close()

def from_classification_to_regression():
    in_path = root + 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000.cPickle'
    f = open(in_path,'r')
    test_prob = cPickle.load(f)
    f.close()

    predict_score = [i[0][1] for i in test_prob]
    out_path = root + 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict.cPickle'
    f = open(out_path, 'wb')
    cPickle.dump(predict_score, f)
    f.close()

def check_result():
    in_path = root + 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000.cPickle'
    f = open(in_path,'r')
    test_prob = cPickle.load(f)
    f.close()

    test_result = [int(i[0][1]>0.5) for i in test_prob]

    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_dict_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    threshold = 0.675
    count = 0
    scores = []
    for img_id in img_ids:
        training_score = ground_truth_training[img_id]
        #if training_score
        training_score_normalized = (training_score + 2)/4
        if training_score_normalized <= threshold:
            scores.append(0)
        if training_score_normalized > threshold:
            scores.append(1)
        count += 1

    a = [(i+j)%2-1 for (i,j) in zip(scores, test_result)]
    print float(sum(a))/len(a)
    print len(a)

def check_result_training():
    in_path = root + 'classify_CNN_wedding/training_prob_classification0.55_0.8_iter_10000.cPickle'
    f = open(in_path,'r')
    test_prob = cPickle.load(f)
    f.close()

    test_result = [int(i[0][1]>0.5) for i in test_prob]

    f = open(root + 'baseline_wedding_test/vgg_wedding_training_result_dict_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_training_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    threshold = [0.8, 0.55]
    count = 0
    scores = []
    idx = []
    for img_id in img_ids:
        training_score = ground_truth_training[img_id]
        #if training_score
        training_score_normalized = (training_score + 2)/4
        if training_score_normalized <= threshold[1] or training_score_normalized >= threshold[0]:
            idx.append(count)
            scores.append(int(training_score_normalized >= threshold[0]))
        count += 1
    test_result = [test_result[i] for i in idx]

    a = [(i+j)%2-1 for (i,j) in zip(scores, test_result)]
    print float(sum(a))/len(a)
    print len(a)

def show_test_predict_img():
    f = open(root + 'classify_CNN_wedding/test_prob_classification0.55_0.8_iter_10000_predict_dict.cPickle','r')
    test_prediction_event = cPickle.load(f)
    f.close()
    root1 = root + 'present_htmls_test/'

    for event_id in test_prediction_event:
        this_event = test_prediction_event[event_id]
        img_ids = []
        img_urls = []
        img_scores = []
        for img in this_event:
            img_ids.append(img[0])
            img_urls.append(img[1])
            img_scores.append(img[2])

        html_path = root1 + event_id + '/prob_classification0.55_0.8_iter_10000.html'
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
                    if score > 1:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("highlight");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >0.8:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("selected_sw");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score >= 0:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("neutral");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'
                    elif score < 0:
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").addClass("irrelevant");\n'
                            line_stack[kk]+='$("label[name=\'image'+str(i+1) + '\']").removeClass("not_selected");\n'

        f = open(html_path, 'wb')
        for line in line_stack:
            f.write(line)
        f.close()



if __name__ == '__main__':
    #find_valid_examples()
    #check_result()
    #check_result_training()
    show_test_predict_img()

    '''
    in_path = root + 'wedding_CNN_net/test_fc8_value_vgg_euclidean_iter_290000.cPickle'
    f = open(in_path,'r')
    test_score = cPickle.load(f)
    f.close()
    test_score = [i[0][0] for i in test_score]
    test_result = [int(i >= 0.675) for i in test_score]

    f = open(root + 'baseline_wedding_test/vgg_wedding_test_result_dict_v2.cPickle','r')
    ground_truth_training = cPickle.load(f)
    f.close()

    f = open(root + 'baseline_wedding_test/wedding_test_image_ids.cPickle','r')
    img_ids = cPickle.load(f)
    f.close()

    thresholds = [0.5,0.675, 0.7, 0.75, 0.8]
    for threshold in thresholds:
        count = 0
        scores = []
        for img_id in img_ids:
            training_score = ground_truth_training[img_id]
            #if training_score
            training_score_normalized = (training_score + 2)/4
            if training_score_normalized <= threshold:
                scores.append(0)
            if training_score_normalized > threshold:
                scores.append(1)
            count += 1

        a = [(i+j)%2-1 for (i,j) in zip(scores, test_result)]
        print float(sum(a))/len(a)
        print len(a)
    '''