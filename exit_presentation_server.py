__author__ = 'wangyufei'

import os
caffe_root = '/home/feiyu1990/local/caffe-mine/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import cPickle
import numpy as np

dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}


root = '/home/feiyu1990/local/event_curation/demo/'
#'img_set/' -> 'img_list.txt'
def create_dataset_path(folder):
    files = [root + folder +'/'+ f for f in os.listdir(root + folder) if f.endswith('jpg') or f.endswith('JPG') or f.endswith('png') or f.endswith('JPG')]
    out_file = root + 'img_list.txt'
    f = open(out_file, 'w')
    f.write('\n'.join(files) + '\n')
    f.close()
    return files

#'img_list.txt' -> 'feature.cPickle'
def extract_feature(name=root+'img_list.txt', net_name='segment_iter_70000', model_name='python_deploy_siamese_old.prototxt', blob_name='sigmoid9', img_size = 227):
    imgs = []
    img_file = name
    with open(img_file, 'r') as data:
        for line in data:
            imgs.append(line[:-1])
    model_name = '/home/feiyu1990/local/event_curation/CNN_all_event/' + model_name
    weight_name = '/home/feiyu1990/local/event_curation/CNN_all_event/snapshot/' + net_name + '.caffemodel'
    mean_file = np.load('/home/feiyu1990/local/caffe-mine/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    caffe.set_mode_gpu()
    net = caffe.Net(model_name,
                weight_name,
                caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_file.mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(1,3,img_size,img_size)
    features = []
    count = 0
    for img in imgs:
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
        out = net.forward()
        a = net.blobs[blob_name].data.copy()
        features.append(a)
        count += 1
    return features

def create_html(folder, event_name):
    files = create_dataset_path(folder)
    features_ = extract_feature()
    idx = dict_name2[event_name]-1
    features = [i[0][idx] for i in features_]
    file_importance = zip(files, features)
    file_importance.sort(key=lambda x: x[1], reverse=True)

    html_path = root + folder +'_' + event_name+ '_present.html'
    f_out = open(html_path,'w')
    f_out.write('<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js">'
                '</script><script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>'
                '<link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />')

    f_out.write('<section id="EventCuration" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">')
    f_out.write('<center>')
    f_out.write('<div class="panel panel-primary"><div class="panel-heading"><strong>'+folder+'_dataset with ranking</strong> </div></div>')

    f_out.write('<center>')
    f_out.write('<table border="1" style="width:500">\n')
    img_count = 0
    for i in xrange(len(file_importance)):
        this_img = file_importance[i]
        if img_count % 10 == 0:
            f_out.write('\t</tr><tr>\n')
        img_count += 1
        f_out.write('\t\t<td align=\"center\" valign=\"center\">\n')
        f_out.write('\t\t\t<img src=\"'+'/'.join(this_img[0].split('/')[-2:])+'\" alt=Loading... /></td>\n')
    f_out.write('</tr></table></section>\n')
    f_out.write('<style type=\"text/css\">img {width:240px;height:auto;}'
                '.selected_sw {border: 3px solid #9900FF !important;opacity:0.7 !important;cursor: pointer;}'
                '.selected {border: 3px solid #FF0000 !important;opacity:1.0 !important;cursor: pointer;}'
                'table, th, td { border: 3px solid white; font-size: 12px;}'
                '</style>\n')
    f_out.close()

if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 2
    folder = args[1]
    event_name = args[2]
    create_html(folder, event_name)

