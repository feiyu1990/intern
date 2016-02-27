__author__ = 'wangyufei'
import csv
import cPickle
import random
root = '/Users/wangyufei/Documents/Study/intern_adobe/'
root = '/home/feiyu1990/local/event_curation/'
dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

def separate_train_val(event_name):
    in_path = root + 'baseline_all/'+event_name+'/test_event_id.cPickle'
    f = open(in_path, 'r')
    event_test = cPickle.load(f)
    f.close()
    len_test = int(float(len(event_test))/2)
    print event_name, len_test
    #print event_test
    random.shuffle(event_test)
    val_ = event_test[:len_test]
    test_ = event_test[len_test:]
    out_path = root + 'baseline_all/'+event_name+'/val_validation_event_id.cPickle'
    f = open(out_path,'w')
    cPickle.dump(val_, f)
    f.close()
    out_path = root + 'baseline_all/'+event_name+'/val_test_event_id.cPickle'
    f = open(out_path,'w')
    cPickle.dump(test_, f)
    f.close()

    create_path(event_name)
    from_npy_to_dicts(event_name, 'val_validation')
    from_npy_to_dicts(event_name, 'val_test')
def create_path(check_type):
    in_path = root + 'baseline_all/'+check_type+'/val_validation_event_id.cPickle'
    f = open(in_path, 'r')
    events_val = cPickle.load(f)
    f.close()
    in_path = root + 'baseline_all/'+check_type+'/val_test_event_id.cPickle'
    f = open(in_path, 'r')
    events_test = cPickle.load(f)
    f.close()
    load_path = root+'all_output/all_images_curation.txt'
    save_paths1 = root + 'baseline_all/'+check_type+'/guru_val_validation_path.txt'
    f1 = open(save_paths1, 'wb')
    save_paths2 = root + 'baseline_all/'+check_type+'/guru_val_test_path.txt'
    f2 = open(save_paths2, 'wb')
    prefix = '/home/feiyu1990/local/event_curation/curation_images/'
    with open(load_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            if meta[1]+'_'+meta[3] in events_val:
                string = prefix + event_name + '/' + meta[3]+'/'+meta[2] + '.jpg 0\n'
                f1.write(string)
            elif meta[1]+'_'+meta[3] in events_test:
                string = prefix + event_name + '/' + meta[3]+'/'+meta[2] + '.jpg 0\n'
                f2.write(string)
    f1.close()
    f2.close()
def from_npy_to_dicts(check_type, type = 'training'):
    #f = open(root + 'baseline_wedding_test/wedding_'+type+'_features.cPickle','r')
    #features = cPickle.load(f)
    #f.close()
    image_ids = []
    in_path = root + 'baseline_all/'+check_type+'/guru_'+type+'_path.txt'
    with open(in_path,'r') as data:
        for line in data:
            name = line.split('/')[-1]
            name = name.split('.')[0]
            image_ids.append(name)

    event_ids = []
    i = 0
    in_path = root+'all_output/all_images_curation.txt'
    with open(in_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            id = meta[2]
            if id == image_ids[i]:
                i += 1
                event_ids.append(meta[1] + '_' + meta[3] + '/' + meta[2])
                if i == len(image_ids):
                    break
    f = open(root + 'baseline_all/' + check_type + '/' + type+'_image_ids.cPickle','wb')
    cPickle.dump(event_ids, f)
    f.close()
def create_ground_truth(event_name, type = 'validation'):
    in_path = root + 'baseline_all/' + event_name + '/val_'+type+'_event_id.cPickle'
    f = open(in_path,'r')
    test_events = cPickle.load(f)
    f.close()

    in_path = root + 'baseline_all/' +event_name + '/alexnet6k_predict_result_10_dict.cPickle'
    f = open(in_path,'r')
    a = cPickle.load(f)
    f.close()
    dict_ = {}
    out_path = root + 'baseline_all/' + event_name + '/val_'+type+'_alexnet6k_predict_result_10_dict.cPickle'
    for event_ in test_events:
        dict_[event_] = a[event_]
    f = open(out_path,'w')
    cPickle.dump(dict_, f)
    f.close()

    in_path = root + 'baseline_all/' +event_name + '/vgg_predict_result_10_dict.cPickle'
    f = open(in_path,'r')
    a = cPickle.load(f)
    f.close()
    dict_ = {}
    out_path = root + 'baseline_all/' + event_name + '/val_'+type+'_vgg_predict_result_10_dict.cPickle'
    for event_ in test_events:
        dict_[event_] = a[event_]
    f = open(out_path,'w')
    cPickle.dump(dict_, f)
    f.close()


    in_path = root + 'baseline_all/' +event_name + '/vgg_test_result_v2.cPickle'
    f = open(in_path,'r')
    a = cPickle.load(f)
    f.close()
    dict_ = {}
    out_path = root + 'baseline_all/' + event_name + '/vgg_val_'+type+'_result_v2.cPickle'
    for event_ in test_events:
        dict_[event_] = a[event_]
    f = open(out_path,'w')
    cPickle.dump(dict_, f)
    f.close()

def create_csv(name):
    f = open(root + 'baseline_all/'+name+'/val_validation_event_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all/'+name+'/val_validation.csv','wb')
    writer = csv.writer(f)
    line_count = 0
    input_path = root + 'all_output/all_output.csv'
    with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    writer.writerow(meta)
                else:
                    if meta[28] in event_ids:
                        writer.writerow(meta)
                line_count += 1
    f.close()

    f = open(root + 'baseline_all/'+name+'/val_test_event_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    f = open(root + 'baseline_all/'+name+'/val_test.csv','wb')
    writer = csv.writer(f)
    line_count = 0
    input_path = root + 'all_output/all_output.csv'
    with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    writer.writerow(meta)
                else:
                    if meta[28] in event_ids:
                        writer.writerow(meta)
                line_count += 1
    f.close()
def create_dict_url(check_type):
    path = root + 'baseline_all/' + check_type + '/val_validation_image_ids.cPickle'
    f = open(path, 'r')
    image_ids = cPickle.load(f)
    f.close()
    dict = {}

    path = root+'all_output/all_images_curation.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            image_id = meta[1] + '_' + meta[3] + '/'+meta[2]
            if image_id in image_ids:
                dict[image_id] = meta[16]

    f = open(root + 'baseline_all/'+check_type+'/val_validation_ulr_dict.cPickle', 'wb')
    cPickle.dump(dict, f)
    f.close()
    path = root + 'baseline_all/' + check_type + '/val_test_image_ids.cPickle'
    f = open(path, 'r')
    image_ids = cPickle.load(f)
    f.close()


    dict = {}
    path = root+'all_output/all_images_curation.txt'
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            image_id = meta[1] + '_' + meta[3] + '/'+meta[2]
            if image_id in image_ids:
                dict[image_id] = meta[16]

    f = open(root + 'baseline_all/'+check_type+'/val_test_ulr_dict.cPickle', 'wb')
    cPickle.dump(dict, f)
    f.close()

def to_guru_file():
    for event_name in dict_name2:
        if event_name == 'Wedding':
            continue
        in_path = root + 'baseline_all/'+event_name+'/guru_val_test_path.txt'
        out_path = root + 'face_heatmap/data/'+event_name+'_val_test_path.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
                for line in data:
                    img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' + '/'.join(line.split('/')[-2:])
                    f.write(img_path_new)
        f.close()

        in_path = root + 'baseline_all/'+event_name+'/guru_val_validation_path.txt'
        out_path = root + 'face_heatmap/data/'+event_name+'_val_validation_path.txt'
        f = open(out_path, 'w')
        with open(in_path,'r') as data:
                    for line in data:
                        img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped_allevent/' + '/'.join(line.split('/')[-2:])
                        f.write(img_path_new)
        f.close()
    event_name = 'Wedding'
    in_path = root + 'baseline_all/'+event_name+'/guru_val_test_path.txt'
    out_path = root + 'face_heatmap/data/'+event_name+'_val_test_path.txt'
    f = open(out_path, 'w')
    with open(in_path,'r') as data:
                for line in data:
                    img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped/' + '/'.join(line.split('/')[-2:])
                    f.write(img_path_new)
    f.close()
    in_path = root + 'baseline_all/'+event_name+'/guru_val_validation_path.txt'
    out_path = root + 'face_heatmap/data/'+event_name+'_val_validation_path.txt'
    f = open(out_path, 'w')
    with open(in_path,'r') as data:
                for line in data:
                    img_path_new = '/home/feiyu1990/local/event_curation/person_heatmap_images_cropped/' + '/'.join(line.split('/')[-2:])
                    f.write(img_path_new)
    f.close()

if __name__ == '__main__':

    for event_name in ['BusinessActivity']: #dict_name2:
        separate_train_val(event_name)
        create_ground_truth(event_name)
        create_ground_truth(event_name, 'test')
        create_csv(event_name)
        create_dict_url(event_name)
    to_guru_file()