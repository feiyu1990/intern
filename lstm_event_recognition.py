import cPickle
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict
import random
dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}


def preprocess_data():
    root = '/home/ubuntu/event_curation/CNN_all_event_1205/'
    feature_test = []
    for event_name in dict_name2:
        with open(root + 'features/' + event_name + '_test_fc7_event_recognition_expand_balanced_3_iter_100000.cPickle') as f:
            feature_test.extend(cPickle.load(f))

    feature_training = []
    for event_name in dict_name2:
        with open(root + 'features/' + event_name + '_training_fc7_event_recognition_expand_balanced_3_iter_100000.cPickle') as f:
            feature_training.extend(cPickle.load(f))

    feature_test = np.asarray(feature_test)
    feature_training = np.asarray(feature_training)
    print feature_test.shape
    print feature_training.shape

    # np.save(root + 'features/training_fc7_event_recognition_expand_balanced_3_iter_100000.npy', feature_training)
    # np.save(root + 'features/test_fc7_event_recognition_expand_balanced_3_iter_100000.npy', feature_test)

    root = '/home/ubuntu/event_curation/'
    # test_feature = np.load(root + '../lstm/data/test_fc7_event_recognition_expand_balanced_3_iter_100000_pca128.npy')
    # training_feature = np.load(root + '../lstm/data/training_fc7_event_recognition_expand_balanced_3_iter_100000_pca128.npy')
    training_event_img_dict = defaultdict(list)
    event_type_dict = dict()
    count = 0
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            img_event_list = cPickle.load(f)
        for i in img_event_list:
            event_type_dict[i.split('/')[0]] = event_name
            training_event_img_dict[i.split('/')[0]].append(count)
            count += 1
    with open(root + '../lstm/data/training_event_img_dict.pkl', 'w') as f:
        cPickle.dump(training_event_img_dict, f)
    test_event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            img_event_list = cPickle.load(f)
        for i in img_event_list:
            event_type_dict[i.split('/')[0]] = event_name
            test_event_img_dict[i.split('/')[0]].append(count)
            count += 1
    with open(root + '../lstm/data/test_event_img_dict.pkl', 'w') as f:
        cPickle.dump(test_event_img_dict, f)

    lstm_training_feature = []
    lstm_training_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    validation_event_id = random.sample(training_event_img_dict.keys(), 50)
    for event_id in training_event_img_dict:
        if event_id in validation_event_id:
            lstm_validation_feature.append(training_event_img_dict[event_id])
            lstm_validation_label.append(dict_name2[event_type_dict[event_id]] - 1)
        else:
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id]] - 1)

    lstm_test_feature = []
    lstm_test_label = []
    for event_id in test_event_img_dict:
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id]] - 1)

    with open(root + '../lstm/data/validation_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root + '../lstm/data/training_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root + '../lstm/data/test_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)

    features_all = np.concatenate((feature_training, feature_test), axis=0)
    print features_all.shape

    feature_training = []
    for i in lstm_training_feature:
        for j in i:
            feature_training.append(features_all[j, :])

    pca = PCA(n_components=128)
    feature_training_pca = pca.fit_transform(np.asarray(feature_training))

    features_all_pca = np.zeros((features_all.shape[0], 128))
    count = 0
    for i in lstm_training_feature:
        for j in i:
            features_all_pca[j, :] = feature_training_pca[count, :]
            count += 1

    np.save(root + '../lstm/data/feature_training_pca.npy', feature_training_pca)

    feature_test = []
    for i in lstm_test_feature:
        for j in i:
            feature_test.append(features_all[j, :])
    feature_test_pca = pca.transform(np.asarray(feature_test))
    count = 0
    for i in lstm_test_feature:
        for j in i:
            features_all_pca[j, :] = feature_test_pca[count, :]
            count += 1

    feature_valid = []
    for i in lstm_validation_feature:
        for j in i:
            feature_valid.append(features_all[j, :])
    feature_valid_pca = pca.transform(np.asarray(feature_valid))
    count = 0
    for i in lstm_validation_feature:
        for j in i:
            features_all_pca[j, :] = feature_valid_pca[count, :]
            count += 1

    print feature_valid_pca.shape
    print feature_test_pca.shape
    np.save(root + '../lstm/data/feature_test_pca.npy', feature_test_pca)
    np.save(root + '../lstm/data/feature_validation_pca.npy', feature_valid_pca)

    np.save(root + '../lstm/data/feature_all_pca.npy', features_all_pca)


    # np.save(root + 'features/training_fc7_event_recognition_expand_balanced_3_iter_100000_pca128.npy', feature_training)
    # np.save(root + 'features/test_fc7_event_recognition_expand_balanced_3_iter_100000_pca128.npy', feature_test)




def create_wemb():
    # root = '/home/feiyu1990/local/event_curation/'
    root = '/home/ubuntu/event_curation/'
    test_feature = np.load(root + '../lstm/data/test_fc7_event_recognition_expand_balanced_3_iter_100000_pca128.npy')
    training_feature = np.load(root + '../lstm/data/training_fc7_event_recognition_expand_balanced_3_iter_100000_pca128.npy')
    training_event_img_dict = defaultdict(list)
    event_type_dict = dict()
    count = 0
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            img_event_list = cPickle.load(f)
        for i in img_event_list:
            event_type_dict[i.split('/')[0]] = event_name
            training_event_img_dict[i.split('/')[0]].append(count)
            count += 1
    with open(root + '../lstm/data/training_event_img_dict.pkl', 'w') as f:
        cPickle.dump(training_event_img_dict, f)
    test_event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            img_event_list = cPickle.load(f)
        for i in img_event_list:
            event_type_dict[i.split('/')[0]] = event_name
            test_event_img_dict[i.split('/')[0]].append(count)
            count += 1
    with open(root + '../lstm/data/test_event_img_dict.pkl', 'w') as f:
        cPickle.dump(test_event_img_dict, f)

    features_all = np.concatenate((training_feature, test_feature), axis=0)
    print features_all.shape
    np.save(root + '../lstm/data/feature_all.npy', features_all)
    print count

    lstm_training_feature = []
    lstm_training_label = []
    # lstm_validation_feature = []
    # lstm_validation_label = []
    # validation_event_id = random.sample(training_event_img_dict.keys(), 50)
    for event_id in training_event_img_dict:
        # if event_id in validation_event_id:
        #     lstm_validation_feature.append(training_event_img_dict[event_id])
        #     lstm_validation_label.append(dict_name2[event_type_dict[event_id]] - 1)
        # else:
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id]] - 1)

    lstm_test_feature = []
    lstm_test_label = []
    for event_id in test_event_img_dict:
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id]] - 1)

    # with open(root + '../lstm/data/validation_imdb.pkl', 'w') as f:
        # cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root + '../lstm/data/training_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root + '../lstm/data/test_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)


if __name__ == '__main__':
    # preprocess_data()
    create_wemb()