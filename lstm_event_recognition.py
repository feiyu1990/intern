import cPickle
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict
import random
import random
import ujson
dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}


dict_reverse = dict([(dict_name2[i], i) for i in dict_name2])
map_to_new = {'birthday':'Birthday',
              'children_birthday':'Birthday',
              'christmas':'Christmas',
              'cruise': 'Cruise',
              # 'exhibition': 'BusinessActivity',
              'exhibition': 'Museum',
              'halloween':'Halloween',
              'road_trip': 'NatureTrip',
              # 'skiing': 'PersonalSports',
              'concert': 'Show',
              'graduation': 'Graduation',
              'hiking': 'NatureTrip',
              'wedding': 'Wedding'}
new_ind = {4:0, 6:1, 7:2, 9:3, 17:4, 18:5, 19:6, 20:7, 21:8, 22:9}#, 15:10}


def preprocess_data_expand():
    root1 = '/home/ubuntu/lstm/data/'
    # feature = np.load(root1 + 'training_fc7_event_recognition_expand_balanced_3_iter_100000.npy')
    # feature_more = np.load(root1 + 'new_training_img_feature.npy')
    # print feature.shape, feature_more.shape
    # feature_training = np.concatenate((feature, feature_more), axis=0)
    #
    # feature_test = np.load(root1 + 'test_fc7_event_recognition_expand_balanced_3_iter_100000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'expand_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'expand_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'expand_feature_all.npy', feature_all)

    print 'generating training/validation/test'
    with open(root1 + 'combine_event_list.pkl') as f:
        temp = cPickle.load(f)
    event_type_dict = dict()
    for event_type in temp:
        for event_name in temp[event_type]:
            event_type_dict[event_name] = event_type
    with open(root1 + 'combine_event_type_dict.pkl', 'w') as f:
        cPickle.dump(event_type_dict, f)

    root = '/home/ubuntu/event_curation/'
    training_event_img_dict = defaultdict(list)
    count = 0
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/training_image_ids.cPickle') as f:
            img_event_list = cPickle.load(f)
        for i in img_event_list:
            training_event_img_dict[i.split('/')[0]].append(count)
            count += 1
    print count
    with open(root1 + 'new_training_img_list.txt') as f:
        for line in f:
            event_name = line.split('/')[-2]
            training_event_img_dict[event_name].append(count)
            count += 1
    print count

    with open(root1 + 'new_combine_event_list.pkl') as f:
        temp = cPickle.load(f)
    event_type_dict = dict()
    for event_type in temp:
        for event_name in temp[event_type]:
            event_type_dict[event_name] = event_type

    event_train_all_dict = defaultdict(list)
    for event_type in temp:
        for event_name in temp[event_type]:
            event_train_all_dict[event_type].append(event_name)
    event_new_dict = defaultdict(list)
    for event_type in event_train_all_dict:
        with open(root + 'baseline_all_0509/' + event_type + '/training_event_id.cPickle') as f:
            temp = cPickle.load(f)
        for i in event_train_all_dict[event_type]:
            if i in temp:
                continue
            event_new_dict[event_type].append(i)
    for i in event_new_dict:
        print i, len(event_new_dict[i])


    test_event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        with open(root + 'baseline_all_0509/' + event_name + '/test_image_ids.cPickle') as f:
            img_event_list = cPickle.load(f)
        for i in img_event_list:
            test_event_img_dict[i.split('/')[0]].append(count)
            event_type_dict[i.split('/')[0]] = event_name
            count += 1
    print count

    with open(root1 + 'new_combine_event_type_dict.pkl', 'w') as f:
        cPickle.dump(event_type_dict, f)


    with open(root1 + 'expand_test_event_img_dict.pkl', 'w') as f:
        cPickle.dump(test_event_img_dict, f)

    with open(root1 + 'expand_training_event_img_dict.pkl', 'w') as f:
        cPickle.dump(training_event_img_dict, f)


    lstm_training_feature = []
    lstm_training_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    validation_event_id = random.sample(training_event_img_dict.keys(), 50)

    for event_type in event_train_all_dict:
        with open(root + 'baseline_all_0509/' + event_type + '/training_event_id.cPickle') as f:
            temp = cPickle.load(f)
        len_total_need = len(temp) * 3
        event_have = event_train_all_dict[event_type]
        event_to_add = event_have
        print event_type, len_total_need, len(event_have)
        if len_total_need > len(event_have):
            len_to_add = len_total_need - len(event_have)
            for i in xrange(len_to_add):
                event_to_add.append(random.sample(event_have, 1)[0])
        print len(event_to_add), event_to_add
        for event_id in event_to_add:
            # print event_id, training_event_img_dict[event_id]
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id]] - 1)
            if event_id in validation_event_id:
                lstm_validation_feature.append(training_event_img_dict[event_id])
                lstm_validation_label.append(dict_name2[event_type_dict[event_id]] - 1)

    ind = range(len(lstm_training_label))
    random.shuffle(ind)
    lstm_training_feature = [lstm_training_feature[i] for i in ind]
    lstm_training_label = [lstm_training_label[i] for i in ind]


    lstm_test_feature = []
    lstm_test_label = []
    for event_id in test_event_img_dict:
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id]] - 1)

    with open(root1 + 'expand_validation_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root1 + 'expand_training_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root1 + 'expand_test_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)


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
    event_id_list = []
    for event_id in test_event_img_dict:
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id]] - 1)
        event_id_list.append(event_id)
    # with open(root + '../lstm/data/validation_imdb.pkl', 'w') as f:
        # cPickle.dump((lstm_validation_feature, lstm_validation_label), f)

    with open(root + '../lstm/data/test_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)
    with open(root + '../lstm/data/training_imdb.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root + '../lstm/data/test_event_list.pkl', 'w') as f:
        cPickle.dump(event_id_list, f)


def oversample_training_img(file_name, oversample=20):
    root = '/home/ubuntu/event_curation/'
    with open(root + '../lstm/data/'+file_name+'.pkl') as f:
        lstm_training_feature, lstm_training_label = cPickle.load(f)
    oversample_label = []; oversample_feature = []
    for feature, label in zip(lstm_training_feature, lstm_training_label):
        for ind in xrange(oversample):
            print (random.random() * 3 / 4 + 0.25)
            len_event = int((random.random() * 3 / 4 + 0.25) * len(feature))
            oversample_feature.append([feature[i] for i in sorted(random.sample(xrange(len(feature)), len_event))])
            oversample_label.append(label)
            # print len(oversample_feature)

    index_shuf = range(len(oversample_feature))
    random.shuffle(index_shuf)
    oversample_feature_shuffled = [oversample_feature[i] for i in index_shuf]
    oversample_label_shuffled = [oversample_label[i] for i in index_shuf]

    with open(root + '../lstm/data/'+file_name+'_oversample_20_0.75.pkl', 'w') as f:
        cPickle.dump((oversample_feature_shuffled, oversample_label_shuffled), f)



def preprocess_data_multilabel():
    root1 = '/home/ubuntu/lstm/data/'
    # feature_training = np.load(root1 + 'training_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'multilabel_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'multilabel_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'multilabel_feature_all.npy', feature_all)

    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round'+weighted+'.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_fc7_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    count = 0
    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count


    lstm_training_feature = []
    lstm_training_label = []
    lstm_test_feature = []
    lstm_test_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    validation_event_id = random.sample(test_event_img_dict.keys(), 50)
    for event_id in training_event_img_dict:
        if len(event_type_dict[event_id]) == 1:
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
        else:
            for event_type in event_type_dict[event_id]:
                lstm_training_feature.append(training_event_img_dict[event_id])
                lstm_training_label.append(dict_name2[event_type[0]] - 1)
    temp = range(len(lstm_training_label))
    random.shuffle(temp)
    lstm_training_label = [lstm_training_label[i] for i in temp]
    lstm_training_feature = [lstm_training_feature[i] for i in temp]

    lstm_test_feature_all = []
    lstm_test_label_all = []

    for event_id in test_list:
        lstm_test_feature_all.append(test_event_img_dict[event_id])
        lstm_test_label_all.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
        if event_id not in validation_event_id:
            continue
        lstm_validation_feature.append(test_event_img_dict[event_id])
        lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)


    # with open(root1 + 'multilabel_iter1w_validation_imdb.pkl', 'w') as f:
    #     cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root1 + 'multilabel_training_old'+weighted+'.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root1 + 'multilabel_test_old'+weighted+'.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)
    with open(root1 + 'multilabel_test_all_old'+weighted+'.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)

def preprocess_data_multilabel_balanced():
    root1 = '/home/ubuntu/lstm/data/'
    # feature_training = np.load(root1 + 'training_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'multilabel_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'multilabel_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'multilabel_feature_all.npy', feature_all)

    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    count = 0
    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count


    lstm_training_feature = []
    lstm_training_label = []
    lstm_test_feature = []
    lstm_test_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    validation_event_id = random.sample(test_event_img_dict.keys(), 50)
    for event_id in training_event_img_dict:
        if len(event_type_dict[event_id]) == 1:
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
        else:
            for event_type in event_type_dict[event_id]:
                lstm_training_feature.append(training_event_img_dict[event_id])
                lstm_training_label.append(dict_name2[event_type[0]] - 1)
    temp = range(len(lstm_training_label))
    random.shuffle(temp)
    lstm_training_label = [lstm_training_label[i] for i in temp]
    lstm_training_feature = [lstm_training_feature[i] for i in temp]

    lstm_test_feature_all = []
    lstm_test_label_all = []

    for event_id in test_list:
        lstm_test_feature_all.append(test_event_img_dict[event_id])
        lstm_test_label_all.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
        if event_id not in validation_event_id:
            continue
        lstm_validation_feature.append(test_event_img_dict[event_id])
        lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)


    # with open(root1 + 'multilabel_iter1w_validation_imdb.pkl', 'w') as f:
    #     cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root1 + 'vote_multilabel_training.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root1 + 'vote_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)
    with open(root1 + 'vote_multilabel_test_all.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)


def preprocess_data_multilabel_softtarget():
    root1 = '/home/ubuntu/lstm/data/'
    # feature_training = np.load(root1 + 'training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'vote_soft_multilabel_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'vote_soft_multilabel_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'vote_soft_multilabel_feature_all.npy', feature_all)


    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_softmaxall_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    count = 0
    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count


    lstm_training_feature = []
    lstm_training_label = []
    lstm_test_feature = []
    lstm_test_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    validation_event_id = random.sample(test_event_img_dict.keys(), 50)
    for event_id in training_event_img_dict:
        lstm_training_feature.append(training_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ))
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_training_label.append(label_this)

    temp = range(len(lstm_training_label))
    random.shuffle(temp)
    lstm_training_label = [lstm_training_label[i] for i in temp]
    lstm_training_feature = [lstm_training_feature[i] for i in temp]

    lstm_test_feature_all = []
    lstm_test_label_all = []

    for event_id in test_list:
        lstm_test_feature_all.append(test_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ), dtype='float32')
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_test_label_all.append(label_this)

    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ), dtype='float32')
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_test_label.append(label_this)

    # with open(root1 + 'multilabel_iter1w_validation_imdb_softtarget.pkl', 'w') as f:
    #     cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root1 + 'vote_softall_multilabel_training.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root1 + 'vote_softall_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)
    with open(root1 + 'vote_softall_multilabel_test_all.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)


def preprocess_data_multilabel_softtarget_crossvalidation(fold = 5):
    root1 = '/home/ubuntu/lstm/data/'
    # feature_training = np.load(root1 + 'training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'vote_soft_multilabel_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'vote_soft_multilabel_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'vote_soft_multilabel_feature_all.npy', feature_all)


    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_softmaxall_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    count = 0
    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    # print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    # print count



    temp_list = training_event_img_dict.keys()
    random.shuffle(temp_list)
    len_ = len(temp_list)
    for iii in xrange(fold):
        lstm_training_feature = []
        lstm_training_label = []
        lstm_test_feature = []
        lstm_test_label = []
        lstm_validation_feature = []
        lstm_validation_label = []
        validation_event_id = temp_list[iii*len_/fold: (iii+1)*len_/fold]
        # print validation_event_id
        for event_id in training_event_img_dict:
            # print event_id
            if event_id in validation_event_id:
                lstm_validation_feature.append(training_event_img_dict[event_id])
                temp = event_type_dict[event_id]
                label_this = np.zeros((23, ))
                for i in temp:
                    label_this[dict_name2[i[0]] - 1] = i[1]
                lstm_validation_label.append(label_this)
            else:
                lstm_training_feature.append(training_event_img_dict[event_id])
                temp = event_type_dict[event_id]
                label_this = np.zeros((23, ))
                for i in temp:
                    label_this[dict_name2[i[0]] - 1] = i[1]
                lstm_training_label.append(label_this)
        print len(lstm_validation_label), len(lstm_training_label)
        temp = range(len(lstm_training_label))
        random.shuffle(temp)
        lstm_training_label = [lstm_training_label[i] for i in temp]
        lstm_training_feature = [lstm_training_feature[i] for i in temp]


        with open(root1 + 'vote_softall_multilabel_training_'+str(iii)+'.pkl', 'w') as f:
            cPickle.dump((lstm_training_feature, lstm_training_label), f)
        with open(root1 + 'vote_softall_multilabel_validation_'+str(iii)+'.pkl', 'w') as f:
            cPickle.dump((lstm_validation_feature, lstm_validation_label), f)


    lstm_test_feature = []
    lstm_test_label = []
    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ), dtype='float32')
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_test_label.append(label_this)
        with open(root1 + 'vote_softall_multilabel_test.pkl', 'w') as f:
            cPickle.dump((lstm_test_feature, lstm_test_label), f)


        # for event_id in test_list:
        #     if len(event_type_dict[event_id]) > 1:
        #         continue
        #     lstm_test_feature.append(test_event_img_dict[event_id])
        #     temp = event_type_dict[event_id]
        #     label_this = np.zeros((23, ), dtype='float32')
        #     for i in temp:
        #         label_this[dict_name2[i[0]] - 1] = i[1]
        #     lstm_test_label.append(label_this)

        # with open(root1 + 'multilabel_iter1w_validation_imdb_softtarget.pkl', 'w') as f:
        #     cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
        # with open(root1 + 'vote_softall_multilabel_training.pkl', 'w') as f:
        #     cPickle.dump((lstm_training_feature, lstm_training_label), f)

        # with open(root1 + 'vote_softall_multilabel_test_all.pkl', 'w') as f:
        #     cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)

def preprocess_data_multilabel_balanced_crossvalidation(fold=5):
    root1 = '/home/ubuntu/lstm/data/'
    # feature_training = np.load(root1 + 'training_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'multilabel_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'multilabel_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'multilabel_feature_all.npy', feature_all)

    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    count = 0
    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count



    temp_list = training_event_img_dict.keys()
    random.shuffle(temp_list)
    len_ = len(temp_list)
    for iii in xrange(fold):
        lstm_training_feature = []
        lstm_training_label = []
        lstm_test_feature = []
        lstm_test_label = []
        lstm_validation_feature = []
        lstm_validation_label = []
        validation_event_id = temp_list[iii*len_/fold: (iii+1)*len_/fold]
        # print validation_event_id
        for event_id in training_event_img_dict:
            # print event_id
            if len(event_type_dict[event_id]) == 1:
                if event_id not in validation_event_id:
                    lstm_training_feature.append(training_event_img_dict[event_id])
                    lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
                    lstm_training_feature.append(training_event_img_dict[event_id])
                    lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
                else:

                    lstm_validation_feature.append(training_event_img_dict[event_id])
                    lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
                    lstm_validation_feature.append(training_event_img_dict[event_id])
                    lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)

            else:
                if event_id not in validation_event_id:
                    for event_type in event_type_dict[event_id]:
                        lstm_training_feature.append(training_event_img_dict[event_id])
                        lstm_training_label.append(dict_name2[event_type[0]] - 1)
                else:
                    for event_type in event_type_dict[event_id]:
                        lstm_validation_feature.append(training_event_img_dict[event_id])
                        lstm_validation_label.append(dict_name2[event_type[0]] - 1)




        print len(lstm_validation_label), len(lstm_training_label)
        temp = range(len(lstm_training_label))
        random.shuffle(temp)
        lstm_training_label = [lstm_training_label[i] for i in temp]
        lstm_training_feature = [lstm_training_feature[i] for i in temp]


        with open(root1 + 'vote_multilabel_training_'+str(iii)+'.pkl', 'w') as f:
            cPickle.dump((lstm_training_feature, lstm_training_label), f)
        with open(root1 + 'vote_multilabel_validation_'+str(iii)+'.pkl', 'w') as f:
            cPickle.dump((lstm_validation_feature, lstm_validation_label), f)

    lstm_test_feature = []
    lstm_test_label = []

    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)

    with open(root1 + 'vote_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)


def preprocess_data_multilabel_balanced_crossvalidation_correct(fold=5):
    root1 = '/home/ubuntu/lstm/data/'
    # feature_training = np.load(root1 + 'training_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_multilabel_event_recognition_expand_balanced_3_iter_40000.npy')
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    # print 'testing pca...'
    # feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)
    #
    # np.save(root1 + 'multilabel_feature_training_pca.npy', feature_training_pca)
    # np.save(root1 + 'multilabel_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'multilabel_feature_all.npy', feature_all)

    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    count = 0
    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count



    temp_list = test_event_img_dict.keys()
    random.shuffle(temp_list)
    len_ = len(temp_list)
    for iii in xrange(fold):
        lstm_training_feature = []
        lstm_training_label = []
        lstm_test_feature = []
        lstm_test_label = []
        lstm_validation_feature = []
        lstm_validation_label = []
        validation_event_id = temp_list[iii*len_/fold: (iii+1)*len_/fold]
        # print validation_event_id
        for event_id in training_event_img_dict:
            # print event_id
            if len(event_type_dict[event_id]) == 1:
                if event_id not in validation_event_id:
                    lstm_training_feature.append(training_event_img_dict[event_id])
                    lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
                    lstm_training_feature.append(training_event_img_dict[event_id])
                    lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
                else:

                    lstm_validation_feature.append(training_event_img_dict[event_id])
                    lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
                    lstm_validation_feature.append(training_event_img_dict[event_id])
                    lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)

            else:
                if event_id not in validation_event_id:
                    for event_type in event_type_dict[event_id]:
                        lstm_training_feature.append(training_event_img_dict[event_id])
                        lstm_training_label.append(dict_name2[event_type[0]] - 1)
                else:
                    for event_type in event_type_dict[event_id]:
                        lstm_validation_feature.append(training_event_img_dict[event_id])
                        lstm_validation_label.append(dict_name2[event_type[0]] - 1)




        print len(lstm_validation_label), len(lstm_training_label)
        temp = range(len(lstm_training_label))
        random.shuffle(temp)
        lstm_training_label = [lstm_training_label[i] for i in temp]
        lstm_training_feature = [lstm_training_feature[i] for i in temp]


        with open(root1 + 'vote_multilabel_training_'+str(iii)+'.pkl', 'w') as f:
            cPickle.dump((lstm_training_feature, lstm_training_label), f)
        with open(root1 + 'vote_multilabel_validation_'+str(iii)+'.pkl', 'w') as f:
            cPickle.dump((lstm_validation_feature, lstm_validation_label), f)

    lstm_test_feature = []
    lstm_test_label = []

    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)

    with open(root1 + 'vote_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)



def valid_preprocess_data_multilabel_balanced():
    root1 = '/home/ubuntu/lstm/data_new/'
    pca = cPickle.load(open('/home/ubuntu/lstm/data_old/training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_PCA.pkl'))
    feature_training = np.load('/home/ubuntu/lstm/data_old/vote_multilabel_feature_training_pca.npy')
    feature_test = np.load(root1 + 'test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    feature_validation = np.load(root1 + 'validation_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    feature_test_pca = pca.transform(feature_test)
    print feature_test_pca.shape
    feature_validation_pca = pca.transform(feature_validation)
    print feature_validation_pca.shape

    feature_all = np.concatenate((feature_training, feature_test_pca, feature_validation_pca), axis=0)

    np.save(root1 + 'multilabel_feature_all.npy', feature_all)
    np.save(root1 + 'multilabel_feature_test_pca.npy', feature_test_pca)
    np.save(root1 + 'multilabel_feature_validation_pca.npy', feature_validation_pca)

    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)


    with open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_list.pkl') as f:
        test_list = cPickle.load(f)
    with open(root1 + 'validation_list.pkl') as f:
        validation_list = cPickle.load(f)
    count = 0


    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    validation_event_img_dict= defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in validation_list:
        for img in event_img_dict[event_id]:
            validation_event_img_dict[event_id].append(count)
            count += 1
    print count


    lstm_training_feature = []
    lstm_training_label = []
    lstm_test_feature = []
    lstm_test_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    for event_id in training_event_img_dict:
        if len(event_type_dict[event_id]) == 1:
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
            lstm_training_feature.append(training_event_img_dict[event_id])
            lstm_training_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
        else:
            for event_type in event_type_dict[event_id]:
                lstm_training_feature.append(training_event_img_dict[event_id])
                lstm_training_label.append(dict_name2[event_type[0]] - 1)
    temp = range(len(lstm_training_label))
    random.shuffle(temp)
    lstm_training_label = [lstm_training_label[i] for i in temp]
    lstm_training_feature = [lstm_training_feature[i] for i in temp]

    lstm_test_feature_all = []
    lstm_test_label_all = []

    for event_id in validation_list:
        lstm_validation_feature.append(validation_event_img_dict[event_id])
        lstm_validation_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
    for event_id in test_list:
        lstm_test_feature_all.append(test_event_img_dict[event_id])
        lstm_test_label_all.append(dict_name2[event_type_dict[event_id][0][0]] - 1)
    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        lstm_test_label.append(dict_name2[event_type_dict[event_id][0][0]] - 1)


    # with open(root1 + 'multilabel_iter1w_validation_imdb.pkl', 'w') as f:
    #     cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root1 + 'vote_multilabel_training.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root1 + 'vote_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)
    with open(root1 + 'vote_multilabel_test_all.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)
    with open(root1 + 'vote_multilabel_validation.pkl', 'w') as f:
        cPickle.dump((lstm_validation_feature, lstm_validation_label), f)


def valid_preprocess_data_multilabel_softtarget():
    root1 = '/home/ubuntu/lstm/data_new/'

    pca = cPickle.load(open('/home/ubuntu/lstm/data_old/training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_PCA.pkl'))
    feature_training = np.load('/home/ubuntu/lstm/data_old/vote_soft_multilabel_feature_training_pca.npy')
    feature_test = np.load(root1 + 'test_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    feature_validation = np.load(root1 + 'validation_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    feature_test_pca = pca.transform(feature_test)
    print feature_test_pca.shape
    feature_validation_pca = pca.transform(feature_validation)
    print feature_validation_pca.shape

    feature_all = np.concatenate((feature_training, feature_test_pca, feature_validation_pca), axis=0)

    np.save(root1 + 'soft_multilabel_feature_all.npy', feature_all)
    np.save(root1 + 'soft_multilabel_feature_test_pca.npy', feature_test_pca)
    np.save(root1 + 'soft_multilabel_feature_validation_pca.npy', feature_validation_pca)


    print 'generating training/validation/test'
    with open(root1 + 'new_multiple_result_2round_softmaxall_removedup_vote.pkl') as f:
        event_type_dict = cPickle.load(f)

    root = '/home/ubuntu/event_curation/'
    event_img_dict = defaultdict(list)

    for event_type in dict_name2:
        with open(root + 'baseline_all_0509/' + event_type + '/training_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)
        with open(root + 'baseline_all_0509/' + event_type + '/test_image_ids.cPickle') as f:
            temp = cPickle.load(f)
        for img in temp:
            event_name = img.split('/')[0]
            event_img_dict[event_name].append(img)

    with open(root1 + 'training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000_event_list.pkl') as f:
        training_list = cPickle.load(f)
    with open(root1 + 'test_list.pkl') as f:
        test_list = cPickle.load(f)
    with open(root1 + 'validation_list.pkl') as f:
        validation_list = cPickle.load(f)
    count = 0

    training_event_img_dict = defaultdict(list)
    test_event_img_dict = defaultdict(list)
    validation_event_img_dict= defaultdict(list)
    for event_id in training_list:
        for img in event_img_dict[event_id]:
            training_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in test_list:
        for img in event_img_dict[event_id]:
            test_event_img_dict[event_id].append(count)
            count += 1
    print count
    for event_id in validation_list:
        for img in event_img_dict[event_id]:
            validation_event_img_dict[event_id].append(count)
            count += 1
    print count


    lstm_training_feature = []
    lstm_training_label = []
    lstm_test_feature = []
    lstm_test_label = []
    lstm_validation_feature = []
    lstm_validation_label = []
    for event_id in training_event_img_dict:
        lstm_training_feature.append(training_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ))
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_training_label.append(label_this)

    temp = range(len(lstm_training_label))
    random.shuffle(temp)
    lstm_training_label = [lstm_training_label[i] for i in temp]
    lstm_training_feature = [lstm_training_feature[i] for i in temp]

    lstm_test_feature_all = []
    lstm_test_label_all = []

    for event_id in test_list:
        lstm_test_feature_all.append(test_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ), dtype='float32')
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_test_label_all.append(label_this)

    for event_id in validation_list:
        lstm_validation_feature.append(validation_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ), dtype='float32')
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_validation_label.append(label_this)

    for event_id in test_list:
        if len(event_type_dict[event_id]) > 1:
            continue
        lstm_test_feature.append(test_event_img_dict[event_id])
        temp = event_type_dict[event_id]
        label_this = np.zeros((23, ), dtype='float32')
        for i in temp:
            label_this[dict_name2[i[0]] - 1] = i[1]
        lstm_test_label.append(label_this)

    # with open(root1 + 'multilabel_iter1w_validation_imdb_softtarget.pkl', 'w') as f:
    #     cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
    with open(root1 + 'vote_softall_multilabel_training.pkl', 'w') as f:
        cPickle.dump((lstm_training_feature, lstm_training_label), f)
    with open(root1 + 'vote_softall_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature, lstm_test_label), f)
    with open(root1 + 'vote_softall_multilabel_test_all.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)

    with open(root1 + 'vote_softall_multilabel_validation.pkl', 'w') as f:
        cPickle.dump((lstm_validation_feature, lstm_validation_label), f)
def preprocess_data_pec():
    root1 = '/home/ubuntu/lstm/data/'

    # feature_training = np.load(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    # print feature_training.shape

    feature_test = np.load(root1 + 'pec_all_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')

    print feature_test.shape
    # print 'doing pca...'
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)

    # f = open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_PCA.pkl', 'w')
    # cPickle.dump(pca, f)
    # f.close()

    f = open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_PCA.pkl')
    pca = cPickle.load(f)
    f.close()
    print 'testing pca...'
    feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)

    # np.save(root1 + 'vote_multilabel_feature_training_pca.npy', feature_training_pca)
    np.save(root1 + 'pec_feature_all_pca.npy', feature_test_pca)
    # np.save(root1 + 'pec_feature_all.npy', feature_all)


    with open(root1 + 'train_test.json') as f:
        event_dict = ujson.load(f)
    with open(root1 + 'pec_all_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)

    lstm_test_feature_all = []
    lstm_test_label_all = []

    count = 0
    for event_id in test_list:
        event_type = event_id.split('/')[0]
        album = event_id.split('/')[1]
        event_count = len(event_dict[event_type][album])
        lstm_test_feature_all.append(range(count, count+event_count))
        if event_type in map_to_new:
            lstm_test_label_all.append(dict_name2[map_to_new[event_type]] - 1)
        else:
            lstm_test_label_all.append(22)
        count += event_count
        print count
    # print count

    with open(root1 + 'pec_vote_multilabel_all.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)

def preprocess_data_pec_soft():
    soft = '' #soft='_soft'
    root1 = '/home/ubuntu/lstm/data_new/'

    feature_validation = np.load(root1 + 'pec_training_fc7_vote'+soft + '_multilabel_event_recognition_expand_balanced_3'+soft + '_iter_100000.npy')
    print feature_validation.shape
    feature_test = np.load(root1 + 'pec_test_fc7_vote'+soft + '_multilabel_event_recognition_expand_balanced_3'+soft + '_iter_100000.npy')
    print feature_test.shape
    print 'doing pca...'

    # feature_validation = np.load(root1 + 'pec_training_fc7_vote_soft_multilabel_event_recognition_expand_balanced_3_soft_iter_100000.npy')
    # pca = PCA(n_components=128)
    # feature_training_pca = pca.fit_transform(feature_training)
    f = open(root1 + 'training_fc7_vote'+soft + '_multilabel_event_recognition_expand_balanced_3'+soft + '_iter_100000_PCA.pkl')
    pca = cPickle.load(f)
    f.close()

    print 'testing pca...'
    feature_test_pca = pca.transform(feature_test)
    feature_validation_pca = pca.transform(feature_validation)
    # feature_all = np.concatenate((feature_test_pca, feature_validation_pca), axis=0)
    # np.save(root1 + 'vote_multilabel_feature_training_pca.npy', feature_training_pca)
    np.save(root1 + 'pec'+soft + '_feature_test_pca.npy', feature_test_pca)
    np.save(root1 + 'pec'+soft + '_feature_validation_pca.npy', feature_validation_pca)


    with open(root1 + 'train_test.json') as f:
        event_dict = ujson.load(f)
    with open(root1 + 'pec_test_fc7_vote'+soft + '_multilabel_event_recognition_expand_balanced_3'+soft + '_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)
    with open(root1 + 'pec_training_fc7_vote'+soft + '_multilabel_event_recognition_expand_balanced_3'+soft + '_iter_100000_event_list.pkl') as f:
        validation_list = cPickle.load(f)
    lstm_test_feature_all = []
    lstm_test_label_all = []

    count = 0
    for event_id in test_list:
        event_type = event_id.split('/')[0]
        album = event_id.split('/')[1]
        event_count = len(event_dict[event_type][album])
        lstm_test_feature_all.append(range(count, count+event_count))
        if event_type in map_to_new:
            lstm_test_label_all.append(dict_name2[map_to_new[event_type]] - 1)
        else:
            lstm_test_label_all.append(22)
        count += event_count
        print count

    lstm_validation_label = [];lstm_validation_feature=[]
    count = 0
    for event_id in validation_list:
        event_type = event_id.split('/')[0]
        album = event_id.split('/')[1]
        event_count = len(event_dict[event_type][album])
        lstm_validation_feature.append(range(count, count+event_count))
        if event_type in map_to_new:
            lstm_validation_label.append(dict_name2[map_to_new[event_type]] - 1)
        else:
            lstm_validation_label.append(22)
        count += event_count
        print count
    # print count
    with open(root1 + 'pec_vote_multilabel'+soft + '_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)

    with open(root1 + 'pec_vote_multilabel'+soft + '_validation.pkl', 'w') as f:
        cPickle.dump((lstm_validation_feature, lstm_validation_label), f)

def preprocess_data_pec_usedict():
    root1 = '/home/ubuntu/lstm/data/'

    # feature_training = np.load(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    # print feature_training.shape

    f = open(root1 + 'pec_test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_feature_dict.pkl')
    feature_dict = cPickle.load(f)
    f.close()


    f = open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_PCA.pkl')
    pca = cPickle.load(f)
    f.close()
    print 'testing pca...'
    feature_test_pca = pca.transform(feature_test)
    # feature_all = np.concatenate((feature_training_pca, feature_test_pca), axis=0)

    # np.save(root1 + 'vote_multilabel_feature_training_pca.npy', feature_training_pca)
    np.save(root1 + 'pec_feature_test_pca.npy', feature_test_pca)
    # np.save(root1 + 'pec_feature_all.npy', feature_all)


    with open(root1 + 'test.json') as f:
        event_dict = ujson.load(f)
    with open(root1 + 'pec_test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)

    lstm_test_feature_all = []
    lstm_test_label_all = []

    count = 0
    for event_id in test_list:
        event_type = event_id.split('/')[0]
        album = event_id.split('/')[1]
        event_count = len(event_dict[event_type][album])
        lstm_test_feature_all.append(range(count, count+event_count))
        if event_type in map_to_new:
            lstm_test_label_all.append(dict_name2[map_to_new[event_type]] - 1)
        else:
            lstm_test_label_all.append(22)
        count += event_count
        print count
    # print count

    with open(root1 + 'pec_vote_multilabel_test.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)


def preprocess_data_pec_test():
    root1 = '/home/ubuntu/lstm/data/'

    with open(root1 + 'test.json') as f:
        event_dict = ujson.load(f)
    with open(root1 + 'pec_test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
        test_list = cPickle.load(f)

    lstm_test_feature_all = []
    lstm_test_label_all = []

    count = 0
    for event_id in test_list:
        event_type = event_id.split('/')[0]
        album = event_id.split('/')[1]
        event_count = len(event_dict[event_type][album])
        for i in range(event_count):
            lstm_test_feature_all.append([count])
            count += 1
            if event_type in map_to_new:
                lstm_test_label_all.append(dict_name2[map_to_new[event_type]] - 1)
            else:
                lstm_test_label_all.append(22)
        print count
    # print count

    with open(root1 + 'pec_vote_multilabel_test_TEST.pkl', 'w') as f:
        cPickle.dump((lstm_test_feature_all, lstm_test_label_all), f)


if __name__ == '__main__':
    # valid_preprocess_data_multilabel_balanced()
    # valid_preprocess_data_multilabel_softtarget()
    preprocess_data_pec_soft()
    # preprocess_data_multilabel()
    # preprocess_data_multilabel_balanced()
    # preprocess_data_multilabel_softtarget()
    # preprocess_data_multilabel_balanced()
    # list_ = ['multilabel_training_balanced', 'multilabel_training_balanced_weighted',
    #          'multilabel_training', 'multilabel_training_weighted',
    #          'multilabel_training_softtarget','multilabel_training_softtarget_weighted']
    # list_ = ['multilabel_training_balanced_old_weighted',
    #          'multilabel_training_old_weighted',
    #          'multilabel_training_softtarget_old_weighted']
    # for file_name in list_:
    #     oversample_training_img(file_name)

    # preprocess_data_multilabel_softtarget_crossvalidation()
    # preprocess_data_multilabel_balanced_crossvalidation()
    # preprocess_data_pec()
    # preprocess_data_pec_soft()

    # oversample_training_img('vote_multilabel_training_0')
    # oversample_training_img('vote_multilabel_training_1')
    # oversample_training_img('vote_multilabel_training_2')
    # oversample_training_img('vote_multilabel_training_3')
    # oversample_training_img('vote_multilabel_training_4')
    # # preprocess_data_pec_test()
    #
    # #
    #
    # root1 = '/home/ubuntu/lstm/data/'
    #
    # # feature_training = np.load(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    # # print feature_training.shape
    #
    # feature_test = np.load(root1 + 'test_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000.npy')
    # f = open(root1 + 'training_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_PCA.pkl')
    # pca = cPickle.load(f)
    # f.close()
    # feature_test_pca = pca.transform(feature_test)
    #
    # feature_test_pca1 = np.load(root1 + 'vote_multilabel_feature_test_pca.npy')
    # print feature_test_pca
    # print feature_test_pca1