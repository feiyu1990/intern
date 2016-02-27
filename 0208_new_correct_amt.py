import cPickle
import operator
import random
import os
import csv
from collections import defaultdict
from PIL import Image
import urllib2
import shutil

dict_name2 = {'ThemePark': 1, 'UrbanTrip': 2, 'BeachTrip': 3, 'NatureTrip': 4,
              'Zoo': 5, 'Cruise': 6, 'Show': 7,
              'Sports': 8, 'PersonalSports': 9, 'PersonalArtActivity': 10,
              'PersonalMusicActivity': 11, 'ReligiousActivity': 12,
              'GroupActivity': 13, 'CasualFamilyGather': 14,
              'BusinessActivity': 15, 'Architecture': 16, 'Wedding': 17,
              'Birthday': 18, 'Graduation': 19, 'Museum': 20, 'Christmas': 21,
              'Halloween': 22, 'Protest': 23}

dict_name = {'Theme park': 'ThemePark', 'Urban/City trip': 'UrbanTrip', 'Beach trip': 'BeachTrip',
             'Nature trip': 'NatureTrip', 'Zoo/Aquarium/Botanic garden': 'Zoo',
             'Cruise trip': 'Cruise', 'Show (air show/auto show/music show/fashion show/concert/parade etc.)': 'Show',
             'Sports game': 'Sports', 'Personal sports': 'PersonalSports', 'Personal art activities': 'PersonalArtActivity',
             'Personal music activities': 'PersonalMusicActivity', 'Religious activities': 'ReligiousActivity',
             'Group activities (party etc.)': 'GroupActivity', 'Casual family/friends gathering': 'CasualFamilyGather',
             'Business activity (conference/meeting/presentation etc.)': 'BusinessActivity',
             'Independence Day': 'Independence',
             'Wedding': 'Wedding', 'Birthday': 'Birthday', 'Graduation': 'Graduation', 'Museum': 'Museum',
             'Christmas': 'Christmas', 'Halloween': 'Halloween', 'Protest': 'Protest', 'Architecture/Art': 'Architecture'}

dict_name1 = dict([(dict_name[a], a) for a in dict_name])


def write_csv_curation():
    # event_to_add_curation = {'7_55455788@N00':'Birthday','144_95413346@N00':'Halloween', '1_21856707@N00': 'GroupActivity',
    #                          '0_22928590@N00':'GroupActivity', '14_93241698@N00':'Museum','32_35578067@N00':'Protest',
    #                          '3_60652642@N00':'ReligiousActivity','9_60053005@N00':'GroupActivity'}

    event_to_add_curation  = {'0_70073383@N00':'PersonalArtActivity'}
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_output.csv'
    head_meta = []
    HITs = {}
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        line_count = 0
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                # print meta[28]
                if meta[28] in event_to_add_curation:
                    HITs[meta[28]] = meta
            line_count += 1

    input_index = {}
    i = 0
    for field in head_meta:
        if field.startswith('Input'):
            input_index[field[6:]] = i
        i += 1

    root = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/'
    out_path = root + 'curation_amt_2.csv'
    f = open(out_path, 'w')
    str_towrite = ''
    for field in input_index:
        str_towrite += field + ','
    str_towrite = str_towrite[:-1] + '\n'
    f.write(str_towrite)

    for event in HITs:
        this_hit = HITs[event]
        str_towrite = ''
        for field in input_index:
            if field == 'event_type':
                str_towrite += dict_name1[event_to_add_curation[event]] + ','
            else:
                str_towrite += this_hit[input_index[field]] + ','
        str_towrite = str_towrite[:-1] + '\n'
        f.write(str_towrite)
    f.close()


def write_csv_small_sample():
    event_id_to_add = ['1_88464035@N00','0_7706183@N06','26_21186435@N00','56_74814994@N00','21_49503048699@N01',
                       '442_28004289@N03','2_36319742@N05','12_76384935@N00','4_15251430@N03','3_54218473@N05',
                       '0_40573754@N04','1_75003318@N00','3_26387956@N07','22_32994285@N00','4_53628484@N00','2_12882543@N00',
                       '15_66390637@N08','63_52304204@N00','211_86383385@N00','3_41773804@N08','15_54494252@N00',
                       '6_92248347@N00','40_56008930@N00','1_97889175@N00','15_23471940@N02','38_53628484@N00',
                       '52_32481334@N00','16_18108851@N00','23_89182227@N00','0_70073383@N00']


    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_output.csv'
    head_meta = []
    HITs = {}
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        line_count = 0
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                # print meta[28]
                if meta[28] in event_id_to_add:
                    # print meta[28]
                    HITs[meta[28]] = meta
            line_count += 1

    image_input_index = {}
    i = 0
    for field in head_meta:
        if field.startswith('Input.image'):
            image_input_index[int(field[11:])] = i
        i += 1

    events_and_image = defaultdict(list)

    index_num_image = 27
    index_event_id = 28
    index_distraction = 31
    for event_id in HITs:
        this_hit = HITs[event_id]
        num_images = int(this_hit[index_num_image])
        distract_image = this_hit[index_distraction]
        event_id = this_hit[index_event_id]
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        # events_and_image[event_id].append()
        for i in xrange(1, 1+num_images):
            if i==distract1 or i==distract2:
                continue
            events_and_image[event_id].append(this_hit[image_input_index[i]])

    print events_and_image




    root = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/'
    out_path = root + 'recognition_amt.csv'
    f = open(out_path, 'w')
    max_size = 200
    for i in xrange(1, 11):
        if i == 1:
            f.write('num_image'+str(i))#+',tags'+str(i))
        else:
            f.write(',num_image'+str(i))
        for ii in xrange(max_size):
                f.write(',')
                f.write('image'+str(i)+'_'+str(ii+1))
    # f.write('\n')
    count = 0
    for event in events_and_image:
        print event
        if count % 10 == 0:
            f.write('\n'+ str(len(events_and_image[event])))
        else:
            f.write(',' + str(len(events_and_image[event])))
        for img in events_and_image[event]:
            f.write(',' + img)
        for i in xrange(200 - len(events_and_image[event])):
            f.write(',NA')
        count += 1

    f.write('\n')
    f.close()


def write_csv():
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_output.csv'
    head_meta = []
    HITs = {}
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        line_count = 0
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                HITs[meta[28]] = meta
            line_count += 1

    image_input_index = {}
    i = 0
    for field in head_meta:
        if field.startswith('Input.image'):
            image_input_index[int(field[11:])] = i
        i += 1

    events_and_image = defaultdict(list)

    index_num_image = 27
    index_event_id = 28
    index_tag = 30
    index_distraction = 31
    for event_id in HITs:
        this_hit = HITs[event_id]
        num_images = int(this_hit[index_num_image])
        distract_image = this_hit[index_distraction]
        event_id = this_hit[index_event_id]
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        events_and_image[event_id].append(event_id)
        events_and_image[event_id].append(this_hit[index_tag])

        for i in xrange(1, 1+num_images):
            if i == distract1 or i == distract2:
                continue
            events_and_image[event_id].append(this_hit[image_input_index[i]])
    # print events_and_image

    root = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/'
    out_path = root + 'all_recognition_amt.csv'
    f = open(out_path, 'w')
    max_size = 200

    for i in xrange(1, 11):
        if i == 1:
            f.write('num_image' + str(i) + ',event_id_' + str(i) + ',tags_' + str(i))
        else:
            f.write(',num_image' + str(i) + ',event_id_' + str(i) + ',tags_' + str(i))
        for ii in xrange(max_size):
                f.write(',')
                f.write('image' + str(i) + '_' + str(ii + 1))
    # f.write('\n')
    count = 0
    print len(events_and_image)
    for event in events_and_image:
        # print event
        if count % 10 == 0:
            f.write('\n' + str(len(events_and_image[event])-2))
        else:
            f.write(',' + str(len(events_and_image[event])-2))
        for img in events_and_image[event]:
            f.write(',' + img)
        for i in xrange(202 - len(events_and_image[event])):
            f.write(',NA')
        count += 1

    f.write('\n')
    f.close()


def find_valid_img():
    # root = '/Users/wangyufei/Documents/Study/intern_adobe/'
    root = '/home/feiyu1990/local/event_curation/'
    input_path = root + '/all_output/all_output.csv'
    head_meta = []
    HITs = {}
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        line_count = 0
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                HITs[meta[28]] = meta
            line_count += 1

    image_input_index = {}
    i = 0
    for field in head_meta:
        if field.startswith('Input.image'):
            image_input_index[int(field[11:])] = i
        i += 1

    events_and_image = defaultdict(list)

    index_num_image = 27
    index_event_id = 28
    index_tag = 30
    index_distraction = 31
    bad_event_imgs = set()
    for event_id in HITs:
        # print event_id
        this_hit = HITs[event_id]
        num_images = int(this_hit[index_num_image])
        distract_image = this_hit[index_distraction]
        event_id = this_hit[index_event_id]
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        events_and_image[event_id].append(event_id)
        events_and_image[event_id].append(this_hit[index_tag])

        for i in xrange(1, 1+num_images):
            if i == distract1 or i == distract2:
                continue
            events_and_image[event_id].append(this_hit[image_input_index[i]])
            # if os.stat(this_hit[image_input_index[i]]).st_size < 4000:
            #     bad_event_id.append(event_id)
            try:
                    imgdata = urllib2.urlopen(this_hit[image_input_index[i]])
                    # print imgdata.info()['content-length']
                    if int(imgdata.info()['content-length'])< 4000:
                        bad_event_imgs.add(this_hit[image_input_index[i]])
                        print this_hit[image_input_index[i]]
            except:
                    print 'ERROR raised', event_id
                    bad_event_imgs.add(this_hit[image_input_index[i]])

    f = open(root + '0208_correction/url_notvalid_img_list.pkl','w')
    cPickle.dump(bad_event_imgs, f)
    f.close()

def write_csv_find_invalid_img():
    # root = '/Users/wangyufei/Documents/Study/intern_adobe/'
    root = '/home/feiyu1990/local/event_curation/'
    input_path = root + '/all_output/all_output.csv'
    head_meta = []
    HITs = {}
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        line_count = 0
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                HITs[meta[28]] = meta
            line_count += 1

    image_input_index = {}
    i = 0
    for field in head_meta:
        if field.startswith('Input.image'):
            image_input_index[int(field[11:])] = i
        i += 1

    events_and_image = defaultdict(list)

    index_num_image = 27
    index_event_id = 28
    index_tag = 30
    index_distraction = 31
    bad_event_id = set()
    for event_id in HITs:
        # print event_id
        this_hit = HITs[event_id]
        num_images = int(this_hit[index_num_image])
        distract_image = this_hit[index_distraction]
        event_id = this_hit[index_event_id]
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        events_and_image[event_id].append(event_id)
        events_and_image[event_id].append(this_hit[index_tag])

        for i in xrange(1, 1+num_images):
            if i == distract1 or i == distract2:
                continue
            events_and_image[event_id].append(this_hit[image_input_index[i]])
            # if os.stat(this_hit[image_input_index[i]]).st_size < 4000:
            #     bad_event_id.append(event_id)
            if event_id not in bad_event_id:
                try:
                    imgdata = urllib2.urlopen(this_hit[image_input_index[i]])
                    # print imgdata.info()['content-length']
                    if int(imgdata.info()['content-length'])< 4000:
                        bad_event_id.add(event_id)
                        print event_id
                except:
                    print 'ERROR raised', event_id
                    bad_event_id.add(event_id)

    f = open(root + '0208_correction/url_notvalid_event_id.pkl','w')
    cPickle.dump(bad_event_id, f)
    f.close()

    # print events_and_image

    root = root + '/0208_correction/'
    out_path = root + 'all_recognition_amt_flickr_notvalid.csv'
    f = open(out_path, 'w')
    max_size = 200

    for i in xrange(1, 11):
        if i == 1:
            f.write('num_image' + str(i) + ',event_id_' + str(i) + ',tags_' + str(i))
        else:
            f.write(',num_image' + str(i) + ',event_id_' + str(i) + ',tags_' + str(i))
        for ii in xrange(max_size):
                f.write(',')
                f.write('image' + str(i) + '_' + str(ii + 1))
    # f.write('\n')
    count = 0
    print len(events_and_image)
    for event in events_and_image:
        if event not in bad_event_id:
            continue
        # print event
        if count % 10 == 0:
            f.write('\n' + str(len(events_and_image[event])-2))
        else:
            f.write(',' + str(len(events_and_image[event])-2))
        for img in events_and_image[event]:
            f.write(',' + img)
        for i in xrange(202 - len(events_and_image[event])):
            f.write(',NA')
        count += 1

    f.write('\n')
    f.close()

def need_download_img():
    root = '/home/feiyu1990/local/event_curation/'
    img_dict = dict()
    for event in dict_name2:
        with open(root + 'baseline_all_0509/' + event + '/training_ulr_dict.cPickle') as f:
            temp = cPickle.load(f)
        img_dict.update(dict([(temp[key], root + 'curation_images/' + event + '/' + key.split('_')[1] + '.jpg') for key in temp]))
        with open(root + 'baseline_all_0509/' + event + '/test_ulr_dict.cPickle') as f:
            temp = cPickle.load(f)
        img_dict.update(dict([(temp[key], root + 'curation_images/' + event + '/' + key.split('_')[1] + '.jpg') for key in temp]))
    print len(img_dict)
    with open(root + '0208_correction/img_url_dict', 'w') as f:
        cPickle.dump(img_dict, f)

    new_img_path = '/home/feiyu1990/local/event_curation/0208_correction/invalid_images/'
    with open(root + '0208_correction/url_notvalid_img_list.pkl') as f:
        invalid_img = cPickle.load(f)
    for img in invalid_img:
        img_old_path = img_dict[img]
        shutil.copy(img_old_path, new_img_path + img_old_path.split('/')[-1])


def write_csv_correct_link():
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_output.csv'
    root = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/'


    with open(root + 'url_notvalid_img_list.pkl') as f:
        need_correct_img = cPickle.load(f)
    with open(root + 'img_url_dict') as f:
        dict_ = cPickle.load(f)
    need_correct_img_dict = dict()
    for i in need_correct_img:
        need_correct_img_dict[i] = dict_[i].split('/')[-1]


    head_meta = []
    HITs = {}
    with open(input_path, 'r') as data:
        reader = csv.reader(data)
        line_count = 0
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                HITs[meta[28]] = meta
            line_count += 1

    image_input_index = {}
    i = 0
    for field in head_meta:
        if field.startswith('Input.image'):
            image_input_index[int(field[11:])] = i
        i += 1

    events_and_image = defaultdict(list)

    index_num_image = 27
    index_event_id = 28
    index_tag = 30
    index_distraction = 31
    for event_id in HITs:
        this_hit = HITs[event_id]
        num_images = int(this_hit[index_num_image])
        distract_image = this_hit[index_distraction]
        event_id = this_hit[index_event_id]
        [distract1, distract2] = distract_image.split(':')
        distract1 = int(distract1)
        distract2 = int(distract2)
        events_and_image[event_id].append(event_id)
        events_and_image[event_id].append(this_hit[index_tag])

        for i in xrange(1, 1+num_images):
            if i == distract1 or i == distract2:
                continue
            img_url = this_hit[image_input_index[i]]
            if img_url in need_correct_img_dict:
                events_and_image[event_id].append('http://acsweb.ucsd.edu/~yuw176/invalid_images/' + need_correct_img_dict[img_url])
            else:
                events_and_image[event_id].append(img_url)
    # print events_and_image

    out_path = root + 'all_recognition_amt_urlcorrected.csv'
    f = open(out_path, 'w')
    max_size = 200

    for i in xrange(1, 11):
        if i == 1:
            f.write('num_image' + str(i) + ',event_id_' + str(i) + ',tags_' + str(i))
        else:
            f.write(',num_image' + str(i) + ',event_id_' + str(i) + ',tags_' + str(i))
        for ii in xrange(max_size):
                f.write(',')
                f.write('image' + str(i) + '_' + str(ii + 1))
    # f.write('\n')
    count = 0
    print len(events_and_image)
    for event in events_and_image:
        # print event
        if count % 10 == 0:
            f.write('\n' + str(len(events_and_image[event])-2))
        else:
            f.write(',' + str(len(events_and_image[event])-2))
        for img in events_and_image[event]:
            f.write(',' + img)
        for i in xrange(202 - len(events_and_image[event])):
            f.write(',NA')
        count += 1

    f.write('\n')
    f.close()


if __name__ == '__main__':
    # write_csv()
    # write_csv_curation()
    #  write_csv_curation()
    # write_csv_find_invalid_img()
    # find_valid_img()
    # need_download_img()
    write_csv_correct_link()
