import cPickle
import operator
import random
import os
import csv
from collections import defaultdict, Counter
from PIL import Image
import urllib2
import shutil
import numpy as np
from operator import itemgetter

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


def write_csv_correct_link_3round():
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/all_output/all_output.csv'
    root = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/'
    # event_names = {'10_71486434@N00', '1_44124452748@N01', '10_8909796@N07', '1_50510658@N04', '84_25965014@N00', '35_30952578@N00',
    #                '3_16976034@N00', '16_95601478@N00', '78_40094880@N06', '6_7826272@N06', '25_37403827@N00', '0_55737440@N02',
    #                '5_54618101@N00', '6_83555001@N00', '5_78969707@N00', '0_78174327@N00', '5_32991505@N00', '12_78147607@N00',
    #                '0_28004076@N04', '0_43208246@N00', '24_97863854@N00', '124_13257277@N00', '2_10351901@N00', '1_36030443@N06',
    #                '10_85473133@N00', '35_53746192@N00', '7_81614435@N00', '12_51035598215@N01', '7_28004076@N04', '17_95601478@N00',
    #                '0_53826674@N05', '25_32628129@N04', '15_92186477@N00', '27_82418181@N00', '0_21757951@N00', '1_48889113645@N01',
    #                '9_34993101@N00', '0_25425455@N04', '12_8743691@N02', '9_16569662@N00', '5_54084941@N00', '55_68457656@N00',
    #                '0_37718678739@N01', '20_46971098@N00', '11_19835999@N00', '5_63586135@N00', '3_41838028@N00', '20_90585585@N00',
    #                '5_7826272@N06', '7_33507204@N00', '34_90908304@N00', '0_55195844@N05', '71_14678786@N00', '17_44124461706@N01',
    #                '0_17956027@N00', '0_23451880@N04', '9_37718678739@N01', '88_99357189@N00', '13_85121963@N00', '112_10011817@N00',
    #                '1_61762156@N00', '11_28817993@N00', '124_7702423@N04', '12_30952578@N00', '0_30613195@N00', '3_74628614@N00',
    #                '8_83838608@N00', '6_29753028@N00', '9_52725445@N00', '59_97863854@N00', '0_23619180@N00', '16_50517642@N00',
    #                '3_33415362@N06', '8_79034573@N00', '44_30952578@N00', '7_72918555@N00', '2_33659625@N00', '6_23244282@N00',
    #                '2_29204155@N08', '2_32234946@N00', '67_40094880@N06', '1_8643216@N02', '0_31998658@N06', '27_60756254@N07',
    #                '13_17868205@N00', '15_44124461706@N01', '34_62929416@N00', '29_7877597@N08', '3_63848257@N06', '25_88483799@N00',
    #                '11_91605789@N00', '92_14678786@N00', '0_47657931@N00', '12_99706198@N00', '1_49503181455@N01', '19_63262340@N00',
    #                '2_80185309@N00', '5_54634670@N03', '1_49231590@N07', '7_54788523@N00', '4_8685194@N06', '2_32974793@N00',
    #                '10_27813593@N00', '9_60053005@N00', '76_88607481@N00', '8_66969128@N00', '3_60311165@N04', '40_7471115@N08',
    #                '3_81103342@N00', '25_99145208@N00', '55_8386148@N06', '29_21186435@N00', '0_52046304@N00', '32_7614607@N05',
    #                '1_22484455@N02', '4_82386510@N00', '1_89225742@N00', '0_38943965@N05', '47_71401718@N00', '0_43453795@N00'}
    event_names = {'1_89803116@N00', '12_7273032@N04', '78_40094880@N06', '5_32991505@N00', '108_7339259@N08', '3_33415362@N06',
                   '109_32806640@N00', '1_74135127@N00', '6_22298070@N00', '4_90514086@N00', '6_16501075@N00', '21_40305917@N00',
                   '4_12732700@N02', '103_99357189@N00', '10_41894151021@N01', '58_8743691@N02', '3_54084941@N00', '44_8743691@N02',
                   '49_18412989@N00', '4_53746192@N00'}

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
        if event_id not in event_names:
            continue
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

    out_path = root + 'all_input_and_result/all_recognition_4round_for_me.csv'
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


def write_csv_correct_link_2round():
    # event_names = {'39_66922282@N00','0_27663842@N00','32_7988353@N04','8_75945450@N00','7_87774367@N00',
    #              '33_7488461@N05','2_24250990@N04','0_9137715@N05','0_77343377@N00','8_34286231@N00',
    #                '32_52336189@N00','14_84764695@N04','341_86383385@N00','439_86383385@N00','2_24250990@N04',
    #                '14_84764695@N04','10_40305917@N00','20_86383385@N00','55_38345529@N05','143_24576031@N04',
    #                '32_35578067@N00','13_34993101@N00'}
    event_names = {'0_47383196@N00','0_75404268@N00','2_33237963@N00','3_69754957@N00','23_22530195@N05','0_40215530@N00',
                   '87_7702423@N04','0_27663842@N00','0_97786161@N00','3_7768124@N08','414_44124466908@N01','3_81103342@N00',
                   '27_79172203@N00','24_12693966@N07','3_42619839@N00','3_27354738@N00','7_87774367@N00','8_34286231@N00',
                   '67_57810730@N03','38_53628484@N00'}
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
        if event_id not in event_names:
            continue
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

    out_path = root + 'all_recognition_amt_urlcorrected_3round.csv'
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


def read_recognition_result_multiple():
    input_path = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/result_first200.csv'
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []

    out_index_morethanone = []
    output_index = []
    input_index = []
    tag_index = []
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            if meta.endswith('_morethan1'):
                out_index_morethanone.append(i)
            else:
                output_index.append(i)
        if 'event_id' in meta:
            input_index.append(i)
        if 'tag' in meta:
            tag_index.append(i)
    #feed_back_index = output_index[0]
    #output_index = output_index[1:]
    output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
    output_index = output_index_new
    HIT_result = []
    for meta in metas:
        HITids.append(meta[0])
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k]] for (j,i,k) in zip(input_index, output_index,tag_index)]])

    HIT_result_dic = {}
    for meta in HIT_result:
        HIT = meta[0]
        result = meta[1]
        if HIT in HIT_result_dic:
            HIT_result_dic[HIT].append(result)
        else:
            HIT_result_dic[HIT] = [result]
    count = 0
    agreement_event_id = []

    ground_truth = dict()


    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        if len(results) > 1:
            count+=1
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            for result in results:
                temp = result[i][2].split('|')
                for result_multi in temp:
                    if result_multi in result_this:
                        result_this[result_multi] += 1
                    else:
                        result_this[result_multi] = 1
            ground_truth[results[0][i][1]] = sorted(result_this.iteritems(), key=operator.itemgetter(1), reverse=True)
            # ground_truth[results[0][i][1]] = [ii for ii in ground_truth[results[0][i][1]]  if ii[1] > 1]
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            #if max_result[1] >= 2 and max_result[1] > len_vote/2:
            # if max_result[1] == len(result_this) and max_result[1] > len_vote/2:
            if max_result[1] > len_vote * 2 / 3:
                agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
    print "number of events with more than 1 vote:", count*10
    print "number of events with more than 2 same votes:",len(agreement_event_id)

    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/result_first200_allvotes.pkl', 'w') as f:
        cPickle.dump(ground_truth, f)

    return agreement_event_id


def softmax(groundtruth_list , threshold=0.3):
    groundtruth_list = [i for i in groundtruth_list if i[0] in dict_name2 and i[1] > 1]
    # groundtruth_list = [i for i in groundtruth_list if i[0] in dict_name2]
    w = [i[1] for i in groundtruth_list]
    t = np.sum(w) / 3
    # t = 5
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    # if len(dist) == 0:
    #     print 'HI!'
    ind = np.where([i > threshold for i in dist])[0]
    if len(ind) < 1:
        num = np.max(dist)
        ind = np.where([i == num for i in dist])[0]
    # print [(groundtruth_list[i][0], dist[i]) for i in ind]
    result_and_prob = sorted([(groundtruth_list[i][0], dist[i]) for i in ind], key=itemgetter(1), reverse=True)

    return result_and_prob

def softmax_all(groundtruth_list , threshold=0):
    groundtruth_list = [i for i in groundtruth_list if i[0] in dict_name2 and i[1] > 1]
    # groundtruth_list = [i for i in groundtruth_list if i[0] in dict_name2]
    w = [i[1] for i in groundtruth_list]
    t = np.sum(w) / 3
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    # if len(dist) == 0:
    #     print 'HI!'
    ind = np.where([i > threshold for i in dist])[0]
    # print [(groundtruth_list[i][0], dist[i]) for i in ind]
    result_and_prob = sorted([(groundtruth_list[i][0], dist[i]) for i in ind], key=itemgetter(1), reverse=True)
    return result_and_prob

def softmax_weighted(groundtruth_list , threshold=0.3):
    groundtruth_list = [i for i in groundtruth_list if i[0] in dict_name2 and i[1] > 0.8]
    w = [i[1] for i in groundtruth_list]
    t = np.sum(w) / 3
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    ind = np.where([i > threshold for i in dist])[0]
    if len(ind) < 1:
        # num = np.max(dist)
        ind = np.where([i >= float(1) / len(dist) for i in dist])[0]
    result_and_prob = sorted([(groundtruth_list[i][0], dist[i]) for i in ind], key=itemgetter(1), reverse=True)

    return result_and_prob


def vote(groundtruth_list, threshold=0.3):
    dict_result = dict()
    for worker in groundtruth_list:
        temp = groundtruth_list[worker]
        len_ = len(temp)
        for i in temp:
            if i in dict_result:
                dict_result[i] += float(1) / len_
            else:
                dict_result[i] = float(1) / len_
    dict_result_new = dict()
    for i in dict_result:
        if dict_result[i] > 1:
            dict_result_new[i] = dict_result[i]

    sum_ = np.sum([dict_result_new[i] for i in dict_result_new])
    event_type, dist = zip(*[(i, dict_result_new[i] / sum_) for i in dict_result_new])
    ind = np.where([i > threshold for i in dist])[0]

    if len(ind) < 1:
        # num = np.max(dist)
        ind = np.where([i > float(1) / len(dist) for i in dist])[0]
        print dist, ind
    result_and_prob = sorted([(event_type[i], dist[i]) for i in ind], key=itemgetter(1), reverse=True)

    return result_and_prob


def combine_from_sources():
    list_csv = [
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_rejected_redo.csv',
        'result_rejected_redo_2round.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50_part.csv'
    ]

    groundtruth_old = dict()
    old_list = ['0.csv','1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8_pre.csv']
    for file in old_list:
        temp, temp_posttag = read_recognition_result_multiple_seetag(file, '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
        for event in temp:
            if event in correct_list:
                groundtruth_old[event] = [(correct_list[event], 3)]
            else:
                groundtruth_old[event] = [i for i in temp[event] if 'NOT' not in i[0]]
        print len(temp), len(groundtruth_old)


    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict_old = cPickle.load(f)
    corrected_event_type_dict_old = dict()
    for event in event_type_dict_old:
        if event in correct_list:
            corrected_event_type_dict_old[event] = correct_list[event]
        else:
            corrected_event_type_dict_old[event] = event_type_dict_old[event]

    for event in corrected_event_type_dict_old:
        if event not in groundtruth_old:
            continue
        if corrected_event_type_dict_old[event] != groundtruth_old[event][0][0]:
            groundtruth_old.pop(event, None)
            # print groundtruth_old[event], corrected_event_type_dict_old[event]
            # groundtruth_old[event] = [(corrected_event_type_dict_old[event], 3)]


    groundtruth = dict()
    groundtruth_posttag = dict()
    for file in list_csv:
        temp, temp_posttag = read_recognition_result_multiple_seetag(file)
        for event in temp:
            if event not in groundtruth:
                groundtruth[event] = temp[event]
                groundtruth_posttag[event] = temp_posttag[event]
            else:
                this_list = groundtruth[event]
                add_list = temp[event]
                count_list = Counter(dict(this_list))
                for event_type in add_list:
                    count_list[event_type[0]] += event_type[1]
                temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth[event] = temp_list

                this_list = groundtruth_posttag[event]
                add_list = temp_posttag[event]
                count_list = Counter(dict(this_list))
                for event_type in add_list:
                    count_list[event_type[0]] += event_type[1]
                temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth_posttag[event] = temp_list
    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_first200.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(no_posttag_csv)
    for event in temp:
        if event in groundtruth_posttag and groundtruth_posttag[event][0][1] != 1:
            continue
        groundtruth_posttag[event] = temp[event]
        groundtruth[event] = temp[event]
    print len(groundtruth)

    no_posttag_csv = 'result_2round.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(no_posttag_csv)
    for event in temp:
        count_old = Counter(dict(groundtruth[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth[event] = temp[event]
        # groundtruth[event] = temp_list

        count_old = Counter(dict(groundtruth_posttag[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        # groundtruth_posttag[event] = temp_list
        groundtruth_posttag[event] = temp[event]


    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_3round.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(no_posttag_csv)
    for event in temp:
        count_old = Counter(dict(groundtruth[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        # groundtruth[event] = temp[event]
        groundtruth[event] = temp_list

        count_old = Counter(dict(groundtruth_posttag[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth_posttag[event] = temp_list
        # groundtruth_posttag[event] = temp[event]

    use_groundtruth = dict()
    for event in groundtruth:
        temp = Counter(dict(groundtruth[event]))
        temp_this = softmax(groundtruth_posttag[event])
        list = ['Birthday', 'Wedding', 'Graduation', 'Protest', 'Themepark', 'ReligiousActivity']
        for i in list:
            if groundtruth[event][0][0] != i and i in [i[0] for i in temp_this]:
                temp = Counter(dict(groundtruth_posttag[event]))
                break
        list = ['Christmas', 'Halloween']
        for i in list:
            if groundtruth[event][0][0] != i and i in [i[0] for i in temp_this]:
                temp.update(Counter(dict(groundtruth_posttag[event])))
                break

        # n = temp.most_common(1)[0][1]
        # event_list = []
        # for i in temp:
        #     if temp[i] == n:
        #         event_list.append(i)
        # if event == '27_79172203@N00':
        #     print 'HI!'
        if event in groundtruth_old:# and groundtruth_old[event][0][0] not in event_list:
            temp.update(Counter(dict(groundtruth_old[event])))
        use_groundtruth[event] = sorted(temp.iteritems(), key=operator.itemgetter(1), reverse=True)
        # if event == '1_53436302@N02':
        #     print use_groundtruth[event]
    for event in use_groundtruth:
        use_groundtruth[event] = softmax(use_groundtruth[event])

    count = 0
    # for event_id in event_type_dict_old:
    #     if event_id in correct_list:
    #         event_type_dict_old[event_id] = correct_list[event_id]

    for event_type in use_groundtruth:
        if len(use_groundtruth[event_type]) > 2:
            count += 1
            print event_type, use_groundtruth[event_type], groundtruth[event_type], groundtruth_posttag[event_type]#, event_type_old
    print count
    #
    # for event_type in use_groundtruth:
    #     if event_type not in corrected_event_type_dict_old:
    #         print event_type
    #         continue
    #     event_type_old = corrected_event_type_dict_old[event_type]
    #
    #     if event_type_old in [i[0] for i in use_groundtruth[event_type]]:
    #         continue
    #     print event_type, use_groundtruth[event_type], groundtruth[event_type], groundtruth_posttag[event_type], event_type_old
    #     shutil.copy('/Users/wangyufei/Documents/Study/intern_adobe/present_htmls_new/' + event_type_dict_old[event_type] + '/' + event_type + '/present_groundtruth.html',
    #                 '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/htmls/' + event_type_old + '->' + '+'.join([i[0] for i in use_groundtruth[event_type]]) + '_' + event_type + '.html')
    #     count += 1
    # print count

def create_html_need_check():
    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict_old = cPickle.load(f)
    corrected_event_type_dict_old = dict()
    for event in event_type_dict_old:
        if event in correct_list:
            corrected_event_type_dict_old[event] = correct_list[event]
        else:
            corrected_event_type_dict_old[event] = event_type_dict_old[event]


    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_weighted.pkl') as f:
        use_groundtruth_weighted = cPickle.load(f)

    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_new.pkl') as f:
        use_groundtruth = cPickle.load(f)
    count = 0

    for event_type in use_groundtruth:
        if event_type not in corrected_event_type_dict_old:
            print event_type
            continue
        event_type_old = corrected_event_type_dict_old[event_type]
        # if len(use_groundtruth[event_type]) <= 2:
        #     continue
        # if event_type_old in [i[0] for i in use_groundtruth[event_type]]:
        #     continue
        no_weight_top = [i[0] for i in use_groundtruth[event_type] if i[1] == np.max([ii[1] for ii in use_groundtruth[event_type]])]
        if use_groundtruth_weighted[event_type][0][0] in no_weight_top:
            continue
        print event_type, use_groundtruth[event_type],use_groundtruth_weighted[event_type], event_type_old
        shutil.copy('/Users/wangyufei/Documents/Study/intern_adobe/present_htmls_new/' + event_type_dict_old[event_type] + '/' + event_type + '/present_groundtruth.html',
                    '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/htmls/' + event_type_old + '->' + '+'.join([i[0] for i in use_groundtruth_weighted[event_type]]) + '_' + event_type + '.html')
        count += 1
    print count

def write_csv_curation_2round():
    event_to_add_curation = {'68_9752912@N05':['Zoo', 'NatureTrip', 'UrbanTrip'],
                             '25_8123170@N06':['Show', 'BusinessActivity'],
                             '30_12873985@N00': ['UrbanTrip','Architecture'],
                             '127_60258967@N00':['NatureTrip','BeachTrip'],
                             '94_43162195@N00':['UrbanTrip'],
                             '61_30335727@N00':['Halloween','PersonalArtActivity'],
                             '414_44124466908@N01':['Christmas','Protest'],
                             '27_17616316@N00':['UrbanTrip','Architecture'],
                             '24_12693966@N07':['UrbanTrip', 'ThemePark'],
                             '0_10779871@N00':['NatureTrip','UrbanTrip']}

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
    out_path = root + 'curation_amt_2round.csv'
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
                str_towrite += '; '.join([dict_name1[i] for i in event_to_add_curation[event]]) + ','
            else:
                str_towrite += this_hit[input_index[field]] + ','
        str_towrite = str_towrite[:-1] + '\n'
        f.write(str_towrite)
    f.close()


def find_reject():
    rejected_already_labeled = []
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/result_rejected_redo.csv') as f:
        reader = csv.reader(f)
        for meta in reader:
            rejected_already_labeled.append(meta[28])
    rejected_index = []; list_worker = []
    list_csv = [
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_all_recognition_amt_urlcorrected_first200.csv'
    ]
    for name in list_csv:
        input_path = '/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/' + name
        line_count = 0
        head_meta = []
        metas = []
        with open(input_path, 'rb') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    head_meta = meta
                else:
                    metas.append(meta)
                line_count += 1
        HITids = []
        out_index_morethanone = [0] * 10
        output_index = [0] * 10
        out_index_morethanone_posttag = [0] * 10
        output_index_posttag = [0] * 10
        input_index = [0] * 10
        tag_index = [0] * 10
        worker_index = 0
        Accept_De_index = 0
        for i in xrange(len(head_meta)):
            meta = head_meta[i]
            if 'Answer.type' in meta:
                if meta.endswith('_morethan1'):
                    if 'posttag' in meta:
                        out_index_morethanone_posttag[int(meta.split('_')[0][11:]) - 1] = i
                    else:
                        out_index_morethanone[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    if 'posttag' in meta:
                        output_index_posttag[int(meta.split('_')[0][11:]) - 1] = i
                    else:
                        output_index[int(meta.split('_')[0][11:]) - 1] = i
            if 'event_id' in meta:
                input_index[int(meta.split('_')[-1]) - 1] = i
            if 'tags' in meta:
                tag_index[int(meta.split('_')[-1]) - 1] = i
            if 'WorkerId' == meta:
                worker_index = i
            if 'AssignmentStatus' == meta:
                Accept_De_index = i

        HIT_result = []
        for meta in metas:
            if meta[Accept_De_index] != 'Approved':
                rejected_index.append(meta[input_index[0]])
            HITids.append(meta[0])
            list_worker.append(meta[worker_index])
            HIT_result.append([meta[0], [[meta[j], meta[i], meta[jj], meta[k]]
                                         for (j, i, k, jj) in zip(input_index, output_index,tag_index, output_index_posttag)]])
        for i in HIT_result:
            for j in i[1]:
                if j[2] == '':
                    j[2] = j[1]
    line_count = 0
    f = open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/rejected_redo_2round.csv', 'w')
    writer = csv.writer(f)
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_recognition_amt_urlcorrected.csv') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
                writer.writerow(head_meta)
            else:
                if (meta[1] in rejected_index and meta[1] not in rejected_already_labeled):
                    writer.writerow(meta)
            line_count += 1
    f.close()

def read_recognition_result_multiple_seetag(name, worker_weight_dict, root='/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/'):
    input_path = root + name
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []
    out_index_morethanone = [-1] * 10
    output_index = [-1] * 10
    out_index_morethanone_posttag = [-1] * 10
    output_index_posttag = [-1] * 10
    input_index = [-1] * 10
    tag_index = [-1] * 10
    worker_index = -1
    Accept_De_index = -1
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            if meta.endswith('_morethan1'):
                if 'posttag' in meta:
                    out_index_morethanone_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    out_index_morethanone[int(meta.split('_')[0][11:]) - 1] = i
            else:
                if 'posttag' in meta:
                    output_index_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    output_index[int(meta.split('_')[0][11:]) - 1] = i
        if 'event_id' in meta:
            try:
                input_index[int(meta.split('_')[-1]) - 1] = i
            except:
                input_index[int(meta.split('_id')[-1]) - 1] = i
        if 'tags' in meta:
            try:
                tag_index[int(meta.split('_')[-1]) - 1] = i
            except:
                tag_index[int(meta.split('tags')[-1]) - 1] = i
        if 'WorkerId' == meta:
            worker_index = i
        if 'AssignmentStatus' == meta:
            Accept_De_index = i

    if output_index_posttag[0] == -1:
        output_index_posttag = output_index
        out_index_morethanone_posttag = out_index_morethanone
    HIT_result = []
    rejected_index = []
    for meta in metas:
        if meta[Accept_De_index] == 'Rejected':
            rejected_index.append(meta[input_index[0]])
        else:
            HITids.append(meta[0])
            HIT_result.append([meta[0], [[meta[j], meta[i], meta[jj], meta[k]]
                                         for (j, i, k, jj) in zip(input_index, output_index,tag_index, output_index_posttag)]])
    for i in HIT_result:
        for j in i[1]:
            if j[2] == '':
                j[2] = j[1]

    # print count
    # print len(HIT_result) * 10
    # print rejected_index

    HIT_result_dic = {}
    for meta in HIT_result:
        HIT = meta[0]
        result = meta[1]
        if HIT in HIT_result_dic:
            HIT_result_dic[HIT].append(result)
        else:
            HIT_result_dic[HIT] = [result]
    agreement_event_id = []
    ground_truth = dict()
    ground_truth_posttag = dict()


    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = defaultdict(int)
            result_this_posttag = defaultdict(int)
            for result in results:
                temp = result[i][1].split('|')
                temp_posttag = result[i][2].split('|')
                for result_multi in temp:
                    if result_multi in result_this:
                        result_this[result_multi] += 1
                    else:
                        result_this[result_multi] = 1
                for result_multi in temp_posttag:
                    if result_multi in result_this_posttag:
                        result_this_posttag[result_multi] += 1
                    else:
                        result_this_posttag[result_multi] = 1
            if results[0][i][0] in ground_truth:
                print results[0][i][0]
                old_ground_truth_ = ground_truth[results[0][i][0]]
                for ii in old_ground_truth_:
                    result_this[ii[0]] += ii[1]
                old_ground_truth_ = ground_truth_posttag[results[0][i][0]]
                for ii in old_ground_truth_:
                    result_this_posttag[ii[0]] += ii[1]
            ground_truth_posttag[results[0][i][0]] = sorted(result_this_posttag.iteritems(), key=operator.itemgetter(1), reverse=True)
            ground_truth[results[0][i][0]] = sorted(result_this.iteritems(), key=operator.itemgetter(1), reverse=True)

    return ground_truth, ground_truth_posttag
def read_recognition_result_multiple_seetag_worker(name, root='/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/'):
    input_path = root + name
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []
    out_index_morethanone = [-1] * 10
    output_index = [-1] * 10
    out_index_morethanone_posttag = [-1] * 10
    output_index_posttag = [-1] * 10
    input_index = [-1] * 10
    tag_index = [-1] * 10
    worker_index = -1
    Accept_De_index = -1
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            if meta.endswith('_morethan1'):
                if 'posttag' in meta:
                    out_index_morethanone_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    out_index_morethanone[int(meta.split('_')[0][11:]) - 1] = i
            else:
                if 'posttag' in meta:
                    output_index_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    output_index[int(meta.split('_')[0][11:]) - 1] = i
        if 'event_id' in meta:
            try:
                input_index[int(meta.split('_')[-1]) - 1] = i
            except:
                input_index[int(meta.split('_id')[-1]) - 1] = i
        if 'tags' in meta:
            try:
                tag_index[int(meta.split('_')[-1]) - 1] = i
            except:
                tag_index[int(meta.split('tags')[-1]) - 1] = i
        if 'WorkerId' == meta:
            worker_index = i
        if 'AssignmentStatus' == meta:
            Accept_De_index = i

    if output_index_posttag[0] == -1:
        output_index_posttag = output_index
        out_index_morethanone_posttag = out_index_morethanone
    HIT_result = []
    rejected_index = []
    for meta in metas:
        if meta[Accept_De_index] == 'Rejected':
            rejected_index.append(meta[input_index[0]])
        else:
            HITids.append(meta[0])
            HIT_result.append([meta[0], [[meta[j], meta[i], meta[jj], meta[k], meta[worker_index]]
                                         for (j, i, k, jj) in zip(input_index, output_index,tag_index, output_index_posttag)]])
    for i in HIT_result:
        for j in i[1]:
            if j[2] == '':
                j[2] = j[1]


    worker_result_dic = defaultdict(dict)
    for meta in HIT_result:
        result = meta[1]
        for i in result:
            event_id = i[0]
            output = i[1]
            output_posttag = i[2]
            worker_id = i[4]
            worker_result_dic[worker_id][event_id] = (output, output_posttag)
    return worker_result_dic
def read_recognition_result_multiple_seetag_weighted(name, worker_weight_dict, root='/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/'):
    input_path = root + name
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []
    out_index_morethanone = [-1] * 10
    output_index = [-1] * 10
    out_index_morethanone_posttag = [-1] * 10
    output_index_posttag = [-1] * 10
    input_index = [-1] * 10
    tag_index = [-1] * 10
    worker_index = -1
    Accept_De_index = -1
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            if meta.endswith('_morethan1'):
                if 'posttag' in meta:
                    out_index_morethanone_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    out_index_morethanone[int(meta.split('_')[0][11:]) - 1] = i
            else:
                if 'posttag' in meta:
                    output_index_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    output_index[int(meta.split('_')[0][11:]) - 1] = i
        if 'event_id' in meta:
            try:
                input_index[int(meta.split('_')[-1]) - 1] = i
            except:
                input_index[int(meta.split('_id')[-1]) - 1] = i
        if 'tags' in meta:
            try:
                tag_index[int(meta.split('_')[-1]) - 1] = i
            except:
                tag_index[int(meta.split('tags')[-1]) - 1] = i
        if 'WorkerId' == meta:
            worker_index = i
        if 'AssignmentStatus' == meta:
            Accept_De_index = i

    if output_index_posttag[0] == -1:
        output_index_posttag = output_index
        out_index_morethanone_posttag = out_index_morethanone
    HIT_result = []
    rejected_index = []
    for meta in metas:
        if meta[Accept_De_index] == 'Rejected':
            rejected_index.append(meta[input_index[0]])
        else:
            HITids.append(meta[0])
            HIT_result.append([meta[0], [[meta[j], meta[i], meta[jj], meta[k], meta[worker_index]]
                                         for (j, i, k, jj) in zip(input_index, output_index,tag_index, output_index_posttag)]])
    for i in HIT_result:
        for j in i[1]:
            if j[2] == '':
                j[2] = j[1]

    # print count
    # print len(HIT_result) * 10
    # print rejected_index

    HIT_result_dic = {}
    for meta in HIT_result:
        HIT = meta[0]
        result = meta[1]
        if HIT in HIT_result_dic:
            HIT_result_dic[HIT].append(result)
        else:
            HIT_result_dic[HIT] = [result]
    agreement_event_id = []
    ground_truth = dict()
    ground_truth_posttag = dict()


    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = defaultdict(int)
            result_this_posttag = defaultdict(int)
            for result in results:
                temp = result[i][1].split('|')
                temp_posttag = result[i][2].split('|')
                try:
                    worker_weight = worker_weight_dict[result[i][-1]]
                except:
                    worker_weight = -np.Inf
                for result_multi in temp:
                    if result_multi in result_this:
                        result_this[result_multi] += worker_weight
                    else:
                        result_this[result_multi] = worker_weight
                for result_multi in temp_posttag:
                    if result_multi in result_this_posttag:
                        result_this_posttag[result_multi] += worker_weight
                    else:
                        result_this_posttag[result_multi] = worker_weight
            if results[0][i][0] in ground_truth:
                print results[0][i][0]
                old_ground_truth_ = ground_truth[results[0][i][0]]
                for ii in old_ground_truth_:
                    result_this[ii[0]] += ii[1]
                old_ground_truth_ = ground_truth_posttag[results[0][i][0]]
                for ii in old_ground_truth_:
                    result_this_posttag[ii[0]] += ii[1]
            ground_truth_posttag[results[0][i][0]] = sorted(result_this_posttag.iteritems(), key=operator.itemgetter(1), reverse=True)
            ground_truth[results[0][i][0]] = sorted(result_this.iteritems(), key=operator.itemgetter(1), reverse=True)

    return ground_truth, ground_truth_posttag



def create_weighted_worker_groundtruth_new():
    list_csv = [
        # '0.csv','1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8_pre.csv',
        'result_all_recognition_amt_urlcorrected_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50_part.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_rejected_redo.csv',
        'result_rejected_redo_2round.csv',
        'result_2round.csv',
        'result_all_recognition_amt_urlcorrected_3round.csv',
        'result_all_recognition_3round_for_me.csv',
        'result_all_recognition_4round_for_me.csv',
        'result_all_recognition_4round_for_worker.csv'
    ]
    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict = cPickle.load(f)

    event_result_dict = defaultdict(list)
    event_result_posttag_dict = defaultdict(list)
    # for path in list_csv:
    #     try:
    #         ground_truth, ground_truth_posttag = read_recognition_result_multiple_seetag(path, None)
    #     except:
    #         ground_truth, ground_truth_posttag = read_recognition_result_multiple_seetag(path,None,  '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
    #     for event in ground_truth:
    #         if event not in event_type_dict:
    #             continue
    #         event_result_dict[event].extend(ground_truth[event])
    #         event_result_posttag_dict[event].extend(ground_truth_posttag[event])
    # for event in event_result_dict:
    #     if event not in event_type_dict:
    #         continue
    #     list_ = event_result_dict[event]
    #     temp_dict = defaultdict(int)
    #     for i in list_:
    #         temp_dict[i[0]] += i[1]
    #     count = Counter(temp_dict)
    #     temp_list = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
    #     event_result_dict[event] = temp_list
    #
    #
    #     list_ = event_result_posttag_dict[event]
    #     temp_dict = defaultdict(int)
    #     for i in list_:
    #         temp_dict[i[0]] += i[1]
    #     count = Counter(temp_dict)
    #     temp_list = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
    #     event_result_posttag_dict[event] = temp_list

    worker_result_dic = defaultdict(dict)
    count = 0
    for path in list_csv:
        try:
            worker_dict_new = read_recognition_result_multiple_seetag_worker(path)
        except:
            worker_dict_new = read_recognition_result_multiple_seetag_worker(path, '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
        for worker in worker_dict_new:
            for event in worker_dict_new[worker]:
                # if event in worker_result_dic[worker]:
                #     worker_result_dic[worker][event].append(worker_dict_new[worker][event])
                #     print worker_result_dic[worker][event]
                # else:
                    worker_result_dic[worker][event] = worker_dict_new[worker][event]

    print count
    worker_result_dic_new = defaultdict(dict)
    for worker in worker_result_dic:
        worker_new = dict()
        for event in worker_result_dic[worker]:
            if event in event_type_dict:
                worker_new[event] = worker_result_dic[worker][event]
        if len(worker_new) > 0:
            worker_result_dic_new[worker] = worker_new

    for worker in worker_result_dic_new:
        for event in worker_result_dic_new[worker]:
            event_result_dict[event].extend(worker_result_dic_new[worker][event][0].split('|'))
            event_result_posttag_dict[event].append(worker_result_dic_new[worker][event][1].split('|'))

    for event in event_result_dict:
        event_result_dict[event] = sorted(Counter(event_result_dict[event]).iteritems(), key=operator.itemgetter(1), reverse=True)
    similarity_worker_dict = dict()
    for worker in worker_result_dic_new:
        similarity_worker = []
        for event in worker_result_dic_new[worker]:
            result_this = event_result_dict[event]
            result_this_except = dict(result_this)
            for tpye_ in (worker_result_dic_new[worker][event][0]).split('|'):
                result_this_except[tpye_] -= 1
            len_votes = np.sum(result_this_except[i] for i in result_this_except)
            similarity = 0
            for type_ in (worker_result_dic_new[worker][event][0]).split('|'):
                if type_ in result_this_except:
                    similarity += float(result_this_except[type_])/ len_votes
                # print similarity
                similarity_worker.append(similarity)
        if np.mean(similarity_worker) == 0:
            print worker
        similarity_worker_dict[worker] = np.mean(similarity_worker)
    similar_value = [similarity_worker_dict[i] for i in similarity_worker_dict]
    temp = np.histogram(similar_value)
    print temp
    print similarity_worker_dict
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/worker_trust_weight_removedup.pkl','w') as f:
        cPickle.dump(similarity_worker_dict, f)


def new_multiple_corrected_recognition():
    list_csv = [
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_rejected_redo.csv',
        'result_rejected_redo_2round.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50_part.csv'
    ]

    groundtruth_old = dict()
    old_list = ['0.csv','1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8_pre.csv']
    for file in old_list:
        temp, temp_posttag = read_recognition_result_multiple_seetag(file, '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
        for event in temp:
            if event in correct_list:
                groundtruth_old[event] = [(correct_list[event], 3)]
            else:
                groundtruth_old[event] = [i for i in temp[event] if 'NOT' not in i[0]]
        print len(temp), len(groundtruth_old)


    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict_old = cPickle.load(f)
    corrected_event_type_dict_old = dict()
    for event in event_type_dict_old:
        if event in correct_list:
            corrected_event_type_dict_old[event] = correct_list[event]
        else:
            corrected_event_type_dict_old[event] = event_type_dict_old[event]

    for event in corrected_event_type_dict_old:
        if event not in groundtruth_old:
            continue
        if corrected_event_type_dict_old[event] != groundtruth_old[event][0][0]:
            groundtruth_old.pop(event, None)
            # print groundtruth_old[event], corrected_event_type_dict_old[event]
            # groundtruth_old[event] = [(corrected_event_type_dict_old[event], 3)]


    groundtruth = dict()
    groundtruth_posttag = dict()
    for file in list_csv:
        temp, temp_posttag = read_recognition_result_multiple_seetag(file)
        for event in temp:
            if event not in groundtruth:
                groundtruth[event] = temp[event]
                groundtruth_posttag[event] = temp_posttag[event]
            else:
                count_old = Counter(dict(groundtruth[event]))
                count_old.update(Counter(dict(temp[event])))
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth[event] = temp_list

                count_old = Counter(dict(groundtruth_posttag[event]))
                count_old.update(Counter(dict(temp[event])))
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth_posttag[event] = temp_list
    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_first200.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(no_posttag_csv)
    for event in temp:
        if event in groundtruth_posttag and groundtruth_posttag[event][0][1] != 1:
            continue
        if event in groundtruth_posttag:
            count_old = Counter(dict(groundtruth[event]))
            count_old.update(Counter(dict(temp[event])))
            temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
            groundtruth[event] = temp_list

            count_old = Counter(dict(groundtruth_posttag[event]))
            count_old.update(Counter(dict(temp[event])))
            temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
            groundtruth_posttag[event] = temp_list
        else:
            groundtruth[event] = temp[event]
            groundtruth_posttag[event] = temp_posttag[event]


    count = []
    for event in groundtruth_posttag:
        count.append(np.sum([i[1] for i in groundtruth_posttag[event]]))
    print Counter(count)


    print len(groundtruth)

    no_posttag_csv = 'result_2round.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(no_posttag_csv)
    for event in temp:
        count_old = Counter(dict(groundtruth[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth[event] = temp[event]
        # groundtruth[event] = temp_list

        count_old = Counter(dict(groundtruth_posttag[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        # groundtruth_posttag[event] = temp_list
        groundtruth_posttag[event] = temp[event]


    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_3round.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(no_posttag_csv)
    for event in temp:
        count_old = Counter(dict(groundtruth[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        # groundtruth[event] = temp[event]
        groundtruth[event] = temp_list

        count_old = Counter(dict(groundtruth_posttag[event]))
        count_old.update(Counter(dict(temp[event])))
        temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth_posttag[event] = temp_list
        # groundtruth_posttag[event] = temp[event]


    csv_more = 'result_all_recognition_3round_for_me.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(csv_more)
    for event in temp:
        this_list = groundtruth[event]
        add_list = temp[event]
        count_list = Counter(dict(this_list))
        for event_type in add_list:
            count_list[event_type[0]] += event_type[1]
        temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth[event] = temp_list

        this_list = groundtruth_posttag[event]
        add_list = temp_posttag[event]
        count_list = Counter(dict(this_list))
        for event_type in add_list:
            count_list[event_type[0]] += event_type[1]
        temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth_posttag[event] = temp_list


    csv_more = 'result_all_recognition_4round_for_me.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(csv_more)
    for event in temp:
        this_list = groundtruth[event]
        add_list = temp[event]
        count_list = Counter(dict(this_list))
        for event_type in add_list:
            count_list[event_type[0]] += event_type[1]
        temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth[event] = temp_list

        this_list = groundtruth_posttag[event]
        add_list = temp_posttag[event]
        count_list = Counter(dict(this_list))
        for event_type in add_list:
            count_list[event_type[0]] += event_type[1]
        temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth_posttag[event] = temp_list

    csv_more = 'result_all_recognition_4round_for_worker.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag(csv_more)
    for event in temp:
        this_list = groundtruth[event]
        add_list = temp[event]
        count_list = Counter(dict(this_list))
        for event_type in add_list:
            count_list[event_type[0]] += event_type[1]
        temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth[event] = temp_list

        this_list = groundtruth_posttag[event]
        add_list = temp_posttag[event]
        count_list = Counter(dict(this_list))
        for event_type in add_list:
            count_list[event_type[0]] += event_type[1]
        temp_list = sorted(count_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        groundtruth_posttag[event] = temp_list

    use_groundtruth = dict()
    for event in groundtruth:
        temp = Counter(dict(groundtruth[event]))
        temp_this = softmax(groundtruth_posttag[event])
        list = ['Birthday', 'Wedding', 'Graduation', 'Protest', 'Themepark', 'ReligiousActivity']
        for i in list:
            if groundtruth[event][0][0] != i and i in [i[0] for i in temp_this]:
                temp = Counter(dict(groundtruth_posttag[event]))
                break
        list = ['Christmas', 'Halloween']
        for i in list:
            if groundtruth[event][0][0] != i and i in [i[0] for i in temp_this]:
                temp.update(Counter(dict(groundtruth_posttag[event])))
                break

        if event in groundtruth_old:# and groundtruth_old[event][0][0] not in event_list:
            temp.update(Counter(dict(groundtruth_old[event])))
        use_groundtruth[event] = sorted(temp.iteritems(), key=operator.itemgetter(1), reverse=True)

    count = []
    for event in use_groundtruth:
        count.append(np.sum([i[1] for i in use_groundtruth[event]]))
    print Counter(count)


    use_groundtruth_all = dict()
    for event in use_groundtruth:
        use_groundtruth_all[event] = softmax_all(use_groundtruth[event])

    for event in use_groundtruth:
        use_groundtruth[event] = softmax(use_groundtruth[event])


    count = 0
    for event in corrected_event_type_dict_old:
        if corrected_event_type_dict_old[event] not in [i[0] for i in use_groundtruth[event]]:
            print use_groundtruth[event], corrected_event_type_dict_old[event]
            count += 1
    print count

    correct_manual = {'4_57013876@N00': [('PersonalArtActivity',1)],
                      '13_34993101@N00': [('Birthday',0.5),('CasualFamilyGather',0.5)],
                      '20_9674366@N08': [('Sports', 0.5), ('PersonalSports',0.5)],
                      '24_12693966@N07': [('UrbanTrip', 0.5),('ThemePark',0.5)]
                      }
    for i in correct_manual:
        use_groundtruth[i] = correct_manual[i]
        use_groundtruth_all[i] = correct_manual[i]
        # print i, use_groundtruth[i]
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round.pkl','w') as f:
        cPickle.dump(use_groundtruth, f)
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_softmaxall.pkl','w') as f:
        cPickle.dump(use_groundtruth_all, f)
def new_multiple_corrected_recognition_worker_weighted():
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/worker_trust_weight.pkl') as f:
        worker_weight_dict = cPickle.load(f)
    read_result_this = read_recognition_result_multiple_seetag_weighted
    softmax_this = softmax_weighted

    # read_result_this = read_recognition_result_multiple_seetag
    # softmax_this = softmax

    list_csv = [
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50_part.csv'
    ]

    groundtruth_old = dict()
    old_list = ['0.csv','1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8_pre.csv']
    for file in old_list:
        temp, temp_posttag = read_result_this(file, worker_weight_dict, '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
        for event in temp:
            # if event in correct_list:
            #     groundtruth_old[event] = [(correct_list[event], 3)]
            # else:
                groundtruth_old[event] = [i for i in temp[event] if i[0] in dict_name2]
        print len(temp), len(groundtruth_old)


    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict_old = cPickle.load(f)
    corrected_event_type_dict_old = dict()
    for event in event_type_dict_old:
        if event in correct_list:
            corrected_event_type_dict_old[event] = correct_list[event]
        else:
            corrected_event_type_dict_old[event] = event_type_dict_old[event]

    for event in corrected_event_type_dict_old:
        if event not in groundtruth_old:
            continue
        if corrected_event_type_dict_old[event] != groundtruth_old[event][0][0]:
            groundtruth_old.pop(event, None)


    groundtruth = dict()
    groundtruth_posttag = dict()
    for file in list_csv:
        temp, temp_posttag = read_result_this(file, worker_weight_dict)
        for event in temp:
            if event not in groundtruth:
                groundtruth[event] = temp[event]
                groundtruth_posttag[event] = temp_posttag[event]
            else:
                count_old =dict(groundtruth[event])
                for i in temp[event]:
                    if i[0] not in count_old:
                        count_old[i[0]] = i[1]
                    else:
                        count_old[i[0]] += i[1]
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth[event] = temp_list

                count_old =dict(groundtruth_posttag[event])
                for i in temp_posttag[event]:
                    if i[0] not in count_old:
                        count_old[i[0]] = i[1]
                    else:
                        count_old[i[0]] += i[1]
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth_posttag[event] = temp_list
    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_first200.csv'
    temp, temp_posttag = read_result_this(no_posttag_csv, worker_weight_dict)
    for event in temp:
        if event in groundtruth_posttag and groundtruth_posttag[event][0][1] != 1:
            continue
        if event in groundtruth_posttag:
                count_old =dict(groundtruth[event])
                for i in temp[event]:
                    if i[0] not in count_old:
                        count_old[i[0]] = i[1]
                    else:
                        count_old[i[0]] += i[1]
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth[event] = temp_list

                count_old =dict(groundtruth_posttag[event])
                for i in temp_posttag[event]:
                    if i[0] not in count_old:
                        count_old[i[0]] = i[1]
                    else:
                        count_old[i[0]] += i[1]
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth_posttag[event] = temp_list
        else:
            groundtruth[event] = temp[event]
            groundtruth_posttag[event] = temp_posttag[event]

    no_posttag_csv = 'result_2round.csv'
    temp, temp_posttag = read_result_this(no_posttag_csv, worker_weight_dict)
    for event in temp:
        groundtruth[event] = temp[event]
        groundtruth_posttag[event] = temp[event]


    append_csv_list = ['result_rejected_redo.csv',
                       'result_rejected_redo_2round.csv',
                       'result_all_recognition_amt_urlcorrected_3round.csv',
                       'result_all_recognition_3round_for_me.csv',
                       'result_all_recognition_4round_for_me.csv',
                       'result_all_recognition_4round_for_worker.csv'
                       ]

    for csv_this in append_csv_list:
        temp, temp_posttag = read_result_this(csv_this, worker_weight_dict)
        for event in temp:
                count_old =dict(groundtruth[event])
                for i in temp[event]:
                    if i[0] not in count_old:
                        count_old[i[0]] = i[1]
                    else:
                        count_old[i[0]] += i[1]
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth[event] = temp_list

                count_old =dict(groundtruth_posttag[event])
                for i in temp_posttag[event]:
                    if i[0] not in count_old:
                        count_old[i[0]] = i[1]
                    else:
                        count_old[i[0]] += i[1]
                temp_list = sorted(count_old.iteritems(), key=operator.itemgetter(1), reverse=True)
                groundtruth_posttag[event] = temp_list

    use_groundtruth = dict()
    for event in groundtruth:
        temp = dict(groundtruth[event])
        temp_this = softmax_this(groundtruth_posttag[event])
        list = ['Birthday', 'Wedding', 'Graduation', 'Protest', 'Themepark', 'ReligiousActivity']
        for i in list:
            if groundtruth[event][0][0] != i and i in [i[0] for i in temp_this]:
                temp = dict(groundtruth_posttag[event])
                break
        list = ['Christmas', 'Halloween']
        for i in list:
            if groundtruth[event][0][0] != i and i in [i[0] for i in temp_this]:
                for i in groundtruth_posttag[event]:
                    if i[0] in temp:
                        temp[i[0]] += i[1]
                    else:
                        temp[i[0]] = i[1]
                break

        if event in groundtruth_old:# and groundtruth_old[event][0][0] not in event_list:
            for i in groundtruth_old[event]:
                if i[0] in temp:
                    temp[i[0]] += i[1]
                else:
                    temp[i[0]] = i[1]
        use_groundtruth[event] = sorted(temp.iteritems(), key=operator.itemgetter(1), reverse=True)

    use_groundtruth_all = dict()
    for event in use_groundtruth:
        use_groundtruth_all[event] = softmax_this(use_groundtruth[event], threshold=0)

    for event in use_groundtruth:
        use_groundtruth[event] = softmax_this(use_groundtruth[event])


    count = 0
    for event in corrected_event_type_dict_old:
        if corrected_event_type_dict_old[event] not in [i[0] for i in use_groundtruth[event]]:
            print use_groundtruth[event], corrected_event_type_dict_old[event]
            count += 1
    print count

    correct_manual = {'4_57013876@N00': [('PersonalArtActivity',1)],
                      '13_34993101@N00': [('Birthday',0.5),('CasualFamilyGather',0.5)],
                      '20_9674366@N08': [('Sports', 0.5), ('PersonalSports',0.5)],
                      '24_12693966@N07': [('UrbanTrip', 0.5),('ThemePark',0.5)]
                      }
    for i in correct_manual:
        use_groundtruth[i] = correct_manual[i]
        use_groundtruth_all[i] = correct_manual[i]
        # print i, use_groundtruth[i]
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_weighted.pkl','w') as f:
        cPickle.dump(use_groundtruth, f)
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_softmaxall_weighted.pkl','w') as f:
        cPickle.dump(use_groundtruth_all, f)


def read_recognition_result_multiple_seetag_new(name, root='/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/'):
    input_path = root + name
    line_count = 0
    head_meta = []
    metas = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            else:
                metas.append(meta)
            line_count += 1
    HITids = []
    out_index_morethanone = [-1] * 10
    output_index = [-1] * 10
    out_index_morethanone_posttag = [-1] * 10
    output_index_posttag = [-1] * 10
    input_index = [-1] * 10
    tag_index = [-1] * 10
    worker_index = -1
    Accept_De_index = -1
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            if meta.endswith('_morethan1'):
                if 'posttag' in meta:
                    out_index_morethanone_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    out_index_morethanone[int(meta.split('_')[0][11:]) - 1] = i
            else:
                if 'posttag' in meta:
                    output_index_posttag[int(meta.split('_')[0][11:]) - 1] = i
                else:
                    output_index[int(meta.split('_')[0][11:]) - 1] = i
        if 'event_id' in meta:
            try:
                input_index[int(meta.split('_')[-1]) - 1] = i
            except:
                input_index[int(meta.split('_id')[-1]) - 1] = i
        if 'tags' in meta:
            try:
                tag_index[int(meta.split('_')[-1]) - 1] = i
            except:
                tag_index[int(meta.split('tags')[-1]) - 1] = i
        if 'WorkerId' == meta:
            worker_index = i
        if 'AssignmentStatus' == meta:
            Accept_De_index = i

    if output_index_posttag[0] == -1:
        output_index_posttag = output_index
        out_index_morethanone_posttag = out_index_morethanone
    HIT_result = []
    rejected_index = []
    for meta in metas:
        if meta[Accept_De_index] == 'Rejected':
            rejected_index.append(meta[input_index[0]])
        else:
            HITids.append(meta[0])
            HIT_result.append([meta[0], [[meta[j], meta[i], meta[jj], meta[k], meta[worker_index]]
                                         for (j, i, k, jj) in zip(input_index, output_index,tag_index, output_index_posttag)]])
    for i in HIT_result:
        for j in i[1]:
            if j[2] == '':
                j[2] = j[1]

    HIT_result_dic = {}
    for meta in HIT_result:
        HIT = meta[0]
        result = meta[1]
        if HIT in HIT_result_dic:
            HIT_result_dic[HIT].append(result)
        else:
            HIT_result_dic[HIT] = [result]
    agreement_event_id = []
    ground_truth = defaultdict(dict)
    ground_truth_posttag = defaultdict(dict)


    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = defaultdict(list)
            result_this_posttag = defaultdict(list)
            for result in results:
                temp = result[i][1].split('|')
                temp_posttag = result[i][2].split('|')
                worker = result[i][-1]
                result_this[worker] = [ii for ii in temp if ii in dict_name2]
                result_this_posttag[worker] = [ii for ii in temp_posttag if ii in dict_name2]
            for worker in result_this:
                ground_truth[results[0][i][0]][worker] = result_this[worker]
                ground_truth_posttag[results[0][i][0]][worker] = result_this_posttag[worker]

    return ground_truth, ground_truth_posttag
def new_multiple_corrected_recognition_removedup(weighted=False):
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/worker_trust_weight_removedup.pkl') as f:
        worker_weight_dict = cPickle.load(f)

    list_csv = [
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50_part.csv'
    ]

    groundtruth_old = defaultdict(dict)
    old_list = ['0.csv','1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8_pre.csv']
    for file in old_list:
        temp, temp_posttag = read_recognition_result_multiple_seetag_new(file, '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
        for event in temp:
                groundtruth_old[event] = temp[event]
        print len(temp), len(groundtruth_old)


    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict_old = cPickle.load(f)
    corrected_event_type_dict_old = dict()
    for event in event_type_dict_old:
        if event in correct_list:
            corrected_event_type_dict_old[event] = correct_list[event]
        else:
            corrected_event_type_dict_old[event] = event_type_dict_old[event]

    for event in corrected_event_type_dict_old:
        if event not in groundtruth_old:
            continue
        event_list = [j for i in groundtruth_old[event] for j in groundtruth_old[event][i]]
        if corrected_event_type_dict_old[event] != Counter(event_list).most_common(1)[0][0]:
            groundtruth_old.pop(event, None)
            # print groundtruth_old[event], corrected_event_type_dict_old[event]
            # groundtruth_old[event] = [(corrected_event_type_dict_old[event], 3)]


    groundtruth = defaultdict(dict)
    groundtruth_posttag = defaultdict(dict)
    for file in list_csv:
        temp, temp_posttag = read_recognition_result_multiple_seetag_new(file)
        for event in temp:
            for worker in temp[event]:
                groundtruth[event][worker] = temp[event][worker]
                groundtruth_posttag[event][worker] = temp_posttag[event][worker]

    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_first200.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag_new(no_posttag_csv)
    for event in temp:
        if event in groundtruth_posttag and len(groundtruth_posttag[event]) != 1:
            continue
        for worker in temp[event]:
            groundtruth[event][worker] = temp[event][worker]
            groundtruth_posttag[event][worker] = temp_posttag[event][worker]


    no_posttag_csv = 'result_2round.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag_new(no_posttag_csv)
    for event in temp:
        # groundtruth.pop(event, None)
        # groundtruth_posttag.pop(event, None)
        groundtruth[event] = temp[event]
        groundtruth_posttag[event] = temp_posttag[event]

    append_csv_list = ['result_rejected_redo.csv',
                       'result_rejected_redo_2round.csv',
                       'result_all_recognition_amt_urlcorrected_3round.csv',
                       'result_all_recognition_3round_for_me.csv',
                       'result_all_recognition_4round_for_me.csv',
                       'result_all_recognition_4round_for_worker.csv'
                       ]

    for in_file in append_csv_list:
        temp, temp_posttag = read_recognition_result_multiple_seetag_new(in_file)
        for event in temp:
            for worker in temp[event]:
                groundtruth[event][worker] = temp[event][worker]
                groundtruth_posttag[event][worker] = temp_posttag[event][worker]
    count_len = []

    use_groundtruth = dict()
    worker_vote_dict = dict()
    for event in groundtruth:
        temp = Counter([j for i in groundtruth[event] for j in groundtruth[event][i]])
        temp_this = softmax(Counter([j for i in groundtruth_posttag[event] for j in groundtruth_posttag[event][i]]).items())
        temp_notag = softmax(Counter([j for i in groundtruth[event] for j in groundtruth[event][i]]).items())
        # if temp_notag != temp_this:
        #     print temp_notag, temp_this
        list = ['Birthday', 'Wedding', 'Graduation', 'Protest', 'Themepark', 'ReligiousActivity']
        temp_this_event = groundtruth[event]
        for i in list:
            if temp.most_common(1)[0][0] != i and i in [ii[0] for ii in temp_this]:
                temp_this_event = groundtruth_posttag[event]
                break
        list = ['Christmas', 'Halloween']
        for i in list:
            if temp.most_common(1)[0][0] != i and i in [ii[0] for ii in temp_this]:
                for worker in groundtruth_posttag[event]:
                    temp_this_event[worker].extend(groundtruth_posttag[event][worker])
                break
        if event in groundtruth_old:# and groundtruth_old[event][0][0] not in event_list:
            for worker in groundtruth_old[event]:
                if worker not in temp_this_event:
                    temp_this_event[worker] = groundtruth_old[event][worker]
        worker_vote_dict[event] = temp_this_event
        count_len.append(len(temp_this_event))
        if weighted:
            count_this = defaultdict(int)
            for worker in temp_this_event:
                worker_weight = worker_weight_dict[worker]
                for jj in temp_this_event[worker]:
                    count_this[jj] += worker_weight
        else:
            count_this = Counter([j for i in temp_this_event for j in temp_this_event[i]])
        use_groundtruth[event] = sorted(count_this.items(), key=operator.itemgetter(1), reverse=True)

    print Counter(count_len)

    with open ('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_removedup_votes.pkl','w') as f:
        cPickle.dump(worker_vote_dict, f)


    use_groundtruth_all = dict()
    for event in use_groundtruth:
        use_groundtruth_all[event] = softmax_all(use_groundtruth[event])

    for event in use_groundtruth:
        use_groundtruth[event] = softmax(use_groundtruth[event])


    count = 0
    for event in corrected_event_type_dict_old:
        if corrected_event_type_dict_old[event] not in [i[0] for i in use_groundtruth[event]]:
            print use_groundtruth[event], corrected_event_type_dict_old[event]
            count += 1
    print count

    correct_manual = {'4_57013876@N00': [('PersonalArtActivity',1)],
                      '13_34993101@N00': [('Birthday',0.5),('CasualFamilyGather',0.5)],
                      '20_9674366@N08': [('Sports', 0.5), ('PersonalSports',0.5)],
                      '24_12693966@N07': [('UrbanTrip', 0.5),('ThemePark',0.5)]
                      }
    for i in correct_manual:
        use_groundtruth[i] = correct_manual[i]
        use_groundtruth_all[i] = correct_manual[i]
        # print i, use_groundtruth[i]
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_removedup.pkl','w') as f:
        cPickle.dump(use_groundtruth, f)
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_softmaxall_removedup.pkl','w') as f:
        cPickle.dump(use_groundtruth_all, f)


def new_multiple_corrected_recognition_removedup_notsoftmax(weighted=False):
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/worker_trust_weight_removedup.pkl') as f:
        worker_weight_dict = cPickle.load(f)

    list_csv = [
        'result_all_recognition_amt_urlcorrected_followup_next_next_50.csv',
        'result_all_recognition_amt_urlcorrected_followup_first200.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_next.csv',
        'result_all_recognition_amt_urlcorrected_followup_next_next_50_part.csv'
    ]

    groundtruth_old = defaultdict(dict)
    old_list = ['0.csv','1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8_pre.csv']
    for file in old_list:
        temp, temp_posttag = read_recognition_result_multiple_seetag_new(file, '/Users/wangyufei/Documents/Study/intern_adobe/amt/results/')
        for event in temp:
                groundtruth_old[event] = temp[event]
        print len(temp), len(groundtruth_old)


    with open('/Users/wangyufei/Documents/Study/intern_adobe/lstm/event_type_dict.pkl') as f:
        event_type_dict_old = cPickle.load(f)
    corrected_event_type_dict_old = dict()
    for event in event_type_dict_old:
        if event in correct_list:
            corrected_event_type_dict_old[event] = correct_list[event]
        else:
            corrected_event_type_dict_old[event] = event_type_dict_old[event]

    for event in corrected_event_type_dict_old:
        if event not in groundtruth_old:
            continue
        event_list = [j for i in groundtruth_old[event] for j in groundtruth_old[event][i]]
        if corrected_event_type_dict_old[event] != Counter(event_list).most_common(1)[0][0]:
            groundtruth_old.pop(event, None)
            # print groundtruth_old[event], corrected_event_type_dict_old[event]
            # groundtruth_old[event] = [(corrected_event_type_dict_old[event], 3)]


    groundtruth = defaultdict(dict)
    groundtruth_posttag = defaultdict(dict)
    for file in list_csv:
        temp, temp_posttag = read_recognition_result_multiple_seetag_new(file)
        for event in temp:
            for worker in temp[event]:
                groundtruth[event][worker] = temp[event][worker]
                groundtruth_posttag[event][worker] = temp_posttag[event][worker]

    no_posttag_csv = 'result_all_recognition_amt_urlcorrected_first200.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag_new(no_posttag_csv)
    for event in temp:
        if event in groundtruth_posttag and len(groundtruth_posttag[event]) != 1:
            continue
        for worker in temp[event]:
            groundtruth[event][worker] = temp[event][worker]
            groundtruth_posttag[event][worker] = temp_posttag[event][worker]


    no_posttag_csv = 'result_2round.csv'
    temp, temp_posttag = read_recognition_result_multiple_seetag_new(no_posttag_csv)
    for event in temp:
        # groundtruth.pop(event, None)
        # groundtruth_posttag.pop(event, None)
        groundtruth[event] = temp[event]
        groundtruth_posttag[event] = temp_posttag[event]

    append_csv_list = ['result_rejected_redo.csv',
                       'result_rejected_redo_2round.csv',
                       'result_all_recognition_amt_urlcorrected_3round.csv',
                       'result_all_recognition_3round_for_me.csv',
                       'result_all_recognition_4round_for_me.csv',
                       'result_all_recognition_4round_for_worker.csv'
                       ]

    for in_file in append_csv_list:
        temp, temp_posttag = read_recognition_result_multiple_seetag_new(in_file)
        for event in temp:
            for worker in temp[event]:
                groundtruth[event][worker] = temp[event][worker]
                groundtruth_posttag[event][worker] = temp_posttag[event][worker]
    count_len = []

    use_groundtruth = dict()
    worker_vote_dict = dict()
    event_worker_dict = dict()
    for event in groundtruth:
        temp_this = vote(groundtruth_posttag[event])
        temp = Counter([j for i in groundtruth[event] for j in groundtruth[event][i]])
        # temp_notag = vote(groundtruth[event])
        # if temp_notag != temp_this:
        #     print temp_notag, temp_this
        list = ['Birthday', 'Wedding', 'Graduation', 'Protest', 'Themepark', 'ReligiousActivity']
        temp_this_event = groundtruth[event]
        for i in list:
            if temp.most_common(1)[0][0] != i and i in [ii[0] for ii in temp_this]:
                temp_this_event = groundtruth_posttag[event]
                break
        list = ['Christmas', 'Halloween']
        for i in list:
            if temp.most_common(1)[0][0] != i and i in [ii[0] for ii in temp_this]:
                for worker in groundtruth_posttag[event]:
                    temp_this_event[worker].extend(groundtruth_posttag[event][worker])
                break
        if event in groundtruth_old:# and groundtruth_old[event][0][0] not in event_list:
            for worker in groundtruth_old[event]:
                if worker not in temp_this_event:
                    temp_this_event[worker] = groundtruth_old[event][worker]
        worker_vote_dict[event] = temp_this_event
        count_len.append(len(temp_this_event))
        if weighted:
            count_this = defaultdict(int)
            for worker in temp_this_event:
                worker_weight = worker_weight_dict[worker]
                for jj in temp_this_event[worker]:
                    count_this[jj] += worker_weight
        else:
            count_this = Counter([j for i in temp_this_event for j in temp_this_event[i]])
        # use_groundtruth[event] = sorted(count_this.items(), key=operator.itemgetter(1), reverse=True)
        use_groundtruth[event] = temp_this_event
    print Counter(count_len)

    # with open ('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_removedup_votes.pkl','w') as f:
    #     cPickle.dump(worker_vote_dict, f)


    use_groundtruth_all = dict()
    for event in use_groundtruth:
        use_groundtruth_all[event] = vote(use_groundtruth[event], threshold=0)

    for event in use_groundtruth:
        use_groundtruth[event] = vote(use_groundtruth[event])


    count = 0
    for event in corrected_event_type_dict_old:
        if corrected_event_type_dict_old[event] not in [i[0] for i in use_groundtruth[event]]:
            print use_groundtruth[event], corrected_event_type_dict_old[event]
            count += 1
    print count

    correct_manual = {'4_57013876@N00': [('PersonalArtActivity',1)],
                      '13_34993101@N00': [('Birthday',0.5),('CasualFamilyGather',0.5)],
                      '20_9674366@N08': [('Sports', 0.5), ('PersonalSports',0.5)],
                      '24_12693966@N07': [('UrbanTrip', 0.5),('ThemePark',0.5)]
                      }
    for i in correct_manual:
        use_groundtruth[i] = correct_manual[i]
        use_groundtruth_all[i] = correct_manual[i]
        # print i, use_groundtruth[i]
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_removedup_vote.pkl','w') as f:
        cPickle.dump(use_groundtruth, f)
    with open('/Users/wangyufei/Documents/Study/intern_adobe/0208_correction/all_input_and_result/new_multiple_result_2round_softmaxall_removedup_vote.pkl','w') as f:
        cPickle.dump(use_groundtruth_all, f)

if __name__ == '__main__':
    #
    # worker_list = []
    # for result in ['Batch_2290270_batch_results-2.csv']:
    #         worker_list = read_recognition_result_multiple_seetag(result, worker_list)
    #         print len(worker_list)
    # count_worker_list = Counter(worker_list)
    # print count_worker_list
    # print sum([count_worker_list[i] for i in count_worker_list])

    # find_reject()
    # read_recognition_result_multiple_seetag('result_all_recognition_amt_urlcorrected_followup_next_next_50.csv')
    # combine_from_sources()
    # write_csv_correct_link_2round()
    # write_csv_curation_2round()
    # new_multiple_corrected_recognition()
    # write_csv_correct_link_3round()
    # create_html_need_check()
    # create_weighted_worker_groundtruth_new()
    # new_multiple_corrected_recognition_worker_weighted()
    # create_html_need_check()
    # new_multiple_corrected_recognition_removedup()
    new_multiple_corrected_recognition_removedup_notsoftmax()