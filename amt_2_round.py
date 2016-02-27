__author__ = 'wangyufei'
import re
import cPickle
import random
import csv
from collections import Counter
import os
root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_curation/data_prepare'
root='/Users/wangyufei/Documents/Study/intern_adobe/amt/clean_input_and_label/3_event_curation/3_all/check_progress/'
suffix = '_new_0730'

'''some check and correctness'''
def check_event_count():
    path = root + '../../all_images.txt'
    all_event_ids = set()
    last_event = ''
    with open(path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_id = meta[1]+'_'+meta[3]
            if last_event == event_id:
                continue
            last_event = event_id
            if event_id not in all_event_ids:
                all_event_ids.add(event_id)
            else:
                print event_id
    pass


'''create dataset for amt'''
def count_events():
    path=root + '20000_all_events.txt'
    meta_path = path
    event_start = []
    event_id = []
    count = 0
    last_event = -1
    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                event_id.append(this_event)
                event_start.append(count)
                last_event = this_event
            count += 1
    event_start.append(count)
    event_length = []
    for i in xrange(1,len(event_start)):
        event_length.append(event_start[i]-event_start[i-1])
    print count
    print len(event_start)
    print len(event_length)
    print max(event_length)
    sample_ = zip(event_length, event_id)
    length_sort = sorted(sample_,reverse=True)
    f = open(root+'sample_length_event.cPickle','w')
    cPickle.dump(length_sort, f)
    f.close()


def write_example_txt():
    f = open(root+'2_round/data_prepare/event_sampled_l1.cPickle','rb')
    sample_event_meta = cPickle.load(f)
    f.close()
    sample_events = []
    for i in sample_event_meta:
        sample_events.extend(sample_event_meta[i])

    sample_event = [a[0] for a in sample_events]
    read_path1 = root + '20000_all_events.txt'
    read_path2 = root + '2_round/data_prepare/manual_birthday.txt'
    read_path3 = root + '2_round/data_prepare/manual_graduation.txt'
    write_path = root + '2_round/data_prepare/event_sampled_l1.txt'

    count = 0
    i = 0
    f = open(write_path, 'w')
    with open(read_path1, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            #if (i % 1000) == 0:
            #    with open(log_path, 'a') as f1:
            #        f1.write(str(i)+'\n')
            if event_this_id in sample_event:
                f.write(line)
    with open(read_path2, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            if event_this_id in sample_event:
                f.write(line)
    with open(read_path3, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            if event_this_id in sample_event:
                f.write(line)

    f.close()
def write_csv():
    f = open(root+'2_round/distraction.cPickle')
    distraction_url = cPickle.load(f)
    f.close()
    f = open(root+'2_round/data_prepare/event_sampled_l1.cPickle','rb')
    sample_event_meta = cPickle.load(f)
    f.close()
    sample_events = []
    for i in sample_event_meta:
        sample_events.extend(sample_event_meta[i])

    dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
            'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)','Independence':'Independence Day',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}

    sample_event_dict = dict((x, [dict_name[y],a,b]) for x, y,a,b in sample_events)
    path=root + '2_round/data_prepare/event_sampled_l1.txt'
    meta_path = path
    event_start = []
    event_id = []
    tags = []
    count = 0
    last_event = -1

    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                event_id.append(this_event)
                tags.append(meta[0])
                event_start.append(count)
                last_event = this_event
            count += 1
    event_start.append(count)
    event_length = []
    for i in xrange(1,len(event_start)):
        event_length.append(event_start[i]-event_start[i-1])
    print count
    print len(event_start)
    print len(event_length)
    print max(event_length)
    length_sort = sorted(event_length)
    f = open(root+'2_round/data_prepare/log_length_event.txt','w')
    for i in length_sort:
        f.write(str(i)+'\n')
    f.close()

    out_path = root + '2_round/data_prepare/event_sampled_l1.csv'
    f = open(out_path, 'w')
    max_size = max(event_length) + 2
    f.write('num_image,event_id,event_type,tags,distraction_num')
    for i in xrange(max_size):
            f.write(',')
            f.write('image'+str(i+1))
    f.write('\n')
    count = 0
    event_prev = ''
    image_count = 0
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if event_this != event_prev:
                    if count > 0:
                        for kk in xrange(max_size-event_length[count - 1]-2):
                            f.write(',NA')
                        f.write('\n')
                    image_count = 0
                    f.write(str(event_length[count]+2))
                    f.write(','+event_this)
                    indice = sorted(random.sample(xrange(event_length[count]), 2))
                    print indice
                    f.write(','+sample_event_dict[event_this][0])
                    count += 1
                    tag_list = meta[0].split(',')
                    tag_this = ':'.join(tag_list)
                    f.write(','+tag_this)
                    f.write(','+str(indice[0]+1)+':'+str(int(indice[1]+2)))
                    event_prev = event_this
                url = meta[16]
                if image_count in indice:
                    f.write(','+random.sample(distraction_url, 1)[0])
                f.write(','+url)
                image_count += 1

    for kk in xrange(max_size-event_length[count - 1]-2):
        f.write(',NA')
    f.write('\n')
    f.close()
def write_csv_400():
    in_path = root + '2_round/data_prepare/event_sampled_l1.csv'
    csv_count = 0
    out_path = root + '2_round/all_input/event_sampled_l1_'+str(csv_count)+'.csv'
    head_meta = ''
    with open(in_path,'r') as data:
        head_meta = data.readline()
    count = -1
    lines = []
    with open(in_path,'r') as data:
        for line in data:
            count += 1
            if count > 0:
                lines.append(line)
    random.shuffle(lines)
    f = open(out_path, 'w')
    f.write(head_meta)
    count = -1
    for line in lines:
            count += 1
            if count == 0:
                continue
            if count % 400 == 0:
                f.close()
                csv_count += 1
                f = open(root + '2_round/all_input/event_sampled_l1_'+str(csv_count)+'.csv','w')
                f.write(head_meta)
            f.write(line)
    f.close()
def correct_csv():
    miss_event_type = {}
    dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
            'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)','Independence':'Independence Day',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}

    delete_event_id = ['805_28004289@N03','72_52066925@N00','32_40279823@N00','7_30697432@N05','0_47397808@N00','554_35034347371@N01',
                       '13_25095531@N07','2_83011695@N00','1_58003213@N00']
    out_file = root+'2_round/all_input/valid_event_sampled_l1_3.csv'
    f = open(out_file,'w')
    writer = csv.writer(f)
    with open(root+'2_round/all_input/event_sampled_l1_3.csv','rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            event_id = meta[1]
            if event_id in delete_event_id:
                print event_id
                event_type = meta[2]
                if event_type in miss_event_type:
                    miss_event_type[event_type] += 1
                else:
                    miss_event_type[event_type] = 1
            else:
                writer.writerow(meta)
    f.close()
    out_file = root+'2_round/all_input/valid_event_sampled_l1_4.csv'
    f = open(out_file,'w')
    writer = csv.writer(f)
    with open(root+'2_round/all_input/event_sampled_l1_4.csv','rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            event_id = meta[1]
            if event_id in delete_event_id:
                print event_id
                event_type = meta[2]
                if event_type in miss_event_type:
                    miss_event_type[event_type] += 1
                else:
                    miss_event_type[event_type] = 1
            else:
                writer.writerow(meta)

    f.close()
    f = open(root+'2_round/data_prepare/event_sampled_l1.cPickle','rb')
    sample_event_meta = cPickle.load(f)
    f.close()

    sample_events = []
    for i in sample_event_meta:
        sample_events.extend(sample_event_meta[i])
    f = open(root+'2_round/data_prepare/l100_event_label_importantlife.cPickle','r')
    event_labels = cPickle.load(f)
    types = [i[1] for i in event_labels]
    count_type = Counter(types)
    print count_type
    f.close()
    f = open(root+'2_round/distraction.cPickle')
    distraction_url = cPickle.load(f)
    f.close()


    f = open(root+'2_round/data_prepare/sample_event_new_0730.cPickle','r')
    event_labels1 = cPickle.load(f)
    f.close()
    sample_events += event_labels1
    types = [i[1] for i in sample_events]
    count_type = Counter(types)
    print count_type

    event_to_append = []
    sampled_event_id = [i[0] for i in sample_events]
    for event_type in miss_event_type:
        need_number = miss_event_type[event_type]
        this_available = []
        for event in event_labels:
            if event[0] not in sampled_event_id and event[1] in dict_name and dict_name[event[1]] == event_type:
                this_available.append(event)
        if len(this_available) >= need_number:
            event_to_append += random.sample(this_available, need_number)
        else:
            event_to_append += this_available

    print event_to_append
    path=root + '20000_all_events.txt'
    meta_path = path
    event_id_to_append = [i[0] for i in event_to_append]
    max_size =102
    count = 0
    event_prev = ''
    image_count = 0
    out_file = root+'2_round/all_input/valid_event_sampled_l1_4.csv'
    f = open(out_file,'a')
    event_start = []
    tags = []
    last_event = ''
    event_id = []
    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                event_id.append(this_event)
                tags.append(meta[0])
                event_start.append(count)
                last_event = this_event
            count += 1
    event_start.append(count)
    event_length = []
    for i in xrange(1,len(event_start)):
        event_length.append(event_start[i]-event_start[i-1])

    real_count = -1
    count = 0
    real_prev = ''
    last_count = 0
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if real_prev != event_this:
                    real_count += 1
                    real_prev = event_this
                if event_prev != event_this:
                    if event_this not in event_id_to_append:
                        continue

                    if count > 0:
                        for kk in xrange(max_size-last_count-2):
                            f.write(',NA')
                        f.write('\r\n')
                    image_count = 0
                    f.write(str(event_length[real_count]+2))
                    f.write(','+event_this)
                    indice = sorted(random.sample(xrange(event_length[real_count]), 2))
                    print indice
                    last_count = event_length[real_count]
                    count += 1
                    f.write(','+event_to_append[event_id_to_append.index(event_this)][1])
                    f.write(','+event_to_append[event_id_to_append.index(event_this)][3])
                    f.write(','+str(indice[0]+1)+':'+str(int(indice[1]+2)))
                    event_prev = event_this
                url = meta[16]
                if image_count in indice:
                    f.write(','+random.sample(distraction_url, 1)[0])
                f.write(','+url)
                image_count += 1

    for kk in xrange(max_size-last_count-2):
        f.write(',NA')
    f.write('\n')
    f.close()
def find_example_result():
    f = open(root+'2_round/sample_event2.cPickle')
    event_alreadyhave = cPickle.load(f)
    f.close()
    f = open(root+ '20000_all_events_length.cPickle','r')
    event_lengths = cPickle.load(f)
    f.close()
    event_lengths = dict((y, x) for x, y in event_lengths)

    input_path = root +'all_event_label.cPickle'
    f = open(input_path, 'r')
    agreement_events = cPickle.load(f)
    f.close()

    event_types = ['Show', 'UrbanTrip', 'NatureTrip', 'Architecture', 'Sports', 'GroupActivity', 'Zoo', 'PersonalSports',
                    'BusinessActivity', 'CasualFamilyGather', 'BeachTrip', 'Museum', 'Wedding', 'Protest', 'Graduation',
                    'ThemePark', 'Cruise', 'Halloween', 'Christmas', 'ReligiousActivity', 'PersonalArtActivity', 'Birthday',
                    'Independence', 'PersonalMusicActivity']
    event_selected = []
    for event_type in event_types:
        events_this_type = []
        event_not_had = []
        for event in agreement_events:
            if event[1] == event_type:
                event_id = event[0]
                if event_lengths[event_id] < 70:
                    if event_id not in event_alreadyhave:
                        event_not_had.append(event)
                    events_this_type.append(event)
        if len(event_not_had) < 3:
            event_not_had = events_this_type
        sample_event = [events_this_type[i] for i in sorted(random.sample(xrange(len(event_not_had)), 3))]
        event_selected.extend(sample_event)
    write_path = root+'2_round/sample_event'+suffix+'.cPickle'
    f = open(write_path,'w')
    cPickle.dump(event_selected,f)
    f.close()
def refine_example():
    f = open(root+'2_round/sample_event'+suffix+'.cPickle','rb')
    sample_event_meta = cPickle.load(f)
    f.close()
    event_id_new = []
    delete_event = ['14_76701708@N00','5_36584552@N00','13_56183392@N00','43_7759477@N05','2_44613915@N00','9_60453293@N00']
    add_event = ['52_74743437@N00','0_51884345@N00','12_12734746@N00','8_43792569@N00','0_40215530@N00','3_85509292@N00',
                 '12_20675795@N00','4_75897017@N05','0_72295842@N00','0_12138652@N08','7_55455788@N00']
    for event_meta in sample_event_meta:
        event_id = event_meta[0]
        if event_id not in delete_event:
            event_id_new.append(event_id)
    event_id_new.extend(add_event)


    f = open(root+ '20000_all_events_length.cPickle','r')
    event_lengths = cPickle.load(f)
    f.close()
    event_lengths = dict((y, x) for x, y in event_lengths)

    input_path = root +'all_event_label.cPickle'
    f = open(input_path, 'r')
    agreement_events = cPickle.load(f)
    f.close()
    event_meta = []
    for event in agreement_events:
        if event[0] in event_id_new:
            event_meta.append(event)

    write_path = root+'2_round/sample_event_new'+suffix+'.cPickle'
    f = open(write_path,'w')
    cPickle.dump(event_meta,f)
    f.close()
def write_example_txt1():
    f = open(root+'2_round/sample_event'+suffix+'.cPickle','rb')
    sample_event_meta = cPickle.load(f)
    f.close()
    sample_event = [a[0] for a in sample_event_meta]
    read_path = root + '../20000_all_events.txt'
    write_path = root + 'event_sampled_l1.txt'
    count = 0
    i = 0
    event_prev_id = ''
    f = open(write_path, 'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            #if (i % 1000) == 0:
            #    with open(log_path, 'a') as f1:
            #        f1.write(str(i)+'\n')
            if event_this_id in sample_event:
                f.write(line)
    f.close()
def read_distraction():
    distraction_image = []
    path = root+'2_round/distraction.txt'
    with open(path, 'r') as data:
        for line in data:
            if line == '\n':
                continue
            distraction_image.append(line[:-1])
    out_path = root+'2_round/distraction.cPickle'
    f = open(out_path, 'w')
    cPickle.dump(distraction_image, f)
    f.close()
def write_csv1():
    f = open(root+'2_round/distraction.cPickle')
    distraction_url = cPickle.load(f)
    f.close()
    f = open(root+'2_round/sample_event'+suffix+'.cPickle','rb')
    sample_event_meta = cPickle.load(f)
    f.close()
    dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
            'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)','Independence':'Independence Day',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}

    sample_event_dict = dict((x, [dict_name[y],a,b]) for x, y,a,b in sample_event_meta)
    path=root + '2_round/sample_event'+suffix+'.txt'
    meta_path = path
    event_start = []
    event_id = []
    tags = []
    count = 0
    last_event = -1

    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                event_id.append(this_event)
                tags.append(meta[0])
                event_start.append(count)
                last_event = this_event
            count += 1
    event_start.append(count)
    event_length = []
    for i in xrange(1,len(event_start)):
        event_length.append(event_start[i]-event_start[i-1])
    print count
    print len(event_start)
    print len(event_length)
    print max(event_length)
    length_sort = sorted(event_length)
    f = open(root+'log_length_event.txt','w')
    for i in length_sort:
        f.write(str(i)+'\n')
    f.close()

    #chunks = chunk(len(event_length), 1000)
    #print chunks
    out_path = root + '2_round/sample_event'+suffix+'.csv'
    f = open(out_path, 'w')
    max_size = max(event_length) + 2
    f.write('num_image,event_id,event_type,tags,distraction_num')
    for i in xrange(max_size):
            f.write(',')
            f.write('image'+str(i+1))
    f.write('\n')
    count = 0
    event_prev = ''
    image_count = 0
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if event_this != event_prev:
                    if count > 0:
                        for kk in xrange(max_size-event_length[count - 1]-2):
                            f.write(',NA')
                        f.write('\n')
                    image_count = 0
                    f.write(str(event_length[count]+2))
                    f.write(','+event_this)
                    indice = sorted(random.sample(xrange(event_length[count]), 2))
                    print indice
                    f.write(','+sample_event_dict[event_this][0])
                    count += 1
                    tag_list = meta[0].split(',')
                    tag_this = ':'.join(tag_list)
                    f.write(','+tag_this)
                    f.write(','+str(indice[0]+1)+':'+str(int(indice[1]+2)))
                    event_prev = event_this
                url = meta[16]
                if image_count in indice:
                    f.write(','+random.sample(distraction_url, 1)[0])
                f.write(','+url)
                image_count += 1

    for kk in xrange(max_size-event_length[count - 1]-2):
        f.write(',NA')
    f.write('\n')
    f.close()
def auxiliary_data_tag():
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_curation/data_prepare/'
    f = open(root1+'manual_birthday.cPickle','r')
    manual_birthday_id = cPickle.load(f)
    f.close()
    f = open(root1+'manual_graduation.cPickle','r')
    manual_graduation_id = cPickle.load(f)
    f.close()
    read_path = root + 'clean_imgs.txt'

    save_path = root1 + 'manual_birthday.txt'
    f = open(save_path,'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]+'_'+meta[3]
            if event_this in manual_birthday_id:
                f.write(line)

    save_path = root1 + 'manual_graduation.txt'
    f = open(save_path,'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]+'_'+meta[3]
            if event_this in manual_graduation_id:
                f.write(line)
def sample_data_for_2round():
    event_types = ['Show', 'UrbanTrip', 'NatureTrip', 'Architecture', 'Sports', 'GroupActivity', 'Zoo', 'PersonalSports',
                    'BusinessActivity', 'CasualFamilyGather', 'BeachTrip', 'Museum', 'Wedding', 'Protest', 'Graduation',
                    'ThemePark', 'Cruise', 'Halloween', 'Christmas', 'ReligiousActivity', 'PersonalArtActivity', 'Birthday',
                    'Independence', 'PersonalMusicActivity']
    load_path = root + 'l100_event_label_importantlife.cPickle'
    f = open(load_path,'r')
    event_labels = cPickle.load(f)
    f.close()
    events_ = [(event_type,[]) for event_type in event_types]
    events_dict = dict(events_)
    for event in event_labels:
        if event[1] in events_dict:
            events_dict[event[1]].append(event)
    pass

    no_need_to_sample = ['ReligiousActivity','PersonalArtActivity','PersonalMusicActivity','Graduation','Birthday']
    sample_100 = ['Christmas','Halloween','ThemePark','UrbanTrip','PersonalSports','Show','Zoo']
    sample_200 = ['Wedding']
    sample_50 = ['BusinessActivity','Architecture','BeachTrip','CasualFamilyGather','Protest','Cruise',
                 'GroupActivity','Museum','NatureTrip','Sports']
    need_decide_later = ['Independence']
    f = open(root+'../sample_event_new_0730.cPickle','r')
    event_had = cPickle.load(f)
    f.close()
    event_had_id = [i[0] for i in event_had]

    event_sampled = {}
    for event_type in no_need_to_sample:
        events_this = events_dict[event_type]
        event_this_new = []
        count = 0
        for i in events_this:
            if i[0] not in event_had_id:
                event_this_new.append(i)
            else:
                count += 1
        event_sampled[event_type] = event_this_new

    for event_type in sample_100:
        events_this = events_dict[event_type]
        event_this_new = []
        count = 0
        for i in events_this:
            if i[0] not in event_had_id:
                event_this_new.append(i)
            else:
                count += 1
        sample_event = random.sample(event_this_new, 100-count)
        event_sampled[event_type] = sample_event

    for event_type in sample_200:
        events_this = events_dict[event_type]
        event_this_new = []
        count = 0
        for i in events_this:
            if i[0] not in event_had_id:
                event_this_new.append(i)
            else:
                count += 1
        sample_event = random.sample(event_this_new, 200-count)
        event_sampled[event_type] = sample_event

    for event_type in sample_50:
        events_this = events_dict[event_type]
        event_this_new = []
        count = 0
        for i in events_this:
            if i[0] not in event_had_id:
                event_this_new.append(i)
            else:
                count += 1
        sample_event = random.sample(event_this_new, 50-count)
        event_sampled[event_type] = sample_event
    last_event = ''
    event_graduation = []
    with open(root + 'manual_graduation.txt','r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                last_event = this_event
                tag = meta[0]
                tag = ':'.join(tag.split(','))
                event_graduation.append([this_event, 'Graduation',-1,tag])
    last_event = ''
    event_birthday = []
    with open(root + 'manual_birthday.txt','r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                last_event = this_event
                tag = meta[0]
                tag = ':'.join(tag.split(','))
                event_birthday.append([this_event, 'Birthday',-1,tag])

    event_sampled['Graduation'].extend(event_graduation)
    event_sampled['Birthday'].extend(event_birthday)


    event_types = ['Show', 'UrbanTrip', 'NatureTrip', 'Architecture', 'Sports', 'GroupActivity', 'Zoo', 'PersonalSports',
                    'BusinessActivity', 'CasualFamilyGather', 'BeachTrip', 'Museum', 'Wedding', 'Protest', 'Graduation',
                    'ThemePark', 'Cruise', 'Halloween', 'Christmas', 'ReligiousActivity', 'PersonalArtActivity', 'Birthday',
                    'PersonalMusicActivity']
    for k in event_types:
        temp = [i[0] for i in event_sampled[k]]
        if len(Counter(temp)) != len(temp):
            print k, len(Counter(temp)), len(temp)
            k_new = []
            a = sorted(event_sampled[k], key=lambda pair: pair[0], reverse=True)
            last = ''
            for ii in a:
                if last == ii[0]:
                    continue
                last = ii[0]
                k_new.append(ii)
                event_sampled[k] = k_new

    print [ (i, len(event_sampled[i])) for i in event_sampled]
    save_path = root + 'event_sampled_l1.cPickle'
    f = open(save_path,'w')
    cPickle.dump(event_sampled,f)
    f.close()
def label_correction():

    path = root + '../../../2_balance_data/l100_event_label_importantlife.cPickle'
    f = open(path,'r')
    events = cPickle.load(f)
    f.close()
    events_need_to_correct = []
    for event in events:
        if event[1] == 'Wedding' and ('birthday' in event[3] or 'graduation' in event[3]):
            events_need_to_correct.append(event)
        if event[1] == 'Birthday' and ('wedding' in event[3] or 'graduation' in event[3]):
            events_need_to_correct.append(event)
        if event[1] == 'Graduation' and ('wedding' in event[3] or 'birthday' in event[3]):
            events_need_to_correct.append(event)

    events_need_to_correct = events_need_to_correct[1:]
    f = open(root + '../../../2_balance_data/l100_need_correct_more_needcorrect.cPickle','w')
    cPickle.dump(events_need_to_correct,f)
    f.close()

    events_corrected = []
    new_events = []
    for event in events:
        if event[1] == 'Wedding' and 'birthday' in event[3]:
            events_corrected.append([event[0],'Birthday',-1,event[3]])
        elif event[1] == 'Wedding' and 'graduation' in event[3]:
            events_corrected.append([event[0],'Graduation',-1,event[3]])
        #else:
            #if event[2] == -1:
            #    print event
        else:
            new_events.append(event)
    new_events.extend(events_corrected)
    f = open(root + '../../../2_balance_data/l100_event_label_more_importantlife.cPickle','w')
    cPickle.dump(new_events,f)
    f.close()
    f = open(root + '../../../2_balance_data/l100_event_label_more_corrected.cPickle','w')
    cPickle.dump(events_corrected,f)
    f.close()


    event_ids = [i[0] for i in events_corrected]
    f = open(root + '../event_sampled_l1.cPickle','r')
    events_dict = cPickle.load(f)
    f.close()
    event_dict_new = {}
    for event_type in events_dict:
        events = events_dict[event_type]
        new_events = []
        for event in events:
            if event[0] in event_ids:
                #new_events.append(events_corrected[event_ids.index(event[0])])
                print events_corrected[event_ids.index(event[0])]
            else:
                new_events.append(event)
        event_dict_new[event_type] = new_events
    for event in events_corrected:
        event_dict_new[event[1]].append(event)

    f = open(root + '../event_sampled_l1_morecorrected.cPickle','w')
    cPickle.dump(event_dict_new, f)
    f.close()

    in_path = root+'../event_sampled_l1.csv'
    out_path = root+'../event_sampled_l1_morecorrected.csv'
    f = open(out_path,'w')
    writer = csv.writer(f)
    line_count = -1
    corrected_metas = []
    with open(in_path, 'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            line_count += 1
            if line_count == 0:
                head_meta = meta
            if meta[1] in event_ids:
                print meta
                meta[2] = events_corrected[event_ids.index(meta[1])][1]
                corrected_metas.append(meta)
            writer.writerow(meta)
    f.close()


    out_path = root+'../event_sampled_l1_need_correct.csv'
    f = open(out_path,'w')
    writer = csv.writer(f)
    writer.writerow(head_meta)
    for meta in corrected_metas:
        writer.writerow(meta)
    f.close()


'''result'''
def find_result_by_id(event_ids):
    input_path = root +'all_event_label.cPickle'
    f = open(input_path, 'r')
    agreement_events = cPickle.load(f)
    f.close()
    event_selected = []
    for event in agreement_events:
        if event[0] in event_ids:
            event_selected.append(event)

    write_path = root+'2_round/sample_event2.cPickle'
    f = open(write_path,'w')
    cPickle.dump(event_selected,f)
    f.close()
def suspicious_worker():
    in_path = root + 'all_input_sample_result3.csv'
    #in_path = root + 'sample_event_0730/results.csv'
    line_number = -1
    image_index = {}
    suspicious_worker_id = []
    with open(in_path, 'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            line_number += 1
            if line_number == 0:
                head_field = meta
                for i in xrange(len(head_field)):
                    field = head_field[i]
                    if 'Answer.image' in field:
                        number = int(field[12:])
                        image_index[number] = i
                    if 'distraction' in field:
                        distraction_index = i
            else:
                temp = meta[distraction_index]
                temp = temp.split(':')
                number1 = int(temp[0]);number2 = int(temp[1])
                if meta[image_index[number1]] not in ['selected_neutral','selected_irrelevant'] or meta[image_index[number2]] not in ['selected_neutral','selected_irrelevant'] :
                    suspicious_worker_id.append((meta[29],meta[0], meta[15]))
    for i in suspicious_worker_id:
        print i
def find_outlier():
        suspicious_worker_id = {}
        all_worker_id = {}
        root = '/Users/wangyufei/Documents/Study/intern_adobe/all_output/'

        in_path = root + 'all_output.csv'
        line_number = -1
        image_index = {}
        last_HITid = ''
        worker_submissions = []
        worker_ids = []
        images_indice = []
        vote_to_score = {'selected':2, 'selected_sw':1,'selected_neutral':0, 'selected_irrelevant':-1}
        with open(in_path, 'r') as data:
            reader = csv.reader(data)
            for meta in reader:
                line_number += 1
                if line_number == 0:
                    head_field = meta
                    for i in xrange(len(head_field)):
                        field = head_field[i]
                        if 'Answer.image' in field:
                            number = int(field[12:])
                            image_index[number] = i
                    images_indice = [image_index[i] for i in image_index]
                else:
                    HITid = meta[0]
                    workerid = meta[15]
                    if HITid != last_HITid:
                        worker_scores = []
                        for worker_submission in worker_submissions:
                            try:
                                worker_score = [vote_to_score[i] for i in worker_submission]
                            except:
                                continue
                            worker_scores.append(worker_score)
                        for (worker_id, worker_submission) in zip(worker_ids, worker_scores):
                            data = [i for i in worker_scores if i != worker_submission]
                            average_vote = [float(sum(col))/len(col) for col in zip(*data)]
                            #print average_vote
                            difference = [i-j for (i,j) in zip(average_vote, worker_submission)]
                            sq_difference = [i*i for i in difference]
                            #print sq_difference
                            if worker_id in all_worker_id:
                                all_worker_id[worker_id] += [last_HITid]
                            else:
                                all_worker_id[worker_id] = [last_HITid]
                            if sum(sq_difference)>0.85*len(sq_difference):
                                # if not os.path.exists(root + 'outliers_html/' + str(kk)):
                                #     os.mkdir(root + 'outliers_html/' + str(kk))
                                # create_result_html_all_id(in_path, root + 'outliers_html/' + str(kk) + '/' + worker_id + '_'+ ''.join(re.split('/| ',event_type))[:10] + '_', last_HITid, worker_id)
                                if worker_id in suspicious_worker_id:
                                    suspicious_worker_id[worker_id] += [last_HITid]
                                else:
                                    suspicious_worker_id[worker_id] = [last_HITid]
                        last_HITid = HITid
                        worker_ids = []
                        worker_submissions = []
                    event_type = meta[29]
                    worker_ids.append(workerid)
                    worker_submission = []
                    for number in images_indice:
                        if meta[number] != '':
                            worker_submission.append(meta[number])
                    worker_submissions.append(worker_submission)
        for i in all_worker_id:
            if i in suspicious_worker_id:
                print i, '%d/%d' % (len(suspicious_worker_id[i]), len(all_worker_id[i])), float(len(suspicious_worker_id[i]))/float(len(all_worker_id[i]))
            #else:
            #    print i, '0/%d' % len(all_worker_id[i])


'''result display'''
line_break = 30
max_html = 50
def images_for_sample(event_type, col_num=5):
    f = open(root + 'event_sampled_l1.cPickle','r')
    event_sampled = cPickle.load(f)
    f.close()
    if event_type not in event_sampled:
        print 'Event %s not selected.' % event_type
        return
    event_sampled = event_sampled[event_type]
    event_ids = [i[0] for i in event_sampled]

    list_path = root+'../../20000_all_events.txt'
    html_count = 0
    out_path = root+'htmls/'+event_type+'_'+str(html_count)+'.html'
    count = 0
    last_event = -1
    over_all_count = 0

    f = open(out_path, 'w')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = -1
    real_last_event = ''
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tag_common = meta[0]
            this_event = meta[1]+'_'+meta[3]
            meta = meta[1:]
            if this_event not in event_ids:
                continue
            if real_last_event != this_event:
                event_number += 1
                real_last_event = this_event
            if event_number >= 40:
                    if html_count > max_html:
                        break
                    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
                    f.write('</table>\n')
                    f.close()
                    html_count += 1
                    out_path = root+'htmls/'+event_type+'_'+str(html_count)+'.html'
                    f = open(out_path, 'w')
                    f.write('<center>')
                    f.write('<table border="1" style="width:100%">\n')
                    event_number = 0
            if this_event != last_event:
                count = 0

                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + this_event + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common + '</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                #f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
                last_event = this_event

            path = meta[15]
            tags = meta[9]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            if len(tags)>line_break:
                position = []
                for i in xrange(1, 1+len(tags)/line_break):
                    position.append(line_break*i+(i-1)*6)
                for pos in position:
                    tags = tags[:pos]+'<br />'+ tags[pos:]
            #f.write('\t\t\t<br /><b>'+str(over_all_count)+' '+date+'</b><br /> '+tags+'\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.close()
    #print count
def images_for_resample(col_num=5):
    f = open(root + 'auxiliary_1000GB_event_id.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()


    list_path = root+'auxiliary_1000GB_events.txt'
    html_count = 0
    out_path = root+'htmls/auxiliary_GraBir_'+str(html_count)+'.html'
    count = 0
    last_event = -1
    over_all_count = 0

    f = open(out_path, 'w')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = -1
    real_last_event = ''
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tag_common = meta[0]
            this_event = meta[1]+'_'+meta[3]
            meta = meta[1:]
            if this_event not in event_ids:
                continue
            if real_last_event != this_event:
                event_number += 1
                real_last_event = this_event
            if event_number >= 40:
                    if html_count > max_html:
                        break
                    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
                    f.write('</table>\n')
                    f.close()
                    html_count += 1
                    out_path = root+'htmls/auxiliary_GraBir_'+str(html_count)+'.html'
                    f = open(out_path, 'w')
                    f.write('<center>')
                    f.write('<table border="1" style="width:100%">\n')
                    event_number = 0
            if this_event != last_event:
                count = 0

                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + this_event + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common + '</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                #f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
                last_event = this_event

            path = meta[15]
            tags = meta[9]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            if len(tags)>line_break:
                position = []
                for i in xrange(1, 1+len(tags)/line_break):
                    position.append(line_break*i+(i-1)*6)
                for pos in position:
                    tags = tags[:pos]+'<br />'+ tags[pos:]
            #f.write('\t\t\t<br /><b>'+str(over_all_count)+' '+date+'</b><br /> '+tags+'\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.close()
    #print count
def images_for_input(col_num=10):
    input_path = root + 'all_input/valid_event_sampled_l1_4_2.csv'
    line_count = -1
    html_count = 0
    out_path = root+'all_input/htmls/4_'+str(html_count)+'.html'
    f = open(out_path, 'w')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')

    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            line_count+=1
            if line_count == 0:
                continue
            num_image = int(meta[0])
            event_id = meta[1]
            event_type = meta[2]
            tag = meta[3]
            f.write('</table>\n')
            f.write('<br><p><b>Event id:' + event_id + 'Event type:'+event_type+' &nbsp;&nbsp;&nbsp;Event tags:' + tag + '</b></p>')
            f.write('<center><table border="1" style="width:100%">\n')
            f.write('\t<tr>\n')
            count = 0
            for i in xrange(num_image):
                f.write('\t\t<td align=\"center\" valign=\"center\">\n')
                f.write('\t\t\t<img src=\"'+meta[i+5]+'\" alt=Loading... width = "200" />\n')
                f.write('\t\t</td>\n')
                count += 1
                if count % col_num == 0:
                    f.write('\t</tr>\n')
            if line_count % 40 ==0:
                f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
                f.write('</table>\n')
                f.close()
                html_count += 1
                out_path = root+'all_input/htmls/4_'+str(html_count)+'.html'
                f = open(out_path, 'w')
                f.write('<center>')
                f.write('<table border="1" style="width:100%">\n')

    f.write('</table>\n')
    f.close()
def images_for_inconsistent(file_name, max_html = 5, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    f=open(root+'balance_/'+file_name+'.cPickle','r')
    agreement_events = cPickle.load(f)
    f.close()
    event_ids = [i[0] for i in agreement_events]
    #print count_type
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    list_path = root+'20000_all_events.txt'
    html_count = 1
    out_path = root+'balance_/html_correct/'+file_name+'_'+str(html_count)+'.html'

    f = open(out_path, 'w')
    count = 0
    last_event = -1
    over_all_count = 0
    #f.write('<p> <br><br>Occurrence of event type:'+', '.join(map(str, count_type))+' <p>')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = -1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tags = meta[0]
            this_event = meta[1]+'_'+meta[3]
            if this_event not in event_ids:
                continue
            if this_event != last_event:
                event_number += 1
                if event_number == 40:
                    if html_count > max_html:
                        break
                    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
                    f.write('</table>\n')
                    f.close()
                    html_count += 1
                    out_path = root+'balance_/html_correct/'+file_name+'_'+str(html_count)+'.html'
                    f = open(out_path, 'w')
                    f.write('<center>')
                    f.write('<table border="1" style="width:100%">\n')
                    event_number = 0

                count = 0
                f.write('</table>\n')
                this_votes = agreement_events[event_ids.index(this_event)][1]
                str_to_write = [key+':'+ str(value) for key,value in this_votes.iteritems()]
                str_to_write = ', '.join(str_to_write)
                f.write('<br><p><b>Event id:' + str(this_event) + ' &nbsp;&nbsp;&nbsp;Tags:' + tags +' &nbsp;&nbsp;&nbsp;Event type and votes:' + str_to_write +'</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                #if write_tags:
                #    f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
                last_event = this_event
            path = meta[16]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
    f.write('</table>\n')
    f.close()
def images_for_consistent_tag(agreement_events, max_html = 5, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    event_ids = [i[0] for i in agreement_events]
    types = [i[1] for i in agreement_events]
    votes = [i[2] for i in agreement_events]
    print event_ids
    #print count_type
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    list_path = root+'20000_all_events.txt'
    html_count = 1
    out_path = root+'htmls/sample_event'+str(html_count)+'.html'

    f = open(out_path, 'w')
    count = 0
    last_event = -1
    over_all_count = 0
    #f.write('<p> <br><br>Occurrence of event type:'+', '.join(map(str, count_type))+' <p>')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = -1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tags = meta[0]
            this_event = meta[1]+'_'+meta[3]
            if this_event not in event_ids:
                continue
            if this_event != last_event:
                event_number += 1

                if event_number >= 40:
                    if html_count > max_html:
                        break
                    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
                    f.write('</table>\n')
                    f.close()
                    html_count += 1
                    out_path = root+'htmls/sample_event'+str(html_count)+'.html'
                    f = open(out_path, 'w')
                    f.write('<center>')
                    f.write('<table border="1" style="width:100%">\n')
                    event_number = 0

                count = 0
                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + str(this_event) + ' &nbsp;&nbsp;&nbsp;Tags:' + tags +' &nbsp;&nbsp;&nbsp;Event type:' + types[event_ids.index(this_event)] + ' &nbsp;&nbsp;&nbsp;Votes:'+ str(votes[event_ids.index(this_event)])+'</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                #if write_tags:
                #    f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
                last_event = this_event
            path = meta[16]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
    f.write('</table>\n')
    f.close()
def create_input_htmls():
    root1 = root+'all_input/'
    input_path = root1 + 'event_sampled_l1_3.csv'
    line_count = 0
    head_meta = []
    event_ids = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[1] not in event_ids:
                event_ids[meta[1]] = [meta]
            else:
                event_ids[meta[1]].append(meta)
            line_count+=1

    for HITId in event_ids:
        this_hits = event_ids[HITId]
        input_field = []
        output_field = []
        for (field, value1) in zip(head_meta, this_hits[0]):
            #print field, value
            input_field.append(('${'+field+'}', value1))
            if 'event_type' in field:
                    name = value1
                    print name
                    name = ''.join(e for e in name if e.isalnum())
                    name = name[:10]

        out_file = root1 +'htmls/input_'+ name +'_'+ HITId + '.html'
        in_file = root + 'Amazon Mechanical Turk_score_hori_result_present.html'
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
                for (field, value1,value2,value3,value4,value5) in output_field:
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'+value1+'**'+value2+'**'+value3+'**'+value4+'**'+value5+'</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for (field, value1,value2,value3,value4,value5) in output_field:
                    if field == 'feedback':
                        pass
                    else:
                        score = 0
                        count_type = Counter([value1,value2,value3,value4,value5])
                        print count_type
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
                        if score > 8:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >5 and score <= 8:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score >= 0 and score <=5:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if score < 0:
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'

        f = open(out_file, 'w')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_htmls(input_path):
    root1 = root
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            elif meta[0] not in HITs:
                HITs[meta[0]] = [meta]
            else:
                HITs[meta[0]].append(meta)
            line_count+=1

    for HITId in HITs:
        number_1 = 0;number_2=1
        this_hits = HITs[HITId]
        input_field = []
        output_field = []
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_type' in field:
                    name = value1
                    print name
                    name = ''.join(e for e in name if e.isalnum())
                    name = name[:10]
                if 'num_image' in field:
                    print value1
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)

        out_file = root1 +'all_events_html/present_'+ name +'_'+ HITId + '.html'
        in_file = root1 + 'Amazon Mechanical Turk_score_hori_result.html'
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
                            line_stack[i] += '\ndocument.getElementById("'+field+key+'").value="'+str(count_type[key])+'";\n'
                            if key == 'selected':
                                score+=count_type[key]*2
                            if key == 'selected_sw':
                                score+=count_type[key]
                            if key == 'selected_irrelevant':
                                score-=count_type[key]*2
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

        f = open(out_file, 'w')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_html(HITId):
    root1 = root+'2_round/'
    input_path = root1 + 'sample_event_result'+suffix+'.csv'
    this_hits = []
    line_count = 0
    head_meta = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            if meta[0]== HITId:
                this_hits.append(meta)
                HITId = meta[0]
            line_count += 1
    if len(this_hits) == 0:
        print "Cannot find line number:",HITId
        return

    input_field = []
    output_field = []
    for (field, value1, value2,value3,value4,value5) in zip(head_meta, this_hits[0],this_hits[1],this_hits[2],this_hits[3],this_hits[4]):
        #print field, value
        if field.startswith('Input.'):
            input_field.append(('${'+field[6:]+'}', value1))
        if field.startswith('Answer.'):
            output_field.append([field[7:], value1,value2,value3,value4,value5])
    out_file = root1 +'sample_result/'+ HITId + '.html'
    in_file = root1 + 'Amazon Mechanical Turk_score_hori_result.html'
    line_stack = []
    with open(in_file, 'r') as data:
        for line in data:
            line_stack.append(line)

    for i in xrange(len(line_stack)):
        for (field, value) in input_field:
            if field in line_stack[i]:
                line_stack[i]=line_stack[i].replace(field, value)

    for i in xrange(len(line_stack)):
        if 'textarea' in line_stack[i]:
            for (field, value1,value2,value3,value4,value5) in output_field:
                if field == 'feedback':
                    line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'+value1+'**'+value2+'**'+value3+'**'+value4+'**'+value5+'</textarea>\n'

        if '$(document).ready(function()' in line_stack[i]:
            for (field, value1,value2,value3,value4,value5) in output_field:
                if field == 'feedback':
                    pass
                else:
                    score = 0
                    count_type = Counter([value1,value2,value3,value4,value5])
                    print count_type
                    for key in count_type:
                        if key == '':
                            continue
                        line_stack[i] += '\ndocument.getElementById("'+field+key+'").value="'+str(count_type[key])+'";\n'
                        if key == 'selected':
                            score+=count_type[key]*2
                        if key == 'selected_sw':
                            score+=count_type[key]
                        if key == 'selected_irrelevant':
                            score-=count_type[key]*2
                    if score > 5:
                        line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                        line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                    if score < -5:
                        line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                        line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'

    f = open(out_file, 'w')
    for line in line_stack:
        f.write(line)
    f.close()
def create_result_html_id(worker_id):
    input_path = root + 'result5.csv'
    line_count = 0
    head_meta = []
    HITs = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            if meta[15]==worker_id:
                HITs.append(meta)
            line_count += 1
    if len(HITs) == 0:
        print "Cannot find line number:", worker_id
        return
    path = root +'usr_more10_result5/'+worker_id + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    for this_hit in HITs:
        HITId = this_hit[0]
        input_field = []
        output_field = []
        for (field, value) in zip(head_meta, this_hit):
            #print field, value
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value))
            if field.startswith('Answer.'):
                output_field.append((field[7:], value))
        out_file = path + HITId + '.html'
        in_file = root + 'Amazon Mechanical Turk_score_hori_result.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)

        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback" COLS=100 ROWS=6>' in line_stack[i]:
                for (field, value) in output_field:
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'+value+'</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for (field, value) in output_field:
                    if field == 'feedback':
                        pass
                    else:
                        if value == '':
                            continue
                        line_stack[i] += '\ndocument.getElementById("'+field+value+'").value="1";\n'
                        if value == 'selected':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if value == 'selected_sw':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if value == 'selected_neutral':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if value == 'selected_irrelevant':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'

        f = open(out_file, 'w')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_html_all_id(input_path, save_path, HITid, worker_id):
    line_count = 0
    head_meta = []
    HITs = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            if meta[15]==worker_id and meta[0] == HITid:
                HITs.append(meta)
            line_count += 1
    if len(HITs) == 0:
        print "Cannot find line number:", worker_id
        return
    path = root +'suspicious_check/'+worker_id + '_'
    for this_hit in HITs:
        HITId = this_hit[0]
        input_field = []
        output_field = []
        for (field, value) in zip(head_meta, this_hit):
            #print field, value
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value))
            if field.startswith('Answer.'):
                output_field.append((field[7:], value))
        out_file = save_path + HITId + '.html'
        in_file = root + 'Amazon Mechanical Turk_score_hori_result.html'
        line_stack = []
        with open(in_file, 'r') as data:
            for line in data:
                line_stack.append(line)

        for i in xrange(len(line_stack)):
            for (field, value) in input_field:
                if field in line_stack[i]:
                    line_stack[i]=line_stack[i].replace(field, value)

        for i in xrange(len(line_stack)):
            if 'textarea NAME="feedback" COLS=100 ROWS=6>' in line_stack[i]:
                for (field, value) in output_field:
                    if field == 'feedback':
                        line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'+value+'</textarea>\n'

            if '$(document).ready(function()' in line_stack[i]:
                for (field, value) in output_field:
                    if field == 'feedback':
                        pass
                    else:
                        if value == '':
                            continue
                        line_stack[i] += '\ndocument.getElementById("'+field+value+'").value="1";\n'
                        if value == 'selected':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("highlight");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if value == 'selected_sw':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("selected_sw");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if value == 'selected_neutral':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("neutral");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'
                        if value == 'selected_irrelevant':
                            line_stack[i]+='$("label[name=\''+field + '\']").addClass("irrelevant");\n'
                            line_stack[i]+='$("label[name=\''+field + '\']").removeClass("not_selected");\n'

        f = open(out_file, 'w')
        for line in line_stack:
            f.write(line)
        f.close()
def create_result_htmls_rearranged(kkk):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/clean_input_and_label/3_event_curation/all_output/'
    root1 = root
    input_path = root1 + 'result'+str(kkk)+'.csv'
    line_count = 0
    head_meta = []
    HITs = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
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
        for i in xrange(len(head_meta)-2):
            field = head_meta[i]
            value1 = this_hits[0][i]
            if field.startswith('Input.'):
                input_field.append(('${'+field[6:]+'}', value1))
                if 'event_type' in field:
                    name = value1
                    print name
                    name = ''.join(e for e in name if e.isalnum())
                    name = name[:10]
                if 'num_image' in field:
                    print value1
                    num_images = int(value1)
            if field.startswith('Answer.'):
                this_ = [field[7:]]
                for this_hit in this_hits:
                    this_.append(this_hit[i])
                output_field.append(this_)
        if not os.path.exists(root1 + 'all_events_html/'+str(kkk)):
            os.mkdir(root1 + 'all_events_html/' + str(kkk))
        out_file = root1 + 'all_events_html/' + str(kkk)+'/rearranged_'+ name +'_'+ HITId + '.html'
        in_file = root1 + 'Amazon Mechanical Turk_score_hori_rearrange.html'
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
                    line_stack[i] += '\ndocument.getElementById("image'+str(k+1)+'selected").value="'+str(sorted_image_list[k][1])+'";\n'
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




        f = open(out_file, 'w')
        for line in line_stack:
            f.write(line)
        f.close()
def random_show_worker():
    input_path = root + 'all_input_sample_result.csv'
    line_count = 0
    worker_ids = {}
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                line_count += 1
                continue
            if meta[15] not in worker_ids:
                worker_ids[meta[15]] = [meta[0]]
            else:
                worker_ids[meta[15]].append(meta[0])
    check_HIT = []
    for worker_id in worker_ids:
        HITs = worker_ids[worker_id]
        check_HIT.append((random.sample(HITs, 1)[0], worker_id))
    for (HIT, worker_id) in check_HIT:
        create_result_html_all_id(HIT, worker_id)
def show_worker_submitted():
    input_path = root + 'all_input_sample_result2.csv'
    line_count = 0
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                line_count += 1
                continue
            if meta[16] != 'Submitted':
                continue
            create_result_html_all_id(meta[0],meta[15])
def show_worker_discrete():
    input_path = root + 'result5.csv'
    line_count = 0
    worker_ids = {}
    HITs = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                line_count += 1
                continue
            if meta[15] not in worker_ids:
                worker_ids[meta[15]] = 1
            else:
                worker_ids[meta[15]] += 1
    worker_ids_valid = []
    for key in worker_ids:
        if worker_ids[key] > 10:
            worker_ids_valid.append(key)

    print worker_ids_valid
    for id in worker_ids_valid:
        create_result_html_id(id)




if __name__ == '__main__':
    #count_events()
    '''
    find_example_result()
    file_name = root+'sample_event.cPickle'
    f = open(file_name, 'r')
    events = cPickle.load(f)
    f.close()

    images_for_consistent_tag(events)


    event_ids = ['0_7453375@N06', '1_37818606@N0', '1_82847947@N00', '4_39983473@N00', '8_96684313N00', '7_51035661423@N01',
                 '3_87622946@N00', '2_34644138@N02', '8_61146724@N00', '3_12071643@N00', '36_8896423@N04', '2_62425546@N00',
                 '19_41987260@N00', '37_53746192@N00', '9_49868460@N00', '1_29065225@N00', '3_61732101@N00', '4_54781523@N00',
                 '51_8896423@N04', '6_43556229@N07', '6_41946087@N04', '38_28110584@N04', '15_7877597@N08', '37_7877597@N08',
                 '45_43682941@N00', '38_79759083@N00', '4_95504187@N00', '25_95504187@N00', '1_33803690@N04', '6_26416016@N02',
                 '11_12584147@N04', '0_28242862@N00', '2_14053687@N04', '0_26332728@N05', '4_45802067@N03', '1_69073148@N04',
                 '28_23311795@N04', '57_66555845@N00', '0_31025097@N07', '0_41818170@N08', '2_49195286@N00', '8_34782855@N00',
                 '1_49503029779@N01', '22_60597745@N00', '5_26104772@N00', '0_39229907@N04', '4_82847947@N00', '2_28832703@N00',
                 '2_7988353@N04', '3_67248646@N05', '0_75707194@N00', '11_82525539@N00', '1_78191101@N00', '24_46702940@N00',
                 '9_68986342@N00', '19_78672040@N00', '1_71158909@N00', '3_9173558@N05', '7_35034347485@N01', '0_20472512@N07',
                 '2_60756254@N07', '17_52111934@N00', '57_7200789@N06', '14_47800690@N03', '1_52725445@N00', '12_44124450615@N01',
                 '0_21968123@N00', '7_23215551@N03', '71_7791881@N04', '6_26673904@N00', '7_23215551@N03', '1_87749039@N00'
                 '10_8138072@N02', '25_82418181@N00', '0_17746744@N08', '0_12484298@N07', '5_75316073@N00', '7_89819035@N00',
                 '1_10907058@N05', '6_85858735@N00', '6_25855244@N04']
    find_result_by_id(event_ids)

    write_example_txt()
    '''
    #write_csv()
    #create_result_htmls()
    #create_result_html('335HHSX8CDSE4CED2H7AYL8OSXMHD3')
    #find_example_result()
    #write_example_txt()
    #write_csv()
    #create_input_htmls()
    #refine_example()
    #sample_data_for_2round()
    #event_types = ['Show', 'UrbanTrip', 'NatureTrip', 'Architecture', 'Sports', 'GroupActivity', 'Zoo', 'PersonalSports',
    #                'BusinessActivity', 'CasualFamilyGather', 'BeachTrip', 'Museum', 'Wedding', 'Protest', 'Graduation',
    #                'ThemePark', 'Cruise', 'Halloween', 'Christmas', 'ReligiousActivity', 'PersonalArtActivity', 'Birthday',
    #                'Independence', 'PersonalMusicActivity']
    #sample_data_for_2round()
    #for event_type in event_types:
    #    images_for_sample(event_type)
    #sample_data_for_2round()
    #write_csv_300()
    #create_result_htmls()
    #suspicious_worker()
    #create_input_htmls()
    #images_for_input()
    #create_result_html_id('A3MKW3AHMMRE28')
    #show_worker_discrete()
    #create_result_html_all_id('37M4O367VJ55SD0VE854SESSW6RM5L', 'AZRNL0S1QKPMP')
    #random_show_worker()
    #show_worker_discrete()
    #create_result_html_id('A1MIR0TP081SKT')
    #find_outlier()
    #check_event_count()
    #label_correction()
    #show_worker_submitted()
    find_outlier()
    # for i in xrange(7):
    #     create_result_htmls_rearranged(i)