__author__ = 'wangyufei'

import csv
import cPickle
import operator
from collections import Counter
import random
restrict_length = 100
root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'

'''correct birthday/wedding/graduation -> casual family gathering'''
def find_consistent_with_tag(name, tag, event_type):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'
    input_path = root +'results/'+name+'.csv'
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

    output_index = []
    input_index = []
    tag_index = []
    length_index = []
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            output_index.append(i)
        if 'event_id' in meta:
            input_index.append(i)
        if 'tag' in meta:
            tag_index.append(i)
        if 'num_image' in meta:
            length_index.append(i)
    #feed_back_index = output_index[0]
    #output_index = output_index[1:]
    output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
    output_index = output_index_new
    HIT_result = []
    for meta in metas:
        HITids.append(meta[0])
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k],meta[ii]] for (j,i,k,ii) in zip(input_index, output_index, tag_index,length_index)]])

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
    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        if len(results) > 1:
            count+=1
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            if int(results[0][i][4])>restrict_length:
                continue
            if tag not in results[0][i][3]:
                continue
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            #if len(results) > 1 and not max_result[1] > len_vote/2:
            if event_type in result_this and result_this[event_type] <= len_vote/2:
                #print max_result
                agreement_event_id.append([results[0][i][1],result_this, results[0][i][3]])

    print "number of events with more than 1 vote:", count*10
    print "number of events with tag '%s', with event type '%s': %d" %(tag, event_type, len(agreement_event_id))
    return agreement_event_id
def find_correct_CasualFamily(name, tag, event_correct, event_type='CasualFamilyGather'):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'
    input_path = root +'results/'+name+'.csv'
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

    output_index = []
    input_index = []
    tag_index = []
    length_index = []
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
            output_index.append(i)
        if 'event_id' in meta:
            input_index.append(i)
        if 'tag' in meta:
            tag_index.append(i)
        if 'num_image' in meta:
            length_index.append(i)
    #feed_back_index = output_index[0]
    #output_index = output_index[1:]
    output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
    output_index = output_index_new
    HIT_result = []
    for meta in metas:
        HITids.append(meta[0])
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k],meta[ii]] for (j,i,k,ii) in zip(input_index, output_index, tag_index,length_index)]])

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
    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        if len(results) > 1:
            count+=1
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            if int(results[0][i][4])>restrict_length:
                continue
            if tag not in results[0][i][3]:
                continue
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            #if len(results) > 1 and not max_result[1] > len_vote/2:
            if event_type in result_this and result_this[event_type] >= len_vote/2:
                if event_correct not in result_this:
                #print max_result
                    agreement_event_id.append([results[0][i][1],result_this, results[0][i][3]])

    print "number of events with more than 1 vote:", count*10
    print "number of events with tag '%s', with event type '%s': %d" %(tag, event_type, len(agreement_event_id))
    return agreement_event_id
def correct_important_life():
    '''
    files = ['0','1','2','3','4','5','6','7','8_pre']
    Casual_wedding = []; Casual_birthday=[];Casual_graduation=[]

    for file_name in files:
        event_id = find_correct_CasualFamily(file_name, 'wedding','Wedding')
        print len(event_id)
        Casual_wedding.extend(event_id)
        event_id = find_correct_CasualFamily(file_name, 'graduation','Graduation')
        print len(event_id)
        Casual_graduation.extend(event_id)
        event_id = find_correct_CasualFamily(file_name, 'birthday','Birthday')
        print len(event_id)
        Casual_birthday.extend(event_id)

    #f = open(root+'balance_/Casual_to_birthday.cPickle','w');cPickle.dump(Casual_birthday,f);f.close()
    #f = open(root+'balance_/Casual_to_wedding.cPickle','w');cPickle.dump(Casual_wedding,f);f.close()
    #f = open(root+'balance_/Casual_to_graduation.cPickle','w');cPickle.dump(Casual_graduation,f);f.close()


    for file_name in files:
        event_id = find_correct_CasualFamily(file_name, 'wedding','Wedding','GroupActivity')
        print len(event_id)
        Casual_wedding.extend(event_id)
        event_id = find_correct_CasualFamily(file_name, 'graduation','Graduation','GroupActivity')
        print len(event_id)
        Casual_graduation.extend(event_id)
        event_id = find_correct_CasualFamily(file_name, 'birthday','Birthday','GroupActivity')
        print len(event_id)
        Casual_birthday.extend(event_id)

    f = open(root+'balance_/Group_to_birthday.cPickle','w');cPickle.dump(Casual_birthday,f);f.close()
    f = open(root+'balance_/Group_to_wedding.cPickle','w');cPickle.dump(Casual_wedding,f);f.close()
    f = open(root+'balance_/Group_to_graduation.cPickle','w');cPickle.dump(Casual_graduation,f);f.close()


    manual_pick_wedding = ['13_8743691@N02','0_45417981@N00','0_29058452@N06','9_59353887@N00']
    manual_pick_birthday = ['1_89803116@N00', '58_97863854@N00', '1_90427028@N00','1_74628614@N00','3_81487723@N00',
                            '6_31348155@N03','0_19039450@N00','3_54084941@N00','10_27813593@N00','22_64596573@N00',
                            '19_94588149@N00','39_35578067@N00','152_82394503@N00','1_89225742@N00','6_48889048097@N01',
                            '14_84764695@N04','0_85644838@N00','2_69072483@N00','1_40112223@N05','3_67604078@N00',
                            '3_35468143341@N01','50_35118454@N00','6_93995264@N00','103_99357189@N00']
    manual_pick_graduation = ['4_97867946@N00','1_21353497@N00','4_88178000@N00']
    f = open(root+'balance_/CasualGroup_to_birthday.cPickle','r')
    Casual_birthday = cPickle.load(f)
    f.close()
    f = open(root+'balance_/CasualGroup_to_wedding.cPickle','r')
    Casual_wedding = cPickle.load(f)
    f.close()
    f = open(root+'balance_/CasualGroup_to_graduation.cPickle','r')
    Casual_graduation = cPickle.load(f)
    f.close()
    manual_birthday = [];manual_graduation=[];manual_wedding=[]
    for birthday in Casual_birthday:
        if birthday[0] in manual_pick_birthday:
            manual_birthday.append(birthday)
    for graduation in Casual_graduation:
        if graduation[0] in manual_pick_graduation:
            manual_graduation.append(graduation)
    for wedding in Casual_wedding:
        if wedding[0] in manual_pick_wedding:
            manual_wedding.append(wedding)


    f = open(root+'balance_/manual_to_birthday.cPickle','w');cPickle.dump(manual_birthday,f);f.close()
    f = open(root+'balance_/manual_to_wedding.cPickle','w');cPickle.dump(manual_wedding,f);f.close()
    f = open(root+'balance_/manual_to_graduation.cPickle','w');cPickle.dump(manual_graduation,f);f.close()


    files = ['0','1','2','3','4','5','6','7','8_pre']
    event_birthday = []
    for file_name in files:
        event_id = find_consistent_with_tag(file_name, 'birthday','Birthday')
        print len(event_id)
        event_birthday.extend(event_id)
    event_graduation = []
    for file_name in files:
        event_id = find_consistent_with_tag(file_name, 'graduation','Graduation')
        print len(event_id)
        event_graduation.extend(event_id)
    event_wedding = []
    for file_name in files:
        event_id = find_consistent_with_tag(file_name, 'wedding','Wedding')
        print len(event_id)
        event_wedding.extend(event_id)

    f = open(root+'balance_/tag_birthday.cPickle','w');cPickle.dump(event_birthday,f);f.close()
    f = open(root+'balance_/tag_wedding.cPickle','w');cPickle.dump(event_wedding,f);f.close()
    f = open(root+'balance_/tag_graduation.cPickle','w');cPickle.dump(event_graduation,f);f.close()
    '''
    f = open(root+'balance_/manual_to_birthday.cPickle','r');event_birthday=cPickle.load(f);f.close()
    f = open(root+'balance_/manual_to_wedding.cPickle','r');event_wedding = cPickle.load(f);f.close()
    f = open(root+'balance_/manual_to_graduation.cPickle','r');event_graduation=cPickle.load(f);f.close()
    event_birthday_new = []
    for i in event_birthday:
        if i in event_birthday_new:
            continue
        event_birthday_new.append(i)
    event_wedding_new = []
    for i in event_wedding:
        if i in event_wedding_new:
            continue
        event_wedding_new.append(i)
    event_graduation_new = []
    for i in event_graduation:
        if i in event_graduation_new:
            continue
        event_graduation_new.append(i)
    f = open(root+'balance_/manual_to_birthday.cPickle','w');cPickle.dump(event_birthday_new,f);f.close()
    f = open(root+'balance_/manual_to_wedding.cPickle','w');cPickle.dump(event_wedding_new,f);f.close()
    f = open(root+'balance_/manual_to_graduation.cPickle','w');cPickle.dump(event_graduation_new,f);f.close()


    f = open(root+'balance_/tag_birthday.cPickle','r');event_birthday=cPickle.load(f);f.close()
    f = open(root+'balance_/tag_wedding.cPickle','r');event_wedding = cPickle.load(f);f.close()
    f = open(root+'balance_/tag_graduation.cPickle','r');event_graduation=cPickle.load(f);f.close()

    f = open(root+'balance_/manual_to_birthday.cPickle','r');event_birthday2=cPickle.load(f);f.close()
    f = open(root+'balance_/manual_to_wedding.cPickle','r');event_wedding2=cPickle.load(f);f.close()
    f = open(root+'balance_/manual_to_graduation.cPickle','r');event_graduation2=cPickle.load(f);f.close()


    temp = [i[0] for i in event_birthday]
    print sorted(temp)

    for event in event_birthday2:
        if event[0] not in temp:
            event_birthday.append(event)
    temp = [i[0] for i in event_wedding]
    for event in event_wedding2:
        if event[0] not in temp:
            event_wedding.append(event)
    temp = [i[0] for i in event_graduation]
    for event in event_graduation2:
        if event[0] not in temp:
            event_graduation.append(event)

    for i in xrange(len(event_birthday)):
        if event_birthday[i][0] == '12_66922282@N00':
            del[event_birthday[i]]
            break
    for i in xrange(len(event_graduation)):
        if event_graduation[i][0] == '3_11764347@N07':
            del[event_graduation[i]]
            break

    f = open(root+'balance_/all_birthday.cPickle','w');cPickle.dump(event_birthday,f);f.close()
    f = open(root+'balance_/all_wedding.cPickle','w');cPickle.dump(event_wedding,f);f.close()
    f = open(root+'balance_/all_graduation.cPickle','w');cPickle.dump(event_graduation,f);f.close()
def add_important_life():
    f = open(root+'balance_/all_birthday.cPickle','r');event_birthday=cPickle.load(f);f.close()
    f = open(root+'balance_/all_wedding.cPickle','r');event_wedding = cPickle.load(f);f.close()
    f = open(root+'balance_/all_graduation.cPickle','r');event_graduation=cPickle.load(f);f.close()

    f = open(root+'balance_/l100_event_label.cPickle','r')
    event_valid_100 = cPickle.load(f)
    event_ids = [i[0] for i in event_valid_100]
    f.close()

    event_ids = [i[0] for i in event_valid_100]
    types = [i[1] for i in event_valid_100]
    votes = [i[2] for i in event_valid_100]
    count_type = Counter(types)
    count_type = sorted(count_type.items(), key=lambda pair: pair[1], reverse=True)
    print count_type


    for event in event_birthday:
        if event[0] not in event_ids:
            event_valid_100.append([event[0],'Birthday',-1,event[2]])
        else:
            i = event_ids.index(event[0])
            #print event_valid_100[i]
            event_valid_100[i][1] = 'Birthday'
            event_valid_100[i][2] = -1
            #print event_valid_100[i]
    for event in event_wedding:
        if event[0] not in event_ids:
            event_valid_100.append([event[0],'Wedding',-1,event[2]])
        else:
            i = event_ids.index(event[0])
            #print event_valid_100[i]
            event_valid_100[i][1] = 'Wedding'
            event_valid_100[i][2] = -1
            #print event_valid_100[i]
    for event in event_graduation:
        if event[0] not in event_ids:
            event_valid_100.append([event[0],'Graduation',-1,event[2]])
        else:
            i = event_ids.index(event[0])
            #print event_valid_100[i]
            event_valid_100[i][1] = 'Graduation'
            event_valid_100[i][2] = -1
            #print event_valid_100[i]

    event_ids = [i[0] for i in event_valid_100]
    types = [i[1] for i in event_valid_100]
    votes = [i[2] for i in event_valid_100]
    count_type = Counter(types)
    count_type = sorted(count_type.items(), key=lambda pair: pair[1], reverse=True)
    print count_type

    f = open(root+'balance_/l100_event_label_importantlife.cPickle','w');cPickle.dump(event_valid_100,f);f.close()

    f = open(root+'balance_/l100_event_notlabel.cPickle','r');event_notlabel=cPickle.load(f);f.close()
    event_notlabel_new = []
    for event in event_notlabel:
        if event[0] in event_ids:
            print event
        else:
            event_notlabel_new.append(event)
    f = open(root+'balance_/l100_event_notlabel_importantlife.cPickle','w');cPickle.dump(event_notlabel_new,f);f.close()
def length_validation_20000():
    f = open(root+'20000_all_events_length.cPickle','r')
    event_valid_100 = []
    event_length = cPickle.load(f)
    f.close()
    event_length = {a:b for b,a in event_length}
    f = open(root+'all_event_label.cPickle','r')
    event_labels = cPickle.load(f)
    f.close()
    for event_ in event_labels:
        event_id = event_[0]
        length = event_length[event_id]
        if length <= restrict_length:
            event_valid_100.append(event_)


    event_ids = [i[0] for i in event_valid_100]
    types = [i[1] for i in event_valid_100]
    votes = [i[2] for i in event_valid_100]
    count_type = Counter(types)
    count_type = sorted(count_type.items(), key=lambda pair: pair[1], reverse=True)
    print count_type
    f = open(root+'balance_/l100_event_label.cPickle','w')
    cPickle.dump(event_valid_100, f)
    f.close()


    event_valid_100 = []
    f = open(root+'all_event_notlabel.cPickle','r')
    event_labels = cPickle.load(f)
    f.close()
    for event_ in event_labels:
        event_id = event_[0]
        length = event_length[event_id]
        if length <= restrict_length:
            event_valid_100.append(event_)
    f = open(root+'balance_/l100_event_notlabel.cPickle','w')
    cPickle.dump(event_valid_100, f)
    f.close()


if __name__ == '__main__':
    #correct_important_life()
    #length_validation_20000()
    #images_for_inconsistent('all_birthday')
    #images_for_inconsistent('all_wedding')
    #images_for_inconsistent('all_graduation')
    #add_important_life()
    #write_example_txt()
    #write_csv()
    #write_csv_400()
    '''
    event_ids = {}
    for i in xrange(5):
        line_count = 0
        path = root + '2_round/all_input/event_sampled_l1_'+str(i)+'.csv'
        with open(path,'r') as data:
            reader = csv.reader(data)
            for meta in reader:
                if line_count == 0:
                    line_count += 1
                    continue
                if meta[1] in event_ids:
                    event_ids[meta[1]] += 1
                else:
                    event_ids[meta[1]] = 1
    line_count = 0
    path = root + '2_round/data_prepare/event_sampled_l1.csv'
    with open(path,'r') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                line_count += 1
                continue
            if meta[1] not in event_ids:
                print meta
    '''
    correct_csv()





