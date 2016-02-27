__author__ = 'wangyufei'
import cPickle
import operator
import random
import os
import numpy as np
from collections import Counter
import csv
import sys
root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_recognition/'
#root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'
def create_result_html(this_count):
    input_path = root + 'Batch_2029575_batch_results.csv'
    this_hit = []
    line_count = 0
    head_meta = []
    with open(input_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            if line_count == 0:
                head_meta = meta
            if line_count== this_count:
                this_hit = meta
                HITId = meta[0]
            line_count += 1
    if len(this_hit) == 0:
        print "Cannot find line number:",this_count
        return

    input_field = []
    output_field = []
    for (field, value) in zip(head_meta, this_hit):
        #print field, value
        if field.startswith('Input.'):
            input_field.append(('${'+field[6:]+'}', value))
        if field.startswith('Answer.'):
            output_field.append((field[7:], value))
    out_file = root +'temp_result/'+ HITId + '.html'
    in_file = root + 'classify_event_lightbox_10.html'
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
            for (field, value) in output_field:
                if field == 'feedback':
                    line_stack[i] = '<textarea NAME="feedback" COLS=100 ROWS=6>'+value+'</textarea>\n'

        if '$(document).ready(function()' in line_stack[i]:
            for (field, value) in output_field:
                if field == 'feedback':
                    pass
                else:
                    line_stack[i] += '\n$(\'input:radio[name="'+field+'"]\').filter(\'[value="'+value+'"]\').attr(\'checked\',true);\n'
            break
    f = open(out_file, 'w')
    for line in line_stack:
        f.write(line)
    f.close()
def find_consistent_result(name):
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/'

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
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
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
    for HIT in HIT_result_dic:
        results = HIT_result_dic[HIT]
        if len(results) > 1:
            count+=1
        len_vote = len(results)
        for i in xrange(len(results[0])):
            result_this = {}
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            #if max_result[1] >= 2 and max_result[1] > len_vote/2:
            if max_result[1] == len(result_this) and max_result[1] > len_vote/2:
                agreement_event_id.append([results[0][i][1], max_result[0], max_result[1], results[0][i][3]])
    print "number of events with more than 1 vote:", count*10
    print "number of events with more than 2 same votes:",len(agreement_event_id)
    return agreement_event_id

def find_inconsistent_result(name):
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
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
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
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i],meta[k]] for (j,i,k) in zip(input_index, output_index,tag_index)]])

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
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            if len(results) > 1 and not max_result[1] > len_vote/2:
                #print max_result
                agreement_event_id.append([results[0][i][1],result_this,results[0][i][3]])

    print "number of events with more than 1 vote:", count*10
    print "number of events with no agreements:",len(agreement_event_id)
    return agreement_event_id
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
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer.type' in meta:
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
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i], meta[k]] for (j,i,k) in zip(input_index, output_index, tag_index)]])

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
            if tag not in results[0][i][3]:
                continue
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            if len(results) > 1 and not max_result[1] > len_vote/2:
                if event_type in result_this:
                    #print max_result
                    agreement_event_id.append([results[0][i][1],result_this])

    print "number of events with more than 1 vote:", count*10
    print "number of events with tag '%s', with event type '%s': %d" %(tag, event_type, len(agreement_event_id))
    return agreement_event_id

def find_all_result():
    input_path = root + 'results/162_10_0724.csv'
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
    for i in xrange(len(head_meta)):
        meta = head_meta[i]
        if 'Answer' in meta:
            output_index.append(i)
        if 'event_id' in meta:
            input_index.append(i)
    #feed_back_index = output_index[0]
    output_index = output_index[1:]
    output_index_new = [output_index[0]]+output_index[2:]+[output_index[1]]
    output_index = output_index_new
    HIT_result = []
    for meta in metas:
        HITids.append(meta[0])
        HIT_result.append([meta[0], [[head_meta[i][7:], meta[j], meta[i]] for (j,i) in zip(input_index, output_index)]])

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
            for result in results:
                if result[i][2] in result_this:
                    result_this[result[i][2]] += 1
                else:
                    result_this[result[i][2]] = 1
            max_result = max(result_this.iteritems(), key=operator.itemgetter(1))
            agreement_event_id.append([results[0][i][1],result_this])

    #print "number of events with more than 1 vote:", count*10
    #print "number of events with no agreements:",len(agreement_event_id)
    return agreement_event_id
def images_for_event(event_id, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    agreement_events = find_all_result()
    event_ids = [i[0] for i in agreement_events]
    #types = [i[1] for i in agreement_events]
    #votes = [i[2] for i in agreement_events]
    #count_type = Counter(types)
    #count_type = list(count_type.items())

    #print count_type
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    list_path = root+'more_graduation_imgs_0719.txt'
    out_path = root+'results/sample_event_'+event_id+'_148.html'

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
            if this_event != event_id:
                continue
            if this_event != last_event:
                event_number += 1
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

def images_for_consistent(agreement_events,event_type,max_html = 5, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    """

    :param agreement_events:
    :param event_type:
    :param max_html:
    :param col_num:
    :param min_occurrence:
    :param min_tag:
    :param write_tags:
    """
    event_ids = [i[0] for i in agreement_events]
    types = [i[1] for i in agreement_events]
    votes = [i[2] for i in agreement_events]
    count_type = Counter(types)
    count_type = sorted(count_type.items(), key=lambda pair: pair[1], reverse=True)
    print count_type
    list_path = root+'20000_all_events.txt'
    html_count = 1
    out_path = root+'htmls/agreement_'+event_type+'_'+str(html_count)+'.html'
    f = open(out_path, 'w')
    count = 0
    last_event = -1
    over_all_count = 0
    f.write('<p> <br><br>Occurrence of event type:'+', '.join(map(str, count_type))+' <p>')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = 0
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tags = meta[0]
            this_event = meta[1]+'_'+meta[3]
            if this_event not in event_ids:
                continue
            if event_type != types[event_ids.index(this_event)]:
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
                    out_path = root+'htmls/agreement_'+event_type+'_'+str(html_count)+'.html'
                    f = open(out_path, 'w')
                    f.write('<p> <br><br>Occurrence of event type:'+', '.join(map(str, count_type))+' <p>')
                    f.write('<center>')
                    f.write('<table border="1" style="width:100%">\n')
                    event_number = 0
                count = 0
                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + str(this_event) + ' &nbsp;&nbsp;&nbsp;Tags:' + tags +' &nbsp;&nbsp;&nbsp;Event type:' + types[event_ids.index(this_event)] + ' &nbsp;&nbsp;&nbsp;Votes:'+ str(votes[event_ids.index(this_event)])+'</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
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
def images_for_inconsistent(agreement_events, event_type, max_html = 5, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    event_ids = [i[0] for i in agreement_events]
    #print count_type
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    list_path = root+'20000_all_events.txt'
    html_count = 1
    out_path = root+'htmls/not_agreement_'+event_type+'_'+str(html_count)+'.html'

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
            if event_type not in tags:
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
                    out_path = root+'htmls/not_agreement_'+event_type+'_'+str(html_count)+'.html'
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
def images_for_consistent_tag(agreement_events, event_type, max_html = 5, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    event_ids = [i[0] for i in agreement_events]
    #print count_type
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    list_path = root+'20000_all_events.txt'
    html_count = 1
    out_path = root+'htmls/agreement_tag_'+event_type+'_'+str(html_count)+'.html'

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
            if event_type not in tags:
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
                    out_path = root+'htmls/agreement_tag_'+event_type+'_'+str(html_count)+'.html'
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
def tags_for_consistent(agreement_events, col_num=10, min_occurrence=5000,min_tag=3, write_tags = True):
    event_ids = [i[0] for i in agreement_events]
    types = [i[1] for i in agreement_events]
    votes = [i[2] for i in agreement_events]
    count_type = Counter(types)
    count_type = list(count_type.items())

    print count_type
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    list_path = root+'more_graduation_imgs_0719.txt'
    out_path = root+'large_results/012_tag_agreement.html'

    f = open(out_path, 'w')
    count = 0
    last_event = -1
    over_all_count = 0
    f.write('<p> <br><br>Occurrence of event type:'+', '.join(map(str, count_type))+' <p>')
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    event_number = 1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tags = meta[0]
            this_event = meta[1]+'_'+meta[3]
            if this_event not in event_ids:
                continue
            if this_event != last_event:
                event_number += 1
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
            #f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            #f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            #f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('<style type=\"text/css\">img {{ height:auto;width:\"200px\";}}<\style>')
    f.write('</table>\n')
    f.close()


if __name__ == '__main__':
    #
    # f = open(root+'all_event_notlabel.cPickle','r')
    # all_agreements = cPickle.load(f)
    # f.close()
    # images_for_inconsistent(all_agreements, 'Birthday')
    #
    # f = open(root+'all_event_Birthday.cPickle','r')
    # all_agreements = cPickle.load(f)
    # f.close()
    # images_for_consistent_tag(all_agreements, 'Birthday')
    #
    #
    # f = open(root+'all_event_label.cPickle','r')
    # all_agreements = cPickle.load(f)
    # f.close()
    # images_for_consistent(all_agreements, 'NatureTrip')
    # images_for_consistent(all_agreements, 'Sports')
    # images_for_consistent(all_agreements, 'OtherActivity')
    # images_for_consistent(all_agreements, 'OtherImportantLife')
    # images_for_consistent(all_agreements, 'OtherHoliday')

    '''

    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'

    a = find_consistent_result('0')
    b = find_consistent_result('1')
    c = find_consistent_result('2')
    e = find_consistent_result('3')
    f = find_consistent_result('4')
    g = find_consistent_result('5')
    h = find_consistent_result('8_pre')
    i = find_consistent_result('6')
    j = find_consistent_result('7')
    all_agreements = a+b+c+e+f+g+h+i+j
    f = open(root+'all_event_label.cPickle','w')
    cPickle.dump(all_agreements, f)
    f.close()

    event_ids = [i[0] for i in all_agreements]
    types = [i[1] for i in all_agreements]
    votes = [i[2] for i in all_agreements]
    count_type = Counter(types)
    count_type = sorted(count_type.items(), key=lambda pair: pair[1], reverse=True)
    print count_type


    a = find_inconsistent_result('0')
    b = find_inconsistent_result('1')
    c = find_inconsistent_result('2')
    e = find_inconsistent_result('3')
    f = find_inconsistent_result('4')
    g = find_inconsistent_result('5')
    h = find_inconsistent_result('8_pre')
    i = find_inconsistent_result('6')
    j = find_inconsistent_result('7')
    all_agreements = a+b+c+e+f+g+h+i+j
    f = open(root+'all_event_notlabel.cPickle','w')
    cPickle.dump(all_agreements, f)
    f.close()
    pass

    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/all_results/'
    tag = 'birthday'
    event_type = 'Birthday'
    a = find_consistent_with_tag('0',tag,event_type)
    b = find_consistent_with_tag('1',tag,event_type)
    c = find_consistent_with_tag('2',tag,event_type)
    e = find_consistent_with_tag('3',tag,event_type)
    f = find_consistent_with_tag('4',tag,event_type)
    g = find_consistent_with_tag('5',tag,event_type)
    h = find_consistent_with_tag('8_pre',tag,event_type)
    i = find_consistent_with_tag('6',tag,event_type)
    j = find_consistent_with_tag('7',tag,event_type)
    all_agreements = a+b+c+e+f+g+h+i+j
    f = open(root+'all_event_'+event_type+'.cPickle','w')
    cPickle.dump(all_agreements, f)
    f.close()
    '''



    a = find_consistent_result('0')
    b = find_consistent_result('1')
    c = find_consistent_result('2')
    e = find_consistent_result('3')
    f = find_consistent_result('4')
    g = find_consistent_result('5')
    h = find_consistent_result('8_pre')
    i = find_consistent_result('6')
    j = find_consistent_result('7')
    all_agreements = a+b+c+e+f+g+h+i+j
    print all_agreements