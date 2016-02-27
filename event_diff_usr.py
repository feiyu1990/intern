__author__ = 'wangyufei'
import numpy as np
import sys
import random
import os
import optparse
from os.path import isfile, join
from os import listdir
import cPickle
import urllib
import random
import operator

min_occurrence = 5000
min_tag = 3


def find_usr(tag,i=0):
    list_path = '../datasets/all_data/clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/'+str(i)+'.txt'
    usr_dict = {}
    with open(list_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[8]
            tag_individuals = tags.split(',')
            has_key = False
            for tag_this in tag_individuals:
                if tag in tag_this:
                    has_key = True
            if not has_key:
                continue
            #if not tag in tag_individuals:
            #    continue
            usr = meta[1]
            if usr in usr_dict.keys():
                usr_dict[usr] += 1
            else:
                usr_dict[usr] = 1
    usr_dict_sorted = sorted(usr_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    print usr_dict_sorted[:10]
    usrs = list(usr_dict_sorted.keys())[:100]
    usrs_sampled = random.sample(usrs, 20)


def images_for_event(usr_id, event_type, col_num, min_occurrence=5000,min_tag=3, write_tags = True):
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
    out_root = '../datasets/html_events/'

    list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/event_clean_tags/'+usr_id+'.txt'

    events_id = []
    #print events_id
    last_event = -1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            meta = meta[1:]
            this_event = int(meta[0])
            #print meta[7]
            #print meta[8]
            #print meta[9]
            this_tag = meta[9]
            if event_type not in this_tag:
                continue
            if this_event != last_event:
                events_id.append(this_event)
                last_event = this_event
    print events_id
    if len(events_id) > maximum_events:
        events_id = random.sample(events_id, maximum_events)
        #events_id[0]=27
        events_id.sort()
        print 'Only selecting '+str(maximum_events)+' events. Event id selected:', events_id

    out_path = out_root + event_type + '_' +usr_id+'_'+str(min_occurrence)+'_'+str(min_tag)+'.html'
    f = open(out_path, 'w')
    #f.write('<head> <title> USER ID:'+usr_id+'</title></head><h1>Images from user:'+usr_id+'; number of images:'+number_of_images+'</h1>\n' )
    f.write('<head> <title> USER ID:'+usr_id+'</title></head><h1>Images from user:'+usr_id+'</h1>\n' )
    count = 0
    #tag_common = ''
    last_event = -1
    over_all_count = 0
    f.write('<center>')
    #if event_type == 'valid':
    #    f.write('<br><p><b>Event id:' + str(last_event) + ' &nbsp;&nbsp;&nbsp;Event tags:' +tag_common+ '</b></p>')
    #else:
    #    f.write('<br><p><b>Event id:' + str(last_event) + '</b></p>')
    f.write('<table border="1" style="width:100%">\n')
    #f.write('\t<tr>\n')
    event_number = 1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tag_common = meta[0]
            meta = meta[1:]
            this_event = int(meta[0])
            if this_event not in events_id:
                continue

            if this_event != last_event:
                event_number += 1
                if event_number > maximum_events:
                    print 'Write only 20 events! '
                    break
                count = 0
                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + str(this_event) + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common + '</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                if write_tags:
                    f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
                last_event = this_event

            path = meta[15]
            tags = meta[9]
            date = meta[4]
            count += 1
            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... width = "200" />\n')
            if len(tags)>line_break:
                position = []
                for i in xrange(1, 1+len(tags)/line_break):
                    position.append(line_break*i+(i-1)*6)
                for pos in position:
                    tags = tags[:pos]+'<br />'+ tags[pos:]
            f.write('\t\t\t<br /><b>'+str(over_all_count)+' '+date+'</b><br /> '+tags+'\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.close()





if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1
    event_name = (args[1])
    if len(args) > 2:
        i = args[2]
    else:
        i = 0
    find_usr(event_name, i)