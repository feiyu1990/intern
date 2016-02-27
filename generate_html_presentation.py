__author__ = 'wangyufei'
'''
This is for:
images_for_tag(tag) -> images for certain tag (from 1M dataset sample)
    'python generate_html.py tag @TAG'
images_for_usr(usr_rank) -> images for certain user based on its occurrence rank (from 1M dataset sample)
    'python generate_html.py usr @rank'
images_for_usr_all(usr_id) -> images for certain user
    'python generate_html.py usr_all @ID'
images_for_usr_event(usr_id) -> images for certain user, separate events (based on the separate event txt files)
    'python generate_html.py usr_event @ID'
'''

import sys
import os
import random
import cPickle

min_occurrence = 50
min_tag = 3
line_break = 30
maximum_events = 50
def images_for_usr_event(events_id, usr_id, event_type, col_num, min_occurrence=5000,min_tag=3):
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
    out_root = '../datasets/html_events/'

    if event_type == 'all':
        list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events/'+usr_id+'.txt'
    elif event_type == 'valid':
        list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/event_clean_tags/'+usr_id+'.txt'
    elif event_type == 'abandoned':
        list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/event_abandoned/'+usr_id+'.txt'
    else:
        raise ValueError('ACCEPTABLE INPUT: all/valid/abandoned')

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    out_path = out_root + 'pres_' + event_type + '_' +usr_id+'_'+str(min_occurrence)+'_'+str(min_tag)+'.html'
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
            if event_type == 'valid':
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
                if event_type == 'valid':
                    f.write('<br><p><b>Event id:' + str(this_event) + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common + '</b></p>')
                else:
                    f.write('<br><p><b>Event id:' + str(this_event) + '</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
                #f.write('<tr><td colspan='+str(col_num)+'>' + 'Event id: ' + str(this_event) + '</td></tr>\n')
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
            #f.write('\t\t\t<br /><b>'+str(over_all_count)+' '+date+'</b><br /> '+tags+'\n')
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.close()
def images_for_ids(load_name, col_num, min_occurrence=5000,min_tag=3):
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/event_recognition/'
    out_root = root+'/htmls/'

    f = open(root+load_name+'.cPickle','r')
    event_ids = cPickle.load(f)
    f.close()
    list_path = root+'20000_all_events.txt'
    out_path = out_root + load_name + '.html'
    f = open(out_path, 'w')
    f.write('<head> <title> '+load_name+'</title></head>\n')
    count = 0
    last_event = -1
    over_all_count = 0
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    #f.write('\t<tr>\n')
    event_number = 1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            tag_common = meta[0]
            meta = meta[1:]
            this_event = meta[0]+'_'+meta[2]
            if this_event not in event_ids:
                continue
            if this_event != last_event:
                event_number += 1
                if event_number > maximum_events:
                    print 'Write only 50 events! '
                    break
                count = 0
                f.write('</table>\n')
                f.write('<br><p><b>Event id:' + str(this_event) + ' &nbsp;&nbsp;&nbsp;Event tags:' + tag_common + '</b></p>')
                f.write('<center><table border="1" style="width:100%">\n')
                f.write('\t<tr>\n')
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
            f.write('\t\t</td>\n')
            if count % col_num == 0:
                f.write('\t</tr>\n')
            over_all_count += 1
    f.write('</table>\n')
    f.close()


if __name__ == '__main__':
    images_for_ids('Birthday_to_Casual', 5)