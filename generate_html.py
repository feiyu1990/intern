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

min_occurrence = 50
min_tag = 3
line_break = 30
maximum_events = 20
def images_for_tag(tag, num_images=1000, col_num=2):

    out_root = '../datasets/html_tags/'
    list_path = '../datasets/clean_all_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    out_path = out_root + 'tag_'+tag+'.html'
    f = open(out_path, 'w')
    f.write('<head> <title> TAG:'+tag+'</title></head><h1>Images that contain tag:'+tag+'</h1>\n' )
    f.write('<center><table border="1" style="width:500">\n')
    f.write('\t<tr>\n')
    count = 0
    with open(list_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[8]
            path = meta[14]
            tag_individuals = tags.split(',')
            if not tag in tag_individuals:
                continue
            count += 1
            if count > num_images:
                break

            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... />\n')
            if len(tags)>line_break:
                position = []
                for i in xrange(len(tags)/line_break):
                    position.append(line_break*i+6*(i-1))
                for pos in position:
                    tags = tags[:pos]+'<br />'+ tags[pos:]
            f.write('\t\t\t<br /> '+tags+'\n')
            f.write('\t\t</td>\n')

            if count % col_num == 0:
                f.write('\t</tr>\n')
    f.write('</table>\n')
    f.close()
def images_for_usr(usr_rank, col_num=2):
    out_root = '../datasets/html_users/'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    usr_count_path = '../datasets/clean_usr_count_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    list_path = '../datasets/clean_all_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    count = 0
    usr_id = ''
    number_of_images = '0'
    with open(usr_count_path,'r') as data:
        for line in data:
            count += 1
            if count == usr_rank:
                temp = line.split('\t')
                usr_id = temp[0]
                number_of_images = temp[1].split('\n')[0]
                break
    out_path = out_root + str(usr_rank)+'_usr_'+usr_id+'.html'
    f = open(out_path, 'w')
    f.write('<head> <title> USER ID:'+usr_id+'</title></head><h1>Images from user:'+usr_id+'; number of images:'+number_of_images+'</h1>\n' )

    f.write('<center><table border="1" style="width:500">\n')
    f.write('\t<tr>\n')
    count = 0
    images = []
    with open(list_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            this_id = meta[1]
            path = meta[14]
            tags = meta[8]
            date = meta[3]
            if this_id == usr_id:
                #print this_id, meta[2]
                temp = date.split(' ')
                date_info = temp[0]
                time_info = temp[1]
                temp = date_info.split('-')
                #print temp
                m = temp[1]
                d = temp[2]
                y = temp[0]
                h = time_info.split(':')[0]
                minute = time_info.split(':')[1]
                second = time_info.split(':')[2]
                time_this = float(y+m+d+h+minute+second)
                images.append((path,tags,time_this, date))

    images_sorted = sorted(images, key=lambda x:x[2])
    for path, tags, time_this,date in images_sorted:
                count += 1
                f.write('\t\t<td align=\"center\" valign=\"center\">\n')
                f.write('\t\t\t<img src=\"'+path+'\" alt=Loading... />\n')
                if len(tags)>line_break:
                    position = []
                    for i in xrange(len(tags)/line_break):
                        position.append(line_break*i+6*(i-1))
                    for pos in position:
                        tags = tags[:pos]+'<br />'+ tags[pos:]
                f.write('\t\t\t<br /><b>'+date+'</b><br /> '+tags+'\n')
                f.write('\t\t</td>\n')

                if count % col_num == 0:
                    f.write('\t</tr>\n')
    f.write('</table>\n')
    f.close()
def images_for_usr_all(usr_id, col_num=5, min_occurrence=5000,min_tag=3):
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
    out_root = '../datasets/html_users/'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    #usr_count_path = '../datasets/clean_usr_count_'+str(min_occurrence)+'_'+str(min_tag)+'.txt'
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/usr_'+usr_id+'/usr.txt'
    list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/all_users_time/'+usr_id+'.txt'
    count = 0
    #usr_id = ''
    #number_of_images = '0'
    #with open(usr_count_path,'r') as data:
    #    for line in data:
    #        count += 1
    #        if count == usr_rank:
    #            temp = line.split('\t')
    #            usr_id = temp[0]
    #            number_of_images = temp[1].split('\n')[0]
    #            break
    out_path = out_root + 'all_usr_'+usr_id+'_'+str(min_occurrence)+'_'+str(min_tag)+'.html'
    f = open(out_path, 'w')
    #f.write('<head> <title> USER ID:'+usr_id+'</title></head><h1>Images from user:'+usr_id+'; number of images:'+number_of_images+'</h1>\n' )
    f.write('<head> <title> USER ID:'+usr_id+'</title></head><h1>Images from user:'+usr_id+'</h1>\n' )

    f.write('<center><table border="1" style="width:100%">\n')
    f.write('\t<tr>\n')
    count = 0
    images = []
    with open(list_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            this_id = meta[1]
            path = meta[14]
            tags = meta[8]
            date = meta[3]
            if this_id == usr_id:
                #print this_id, meta[2]
                temp = date.split(' ')
                date_info = temp[0]
                time_info = temp[1]
                temp = date_info.split('-')
                #print temp
                m = temp[1]
                d = temp[2]
                y = temp[0]
                h = time_info.split(':')[0]
                minute = time_info.split(':')[1]
                second = time_info.split(':')[2]
                time_this = float(y+m+d+h+minute+second)
                images.append((path,tags,time_this, date))

    images_sorted = sorted(images, key=lambda x:x[2])
    for path, tags, time_this,date in images_sorted:
                count += 1
                if count >= 1000:
                    break
                f.write('\t\t<td align=\"center\" valign=\"center\">\n')
                f.write('\t\t\t<img src=\"'+path+'\" alt=Loading...  width = "200"/>\n')
                if len(tags)>line_break:
                    position = []
                    for i in xrange(len(tags)/line_break):
                        position.append(line_break*i+(i-1)*6)
                    for pos in position:
                        tags = tags[:pos]+'<br />'+ tags[pos:]
                f.write('\t\t\t<br /><b>'+date+'</b><br /> '+tags+'\n')
                f.write('\t\t</td>\n')

                if count % col_num == 0:
                    f.write('\t</tr>\n')
    f.write('</table>\n')
    f.close()
def images_for_usr_event(usr_id, event_type, col_num, min_occurrence=5000,min_tag=3, write_tags = True):
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

    events_id = []
    events_length = []
    curr_length = 0
    #print events_id
    last_event = -1
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            if event_type == 'valid':
                meta = meta[1:]
            this_event = int(meta[0])
            curr_length += 1
            if this_event != last_event:
                events_id.append(this_event)
                events_length.append(curr_length)
                last_event = this_event
                curr_length = 0

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
    num_images = 1000
    col_num = 10
    args = sys.argv
    assert len(args) > 2
    type = args[1]
    if type == 'tag':
        tag = args[2]
        print 'Generating for tag:',tag
        if len(args) > 3:
            num_images = int(args[3])
        if len(args) > 4:
            col_num = int(args[4])
        images_for_tag(tag,num_images=num_images, col_num=col_num)
    elif type == 'usr':
        usr_id = int(args[2])
        print 'Generating for usr rank:',usr_id
        if len(args) > 3:
            num_images = int(args[3])
        if len(args) > 4:
            col_num = int(args[4])
        images_for_usr(usr_id,col_num=col_num)
    elif type == 'usr_all':
        usr_id = args[2]
        print 'Generating for usr:',usr_id
        if len(args) > 3:
            num_images = int(args[3])
        if len(args) > 4:
            col_num = int(args[4])
        images_for_usr_all(usr_id,col_num=col_num)
    elif type == 'usr_event':
        usr_id = args[2]
        print 'Generating for usr:',usr_id
        assert len(args) > 3
        event_type = args[3]
        if len(args) > 4:
            col_num = int(args[4])
        if len(args) > 5:
            write_tags = False
        else:
            write_tags = True
        images_for_usr_event(usr_id, event_type, col_num=col_num, write_tags=write_tags)
    elif type == 'event':
        usr_id = args[2]
        assert len(args) > 3
        event_name = args[3]
        print 'Generating event for usr:',usr_id, 'for event:', event_name
        if len(args) > 4:
            col_num = int(args[4])
        if len(args) > 5:
            write_tags = False
        else:
            write_tags = True
        images_for_event(usr_id, event_name, col_num=col_num, write_tags=write_tags)
    else:
        print 'Wrong input! Has to be tag or usr'


