__author__ = 'wangyufei'
'''
This is for:
images_for_tag(tag) -> images for certain tag (from 1M dataset sample)
    'python generate_html.py tag @TAG'
images_for_usr(usr_rank) -> images for certain user based on its occurrence rank (from 1M dataset sample)
    'python generate_html.py usr @rank'
images_for_usr_all(usr_id) -> images for certain user
    DEPRECATED 'python generate_html.py usr_all @ID'
images_for_usr_event(usr_id) -> images for certain user, separate events (based on the separate event txt files)
    'python generate_html.py usr_event @ID'
images_for_usr_any_event(usr_id) -> images for certain user (event segmented)
    'python generate_html.py usr_all @ID'
'''

import sys
import os
import random

min_occurrence = 50
min_tag = 3
line_break = 30
maximum_events = 20
def images_for_similar(usr_id='7436989@N05'):
    col_num = 2
    root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/'
    list_path = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/dup_pairs_url.txt'
    out_root = '/Users/wangyufei/Documents/Study/intern_adobe/CNN/features_yw/dup_7436989@N05.html'
    #list_path = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+'/events_clean_tags_new/'+usr_id+'.txt'
    #out_root = root+'clean_tags_'+str(min_occurrence)+'_'+str(min_tag)+ '/html_events_0708/'
    f = open(out_root, 'w')
    f.write('<head> <title> USER ID:'+usr_id+'</title></head><h1>Similar images from user:'+usr_id+'</h1>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    count = 0
    with open(list_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            img_1 = meta[0]
            img_2 = meta[1]
            score = meta[2]
            if count % 6 == 0:
                f.write('\t<tr>\n')

            f.write('\t\t<td align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+img_1+'\" alt=Image... width = "200" />\n')
            f.write('\t\t\t<br /><b>'+str(score)+'</b><br />\n')
            f.write('\t\t<td style=\"padding-right:20px\" align=\"center\" valign=\"center\">\n')
            f.write('\t\t\t<img src=\"'+img_2+'\" alt=Image... width = "200" />\n')
            f.write('\t\t\t<br /><b>'+str(score)+'</b><br />\n')
            f.write('\t\t</td>\n')
            if count %6 == 2:
                f.write('\t</tr>\n')

            count += 1

    f.write('</table>\n')
    f.close()

if __name__ == '__main__':
    images_for_similar()

