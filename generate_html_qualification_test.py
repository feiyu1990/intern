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
import csv

min_occurrence = 50
min_tag = 3
line_break = 30
maximum_events = 20
event_ids = ['1_7851526@N08','2_46506950@N04','56_24256658@N06','5_49231590@N07','0_41949984@N00']

def images_for_usr_event():
    root = '/Users/wangyufei/Documents/Study/intern_adobe/amt/input_result/'
    in_path = root + 'sample_20000_10.csv'
    out_path = root + 'qualification_test.html'

    f = open(out_path, 'w')
    f.write('<head> <title> Images for AMT </title></head>\n' )
    f.write('<center>')
    f.write('<table border="1" style="width:100%">\n')
    f.write('\t<tr>\n')
    image_count = 0
    with open(in_path, 'rb') as data:
        reader = csv.reader(data)
        for meta in reader:
            for i in xrange(1,1+203*4,203):
                if meta[i] in event_ids:
                    f.write('<td> Event id:'+meta[i]+'</td></tr><tr>')
                    for j in xrange(3, 203):
                        if meta[i+j] == 'NA':
                            break

                        f.write('\t\t<td>\n')
                        f.write('\t\t\t<img src=\"'+meta[i+j]+'\" alt="Loading..." width = "180" />\n')
                        f.write('\t\t</td>\n')
                        image_count += 1
                        if image_count % 5 == 0:
                            f.write('\t</tr>\n')
                            f.write('\t<tr>\n')
                    f.write('\t</tr>\n')
                    f.write('\t<tr>\n')
                    image_count = 0
    f.write('\t</tr>\n')
    f.write('</table>\n')
    f.close()

if __name__ == '__main__':
    images_for_usr_event()

