__author__ = 'wangyufei'

import cPickle
import operator
import random
import os
root = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_events/'

def create_from_clean():
    write_path = root + 'clean_imgs.txt'
    f = open(write_path, 'w')
    for i in xrange(93):
        number = i*140000+1
        this_path = root+'clean_imgs_'+str(number)+'.txt'
        with open(this_path, 'r') as data:
            for line in data:
                f.write(line)
    f.close()
    write_path = root + 'clean_ids.txt'
    f = open(write_path, 'w')
    for i in xrange(93):
        number = i*140000+1
        this_path = root+'clean_ids_'+str(number)+'.txt'
        with open(this_path, 'r') as data:
            for line in data:
                f.write(line)
    f.close()
    write_path = root + 'clean_urls.txt'
    f = open(write_path, 'w')
    for i in xrange(93):
        number = i*140000+1
        this_path = root+'clean_urls_'+str(number)+'.txt'
        with open(this_path, 'r') as data:
            for line in data:
                f.write(line)
    f.close()
def tags_with_count():
    count = 0
    event_prev = -1
    user_prev = -1
    seen = {}
    read_path = root + 'clean_imgs.txt'

    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            if usr_this != user_prev or event_prev != event_this:
                event_prev = event_this
                user_prev = usr_this
                tag_list = tags.split(',')
                for tag_individual in tag_list:
                    if tag_individual not in seen:
                        seen[tag_individual] = 1
                    else:
                        seen[tag_individual] += 1

    #seen_sorted = sorted(seen, key=lambda x:x[1])
    seen_sorted = sorted(seen.items(), key=operator.itemgetter(1), reverse = True)
    out_file_path = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/clean_tags_event_count.txt'
    out_f = open(out_file_path, 'w')
    for key, value in seen_sorted:
        try:
            out_f.write(key+'\t'+str(value)+'\n')
        except:
            print "write tag failed:", key, value
            #del seen[key]


    f = open('/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/clean_tags_event_count.cPickle', 'w')
    cPickle.dump(seen_sorted,f)
    f.close()

threshold = 250
def select_20000_events():
    event_prev = -1
    user_prev = -1
    read_path = root + 'clean_imgs.txt'
    event_id = []
    all_event_id = []
    count = 0
    f = open('/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/clean_tags_event_count.cPickle', 'rb')
    tags_count = cPickle.load(f)
    f.close()
    dict_tag_count = dict((x, y) for x, y in tags_count)

    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            #if count % 100000 == 0:
            #    print count
            count += 1
            tag_list = tags.split(',')
            if usr_this != user_prev or event_prev != event_this:
                event_prev = event_this
                user_prev = usr_this
                event_this_id = usr_this+'_'+event_this
                all_event_id.append(event_this_id)
                rate = []
                for tag_individual in tag_list:
                    rate.append(dict_tag_count[tag_individual])
                if max(rate) >= threshold:
                    event_id.append(event_this_id)
    print event_id
    print len(event_id), len(all_event_id)
    sample_event = [event_id[i] for i in sorted(random.sample(xrange(len(event_id)), 20000))]
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    if not os.path.exists(root1):
        os.mkdir(root1)
    f = open(root1+'sample_event_id.cPickle','w')
    cPickle.dump(sample_event, f)
    f.close()
    f = open(root1+'all_event_id.cPickle','w')
    cPickle.dump(all_event_id, f)
    f.close()
def write_20000_sample():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    f = open(root1+'sample_event_id.cPickle','rb')
    sample_event = cPickle.load(f)
    f.close()

    read_path = root + 'clean_imgs.txt'
    write_path = root1 + 'sample_20000_imgs_0716.txt'
    count = 0
    i = 0
    event_prev_id = ''
    log_path = root1 + 'log.txt'
    f = open(write_path, 'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = usr_this+'_'+event_this
            i += 1
            if (i % 1000) == 0:
                with open(log_path, 'a') as f1:
                    f1.write(str(i)+'\n')
            if (event_prev_id == event_this_id) or (count < len(sample_event) and event_this_id == sample_event[count]):
                if event_prev_id != event_this_id:
                    count += 1
                    event_prev_id = event_this_id
                f.write(line)
    f.close()
def chunk(ylen, n):
    ys = [i for i in xrange(ylen)]
    random.shuffle(ys)
    #ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in xrange(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in xrange(leftover):
            chunks[i%n].append(ys[edge+i])
    chunks_new = []
    for i in xrange(len(chunks)):
        a = sorted(chunks[i])
        chunks_new.append(a)
    return chunks_new
def tags_with_count_sample():
    event_prev = -1
    user_prev = -1
    seen = {}
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    read_path=root1 + 'sample_20000_imgs_0716.txt'

    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            if usr_this != user_prev or event_prev != event_this:
                event_prev = event_this
                user_prev = usr_this
                tag_list = tags.split(',')
                for tag_individual in tag_list:
                    if tag_individual not in seen:
                        seen[tag_individual] = 1
                    else:
                        seen[tag_individual] += 1

    seen_sorted = sorted(seen.items(), key=operator.itemgetter(1), reverse = True)
    out_file_path = root1 + '200000_tags_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key, value in seen_sorted:
        try:
            out_f.write(key+'\t'+str(value)+'\n')
        except:
            print "write tag failed:", key, value
            #del seen[key]
    out_f.close()

def count_events():
    root2 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    path=root2 + 'sample_20000_imgs_0716.txt'
    #path = root + 'clean_imgs.txt'
    meta_path = path
    event_start = []
    event_id = []
    #tags = []
    count = 0
    last_event = -1
    with open(meta_path,'r') as data:
        for line in data:
            meta = line.split('\t')
            this_event = meta[1]+'_'+meta[3]
            if this_event != last_event:
                event_id.append(this_event)
                #tags.append(meta[0])
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
    f = open(root2+'sample_length_event.txt','w')
    for length, id in length_sort:
        f.write(str(length)+' '+id+'\n')
    f.close()
    f = open(root2+'sample_length_event.cPickle','w')
    cPickle.dump(length_sort, f)
    f.close()

def select_20000_events_valid(c_min=30, c_max=200):
    event_prev_id = ''
    read_path = root + 'clean_imgs.txt'
    event_id = []
    all_event_id = []
    count = 0

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'

    f = open('/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/clean_tags_event_count.cPickle', 'rb')
    tags_count = cPickle.load(f)
    f.close()

    f = open(root1+'clean_length_event.cPickle','r')
    event_length = cPickle.load(f)
    f.close()
    event_length = dict((y, x) for x, y in event_length)
    dict_tag_count = dict((x, y) for x, y in tags_count)
    event_count = 0
    event_length_valid = []
    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            count += 1
            tag_list = tags.split(',')
            event_this_id = event_this + '_' + usr_this
            if event_this_id != event_prev_id:
                event_prev_id = event_this_id
                all_event_id.append(event_this_id)
                length_this = event_length[event_this_id]
                rate = []
                for tag_individual in tag_list:
                    rate.append(dict_tag_count[tag_individual])
                if max(rate) >= threshold and length_this >= c_min and length_this <= c_max:
                    event_id.append(event_this_id)
                    event_length_valid.append(length_this)
                    #print length_this, event_this_id

    print event_id
    print len(event_id),len(event_length_valid), len(event_length), len(all_event_id)
    indice = sorted(random.sample(xrange(len(event_id)), 20000))
    sample_event = [event_id[i] for i in indice]
    sample_length = [event_length_valid[i] for i in indice]

    sample_ = zip(sample_length, sample_event)
    length_sort = sorted(sample_,reverse=True)
    f = open(root1+'precal_sample_length_event.txt','w')
    for length, id in length_sort:
        f.write(str(length)+' '+id+'\n')
    f.close()


    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    if not os.path.exists(root1):
        os.mkdir(root1)
    f = open(root1+'sample_event_id.cPickle','w')
    cPickle.dump(sample_event, f)
    f.close()
    f = open(root1+'sample_event_id.txt','w')
    for i in sample_event:
        f.write(i+'\n')
    f.close()

    f = open(root1+'valid_event_id.cPickle','w')
    cPickle.dump(event_id, f)
    f.close()
def write_20000_sample_valid():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    f = open(root1+'sample_event_id.cPickle','rb')
    sample_event = cPickle.load(f)
    f.close()

    read_path = root + 'clean_imgs.txt'
    write_path = root1 + 'sample_20000_imgs_0716.txt'
    count = 0
    i = 0
    event_prev_id = ''
    log_path = root1 + 'log.txt'
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


def write_csv():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    path=root1 + 'sample_20000_imgs_0716.txt'
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
    f = open(root1+'log_length_event.txt','w')
    for i in length_sort:
        f.write(str(i)+'\n')
    f.close()

    #chunks = chunk(len(event_length), 1000)
    #print chunks
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    out_path = root1 + 'sample_20000.csv'
    f = open(out_path, 'w')
    max_size = max(event_length)
    f.write('num_image,tags')
    for i in xrange(max_size):
            f.write(',')
            f.write('image'+str(i+1))
    f.write('\n')
    count = 0
    event_prev = ''
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if event_this != event_prev:
                    if count > 0:
                        for kk in xrange(max_size-event_length[count - 1]):
                            f.write(',NA')
                        f.write('\n')
                    f.write(str(event_length[count]))
                    count += 1
                    tag_list = meta[0].split(',')
                    tag_this = ':'.join(tag_list)
                    f.write(','+tag_this)
                    event_prev = event_this
                url = meta[16]
                f.write(','+url)
    for kk in xrange(max_size-event_length[count - 1]):
        f.write(',NA')
    f.write('\n')
    f.close()
def write_csv_random(max_size = 200):
    line_stack = []
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    in_path = root1 + 'sample_20000.csv'
    out_path = root1 + 'random_sample_20000.csv'
    count = 0
    with open(in_path, 'r') as data:
        for line in data:
            if count == 0:
                first_line = line
            else:
                line_stack.append(line)
            count += 1
    random.shuffle(line_stack) # << shuffle before print or assignment
    f = open(out_path,'w')
    f.write(first_line)
    for l in line_stack:
        f.write(l)
    f.close()
    out_path = root1 + 'random_5_sample_20000.csv'
    str_to_write = ''
    for k in xrange(5):
        str_to_write += ',num_image'+str(k+1)+',tags'+str(k+1)
        for i in xrange(max_size):
                str_to_write += ',image'+str(k+1)+'_'+str(i+1)
    str_to_write+='\n'
    f = open(out_path, 'w')
    f.write(str_to_write[1:])
    count = 1
    for l in line_stack:
        if count % 5 == 0:
            f.write(l)
        else:
            f.write(l[:-1]+',')
        count+=1
    f.close()


def create_csv_sample():

    chunks = chunk(20000, 100)
    print chunks

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    in_path = root1 + 'random_sample_20000.csv'

    for i in xrange(200):
        out_path= root1 + 'sample_csv/sample_20000_'+str(i)+'.csv'
        count = 0;j=-1
        this_indice = chunks[i]
        f = open(out_path, 'w')
        with open(in_path, 'r') as data:
            for line in data:
                if j == -1:
                    f.write(line)
                if count < len(this_indice) and j == this_indice[count]:
                    count += 1
                    f.write(line)
                j+=1
        f.close()

def write_csv_5():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    path=root1 + 'sample_20000_imgs_0716.txt'
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
    f = open(root1+'log_length_event.txt','w')
    for i in length_sort:
        f.write(str(i)+'\n')
    f.close()

    #chunks = chunk(len(event_length), 1000)
    #print chunks
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    out_path = root1 + '5_sample_20000.csv'
    f = open(out_path, 'w')
    max_size = max(event_length)
    str_to_write = ''
    for k in xrange(5):
        str_to_write += ',num_image'+str(k+1)+',tags'+str(k+1)
        for i in xrange(max_size):
                str_to_write += ',image'+str(k+1)+'_'+str(i+1)
    f.write(str_to_write[1:])
    f.write('\n')

    count = 0
    event_count = 0
    event_prev = ''
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if event_this != event_prev:
                    if count > 0:
                        for kk in xrange(max_size-event_length[count - 1]):
                            f.write(',NA')
                        if event_count % 5 == 0:
                            f.write('\n')
                        else:
                            f.write(',')
                    f.write(str(event_length[count]))
                    count += 1
                    tag_list = meta[0].split(',')
                    tag_this = ':'.join(tag_list)
                    f.write(','+tag_this)
                    event_prev = event_this
                    event_count += 1
                url = meta[16]
                f.write(','+url)
    for kk in xrange(max_size-event_length[count - 1]):
        f.write(',NA')
    f.write('\n')
    f.close()
def create_csv_sample_5():
    chunks = chunk(4000, 50)
    print chunks

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    in_path = root1 + 'random_5_sample_20000.csv'

    with open(in_path, 'r') as data:
        first_line = data.readline()
    j = -1
    out_path= root1 + '5_sample_csv/sample_20000_'+str(0)+'.csv'
    f = open(out_path, 'w')
    with open(in_path, 'r') as data:
        for line in data:
            if j==-1:
                j+=1
                continue
            if j % 80 == 0:
                f.close()
                out_path= root1 + '5_sample_csv/sample_20000_'+str(j/80)+'.csv'
                f = open(out_path, 'w')
                f.write(first_line)
            f.write(line)
            j+=1
    f.close()

def select_more_graduation(tag='graduation', c_min = 30, c_max=200):
    event_prev_id = ''
    read_path = root + 'clean_imgs.txt'
    event_id = []
    all_event_id = []
    count = 0
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'

    f = open(root1+'sample_event_id.cPickle','r')
    pre_event = cPickle.load(f)
    f.close()


    f = open(root1+'clean_length_event.cPickle','r')
    event_length = cPickle.load(f)
    f.close()
    event_length = dict((y, x) for x, y in event_length)
    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            count += 1
            tag_list = tags.split(',')
            event_this_id = event_this + '_' + usr_this
            if event_this_id != event_prev_id:
                event_prev_id = event_this_id
                all_event_id.append(event_this_id)
                length_this = event_length[event_this_id]
                if event_this_id not in pre_event and tag in tag_list:
                    if length_this >= c_min and length_this <= c_max:
                        event_id.append(event_this_id)

    print event_id

    indice = sorted(random.sample(xrange(len(event_id)), 30))
    sample_event_tag = [event_id[i] for i in indice]

    indice = sorted(random.sample(xrange(len(pre_event)), 30))
    sample_event = [pre_event[i] for i in xrange(len(pre_event)) if i not in indice]

    sample_event_all = sample_event_tag + sample_event

    f = open(root1+'graduation_event_id.cPickle','w')
    cPickle.dump(sample_event_all, f)
    f.close()
    f = open(root1+'graduation_event_id.txt','w')
    for i in sample_event_all:
        f.write(i+'\n')
    f.close()
def write_more_graduation():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    f = open(root1+'graduation_event_id.cPickle','rb')
    sample_event = cPickle.load(f)
    f.close()

    read_path = root + 'clean_imgs.txt'
    write_path = root1 + 'more_graduation_imgs_0719.txt'
    i = 0
    f = open(write_path, 'w')
    with open(read_path, 'r') as data:
        for line in data:
            meta = line.split('\t')
            event_this = meta[1]
            usr_this = meta[3]
            event_this_id = event_this + '_' + usr_this
            i += 1
            if event_this_id in sample_event:
                f.write(line)
    f.close()
def graduate_write_csv_10():
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    path=root1 + 'more_graduation_imgs_0719.txt'
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
    f = open(root1+'log_length_event.txt','w')
    for i in length_sort:
        f.write(str(i)+'\n')
    f.close()

    #chunks = chunk(len(event_length), 1000)
    #print chunks
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    out_path = root1 + 'more_graduation_20000.csv'
    f = open(out_path, 'w')
    max_size = max(event_length)
    f.write('num_image,event_id,tags')
    for i in xrange(max_size):
            f.write(',')
            f.write('image'+str(i+1))
    f.write('\n')


    count = 0
    event_count = 0
    event_prev = ''
    with open(meta_path, 'r') as data:
            for line in data:
                meta = line.split('\t')
                event_this = meta[1]+'_'+meta[3]
                if event_this != event_prev:
                    if count > 0:
                        for kk in xrange(max_size-event_length[count - 1]):
                            f.write(',NA')
                        if event_count % 1 == 0:
                            f.write('\n')
                        else:
                            f.write(',')
                    f.write(str(event_length[count]))
                    count += 1
                    f.write(','+event_this)
                    tag_list = meta[0].split(',')
                    tag_this = ':'.join(tag_list)
                    f.write(','+tag_this)
                    event_prev = event_this
                    event_count += 1
                url = meta[16]
                f.write(','+url)
    for kk in xrange(max_size-event_length[count - 1]):
        f.write(',NA')
    f.write('\n')
    f.close()


    in_path = root1 + 'more_graduation_20000.csv'
    lines = []
    with open(in_path, 'r') as data:
        for line in data:
            lines.append(line)
    start_line = lines[0]
    lines = lines[1:]

    random.shuffle(lines)

    out_path = root1 + 'shuffle_more_graduation_20000.csv'
    f = open(out_path,'w')
    f.write(start_line)
    for line in lines:
        f.write(line)
    f.close()

    out_path = root1 + '10_more_graduation_20000.csv'
    f = open(out_path, 'w')
    str_to_write = ''
    for k in xrange(10):
        str_to_write += ',num_image'+str(k+1)+',event_id'+str(k+1)+',tags'+str(k+1)
        for i in xrange(max_size):
                str_to_write += ',image'+str(k+1)+'_'+str(i+1)
    f.write(str_to_write[1:])
    f.write('\n')

    count = 1
    for l in lines:
        if count % 10 == 0:
            f.write(l)
        else:
            f.write(l[:-1]+',')
        count+=1
    f.close()

def create_csv_graduation_10(chunk_size = 50):
    #chunks = chunk(4000, 50)
    #print chunks

    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    in_path = root1 + '10_more_graduation_20000.csv'
    if not os.path.exists(root1 + '10_more_graduation_csv/'):
        os.mkdir(root1 + '10_more_graduation_csv/')
    with open(in_path, 'r') as data:
        first_line = data.readline()
    j = -1
    out_path= root1 + '10_more_graduation_csv/sample_20000_'+str(0)+'.csv'
    f = open(out_path, 'w')
    with open(in_path, 'r') as data:
        for line in data:
            if j==-1:
                j+=1
                continue
            if j % chunk_size == 0:
                f.close()
                out_path= root1 + '10_more_graduation_csv/sample_20000_'+str(j/chunk_size)+'.csv'
                f = open(out_path, 'w')
                f.write(first_line)
            f.write(line)
            j+=1
    f.close()
def create_csv_graduation_10_except1(chunk_size=50):
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    in_path = root1 + '10_more_graduation_20000.csv'
    with open(in_path, 'r') as data:
        first_line = data.readline()
    j = -1
    out_path= root1 + '10_more_graduation_20000_except1.csv'
    f = open(out_path, 'w')
    f.write(first_line)
    with open(in_path, 'r') as data:
        for line in data:
            if j==-1:
                j+=1
                continue
            if j/chunk_size == 1:
                j+=1
                continue
            f.write(line)
            j+=1
    f.close()

def separate_except1(chunk_size=250):
    root1 = '../amt/input_result/'
    in_path= root1 + '10_more_graduation_20000_except1.csv'
    with open(in_path, 'r') as data:
        first_line = data.readline()
    j = -1
    output_path1  =root1+'10_more_graduation_20000_except1_1.csv'
    f = open(output_path1, 'w')
    f.write(first_line)

    with open(in_path, 'r') as data:
        for line in data:
            if j==-1:
                j+=1
                continue
            if j % chunk_size == 0:
                f.close()
                out_path= root1 + '10_more_graduation_20000_except1_'+str(j/chunk_size)+'.csv'
                f = open(out_path, 'w')
                f.write(first_line)
            f.write(line)
            j+=1
    f.close()

def tags_with_count_graduation():
    event_prev = -1
    user_prev = -1
    seen = {}
    root1 = '/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/amt_csv/'
    read_path=root1 + 'more_graduation_imgs_0719.txt'

    with open(read_path, 'r') as data:
        for line in data:
            line = line.split('\n')[0]
            meta = line.split('\t')
            tags = meta[0]
            event_this = meta[1]
            usr_this = meta[3]
            if usr_this != user_prev or event_prev != event_this:
                event_prev = event_this
                user_prev = usr_this
                tag_list = tags.split(',')
                for tag_individual in tag_list:
                    if tag_individual not in seen:
                        seen[tag_individual] = 1
                    else:
                        seen[tag_individual] += 1

    seen_sorted = sorted(seen.items(), key=operator.itemgetter(1), reverse = True)
    out_file_path = root1 + 'more_graduation_tags_with_count.txt'
    out_f = open(out_file_path, 'w')
    for key, value in seen_sorted:
        try:
            out_f.write(key+'\t'+str(value)+'\n')
        except:
            print "write tag failed:", key, value
            #del seen[key]
    out_f.close()

if __name__ == '__main__':
    #create_from_clean()
    #tags_with_count()
    #select_20000_events()
    #write_20000_sample()
    #tags_with_count_sample()
    #write_csv()
    #count_events()
    #select_20000_events_valid()
    #write_20000_sample_valid()
    #write_csv_5()
    #write_csv_random()
    #create_csv_sample()
    #create_csv_sample_5()
    #select_more_graduation()
    #write_more_graduation()
    #
    #graduate_write_csv_10()
    #tags_with_count_graduation()
    #create_csv_graduation_10()
    #create_csv_graduation_10_except1()
    separate_except1()