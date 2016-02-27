import os
import shutil

root = 'C:\\Users\\yuwang\\Documents\\'


dict_name = {'ThemePark':'Theme park', 'UrbanTrip':'Urban/City trip', 'BeachTrip':'Beach trip', 'NatureTrip':'Nature trip',
             'Zoo':'Zoo/Aquarium/Botanic garden','Cruise':'Cruise trip','Show':'Show (air show/auto show/music show/fashion show/concert/parade etc.)',
            'Sports':'Sports game','PersonalSports':'Personal sports','PersonalArtActivity':'Personal art activities',
            'PersonalMusicActivity':'Personal music activities','ReligiousActivity':'Religious activities',
            'GroupActivity':'Group activities (party etc.)','CasualFamilyGather':'Casual family/friends gathering',
            'BusinessActivity':'Business activity (conference/meeting/presentation etc.)','Independence':'Independence Day',
            'Wedding':'Wedding', 'Birthday':'Birthday', 'Graduation':'Graduation', 'Museum':'Museum','Christmas':'Christmas',
            'Halloween':'Halloween', 'Protest':'Protest', 'Architecture':'Architecture/Art'}


def prepare_guru_dataset():
    path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/ranking_reallabel_part_test_all_event_name.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/guru_ranking_reallabel_part_test.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            event = line.split(' ')[-1]
            event = event.split()[0]
            new_line = line.split(' ')[:-1]
            new_line = (' ').join(new_line)
            meta = new_line.split('/')
            new_meta = ['/home/feiyu1990/local/event_curation/curation_images'] + [event] + meta[9:]
            f.write(('/').join(new_meta) + '\n')
    f.close()

    path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/ranking_reallabel_training_all_event_name.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/guru_ranking_reallabel_training.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            event = line.split(' ')[-1]
            event = event.split()[0]
            new_line = line.split(' ')[:-1]
            new_line = (' ').join(new_line)
            meta = new_line.split('/')
            new_meta = ['/home/feiyu1990/local/event_curation/curation_images'] + [event] + meta[9:]
            f.write(('/').join(new_meta) + '\n')
    f.close()

def copy_wedding_file():
    in_path = root + 'baseline_wedding_test\\wedding_training_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])
    for image_path in img_paths:
        new_folder = image_path.split('\\')[-2]
        if os.path.exists(root + 'wedding_images\\' + new_folder):
            os.mkdir(root + 'wedding_images\\' + new_folder)
        new_name = image_path.split('\\')[-1]
        shutil.copyfile(image_path, root + 'wedding_images\\' + new_folder + '\\' + new_name)


    in_path = root + 'baseline_wedding_test\\wedding_test_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])
    for image_path in img_paths:
        new_folder = image_path.split('\\')[-2]
        if os.path.exists(root + 'wedding_images\\' + new_folder):
            os.mkdir(root + 'wedding_images\\' + new_folder)
        new_name = image_path.split('\\')[-1]
        shutil.copyfile(image_path, root + 'wedding_images\\' + new_folder + '\\' + new_name)

def copy_all_file(name):
    in_path = root + 'baseline_all\\' + name + '\\training_path.txt'
    img_paths = []
    if os.path.exists(root + 'curation_images\\' + name):
        os.mkdir(root + 'curation_images\\' + name)
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])
    for image_path in img_paths:
        new_folder = image_path.split('\\')[-2]
        if os.path.exists(root + 'curation_images\\' + name + new_folder):
            os.mkdir(root + 'curation_images\\' + name + new_folder)
        new_name = image_path.split('\\')[-1]
        shutil.copyfile(image_path, root + 'curation_images\\' + name + new_folder + '\\' + new_name)


    in_path = root + 'baseline_all\\' + name + '\\test_path.txt'
    img_paths = []
    with open(in_path,'r') as data:
        for line in data:
            img_paths.append(line.split(' ')[0])
    for image_path in img_paths:
        new_folder = image_path.split('\\')[-2]
        if os.path.exists(root + 'curation_images\\'+ name + new_folder):
            os.mkdir(root + 'curation_images\\'+ name + new_folder)
        new_name = image_path.split('\\')[-1]
        shutil.copyfile(image_path, root + 'curation_images\\'+ name + new_folder + '\\' + new_name)

def prepare_server_dataset(name = 'training'):
    path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/guru_ranking_reallabel_'+name+'.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/ranking_reallabel_'+name+'_nomargin.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            meta = line.split('/')
            new_meta = ['/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition']+ meta[7:]
            f.write(('/').join(new_meta))
    f.close()

    path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/guru_ranking_reallabel_'+name+'_p.txt'
    new_path = '/Users/wangyufei/Documents/Study/intern_adobe/ranking_loss_CNN/training_CNN_all_events_nomargin/ranking_reallabel_'+name+'_nomargin_p.txt'
    f = open(new_path,'w')
    with open(path,'r') as data:
        for line in data:
            meta = line.split('/')
            new_meta = ['/mnt/ilcompf2d0/project/yuwang/datasets/all_data/clean_tags_5000_3_0710/download_event_recognition']+ meta[7:]
            f.write(('/').join(new_meta))
    f.close()



if __name__ == "__main__":
    #for event_name in dict_name:
    #    copy_all_file(event_name)
    #prepare_guru_dataset()
    prepare_server_dataset('part_test')