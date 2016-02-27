__author__ = 'wangyufei'
import h5py
import numpy as np
root = '/Users/wangyufei/Documents/Study/intern_adobe/to_guru/CNN_all_event/data/no_margin/'
def create_check_file():
    f = open(root + 'guru_ranking_reallabel_training_test.txt','w')
    count = 0;lines = 0;hdf5_count = 0
    with open(root + 'guru_ranking_reallabel_training_nomargin.txt', 'r') as data:
        for line in data:
            file_path = line.split(' ')[0]
            if count == lines:
                hdf5_count += 1
                f1 = h5py.File(root + 'training_event_label_nomargin'+str(hdf5_count)+'.h5','r')
                event_label = f1['event_label'].value
                print event_label.shape, len(event_label)
                count = 0
                lines = len(event_label)
            label = np.nonzero(event_label[count])[0][0]

            f.write(file_path + ' ' + str(label) + '\n')
            count += 1
    f.close()
if __name__ == '__main__':
    create_check_file()
