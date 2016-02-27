%% WRITING TO HDF5

filename='test.h5';


num_total_samples=2601;
load('C:/Users/yuwang/Documents/wedding_CNN_net/vgg_wedding_test_result_v2.mat')
test_scores = scores;

load('C:/Users/yuwang/Documents/wedding_CNN_net/imgs_test.mat')
test_data = imgs_all;
chunksz=100;
created_flag=false;
totalct=0;
for batchno=1:num_total_samples/chunksz
  fprintf('batch no. %d\n', batchno);
  last_read=(batchno-1)*chunksz;

  % to simulate maximum data to be held in memory before dumping to hdf5 file 
  batchdata=test_data(:,:,1,last_read+1:last_read+chunksz);
  batchlabs=test_scores(:,last_read+1:last_read+chunksz);

  % store to hdf5
  startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
  curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz);
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
end



filename='train.h5';
% to simulate data being read from disk / generated etc.


load('C:/Users/yuwang/Documents/wedding_CNN_net/vgg_wedding_training_result_v2.mat')
training_scores_all = scores;
num_total_samples=8004;
chunksz=100;
created_flag=false;
totalct=0;

for i = 1:8
    load(strcat('C:/Users/yuwang/Documents/wedding_CNN_net/imgs_train',str(i*1000),'.mat'));
    training_data = imgs_all;
    training_scores = training_scores_all((i-1)*1000+1:i*1000);
    for batchno=1:10
        fprintf('batch no. %d\n', batchno);
        last_read=(batchno-1)*chunksz;
        % to simulate maximum data to be held in memory before dumping to hdf5 file 
        batchdata=training_data(:,:,1,last_read+1:last_read+chunksz);
        batchlabs=training_scores(:,last_read+1:last_read+chunksz);

        % store to hdf5
        startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz);
        created_flag=true;% flag set so that file is created only once
        totalct=curr_dat_sz(end);% updated dataset size (#samples)
    end
end
load(strcat('C:/Users/yuwang/Documents/wedding_CNN_net/imgs_train8004.mat'));
training_data = imgs_all;
training_scores = training_scores_all((i-1)*1000+1:i*1000);
fprintf('batch no. %d\n', batchno);
last_read=0;
% to simulate maximum data to be held in memory before dumping to hdf5 file 
batchdata=training_data(:,:,1,last_read+1:end);   
batchlabs=training_scores(:,last_read+1:end);
% store to hdf5
startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz);
totalct=curr_dat_sz(end);% updated dataset size (#samples)



%mirrored data
read_already = 8004;
for batchno=1:num_total_samples/chunksz
  fprintf('batch no. %d\n', batchno);
  last_read=(batchno-1)*chunksz + read_already;

  % to simulate maximum data to be held in memory before dumping to hdf5 file 
  batchdata=training_data_mirror(:,:,1,last_read+1:last_read+chunksz);
  batchlabs=training_scores(:,last_read+1:last_read+chunksz);

  % store to hdf5
  startloc=struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
  curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz);
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
end

