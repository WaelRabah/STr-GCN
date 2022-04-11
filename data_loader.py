from torch.utils.data import Dataset
from random import randint,shuffle
import wget
import os
import zipfile
import torch
import numpy as np
import torch.nn as nn
import random

def download_DHG_dataset():
    if not os.path.exists("../DHG2016.zip") :
        url = 'http://www-rech.telecom-lille.fr/DHGdataset/DHG2016.zip'
        wget.download(url)
    if not os.path.exists("../dataset") :
        os.mkdir("../dataset")
        with zipfile.ZipFile("../DHG2016.zip", 'r') as zip_ref:
            zip_ref.extractall("../dataset")




#Change the path to your downloaded dataset
dataset_fold = "../dataset"

#change the path to your downloaded DHG dataset


def read_data_from_disk():
    neighbors_by_node={1: [1, 2, 3], 2: [2, 1, 7, 11, 15, 19], 3: [3, 1, 4], 4: [4, 3, 5], 5: [5, 4, 6], 6: [6, 5], 7: [7, 2, 8], 8: [8, 7, 9], 9: [9, 8, 10], 10: [10, 9], 11: [11, 2, 12], 12: [12, 11, 13], 13: [13, 12, 14], 14: [14, 13], 15: [15, 2, 16], 16: [16, 15, 17], 17: [17, 16, 18], 18: [18, 17], 19: [19, 2, 20], 20: [20, 19, 21], 21: [21, 20, 22], 22: [22, 21]}


    def get_edge_index():
        adj=np.zeros(shape=(22,22))
        for key,neighbours in neighbors_by_node.items():
            for n in neighbours :
                adj[key-1,n-1]=1
        # edge_index=torch.from_numpy(adj).to_sparse()
        # return edge_index.indices()
        edge_index=torch.from_numpy(adj)
        return edge_index
    def parse_data(src_file):
        video = []
        for line in src_file:
            line = line.split("\n")[0]
            data = line.split(" ")
            frame = []
            point = []
            for data_ele in data:
                point.append(float(data_ele))
                if len(point) == 3:
                    frame.append(point)
                    point = []
            video.append(frame)
        return video
    
    result = {}

    for g_id in range(1,15):
        print("gesture {} / {}".format(g_id,14))
        for f_id in range(1,3):
            for sub_id in range(1,21):
                for e_id in range(1,6):
                    src_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt".format(g_id, f_id, sub_id, e_id)
                    src_file = open(src_path)
                    video = parse_data(src_file) #the 22 points for each frame of the video
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
                    result[key] = video
                    src_file.close()
    return result, get_edge_index()

def get_valid_frame(video_data):
    # filter frames using annotation
    info_path = dataset_fold + "/informations_troncage_sequences.txt"
    info_file = open(info_path)
    used_key = []
    for line in info_file:
        line = line.split("\n")[0]
        data = line.split(" ")
        g_id =  data[0]
        f_id = data[1]
        sub_id = data[2]
        e_id = data[3]
        key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
        used_key.append(key)
        start_frame = int(data[4])
        end_frame = int(data[5])
        data = video_data[key]
        video_data[key] = data[(start_frame): end_frame + 1]
        #print(key,start_frame,end_frame)
        #print(len(video_data[key]))
        #print(video_data[key][0])
    #print(len(used_key))
    #print(len(video_data))
    return video_data

def split_train_test(test_subjects_ids, filtered_video_data, cfg):
  #split data into train and test
  #cfg = 0 >>>>>>> 14 categories      cfg = 1 >>>>>>>>>>>> 28 cate
    train_data = []
    test_data = []
    for g_id in range(1, 15):
        for f_id in range(1, 3):
            for sub_id in range(1, 21):
                for e_id in range(1, 6):
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

                    #set table to 14 or
                    if cfg == 0:
                        label = g_id
                    elif cfg == 1:
                        if f_id == 1:
                            label = g_id
                        else:
                            label = g_id + 14

                  #split to train and test list
                    data = filtered_video_data[key]
                    sample = {"skeleton":data, "label":label}
                    if sub_id in test_subjects_ids:
                        test_data.append(sample)
                    else:
                        train_data.append(sample)
    if len(test_data) == 0:
        raise "no such test subject"

    return train_data, test_data

def get_train_test_data(test_subjects_ids, cfg):
    print("Downloading DHG dataset.......")
    download_DHG_dataset()
    print("Reading data from disk.......")
    video_data,edge_index = read_data_from_disk()
    print("Filtering frames .......")
    filtered_video_data = get_valid_frame(video_data)
    train_data, test_data = split_train_test(test_subjects_ids, filtered_video_data, cfg)
    return train_data,test_data,edge_index

class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, time_len, use_data_aug, use_upsampling):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """
        self.use_data_aug = use_data_aug
        self.use_upsampling=use_upsampling
        self.data = data
        self.frame_sizes=[ len(data_ele["skeleton"]) for data_ele in self.data]
        self.max_sequence=8
        self.time_len = time_len
        self.compoent_num = 22



    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        #print("ind:",ind)
        data_ele = self.data[ind]


        #hand skeleton
        skeleton = data_ele["skeleton"]
        skeleton = np.array(skeleton)
        # if self.use_upsampling:
        #     skeleton = self.upsample(skeleton,self.max_sequence)
        if self.use_data_aug:
            skeleton = self.data_aug(skeleton)
            # skeleton=torch.from_numpy(skeleton)
        ## **** old code ***
        # sample time_len frames from whole video
        # data_num = skeleton.shape[0]
        # idx_list = self.sample_frame(data_num)
        # skeleton = [torch.unsqueeze(skeleton[idx],dim=0) for idx in idx_list]
        # skeleton = torch.cat(skeleton,dim=0)
        #normalize by palm center
        # skeleton -= torch.clone(skeleton[0][1])
        # skeleton = skeleton.float()
        ## **** old code ***
        # sample time_len frames from whole video
        data_num = skeleton.shape[0]
        idx_list = self.sample_frame(data_num)
        skeleton = [skeleton[idx] for idx in idx_list]
        skeleton = np.array(skeleton)

        #normalize by palm center
        skeleton -= skeleton[0][1]



        skeleton = torch.from_numpy(skeleton).float()
        # label
        label = data_ele["label"] - 1 #

        sample = {'skeleton': skeleton, "label" : label}

        return sample

    def data_aug(self, skeleton):

        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] *= factor
            skeleton = np.array(skeleton)
            return skeleton

        def shift(skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] += offset
            skeleton = np.array(skeleton)
            return skeleton

        def noise(skeleton):
            low = -0.1
            high = -low
            #select 4 joints
            all_joint = list(range(self.compoent_num))
            shuffle(all_joint)
            selected_joint = all_joint[0:4]
            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(self.time_len):
                  skeleton[t][j_id] += noise_offset

                  
            skeleton = np.array(skeleton)
            return skeleton

        def time_interpolate(skeleton):
            skeleton = np.array(skeleton)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []

            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1]#d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i -1] + displace)# r*disp

            while len(result) < self.time_len:
                result.append(result[-1]) #padding
            result = np.array(result)
            return result




        # og_id = np.random.randint(3)
        aug_num = 4
        ag_id = randint(0, aug_num - 1)
        if ag_id == 0:
            skeleton = scale(skeleton)
        elif ag_id == 1:
            skeleton = shift(skeleton)
        elif ag_id == 2:
            skeleton = noise(skeleton)
        elif ag_id == 3:
            skeleton = time_interpolate(skeleton)

        return skeleton


    def upsample(self,skeleton,max_frames):
      tensor=torch.unsqueeze(torch.unsqueeze(torch.from_numpy(skeleton),dim=0),dim=0)

      out=nn.functional.interpolate(tensor, size=[max_frames,tensor.shape[-2],tensor.shape[-1]], mode='nearest')
      tensor=torch.squeeze(torch.squeeze(out,dim=0),dim=0)

      return tensor

    def sample_frame(self, data_num):
        #sample #time_len frames from whole video
        
        sample_size = self.time_len
        each_num = (data_num - 1) / (sample_size - 1)
        idx_list = [0, data_num - 1]
        for i in range(sample_size):
            index = round(each_num * i)
            if index not in idx_list and index < data_num:
                idx_list.append(index)
        idx_list.sort()

        while len(idx_list) < sample_size:
            idx = random.randint(0, data_num - 1)
            if idx not in idx_list:
                idx_list.append(idx)
        idx_list.sort()
        return idx_list    





def init_data_loader(test_subjects_ids, data_cfg, sequence_len, batch_size, workers, device):

    train_data, test_data, edge_index = get_train_test_data(test_subjects_ids, data_cfg)


    train_dataset = Hand_Dataset(train_data, use_upsampling=True, use_data_aug = True, time_len = sequence_len)

    test_dataset = Hand_Dataset(test_data, use_upsampling=True, use_data_aug = False, time_len = sequence_len)

    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))

    print("batch size:", batch_size)
    print("workers:", workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
         pin_memory=True, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
         pin_memory=True, num_workers=workers)

    return train_loader, val_loader, edge_index.to(device)