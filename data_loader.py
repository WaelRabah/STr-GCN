from torch.utils.data import Dataset
from random import randint,shuffle
import wget
import os
import zipfile
import torch
import numpy as np
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split

def download_DHG_dataset():
    print("Downloading DHG 2016 dataset.......")
    if not os.path.exists("DHG2016.zip") :
        url = 'http://www-rech.telecom-lille.fr/DHGdataset/DHG2016.zip'
        wget.download(url)
    if not os.path.exists("./dataset") :
        os.mkdir("./dataset")
        with zipfile.ZipFile("DHG2016.zip", 'r') as zip_ref:
            zip_ref.extractall("./dataset")

def download_SHREC_2017_DHG_dataset():
  
    if not os.path.exists("HandGestureDataset_SHREC2017.rar") :
        print("Downloading DHG SHREC 2017 dataset.......")
        url = 'http://www-rech.telecom-lille.fr/shrec2017-hand/HandGestureDataset_SHREC2017.rar'
        wget.download(url)
    if not os.path.exists("./dataset") :
        os.mkdir("./dataset")
        import time
        start_time = time.time()
        patoolib.extract_archive("HandGestureDataset_SHREC2017.rar", outdir="dataset")
        print("--- %s seconds ---" % (time.time() - start_time))

#Change the path to your downloaded dataset
dataset_fold = "./dataset/HandGestureDataset_SHREC2017"

#change the path to your downloaded DHG dataset

def load_SHREC17_data_from_disk():
  neighbors_by_node={1: [1, 2, 3], 2: [2, 1, 7, 11, 15, 19], 3: [3, 1, 4], 4: [4, 3, 5], 5: [5, 4, 6], 6: [6, 5], 7: [7, 2, 8], 8: [8, 7, 9], 9: [9, 8, 10], 10: [10, 9], 11: [11, 2, 12], 12: [12, 11, 13], 13: [13, 12, 14], 14: [14, 13], 15: [15, 2, 16], 16: [16, 15, 17], 17: [17, 16, 18], 18: [18, 17], 19: [19, 2, 20], 20: [20, 19, 21], 21: [21, 20, 22], 22: [22, 21]}


  def get_edge_index():
    '''
    Generates the adjacency matrix for this dataset
    '''
    adj=np.zeros(shape=(22,22))
    for key,neighbours in neighbors_by_node.items():
      for n in neighbours :
        adj[key-1,n-1]=1
    # edge_index=torch.from_numpy(adj).to_sparse()
    # return edge_index.indices()
    edge_index=torch.from_numpy(adj)
    return edge_index

  def parse_data(src_file):
    '''
    Retrieves the skeletons sequence for each gesture
    '''
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
    
  train_data = {}
  test_data = {}
  print("Loading training data ...")
  with open(dataset_fold+"/train_gestures.txt","r") as f :
    for line in f :
      params=line.split(" ")
      g_id=params[0]
      f_id=params[1]
      sub_id=params[2]
      e_id=params[3]
      src_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(g_id, f_id, sub_id, e_id)
      src_file = open(src_path)
      video = parse_data(src_file) #the 22 points for each frame of the video
      key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
      train_data[key] = video
      src_file.close()

  print("Loading testing data ...")
  with open(dataset_fold+"/test_gestures.txt","r") as f :
    for line in f :
      params=line.split(" ")
      g_id=params[0]
      f_id=params[1]
      sub_id=params[2]
      e_id=params[3]
      src_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(g_id, f_id, sub_id, e_id)
      src_file = open(src_path)
      video = parse_data(src_file) #the 22 points for each frame of the video
      key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
      test_data[key] = video
      src_file.close()
  return train_data, test_data, get_edge_index()


def split_train_test(train_data_dict, test_data_dict, cfg):
  #split data into train and test
  #cfg = 0 >>>>>>> 14 categories      cfg = 1 >>>>>>>>>>>> 28 cate
    dataset = []
    train_data=[]
    test_data=[]
    val_data=[]
    train_path = dataset_fold + "/train_gestures.txt"
    test_path = dataset_fold + "/test_gestures.txt"
    with open(dataset_fold+"/train_gestures.txt","r") as f :
      for line in f :
        params=line.split(" ")
        g_id=params[0]
        f_id=params[1]
        sub_id=params[2]
        e_id=params[3]
        key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

        #set table to 14 or
        if cfg == 0:
            label = int(g_id)
        elif cfg == 1:
            if f_id == 1:
                label = int(g_id)
            else:
                label = int(g_id) + 14

        #split to train and test list
        data = train_data_dict[key]
        sample = {"skeleton":data, "label":label}
        train_data.append(sample)

    with open(dataset_fold+"/test_gestures.txt","r") as f :
      for line in f :
        params=line.split(" ")
        g_id=params[0]
        f_id=params[1]
        sub_id=params[2]
        e_id=params[3]
        key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

        #set table to 14 or
        if cfg == 0:
            label = int(g_id)
        elif cfg == 1:
            if f_id == 1:
                label = int(g_id)
            else:
                label = int(g_id) + 14

        #split to train and test list
        data = test_data_dict[key]
        sample = {"skeleton":data, "label":label}
        test_data.append(sample)


    return train_data, test_data, test_data

def get_train_test_data(test_subject_id,val_subject_id, cfg):
    download_SHREC_2017_DHG_dataset()
    print("Reading data from disk.......")
    train_data, test_data,edge_index = load_SHREC17_data_from_disk()
    print("Splitting data into train/test sets .......")
    #filtered_video_data = get_valid_frame(video_data)
    train_data, test_data, val_data = split_train_test( train_data, test_data, cfg)
    return train_data, test_data, val_data,edge_index

class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, time_len, use_data_aug,use_aug_features):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """
        self.use_data_aug = use_data_aug
        self.data = data
        self.frame_sizes=[ len(data_ele["skeleton"]) for data_ele in self.data]
        self.max_seq_size=time_len
        self.time_len = time_len
        self.compoent_num = 22
        if self.use_data_aug:
            augmented_data=[]
            for data_el in self.data : 
                augmented_skeletons=self.data_aug(data_el['skeleton'])
                for s in augmented_skeletons :
                    augmented_data.append({"skeleton":self.aug_features(torch.from_numpy(s).float()) if use_aug_features else s ,"label":data_el['label']})
            self.data=augmented_data
        else :
            self.data=[{"skeleton": self.aug_features(torch.from_numpy(data_el['skeleton']).float()) if use_aug_features else data_el['skeleton'] ,"label":data_el['label']} for data_el in data]


    def __len__(self):
        return  len(self.data)
    def compute(self,sequence) : 
      def angle_between_pair(x,x_next):
        scalar_product=torch.einsum('jk,jk->j', x, x_next).unsqueeze(-1)
        l2_x=torch.sqrt(torch.einsum('jk,jk->j', x, x).unsqueeze(-1))
        l2_x_next=torch.sqrt(torch.einsum('jk,jk->j', x_next, x_next).unsqueeze(-1))
        angles=torch.arccos(scalar_product/(l2_x*l2_x_next))
        return torch.nan_to_num(angles) 
        
      movement_vectors=torch.tensor([],dtype=torch.float32)
      distances=torch.tensor([],dtype=torch.float32)
      angles=torch.tensor([],dtype=torch.float32)
      for i in range(len(sequence)): 
        if i==len(sequence)-1 :
          x=sequence[i]
          x_next=sequence[i]
          movement_vec=x_next-x
          movement_vectors=torch.cat((movement_vectors,movement_vec.unsqueeze(0)))
          l2_norm=torch.sqrt(torch.sum((movement_vec) ** 2,dim=1)).unsqueeze(-1)
          distances=torch.cat((distances,l2_norm.unsqueeze(0)))
          angle_b_p=angle_between_pair(x,x_next)
          angles=torch.cat((angles,angle_b_p.unsqueeze(0)))
          continue
        x=sequence[i]
        x_next=sequence[i+1]
        movement_vec=(x_next-x)
        movement_vectors=torch.cat((movement_vectors,movement_vec.unsqueeze(0)))
        l2_norm=torch.sqrt(torch.sum((movement_vec) ** 2,dim=1)).unsqueeze(-1)
        distances=torch.cat((distances,l2_norm.unsqueeze(0)))
        angle_b_p=angle_between_pair(x,x_next)
        angles=torch.cat((angles,angle_b_p.unsqueeze(0)))
      return distances, angles, movement_vectors
    def aug_features(self, sample: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [ batch_size, seq_len, n_nodes, n_features]
        """
        distances, angles, movement_vectors=self.compute(sample)
        return torch.cat([sample,distances, angles, movement_vectors],dim=-1)
    def __getitem__(self, ind):

        data_ele = self.data[ind]
        #hand skeleton
        skeleton = data_ele["skeleton"]
        skeleton = np.array(skeleton)

        data_num = skeleton.shape[0]
        if data_num >= self.max_seq_size :
          idx_list = self.sample_frames(data_num)
          skeleton = [skeleton[idx] for idx in idx_list]
          skeleton = np.array(skeleton)
          skeleton=torch.from_numpy(skeleton)
        else :
          skeleton = self.upsample(skeleton,self.max_seq_size)
        #normalize by palm center
        skeleton -= torch.clone(skeleton[0][1])
        skeleton = skeleton.float()
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
                for t in range(skeleton.shape[0]):
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



        skeleton = np.array(skeleton)
        # og_id = np.random.randint(3)
        aug_num = 4
        ag_id = randint(0, aug_num - 1)
        # skeleton_scaled = scale(skeleton)
        # skeleton_shifted = shift(skeleton)
        # skeleton_noise = noise(skeleton)
        # skeleton_time_interpolated = time_interpolate(skeleton)

        skeleton_aug = scale(skeleton)
        skeleton_aug = shift(skeleton_aug)
        skeleton_aug = noise(skeleton_aug)
        skeleton_aug = time_interpolate(skeleton_aug)
        skeletons=[skeleton, skeleton_aug]
        return  skeletons


    def upsample(self,skeleton,max_frames):
      tensor=torch.unsqueeze(torch.unsqueeze(torch.from_numpy(skeleton),dim=0),dim=0)

      out=nn.functional.interpolate(tensor, size=[max_frames,tensor.shape[-2],tensor.shape[-1]], mode='nearest')
      tensor=torch.squeeze(torch.squeeze(out,dim=0),dim=0)

      return tensor

    def sample_frames(self, data_num):
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
            if idx not in idx_list :
                idx_list.append(idx)
        idx_list.sort()
        return idx_list    







def init_data_loader(test_subject_id, val_subject_id, data_cfg, sequence_len, batch_size, workers, device):

    train_data, test_data, val_data, edge_index = get_train_test_data(test_subject_id,val_subject_id, data_cfg)


    train_dataset = Hand_Dataset(train_data, use_data_aug = True, use_aug_features=False, time_len = sequence_len)

    test_dataset = Hand_Dataset(test_data, use_data_aug = False, use_aug_features=False, time_len = sequence_len)

    val_dataset = Hand_Dataset(val_data, use_data_aug = False, use_aug_features=False, time_len = sequence_len)

    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))
    print("val data num: ",len(val_dataset))

    print("batch size:", batch_size)
    print("workers:", workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
         pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
         pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
         pin_memory=True)

    return train_loader, test_loader, val_loader, edge_index.to(device)
    