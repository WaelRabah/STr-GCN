from torch.utils.data import Dataset
import torch.nn as nn
import torch
import numpy as np
from torch import Tensor
from random import shuffle
import random
import torch.nn.functional as F

class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data,
                 time_len,
                 use_data_aug,
                 use_aug_features,
                 normalize=True,
                 scaleInvariance=False,
                 translationInvariance=False,
                 isPadding=False,
                 useSequenceFragments=False,
                 useRandomMoving=False,
                 useMirroring=False,
                 useTimeInterpolation=False,
                 useNoise=False,
                 useScaleAug=False,
                 useTranslationAug=False
                 ):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """

        self.use_data_aug = use_data_aug
        self.data = data
        self.frame_sizes = [len(data_ele["skeleton"])
                            for data_ele in self.data]
        self.max_seq_size = max(self.frame_sizes)
        self.time_len = time_len
        self.compoent_num = 22
        self.normalize = normalize
        self.scaleInvariance = scaleInvariance
        self.translationInvariance = translationInvariance
        # self.transform = transform
        self.isPadding = isPadding
        self.useSequenceFragments = useSequenceFragments
        self.useRandomMoving = useRandomMoving
        self.useMirroring = useMirroring
        self.useTimeInterpolation = useTimeInterpolation
        self.useNoise = useNoise
        self.useScaleAug = useScaleAug
        self.useTranslationAug = useTranslationAug
        if self.use_data_aug:
            augmented_data = []
            for data_el in self.data:
                augmented_skeletons = self.data_aug(self.preprocessSkeleton(
                    torch.from_numpy(np.array(data_el['skeleton'])).float()))
                for s in augmented_skeletons:
                    augmented_data.append({"skeleton": self.aug_features(self.preprocessSkeleton(torch.from_numpy(
                        np.array(data_el['skeleton'])).float())) if use_aug_features else s, "label": data_el['label']})
            self.data = augmented_data
        else:
            self.data = [{"skeleton": self.aug_features(self.preprocessSkeleton(torch.from_numpy(np.array(data_el['skeleton'])).float(
            ))) if use_aug_features else self.preprocessSkeleton(torch.from_numpy(np.array(data_el['skeleton'])).float()), "label":data_el['label']} for data_el in data]

    def __len__(self):
        return len(self.data)

    def compute(self, sequence):
        def angle_between_pair(x, x_next):
            scalar_product = torch.einsum('jk,jk->j', x, x_next).unsqueeze(-1)
            l2_x = torch.sqrt(torch.einsum('jk,jk->j', x, x).unsqueeze(-1))
            l2_x_next = torch.sqrt(torch.einsum(
                'jk,jk->j', x_next, x_next).unsqueeze(-1))
            angles = torch.arccos(scalar_product/(l2_x*l2_x_next))
            return torch.nan_to_num(angles)

        def cartesianToSpherical(inputs):
            x = torch.clone(inputs[:, :, 0]).unsqueeze(-1)
            y = torch.clone(inputs[:, :, 1]).unsqueeze(-1)
            z = torch.clone(inputs[:, :, 2]).unsqueeze(-1)
            # takes list xyz (single coord)
            r = torch.sqrt(torch.tensor(x*x + y*y + z*z))
            theta = torch.arccos(z/r)*180 / torch.pi  # to degrees
            phi = torch.atan2(torch.tensor(y), torch.tensor(x))*180 / torch.pi
            return torch.cat([r, theta, phi], dim=-1)
        movement_vectors = torch.tensor([], dtype=torch.float32)
        distances = torch.tensor([], dtype=torch.float32)
        angles = torch.tensor([], dtype=torch.float32)
        for i in range(0, len(sequence)):
            if i == 0:
                x = sequence[i]
                x_next = sequence[i+1]
                movement_vec = x_next-x
                movement_vectors = torch.cat(
                    (movement_vectors, movement_vec.unsqueeze(0)))
                l2_norm = torch.sqrt(
                    torch.sum((movement_vec) ** 2, dim=1)).unsqueeze(-1)
                distances = torch.cat((distances, l2_norm.unsqueeze(0)))
                angle_b_p = angle_between_pair(x, x_next)
                angles = torch.cat((angles, angle_b_p.unsqueeze(0)))
                continue
            x = sequence[i-1]
            x_next = sequence[i]
            movement_vec = (x_next-x)
            movement_vectors = torch.cat(
                (movement_vectors, movement_vec.unsqueeze(0)))
            l2_norm = torch.sqrt(
                torch.sum((movement_vec) ** 2, dim=1)).unsqueeze(-1)
            distances = torch.cat((distances, l2_norm.unsqueeze(0)))
            angle_b_p = angle_between_pair(x, x_next)
            angles = torch.cat((angles, angle_b_p.unsqueeze(0)))
        spherical_coordinates = cartesianToSpherical(sequence)
        return distances, angles, movement_vectors, spherical_coordinates

    def aug_features(self, sample: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [ batch_size, seq_len, n_nodes, n_features]
        """
        distances, angles, movement_vectors, spherical_coordinates = self.compute(
            sample)
        return torch.cat([sample, distances, angles, movement_vectors, spherical_coordinates], dim=-1)

    def preprocessSkeleton(self, skeleton):
        def translationInvariance(skeleton):
            # normalize by palm center value at frame=1
            skeleton -= torch.clone(skeleton[0][1])
            skeleton = skeleton.float()
            return skeleton

        def scaleInvariance(skeleton):

            x_c = torch.clone(skeleton)

            distance = torch.sqrt(torch.sum((x_c[0, 1]-x_c[0, 0])**2, dim=-1))

            factor = 1/distance

            x_c *= factor

            return x_c

        def normalize(skeleton):

            # if self.transform:
            #     skeleton = self.transform(skeleton.numpy())
            skeleton = F.normalize(skeleton)

            return skeleton
        if self.normalize:
            skeleton = normalize(skeleton)
        if self.scaleInvariance:
            skeleton = scaleInvariance(skeleton)
        if self.translationInvariance:
            skeleton = translationInvariance(skeleton)

        return skeleton

    def __getitem__(self, ind):

        data_ele = self.data[ind]
        # hand skeleton
        skeleton = data_ele["skeleton"]
        skeleton = np.array(skeleton)
        data_num = skeleton.shape[0]
        if self.isPadding:
            skeleton = self.preprocessSkeleton(skeleton)
            # padding
            skeleton = self.auto_padding(skeleton, self.max_seq_size)
            # label
            label = data_ele["label"]-1

            sample = {'skeleton': skeleton, "label": label}

            return sample

        if data_num >= self.time_len:
            idx_list = self.sample_frames(data_num)
            skeleton = [skeleton[idx] for idx in idx_list]
            skeleton = np.array(skeleton)
            skeleton = torch.from_numpy(skeleton)
        else:
            skeleton = self.upsample(skeleton, self.time_len)

        # skeleton = self.preprocessSkeleton(skeleton)
        # label
        label = data_ele["label"]-1
        sample = {'skeleton': skeleton, "label": label}

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
            # select 4 joints
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
                displace = skeleton[i] - skeleton[i - 1]  # d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i - 1] + displace)  # r*disp

            while len(result) < self.time_len:
                result.append(result[-1])  # padding
            result = np.array(result)
            return result

        def random_sequence_fragments(sample):
            samples = [sample]
            sample = torch.from_numpy(sample)
            n_fragments = 2
            T, V, C = sample.shape
            if T <= self.time_len:
                return samples
            for _ in range(n_fragments):

                # fragment_len=int(T*fragment_len)
                fragment_len = self.time_len
                max_start_frame = T-fragment_len

                random_start_frame = random.randint(0, max_start_frame)
                new_sample = sample[random_start_frame:random_start_frame+fragment_len]
                samples.append(new_sample.numpy())

            return samples

        def mirroring(data_numpy):
            T, V, C = data_numpy.shape
            data_numpy[:, :, 0] = np.max(
                data_numpy[:, :, 0]) + np.min(data_numpy[:, :, 0]) - data_numpy[:, :, 0]
            return data_numpy

        def random_moving(data_numpy,
                          angle_candidate=[-10., -5., 0., 5., 10.],
                          scale_candidate=[0.9, 1.0, 1.1],
                          transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                          move_time_candidate=[1]):
            # input: T,V,C
            data_numpy = np.transpose(data_numpy, (2, 0, 1))
            new_data_numpy = np.zeros(data_numpy.shape)
            C, T, V = data_numpy.shape
            move_time = random.choice(move_time_candidate)

            node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
            node = np.append(node, T)
            num_node = len(node)

            A = np.random.choice(angle_candidate, num_node)
            S = np.random.choice(scale_candidate, num_node)
            T_x = np.random.choice(transform_candidate, num_node)
            T_y = np.random.choice(transform_candidate, num_node)

            a = np.zeros(T)
            s = np.zeros(T)
            t_x = np.zeros(T)
            t_y = np.zeros(T)

            # linspace
            for i in range(num_node - 1):
                a[node[i]:node[i + 1]] = np.linspace(
                    A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
                s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                     node[i + 1] - node[i])
                t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                       node[i + 1] - node[i])
                t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                       node[i + 1] - node[i])

            theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                              [np.sin(a) * s, np.cos(a) * s]])

            # perform transformation
            for i_frame in range(T):
                xy = data_numpy[0:2, i_frame, :]
                new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))

                new_xy[0] += t_x[i_frame]
                new_xy[1] += t_y[i_frame]

                new_data_numpy[0:2, i_frame, :] = new_xy.reshape(2, V)

            new_data_numpy[2, :, :] = data_numpy[2, :, :]

            return np.transpose(new_data_numpy, (1, 2, 0))

        skeleton = np.array(skeleton)
        skeletons = [skeleton]
        if self.useTimeInterpolation:
            skeletons.append(time_interpolate(skeleton))

        if self.useNoise:
            skeletons.append(noise(skeleton))

        if self.useScaleAug:
            skeletons.append(scale(skeleton))

        if self.useTranslationAug:
            skeletons.append(shift(skeleton))

        if self.useSequenceFragments:
            n_skeletons = []
            for s in skeletons:
                n_skeletons = [*n_skeletons, *random_sequence_fragments(s)]
            skeletons = [*skeletons, *n_skeletons]

        if self.useRandomMoving:
            # aug_skeletons = []
            # for s in skeletons:
            #     aug_skeletons.append(random_moving(s))
            skeletons.append(random_moving(skeleton))

        if self.useMirroring:
            aug_skeletons = []
            for s in skeletons:
                aug_skeletons.append(mirroring(s))
            skeletons = [*skeletons, *aug_skeletons]

        return skeletons

    def auto_padding(self, data_numpy, size, random_pad=False):
        C, T, V = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((C, size, V))
            data_numpy_paded[:, begin:begin + T, :] = data_numpy
            return data_numpy_paded
        else:
            return data_numpy

    def upsample(self, skeleton, max_frames):
        tensor = torch.unsqueeze(torch.unsqueeze(
            torch.from_numpy(skeleton), dim=0), dim=0)

        out = nn.functional.interpolate(
            tensor, size=[max_frames, tensor.shape[-2], tensor.shape[-1]], mode='trilinear')
        tensor = torch.squeeze(torch.squeeze(out, dim=0), dim=0)

        return tensor

    def sample_frames(self, data_num):
        # sample #time_len frames from whole video

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


class Graph():

    def __init__(self,
                 layout='DHG14/28',
                 strategy='uniform',
                 max_hop=2,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'DHG14/28':
            self.num_node = 22
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1),
                             (0, 2),
                             (1, 0),
                             (1, 6),
                             (1, 10),
                             (1, 14),
                             (1, 18),
                             (2, 0),
                             (2, 3),
                             (3, 2),
                             (3, 4),
                             (4, 3),
                             (4, 5),
                             (5, 4),
                             (6, 1),
                             (6, 7),
                             (7, 6),
                             (7, 8),
                             (8, 7),
                             (8, 9),
                             (9, 8),
                             (10, 1),
                             (10, 11),
                             (11, 10),
                             (11, 12),
                             (12, 11),
                             (12, 13),
                             (13, 12),
                             (14, 1),
                             (14, 15),
                             (15, 14),
                             (15, 16),
                             (16, 15),
                             (16, 17),
                             (17, 16),
                             (18, 1),
                             (18, 19),
                             (19, 18),
                             (19, 20),
                             (20, 19),
                             (20, 21),
                             (21, 20)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "FPHA":
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 0),
                (1, 6),
                (2, 0),
                (2, 7),
                (3, 0),
                (3, 8),
                (4, 0),
                (4, 9),
                (5, 0),
                (5, 10),
                (6, 1),
                (6, 11),
                (7, 2),
                (7, 12),
                (8, 3),
                (8, 13),
                (9, 4),
                (9, 14),
                (10, 5),
                (10, 15),
                (11, 6),
                (11, 16),
                (12, 7),
                (12, 17),
                (13, 8),
                (13, 18),
                (14, 9),
                (14, 19),
                (15, 10),
                (15, 20),
                (16, 11),
                (17, 12),
                (18, 13),
                (19, 14),
                (20, 15)
            ]
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(
            A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

    def normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
