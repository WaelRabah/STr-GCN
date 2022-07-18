import wget
import os
import numpy as np
import patoolib

def download_SHREC_2017_DHG_dataset():

    if not os.path.exists("HandGestureDataset_SHREC2017.rar"):
        print("Downloading DHG SHREC 2017 dataset.......")
        url = 'http://www-rech.telecom-lille.fr/shrec2017-hand/HandGestureDataset_SHREC2017.rar'
        wget.download(url)
    if not os.path.exists("./data/shrec_dataset"):
        os.mkdir("./data/shrec_dataset")
        import time
        start_time = time.time()
        patoolib.extract_archive(
            "HandGestureDataset_SHREC2017.rar", outdir="shrec_dataset")
        print("--- %s seconds ---" % (time.time() - start_time))


# Change the path to your downloaded dataset
dataset_fold = "./data/shrec_dataset/HandGestureDataset_SHREC2017"

# change the path to your downloaded DHG dataset


def load_SHREC17_data_from_disk():


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
                    frame.append(np.array(point))
                    point = []
            frame=np.array(frame)
            video.append(frame)
        return np.array(video)

    train_data = {}
    test_data = {}
    print("Loading training data ...")
    with open(dataset_fold+"/train_gestures.txt", "r") as f:
        for line in f:
            params = line.split(" ")
            g_id = params[0]
            f_id = params[1]
            sub_id = params[2]
            e_id = params[3]
            src_path = dataset_fold + \
                "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(
                    g_id, f_id, sub_id, e_id)
            src_file = open(src_path)
            # the 22 points for each frame of the video
            video = parse_data(src_file)
            key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
            train_data[key] = video
            src_file.close()

    print("Loading testing data ...")
    with open(dataset_fold+"/test_gestures.txt", "r") as f:
        for line in f:
            params = line.split(" ")
            g_id = params[0]
            f_id = params[1]
            sub_id = params[2]
            e_id = params[3]
            src_path = dataset_fold + \
                "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(
                    g_id, f_id, sub_id, e_id)
            src_file = open(src_path)
            # the 22 points for each frame of the video
            video = parse_data(src_file)
            key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
            test_data[key] = video
            src_file.close()
    return train_data, test_data


def split_train_test(train_data_dict, test_data_dict, cfg):
    # split data into train and test
    # cfg = 0 >>>>>>> 14 categories      cfg = 1 >>>>>>>>>>>> 28 cate
    train_data = []
    test_data = []
    train_path = dataset_fold + "/train_gestures.txt"
    test_path = dataset_fold + "/test_gestures.txt"
    with open(train_path, "r") as f:
        for line in f:
            params = line.split(" ")
            g_id = params[0]
            f_id = params[1]
            sub_id = params[2]
            e_id = params[3]
            key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

            # set table to 14 or
            if cfg == 0:
                label = int(g_id)
            elif cfg == 1:
                if f_id == "1":
                    label = int(g_id)
                else:
                    label = int(g_id) + 14

            # split to train and test list
            data = train_data_dict[key]
            sample = {"skeleton": data, "label": label,"sample_name":key}
            train_data.append(sample)

    with open(test_path, "r") as f:
        for line in f:
            params = line.split(" ")
            g_id = params[0]
            f_id = params[1]
            sub_id = params[2]
            e_id = params[3]
            key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

            # set table to 14 or
            if cfg == 0:
                label = int(g_id)
            elif cfg == 1:
                if f_id == "1":
                    label = int(g_id)
                else:
                    label = int(g_id) + 14

            # split to train and test list
            data = test_data_dict[key]
            sample = {"skeleton": data, "label": label,"sample_name":key}
            test_data.append(sample)

    return train_data, test_data, test_data




def load_shrec_data(cfg):
    download_SHREC_2017_DHG_dataset()
    print("Reading data from disk.......")
    train_data, test_data = load_SHREC17_data_from_disk()
    print("Splitting data into train/test sets .......")
    #filtered_video_data = get_valid_frame(video_data)
    train_data, test_data, val_data = split_train_test(
        train_data, test_data, cfg)
    return train_data, test_data, val_data