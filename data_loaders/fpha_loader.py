import os 
def load_FPHA_data():
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
    dataset_folder = "./data/FPHA_data/"
    dataset={}
    for sub_id in range(1, 7):

        gestures = os.listdir(dataset_folder+f"Subject_{sub_id}")
        for g_id,g in enumerate(gestures):
            essais = os.listdir(dataset_folder+f"Subject_{sub_id}/{g}")
            for e in essais :
                src_file = open(dataset_folder+f"Subject_{sub_id}/{g}/{e}/skeleton.txt")
                data = parse_data(src_file)
                dataset[f"Subject_{sub_id}/{g}/{e}"]=data
    
    test_data=[]
    train_data=[]
    add_training=True
    with open("./data/FPHA_data/data_split_action_recognition.txt") as f :
        for line in f : 
            fields=line.split(" ")
            if "Training" in fields :
                add_training=True 
                continue
            if "Test" in fields :
                print("e")
                add_training=False
                continue
            if add_training :
                train_data.append({"skeleton":dataset[fields[0]],"label":int(fields[1])})
            else :
                test_data.append({"skeleton":dataset[fields[0]],"label":int(fields[1])})
    return train_data, test_data, test_data


