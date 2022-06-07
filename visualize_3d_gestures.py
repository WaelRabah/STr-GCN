import os
import matplotlib.pyplot as  plt
import matplotlib
import shutil
from PIL import Image
from datetime import datetime
from multiprocessing import Pool, Lock
import wget
import zipfile
import patoolib
import torch
import numpy as np
matplotlib.use('Agg')
neighbors_by_node={1: [1, 2, 3], 2: [2, 1, 7, 11, 15, 19], 3: [3, 1, 4], 4: [4, 3, 5], 5: [5, 4, 6], 6: [6, 5], 7: [7, 2, 8], 8: [8, 7, 9], 9: [9, 8, 10], 10: [10, 9], 11: [11, 2, 12], 12: [12, 11, 13], 13: [13, 12, 14], 14: [14, 13], 15: [15, 2, 16], 16: [16, 15, 17], 17: [17, 16, 18], 18: [18, 17], 19: [19, 2, 20], 20: [20, 19, 21], 21: [21, 20, 22], 22: [22, 21]}
views=[0,45,90,135,180,225,270,315]

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


def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    mode = 0o666
    os.mkdir(path, mode)
def sort_num(files):
    d={}
    numbers=[]
    for f in files :
        n=f.split(".")[0].split("_")[2]
        numbers.append(int(n))
        d[int(n)]=f
    numbers=sorted(numbers)
    files=[]
    for n in numbers :
        files.append(d[n])
    return files
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
def gen_sequence_GIF(g_id,s_id,e_id,view):
    path=f"./sequences/{g_id}_{s_id}_{e_id}/sequence_{view}"
    
    files=list(filter(lambda s : "png" in s,os.listdir(path)))
    files=sort_num(files)
    # Create the frames
    frames = []
    files
    for i in files:
        new_frame = Image.open(path+"/"+i)
        frames.append(new_frame)
        
    # Save into a GIF file that loops forever  
    if os.path.exists(f"./gestures/{g_id}_{s_id}_{e_id}/gesture_{view}.gif"):
        os.remove(f"./gestures/{g_id}_{s_id}_{e_id}/gesture_{view}.gif")
    
    frames[0].save(f'./gestures/{g_id}_{s_id}_{e_id}/gesture_{view}.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=100, loop=0)

    path=f"./sequences/{g_id}_{s_id}_{e_id}/sequence_{view}_interpolated"
    
    files=list(filter(lambda s : "png" in s,os.listdir(path)))
    files=sort_num(files)
    # Create the frames
    frames = []
    files
    for i in files:
        new_frame = Image.open(path+"/"+i)
        frames.append(new_frame)
        
    # Save into a GIF file that loops forever  
    if os.path.exists(f"./gestures/{g_id}_{s_id}_{e_id}/gesture_{view}_interpolated.gif"):
        os.remove(f"./gestures/{g_id}_{s_id}_{e_id}/gesture_{view}_interpolated.gif")
    
    frames[0].save(f'./gestures/{g_id}_{s_id}_{e_id}/gesture_{view}_interpolated.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=60, loop=0)

def upsample(skeleton,max_frames):
    tensor=torch.unsqueeze(torch.unsqueeze(torch.from_numpy(skeleton),dim=0),dim=0)

    out=torch.nn.functional.interpolate(tensor, size=[max_frames,tensor.shape[-2],tensor.shape[-1]], mode='nearest')
    tensor=torch.squeeze(torch.squeeze(out,dim=0),dim=0)

    return tensor



def save_3d_sequence(g_id,s_id,e_id,view):
    def save_video(video,interpolated=False):
        fig_index=0
        for frame in video :
            fig = plt.figure()
            ax = plt.axes(projection = '3d')
            ax.axis("off")
            ax.grid(False)
            ax.set_aspect("auto")
            ax.view_init(elev=20., azim=view)
            for i in range(0,len(frame)-1) :

                x,y,z=frame[i]
                x_n,y_n,z_n=frame[i+1]
                # Visualize 3D scatter plot
                ax.scatter3D(x, y, z)
                if i+2 in neighbors_by_node[i+1] :
                    ax.plot([x,x_n], [y,y_n], zs=[z,z_n])
                # Give labels
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                
            seq=str(view)+("_interpolated" if interpolated else "")
            # Save figure
            plt.savefig(f"./sequences/{g_id}_{s_id}_{e_id}/sequence_{seq}/3d_scatter_{fig_index}.png", dpi = 300)
            plt.close()
            fig_index+=1
    
    path=f"./dataset/HandGestureDataset_SHREC2017/gesture_{g_id}/finger_1/subject_{s_id}/essai_{e_id}/skeletons_world.txt"
    src_file = open(path)
    # global lock
    # with lock :
    video = parse_data(src_file) #the 22 points for each frame of the video
        
    # Create 3D container
    
    video=np.array(video)
    interpolated_video=upsample(video,16)  
    
    save_video(video)
    save_video(interpolated_video,True)
    




def create_gesture_visualizations(g_id):
    for s_id in range(1,28):

        subject_path=f"./dataset/HandGestureDataset_SHREC2017/gesture_{g_id}/finger_1/subject_{s_id}"
        essais=os.listdir(subject_path) 
        for essai in essais :
            e_id=essai.split("_")[1]
            create_directory(f"./sequences/{g_id}_{s_id}_{e_id}")
            create_directory(f"./gestures/{g_id}_{s_id}_{e_id}")
            for view in views :
                create_directory(f"./sequences/{g_id}_{s_id}_{e_id}/sequence_{view}")
                create_directory(f"./sequences/{g_id}_{s_id}_{e_id}/sequence_{view}_interpolated")
                save_3d_sequence(g_id,s_id,e_id,view)
                gen_sequence_GIF(g_id,s_id,e_id,view)  







if __name__ == '__main__':
    lock = Lock()
    download_SHREC_2017_DHG_dataset()
    create_directory("./sequences")
    create_directory("./gestures")
    _start = datetime.now()

    with Pool() as pool:
        for p in [pool.apply_async(func=create_gesture_visualizations, args=(g_id,))
                  for g_id in range(1,15)]:
            p.wait()
        pool.close()
        pool.join() 
    _end = datetime.now()

    print(f'Duration={_end-_start}')