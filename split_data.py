import glob
import random
import os
import shutil

def create_folder(data_dir,out_dir):
    try:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        
    except:
        os.mkdir(out_dir)
    train_dir = os.path.join(out_dir,"train")
    test_dir = os.path.join(out_dir,"val")
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    labels=os.listdir(data_dir)
    maps={x:labels[x] for x in range(len(labels))} # id : label
    maps_txt=""
    for x in maps.items():
        maps_txt += str(x[0])+" "+str(x[1])+"\n"
    f=open(os.path.join(out_dir,"maps.txt"),"w+")
    f.writelines(maps_txt[:-1])
    for x in maps.items():
        os.mkdir(os.path.join(out_dir,"train",str(x[0])))
        os.mkdir(os.path.join(out_dir,"val",str(x[0])))
    return maps

def split_data(data_dir,out_dir,rate):
    maps=create_folder(data_dir,out_dir)
    for x in maps.items():
        datasub=os.path.join(data_dir,str(x[1]))
        list_img=glob.glob(os.path.join(datasub,"*"))
        random.shuffle(list_img)
        train=list_img[:int(rate/100*len(list_img))]
        test=list_img[:int((100-rate)/100*len(list_img))]
        for item in train:
            shutil.copy(item,os.path.join(out_dir,"train",str(x[0]),item.split("/")[-1]))
        for item in test:
            shutil.copy(item,os.path.join(out_dir,"val",str(x[0]),item.split("/")[-1]))

    
# split_data(data_dir,out_dir,rate)