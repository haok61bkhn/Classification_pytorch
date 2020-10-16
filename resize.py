import cv2
import glob
datadir="./"
for i in range(9):
    for x in glob.glob(datadir+"/"+str(i)+"/*"):
        img=cv2.imread(x)
        img=cv2.resize(img,(400,600))
        cv2.imwrite(x,img)
