import os, cv2
import numpy as np
import ypoften as of

imgpath = "oranges.jpg"
imgname = os.path.basename(imgpath).replace(".jpg","")

# read image
img = cv2.imread(imgpath, 1)

# specify the folder for saving blocks
tffolder = os.path.join("","img block")

# divide the image into 2 x 2 blocks
nblock = 2
sp1d = np.array_split(img, nblock, axis=0) 
sp2d = [np.array_split(x, nblock, axis=1) for x in sp1d]
blocks = [x for sublist in sp2d for x in sublist]

# save each block
for i, block in enumerate(blocks):
    imgsavepath = os.path.join(tffolder, imgname + "." + str(i+1) + '.jpg')
    of.create_path(imgsavepath)
    print(imgsavepath)
    cv2.imwrite(imgsavepath, block)
    print("DONE")

print("DONE"*10)
