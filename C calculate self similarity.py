import os, joblib
import numpy as np
import ypoften as of
from numpy.linalg import norm

# read image name
imgpath = "oranges.jpg"
imgname = os.path.basename(imgpath).replace(".jpg","")

# 2 x 2 blocks
nblock = 2
lenblock = nblock ** 2

resultpath = os.path.join("", "img result",'attr repetition.txt')

# find the folder that saves the results   
exfolder = os.path.join("","img block exfeature","")

# a list that saves similarity measures 
l_similarity = []

# loop over all pairs of blocks
for i in range(1, lenblock):
    for j in range(i + 1, lenblock+1):
        # read vectors
        expath1 = os.path.join(exfolder, imgname + "." + str(i) + ".dat")
        expath2 = os.path.join(exfolder, imgname + "." + str(j) + ".dat")
        ex1 = joblib.load(expath1)
        ex2 = joblib.load(expath2)
        # calculate cosine similarity
        sim = np.inner(ex1, ex2) / (norm(ex1) * norm(ex2)) 
        l_similarity.append(sim)

# get average similarity
average_sim = np.mean(l_similarity)
print(imgname, average_sim)
    
# save results
wlist = [imgname] + l_similarity[:] + [average_sim]
of.save_list_to_txt(wlist, resultpath)

print("DONE"*50)