import os, sys, joblib 
import numpy as np
import ypoften as of
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

# build the pre-trained model
base_model = VGG16(weights= "imagenet", include_top=True)
print(base_model.summary())
feature_model = Model(base_model.input, base_model.get_layer('fc2').output)
img_size = (224, 224)
print(feature_model.summary())

# the image was split into 2 x 2 blocks
nblock = 2

# the folder that stores split iamges
blockfolder = os.path.join("", "img block", "")

# read image name
imgpath = "oranges.jpg"
imgname = os.path.basename(imgpath).replace(".jpg","")

# transform each block into vector with pre-trained model
for p in range(1,nblock*nblock+1): 

    # get image path
    imgpath = os.path.join(blockfolder, imgname + "." + str(p) + '.jpg')

    # load the image 
    img = image.load_img(imgpath, target_size=img_size) 
    # convert the image to a numpy array
    image_array = image.img_to_array(img)
    # add one dimension so it can be read by the pre-trained model
    image_expand = np.expand_dims(image_array, 0)
    # normalize image data to 0-1 range
    x_train = preprocess_input(image_expand)
    # extract features for each image
    features_x = feature_model.predict(x_train)
 
    # specify the folder to save the results   
    exfolder = os.path.join("","img block exfeature","")
    featuresavepath = os.path.join(exfolder, imgname + "." + str(p) + ".dat")

    of.create_path(featuresavepath)
    joblib.dump(features_x[0], featuresavepath)
    print("DONE")

print("DONE"*20)