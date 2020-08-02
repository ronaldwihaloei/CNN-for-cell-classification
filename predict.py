# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from scipy import misc
import glob

norm_size = 64

def predict(path):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("G:/2000.h5")
    MCF7 = 0
    MCF10A = 0
    MB231 = 0
    # load the image
    filelist = glob.glob('{}/*.tif'.format(path))
    for file in filelist:
        image = misc.imread(file)
        orig = image.copy()
    # pre-process the image for classification
        image = cv2.resize(image, (norm_size, norm_size))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

    # classify the input image
        result = model.predict(image)[0]
    # print (result.shape)
        proba = np.max(result)
        label = str(np.where(result == proba)[0])
        print(label)
        if label =="[0]":
            MCF7 += 1
        elif label =="[2]":
            MB231 += 1
        else:
            MCF10A +=1
    print("MCF-7", MCF7 / (MCF7 + MB231 + MCF10A))
    print("MCF-10a", MCF10A / (MCF7 + MB231 + MCF10A))
    print("MB-231", MB231 / (MCF7 + MB231 + MCF10A))


"""
print(label)
    if label=="[1]":
        label = 'MCF-7'
    elif label=="[0]":
        label = 'MCF-10A'
    elif label=="[2]":
        label = 'MCF-MB-231'
    else:
        label = label
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)

    if 1 ==1:
        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)
"""



# python predict.py --model traffic_sign.model -i ../2.png -s
if __name__ == '__main__':
    #args = args_parse()
    predict("G:\研一\cell/ce/test/0")