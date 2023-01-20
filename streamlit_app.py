#IMPORT LIBRARY
from collections import namedtuple
from PIL import Image
from skimage.segmentation import slic
from skimage.color import label2rgb
from scipy.spatial import distance
import altair as alt
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import math
import collections
import matplotlib as plt

# streamlit run streamlit_app.py
# path = C:\Users\tansi\Documents\SEM 1\VISUAL INFORMATION PROCESSING\github\VisualProcessing

def histogram(img,  binsize=128):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    segments_slic = slic(img, n_segments=300, start_label=1)
    colorized = label2rgb(segments_slic, image=img, kind='avg')
    
    r, g, b = colorized[:,:,0], colorized[:,:,1], colorized[:,:,2]
    r_hist, r_bin = np.histogram(r, binsize, density=True)
    g_hist, g_bin = np.histogram(g, binsize, density=True)
    b_hist, b_bin = np.histogram(b, binsize, density=True)
    rgb_hist = np.concatenate((r_hist, g_hist, b_hist))

    return rgb_hist

imagedataset = pd.read_csv('Test_data.csv')
imagedataset['Color_Histogram'] = imagedataset['Color_Histogram'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

st.title('Logo Retrieval and Recognition System')
upload_file = st.file_uploader('Please upload an Image file', type=["jpg", "jpeg", "png","jfif"])


if upload_file is not None:
    st.success("Image file uploaded")
    # Read the file to a dataframe using cv2
    st.image(upload_file, caption='Uploaded Image', use_column_width=True)                

    # convert to opencv_image
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    ###########################################  CROP IMAGE ###########################################
    
    # Create the sliders for the top left point
    x1 = st.slider('Select the X coordinate for the top left point', 0, opencv_image.shape[1], 0)
    y1 = st.slider('Select the Y coordinate for the top left point', 0, opencv_image.shape[0], 0)

    # Create the sliders for the bottom right point
    x2 = st.slider('Select the X coordinate for the bottom right point', x1, opencv_image.shape[1], opencv_image.shape[1])
    y2 = st.slider('Select the Y coordinate for the bottom right point', y1, opencv_image.shape[0], opencv_image.shape[0])
        
    # Create the cropping function
    def crop_image(img, x1, y1, x2, y2):
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            st.warning("Cannot crop image with width or height of 0.")
            return None
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return pil_img.crop((x1, y1, x2, y2))

    # Create the cropped image
    cropped_image = crop_image(opencv_image, x1, y1, x2, y2)

    # Show the image
    if cropped_image:
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)
    
    cropped_image = np.asarray(cropped_image, dtype=np.uint8)

    # Compute the mean and standard deviation of the image
    mean, std = cv2.meanStdDev(cropped_image)

    # Create a copy of the image
    normalized_image = cropped_image.copy()

    # Normalize the image
    cv2.normalize(cropped_image, normalized_image, mean[0][0], std[0][0], cv2.NORM_MINMAX)

    # Show the image
    # st.image(normalized_image, caption="Normalized Image", use_column_width=True)

    # Blur the image
    blurred = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Compute the difference image
    difference = normalized_image - blurred
    strength = 1

    # Add the difference image to the original image
    sharpened = normalized_image + strength * difference

    # Show the image
    # st.image(sharpened, caption="Sharpened Image", use_column_width=True)
    
    # let the user select threshold value
    threshold_value = st.slider("Select Threshold Value", 0, 255, 120)

    # perform gaussianBlur
    img_blur = cv2.GaussianBlur(sharpened, (5, 5), 0)

    # convert colorspace
    grayImage = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # thresholding
    ret, thres = cv2.threshold(grayImage, threshold_value, 255, cv2.THRESH_BINARY)
    st.image(thres, caption='Processed Image', use_column_width=True)

    if st.button("Invert Colors"):
        thres = cv2.bitwise_not(thres)

    #cropped_image = np.array(cropped_image)

    # Perform bitwise_and operation between the original image and the thresholded image
    result = cv2.bitwise_and(cropped_image, cropped_image, mask=thres)

    # Show the image
    st.image(result, caption="Background removed image", use_column_width=True)

    edges = cv2.Canny(thres, 50, 150)
    st.image(edges, caption='Edged Image', use_column_width=True)

    ###########################################  COLOR HISTOGRAM ###########################################
    
    if st.button("Search"):
        query_Image = histogram(cropped_image)
        best_matches = []

        for index, row in imagedataset.iterrows():
            histogramrow = row['Color_Histogram']
            # histogramrow = histogramrow.flatten()
            # query_Image = query_Image.flatten()
            dist = distance.euclidean(histogramrow, query_Image)
            best_matches.append((row['filename'], dist))

        best_matches = sorted(best_matches, key=lambda x: x[1])

        import os
        folder_path = os.getcwd() + '\datasets'
        file_path = os.path.join(folder_path, best_matches[0][0])

        st.text(best_matches[0][1])
        st.image(cv2.imread(file_path), caption='Similar Image', use_column_width=True)

    ########################################################################################################

#    if(st.button("CROP")):
#        r = cv2.selectROI(edges)
#        img_cropped = edges[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        st.image(img_cropped, caption='Edged Image', use_column_width=True)