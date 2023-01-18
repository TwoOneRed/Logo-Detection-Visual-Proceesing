#IMPORT LIBRARY
from collections import namedtuple
from PIL import Image, ImageDraw
import altair as alt
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import math
import collections

# streamlit run streamlit_app.py
# path = C:\Users\tansi\Documents\SEM 1\VISUAL INFORMATION PROCESSING\github\VisualProcessing

st.title('Logo Retrieval and Recognition System')
upload_file = st.file_uploader('Please upload an Image file', type=["jpg", "jpeg", "png","jfif"])


if upload_file is not None:
    st.success("Image file uploaded")
    # Read the file to a dataframe using cv2
    st.image(upload_file, caption='Uploaded Image', use_column_width=True)                

    # convert to opencv_image
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

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

    status = False
    # Show the image
    if cropped_image:
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)
    else:
        st.image(opencv_image, caption='Cropped Image', use_column_width=True)

    processimage = np.asarray(cropped_image, dtype=np.uint8)
    

    # Compute the mean and standard deviation of the image
    mean, std = cv2.meanStdDev(processimage)

    # Create a copy of the image
    normalized_image = processimage.copy()

    # Normalize the image
    cv2.normalize(processimage, normalized_image, mean[0][0], std[0][0], cv2.NORM_MINMAX)

    # Show the image
    st.image(normalized_image, caption="Normalized Image", use_column_width=True)


    # Blur the image
    blurred = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Compute the difference image
    difference = normalized_image - blurred

    strength = 2.0

    # Add the difference image to the original image
    sharpened = normalized_image + strength * difference

    # Show the image
    st.image(sharpened, caption="Sharpened Image", use_column_width=True)
    # let the user select threshold value
    threshold_value = st.slider("Select Threshold Value", 0, 255, 150)

    # perform gaussianBlur
    img_blur = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # convert colorspace
    grayImage = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # thresholding
    ret, thres = cv2.threshold(grayImage, threshold_value, 255, cv2.THRESH_BINARY)
    st.image(thres, caption='Processed Image', use_column_width=True)

    edges = cv2.Canny(thres, 50, 150)
    st.image(edges, caption='Edged Image', use_column_width=True)

    










#    if(st.button("CROP")):
#        r = cv2.selectROI(edges)
#        img_cropped = edges[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        st.image(img_cropped, caption='Edged Image', use_column_width=True)