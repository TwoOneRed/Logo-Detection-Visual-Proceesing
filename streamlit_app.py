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
        # Draw a black rectangle over the portion of the image that will be cropped
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return pil_img.crop((x1, y1, x2, y2))

    # Create the cropped image
    cropped_image = crop_image(opencv_image, x1, y1, x2, y2)

    # Show the image
    st.image(cropped_image, use_column_width=True)

    # Create the button
    if st.button('Crop'):
        st.success("Image has been cropped")


    # let the user select threshold value
    threshold_value = st.slider("Select Threshold Value", 0, 255, 150)

    # perform gaussianBlur
    img_blur = cv2.GaussianBlur(opencv_image, (5, 5), 0)

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