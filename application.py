#IMPORT LIBRARY
from PIL import Image
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import matplotlib as plt
import base64
from concurrent.futures import ThreadPoolExecutor

#APPLICATION TITLE
st.title('Logo Retrieval and Recognition System')

#UPLOAD IMAGE SECTIONS
upload_file = st.file_uploader('Please upload an Image file', type=["jpg", "jpeg", "png","jfif"])

if upload_file is not None:
    #PRINT SUCCESS MESSAGE
    st.success("Image file uploaded")
    #DISPLAY UPLOADED IMAGE
    st.image(upload_file, caption='Uploaded Image', use_column_width=True)   

    #CONVERT UPLOADED IMAGE TO OPENCV IMAGE FORMAT
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    #CONVERT TO RGB
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)


    #############################################################
    #CROP IMAGE
    #############################################################
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
            pil_img = Image.fromarray(img)
            return pil_img.crop((x1, y1, x2, y2))

    # Create the cropped image
    cropped_image = crop_image(opencv_image, x1, y1, x2, y2)

    # Show the image
    if cropped_image:
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)

    #############################################################
    #NORMALIZE IMAGE
    #############################################################
    cropped_image = np.asarray(cropped_image, dtype=np.uint8)

    # Compute the mean and standard deviation of the image
    mean, std = cv2.meanStdDev(cropped_image)

    # Create a copy of the image
    normalized_image = cropped_image.copy()

    # Normalize the image
    cv2.normalize(cropped_image, normalized_image, mean[0][0], std[0][0], cv2.NORM_MINMAX)

    # Show the image
    st.image(normalized_image, caption="Normalized Image", use_column_width=True)

    #############################################################
    #SHARPEN IMAGE
    #############################################################
    # Blur the image
    blurred = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Compute the difference image
    difference = normalized_image - blurred
    strength = 2

    # Add the difference image to the original image
    sharpened = normalized_image + strength * difference

    # Show the image
    st.image(sharpened, caption="Sharpened Image", use_column_width=True)

    #############################################################
    #REMOVE BACKGROUND IMAGE
    #############################################################
    # let the user select threshold value
    threshold_value = st.slider("Select Threshold Value", 0, 255, 120)

    # perform gaussianBlur
    img_blur = cv2.GaussianBlur(sharpened, (5, 5), 0)

    # convert colorspace
    grayImage = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # thresholding
    ret, thres = cv2.threshold(grayImage, threshold_value, 255, cv2.THRESH_BINARY)

    #Perform bitwise_and operation between the original image and the thresholded image
    result = cv2.bitwise_and(cropped_image, cropped_image, mask=thres)
    st.image(result, caption="Background removed image", use_column_width=True)

    #############################################################
    #EDGE
    #############################################################
    edges = cv2.Canny(thres, 50, 150)
    st.image(edges, caption='Edged Image', use_column_width=True)

    if st.button("Search Image"):
        #READ IMAGE DATA FROM PARQUET FILES
        imagedataset = pd.read_parquet('data.parquet')

        # FASTER WAY (around 3 secs)
        img = cv2.resize(cropped_image, (0,0), fx=0.4, fy=0.4)
        query_Image = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        best_matches_color = []

        # Define a function to run in parallel
        def find_best_match(i):
            # Decode the base64 string to a NumPy array
            image_array = np.frombuffer(base64.b64decode(imagedataset.loc[i]['image']), np.uint8)

            # Convert the NumPy array back to an image
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            histIMG = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            distance = cv2.compareHist(query_Image, histIMG, cv2.HISTCMP_BHATTACHARYYA)
            best_matches_color.append((imagedataset.loc[i]['filename'], distance))

        # Use the concurrent.futures library to run the function in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(find_best_match, range(imagedataset.shape[0]))

        best_matches_color = sorted(best_matches_color, key=lambda x: x[1])
    
        colourFilter = [item[0] for item in best_matches_color]
        colourFilter = colourFilter[:500]

        colourDataset = imagedataset[imagedataset.filename.isin(colourFilter)]

        colourDataset = colourDataset.reset_index(drop=True)
        
        def closest_pair(des1, des2, top_k=5):  
            # Match SIFT descriptors
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Get the top k matches
            top_k_matches = matches[:top_k]
            
            # Get the distances of the top k matches
            top_k_distances = [match.distance for match in top_k_matches]
            
            # Sum up the distances of the top k pairs
            score = sum(top_k_distances)
            
            # Return the score
            return score

        # FASTER WAY (around 3 secs)
        sift = cv2.xfeatures2d.SIFT_create()
        bf = cv2.BFMatcher()

        grayimg = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        # Detect keypoints and descriptors in the image
        kp, des = sift.detectAndCompute(grayimg, None)

        best_matches_sift = []

        # Define a function to run in parallel
        def find_best_match_sift(i):
            # Decode the base64 string to a NumPy array
            image_array = np.frombuffer(base64.b64decode(colourDataset.loc[i]['image']), np.uint8)

            # Convert the NumPy array back to an image
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            grayimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and descriptors in the image
            kp, des1 = sift.detectAndCompute(grayimg, None)
            
            if kp !=():
                score = closest_pair(des,des1)
                best_matches_sift.append((colourDataset.loc[i]['filename'], score))

        # Use the concurrent.futures library to run the function in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(find_best_match_sift, range(colourDataset.shape[0]))

        best_matches_sift = sorted(best_matches_sift, key=lambda x: x[1])

        for i in range(10):
            st.text(best_matches_sift[i])


