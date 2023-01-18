#IMPORT LIBRARY
from collections import namedtuple
import altair as alt
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import math
import collections

st.Title('Logo Retrieval and Recognition System')
upload_file = st.file_uploader('Please upload an Image file', type=["jpg", "jpeg", "png", "gif", "jfif"])

if upload_file is not None:
   st.success("Image file uploaded")

   # Read the file to a dataframe using cv2
   img = cv2.imread('upload_file', cv2.IMREAD_GRAYSCALE)
   st.image(img, caption='Uploaded Image', use_column_width=True)                
