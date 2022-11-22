from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


data = pd.read_csv('data.csv')
data["point"] = data["lat"] + "," + data["lon"]

st.write('Longitude Latitude:', data['point'])

option = st.selectbox(
    'Select A Location?',
    ('Email', 'Home phone', 'Mobile phone'))



st.map(data)


