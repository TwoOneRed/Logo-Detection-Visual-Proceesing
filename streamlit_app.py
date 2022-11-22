from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


data = pd.read_csv('data.csv')

data['lat'] = data['lat'].astype(str)
data['lon'] = data['lon'].astype(str)

data["point"] = data['lat'] +', ' + data['lon']

st.text(data['point'])

option = st.selectbox(
    'Select A Location?',
    (data['point']))

loc = option.split(',')

st.text(loc[0], loc[1])

st.map(loc[0], loc[1])
