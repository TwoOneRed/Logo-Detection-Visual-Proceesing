from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


data = pd.read_csv('data.csv')

data['lat'] = data['lat'].astype(str)
data['lon'] = data['lon'].astype(str)

data["point"] = data['lat'] +',' + data['lon']

option = st.selectbox(
    'Select A Location?',
    (data['point']))


loc = [float(idx) for idx in option.split(',')]

st.text(loc[0])
st.text(loc[1])

df = pd.DataFrame(
    {'lat' : [loc[0]] , 'lon': loc[1] } )

st.map(df)
