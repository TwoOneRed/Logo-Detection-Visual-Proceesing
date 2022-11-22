from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


data = pd.read_csv('data.csv')
data["point"] = str(data["lat"]) + "," + str(data["lon"])


st.text(data['point'])

option = st.selectbox(
    'Select A Location?',
    ('Email', 'Home phone', 'Mobile phone'))



st.map(data)


