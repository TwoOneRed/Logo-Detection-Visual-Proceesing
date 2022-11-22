from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

data = pd.read_csv('data.csv')

st.text(data['SmokerStatus'].unique())

st.map(data)
