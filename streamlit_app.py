from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import collections

#SELECT POINT FROM MAP
data = pd.read_csv('data.csv')

data['lat'] = data['lat'].astype(str)
data['lon'] = data['lon'].astype(str)

data["point"] = data['lat'] +', ' + data['lon']

option = st.selectbox(
    'Select A Location?',
    (data['point']))


loc = [float(idx) for idx in option.split(', ')]

st.text(loc[0])
st.text(loc[1])

df = pd.DataFrame(
    {'lat' : [loc[0]] , 'lon': loc[1] } )

st.map(df)

#TOP 5 RETAILS
mmu = pd.read_csv('MMU.csv')
komtar = pd.read_csv('Komtar.csv')
bbotanic =pd.read_csv('BBotanic.csv')

def clean(data):
    cleandata = data['name'].values.tolist()
    
    retailFilterList = ['a&w','boat noodle','burger king', 'che go','d laksa','dragon-i', 'go noodle','i love yoo!', 
                        'kfc','kim gary','nando','pop meals','texas', 'old town','myburgerlab', 'oldtown']
    
    for i in range(len(cleandata)):
        for b in retailFilterList:
            if (b.lower() in cleandata[i].lower()):
                if 'oldtown' in cleandata[i].lower():
                    cleandata[i] = 'old town'
                else:
                    cleandata[i] = b
                
    return cleandata


location1 = clean(mmu)
location2 = clean(komtar)
location3 = clean(bbotanic)

shops = location1 + location2 + location3

shops_count = collections.Counter(shops)
shops_count_sorted = sorted(shops_count.items(), key=lambda x:x[1], reverse=True)

print("Top 1 = ", shops_count_sorted[0][0])
print("Top 2 = ", shops_count_sorted[1][0])
print("Top 3 = ", shops_count_sorted[2][0])
print("Top 4 = ", shops_count_sorted[3][0])
print("Top 5 = ", shops_count_sorted[4][0])
