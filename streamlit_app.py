from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import collections
import numpy as np
import math

#Clean Data
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

#Distance
def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = math.radians(lat1)
   phi2 = math.radians(lat2)
   delta_phi = math.radians(lat2 - lat1)
   delta_lambda = math.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
   return np.round(res, 2)


######################################################################
#TOP 5 RETAILS
######################################################################
st.title('TOP 5 RETAILS')

mmu = pd.read_csv('MMU.csv')
komtar = pd.read_csv('Komtar.csv')
bbotanic =pd.read_csv('BBotanic.csv')

location1 = clean(mmu)
location2 = clean(komtar)
location3 = clean(bbotanic)

shops = location1 + location2 + location3

shops_count = collections.Counter(shops)
shops_count_sorted = sorted(shops_count.items(), key=lambda x:x[1], reverse=True)

st.text("Top 1 = " + shops_count_sorted[0][0] + ' ('+ str(shops_count_sorted[0][1]) + ')')
st.text("Top 2 = " + shops_count_sorted[1][0] + ' ('+ str(shops_count_sorted[1][1]) + ')')
st.text("Top 3 = " + shops_count_sorted[2][0] + ' ('+ str(shops_count_sorted[2][1]) + ')')
st.text("Top 4 = " + shops_count_sorted[3][0] + ' ('+ str(shops_count_sorted[3][1]) + ')')
st.text("Top 5 = " + shops_count_sorted[4][0] + ' ('+ str(shops_count_sorted[4][1]) + ')')

######################################################################
#SELECT POINT FROM MAP
######################################################################
st.title('POINT FROM MAP')
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

######################################################################
#COUNT DISTANCE
######################################################################
distances_km = []
for row in data.itertuples(index=False):
    distances_km.append(haversine_distance(float(loc[0]), float(loc[1]), float(row.lat), float(row.lon)))
    
data['distance'] = distances_km

twokmdata = data[data['distance'] <= 2]

st.text("Average " + twokmdata['FamilyExpenses_monthly'].mean())
st.text("Occupation " + twokmdata['Occupation'].unique())
                      
