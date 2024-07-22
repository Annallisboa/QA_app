import streamlit as st
from agent import Agent
import folium 
import pandas as pd

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, MeasureControl
import numpy as np
import json

agent = Agent("sk-proj-VbB6ErYVhSuNoBhdXRpiT3BlbkFJal67RcYPlxXXMjQj24R8")

st.set_page_config(layout='wide')
st.title('Q&A sobre como usar um smartphone')

def initialize_session_state():
    if "center" not in st.session_state:
        st.session_state.center = [48.9, 2.4]
    if "zoom" not in st.session_state:
        st.session_state.zoom = 10
    if "marker" not in st.session_state:
        st.session_state.marker = []
        
# def initialize_map(center, zoom):
#     if "map" not in st.session_state or st.session_state.map is None:
#         st.session_state.center = center
#         st.session_state.zoom = zoom
#         folium_map = folium.Map(location=st.session_state.center,
#                                 zoom_start=st.session_state.zoom)
#         st.session_state.map = folium_map
#     return st.session_state.map

def reset_session_state():
    for key in st.session_state.keys():
        if key in ["center", "zoom"]:
            continue
        del st.session_state[key]
    initialize_session_state()  

st.write("""
    Esse aplicativo ajuda pessoas mais velhas a usar um **smartphone android**!         
""")

initialize_session_state()
col1, col2 = st.columns(2)

with col1:
    request = st.text_area("Qual a sua dúvida?")
    button = st.button("Responder pergunta")
    box = st.container(height=300)
    with box:
        container = st.empty()
        container.header("Resposta")
if button and request:
    reset_session_state()
    itinerary = agent.get_itinerary(request)
    try:
        container.write(itinerary['agent_suggestion'])
    except KeyError:
        container.write("Desculpe, não entendi sua pergunta")
        
    try:
        points_coordinates = []
        days = json.loads(itinerary['coordinates'])
        for day in days.values():
            for item in day:
                locations = item["locations"]
                for loc in locations:
                    points_coordinates.append((loc['lat'], loc['lon']))
        st.session_state["marker"] = [folium.Marker(location=point)
                                      for point in points_coordinates]
    except KeyError:
        pass

    center_info = json.loads(itinerary['center_info'])
    center = center_info['center']
    zoom = center_info['zoom']
    st.session_state.center = center
    st.session_state.zoom = zoom

# with col2:
#     folium_map = initialize_map(st.session_state.center, st.session_state.zoom)
#     fg = folium.FeatureGroup(name="Markers")
#     for marker in st.session_state["marker"]:
#         fg.add_child(marker)
#     fg.add_to(folium_map)
#     folium_static(folium_map)