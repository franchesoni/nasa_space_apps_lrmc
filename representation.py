# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, MDS
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler

file = '/Near-Earth_Comets_-_Orbital_Elements.csv'
data = pd.read_csv(file)
data = data.drop(labels=['ref', 'Object', 'Object_name'], axis='columns')

scaler = MinMaxScaler() 
data = scaler.fit_transform(data)

data_arr = np.array(data)

n = 10
nn = 4
n_frame_start = 2
n_frame_end = 31
#known_mask = ~np.isnan(data.iloc[:, n])

X_t = TSNE(n_components=3).fit_transform(data_arr[:, 0:-nn])
data_t = pd.DataFrame(np.concatenate((X_t,
                                      np.expand_dims(data_arr[:, -nn+1],
                                                     axis=1)), axis=1)[n_frame_start:n_frame_end, :], 
                     columns=['d1', 'd2', 'd3', 'missing'])

data_t = data_t.fillna(0)
fig = px.scatter_3d(data_t, x='d1', y='d2', z='d3',
              color='missing', color_continuous_scale='geyser')
fig.show()


