# -*- coding: utf-8 -*-
"""
Script that that uses T-distributed stocastic neighbour embedding to create a visual representation of some data points of the Nasa Data set :Near Earth Comets and Orbital Elements.
author: Feeling the data
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, MDS
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
%matplotlib inline

file = r"C:\Users\JuanElenter\Desktop\docs\feelingdata\Near-Earth_Comets_-_Orbital_Elements.csv"
data = pd.read_csv(file)
data = data.drop(labels=['ref', 'Object', 'Object_name'], axis='columns')

scaler = MinMaxScaler() 
data = scaler.fit_transform(data)

data_arr = np.array(data)

n = 10
nn = 4
valid_rows = (2,3,4,5,7,8,9,10,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,94,96,99,100,105,108,109,113,119,120,132,157)
#known_mask = ~np.isnan(data.iloc[:, n])

X_t = TSNE(n_components=3).fit_transform(data_arr[:, 0:-nn])
data_t = pd.DataFrame(np.concatenate((X_t,
                                      np.expand_dims(data_arr[:, -nn+1],
                                                     axis=1)), axis=1)[valid_rows, :], 
                     columns=['d1', 'd2', 'd3', 'missing'])



x = data_t[['missing']].values.astype(float)
scaler_t = MinMaxScaler((1,30))
x_s = scaler_t.fit_transform(x)
data_t["missing"] = x_s

data_t = data_t.fillna(30)

fig = px.scatter_3d(data_t, x='d3', y='d2', z='d1', size= "missing",
              color='missing', color_continuous_scale='geyser', template = "plotly_dark")
fig.show()
