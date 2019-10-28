## Feeling the data!
Implementation of Low Rank Matrix Completion by Proximal Gradient,
and data visualization using t-distributed Stochastic Neighbor Embedding.
There is an example of matrix completion where YaleB-Dataset (of faces) is used.
Visualization examples use "Near Earth Comets" and "Meteorite_Landings" datasets,
which were provided by NASA as example resources on SpaceApps challenge "Chasers of the lost Data".


## Dependencies

* numpy
* pandas
* sklearn
* plotly
* matplotlib

## Demo


The implemented function is in
- lrmc.py
and uses auxiliary functions that are in
- aux_functions.py

The script that runs the function and stablish the parameters is
- lrmc_test.py  

The script that generates the output images is
- vizualization.py

This can also be ran from the interactive notebook called "FEELING DATA.ipynb"

The code will use YaleB-Dataset.zip uncompressed in this directory.


