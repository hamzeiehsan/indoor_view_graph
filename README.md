# Indoor View Graph: A Model to Capture Route and Configurational Information

## Abstract
This paper presents a graph model to simultaneously store route and configurational information about indoor spaces. Existing models either capture route information to compute shortest paths and generate route descriptions or store configurational information to enable spatial querying and inference. Consequently, multiple representations of indoor environments must be stored in computing systems to perform these tasks. In this paper, we propose a *decomposed view graph* which is able to capture both types of information in a unified manner. The view graph is the dual representation of connected lines of sight called *views*. Views can represent continuous movement in an indoor environment based on their connectivity, and at the same time, the visible configurational information from each view can be explicitly captured. In this paper, we discuss the conceptual design of the model and an automatic approach to derive the view graph from floorplans. Finally, we show the capabilities of our model in performing different tasks including calculating shortest paths, generating route descriptions, and deriving place graphs.


## Installation
Here, you can find the information for dependencies and their version. Make sure that the versions are compatible to what is specified below:

|name|version|
|:----|:------|
|python| 3.9.7 |
|geojson| 2.5.0 |
|geopandas| 0.10.2|
|matplotlib| 3.5.0 |
|networkx| 2.6.3 |
|numpy| 1.20.3|
|py2d-fixed| 0.1   |
|pyvis| 0.2.1 |
|scikit-geometry| 0.1.2 |
|shapely| 1.8.2 |
|visilibity| 1.0.10|

## Test cases
All can be found in the 'envs' folder, which includes geojson files that represent the containers and doors:
1. Real-world floorplan: mc-floor-5
2. Hypothetical floorplans: basic and hypo environment


## Running the code
You can either try running and modifying the jupyter notebooks or the main.py

```commandline
python main.py
```


```commandline
jupyter notebook
```
