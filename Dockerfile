FROM continuumio/miniconda3

COPY . /viewgraph
WORKDIR /viewgraph

RUN conda create -n viewgraph python=3.9.7 jupyter notebook

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "viewgraph", "/bin/bash", "-c"]

RUN conda config --add channels conda-forge
RUN conda install -c conda-forge geopandas scikit-geometry jpype1
RUN pip install geojson networkx py2d-fixed pyvis VisiLibity
EXPOSE 8888
CMD jupyter notebook --port=8888