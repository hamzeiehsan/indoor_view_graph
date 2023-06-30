FROM conda/miniconda3
COPY . /viewgraph
WORKDIR /viewgraph
RUN conda update conda
RUN conda install python=3.9.6
RUN conda config --add channels conda-forge
RUN conda install -c conda-forge scikit-geometry
RUN conda install geopandas
RUN conda install -c conda-forge jpype1
RUN conda install jupyter pandas numpy
RUN pip install SALib seaborn
RUN pip install geojson networkx py2d-fixed pyvis
RUN pip install VisiLibity
EXPOSE 8888
CMD jupyter notebook --port=8888