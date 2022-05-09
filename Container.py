from Utility import Utility

class Container:
    def __init__(self, address, polygon_file, holes_file, doors_file, dpoints_file, landmarks_file):
        print('reading GeoJSON files (boundary, holes, doors and decision points)')
        self.boundary = Utility.read_geojson(address, polygon_file)['features'][0]['geometry']['coordinates'][0][0]
        if holes_file is not None:
            self.holes = Utility.read_geojson(address, holes_file)['features']
        else:
            self.holes = []
        self.doors = Utility.read_geojson(address, doors_file)['features']
        if dpoints_file is not None:
            self.dpoints = Utility.read_geojson(address, dpoints_file)['features']
        else:
            self.dpoints = []
        if landmarks_file is not None:
            self.landmarks = Utility.read_geojson(address, landmarks_file)['features']
        else:
            self.landmarks = []