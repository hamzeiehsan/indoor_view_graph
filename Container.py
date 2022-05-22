from Utility import Utility

class Container:
    def __init__(self, address, polygon_file, holes_file, doors_file, dpoints_file, landmarks_file):
        print('reading GeoJSON files (boundary, holes, doors and decision points)')
        bound = Utility.read_geojson(address, polygon_file)['features'][0]
        self.boundary = bound['geometry']['coordinates'][0][0]
        self.name = 'Container'
        self.name_id = 'Container'
        if 'properties' in bound.keys() and 'full_name' in bound['properties'].keys():
            self.name = bound['properties']['full_name']
            self.name_id = bound['properties']['name']
        if holes_file is not None:
            self.holes = Utility.read_geojson(address, holes_file)['features']
        else:
            self.holes = []
        self.doors = Utility.read_geojson(address, doors_file)['features']
        self.door_names = []
        for d in self.doors:
            if 'properties' in d.keys() and 'container1' in d['properties'].keys():
                if d['properties']['container1'] != self.name_id:
                    self.door_names.append('the door to {}'.format(d['properties']['container1']))
                else:
                    self.door_names.append('the door to {}'.format(d['properties']['container2']))
            else:
                self.door_names.append('door {}'.format(self.doors.index(d)))
        if dpoints_file is not None:
            self.dpoints = Utility.read_geojson(address, dpoints_file)['features']
            for dp in self.dpoints:
                self.door_names.append('decision point {}'.format(self.dpoints.index(dp)))
        else:
            self.dpoints = []
        self.landmark_names = []
        if landmarks_file is not None:
            self.landmarks = Utility.read_geojson(address, landmarks_file)['features']
            for l in self.landmarks:
                if 'properties' in l.keys() and 'name' in l['properties'].keys():
                    self.landmark_names.append(l['properties']['name'])
                else:
                    self.landmark_names.append('landmark {}'.format(self.landmarks.index(l)))
        else:
            self.landmarks = []