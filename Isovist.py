import visilibity as vis
from shapely.geometry import shape, MultiPolygon, MultiLineString, GeometryCollection

from Parameters import Parameters
from Utility import Utility


class Isovist:
    def __init__(self, boundary, holes, doors, dpoints=[], landmarks=[]):
        # boundary
        self.space_poly = Utility.reformat_polygon(boundary, is_hole=False)
        self.space_x, self.space_y = Utility.save_print_geojson(boundary)
        self.space_shp = Utility.to_polygon_shape(Utility.to_polygon_geojson(self.space_x, self.space_y),
                                                  None, None)

        # holes
        self.holes = holes
        self.holes_list = []
        self.holes_shape = []
        self.holes_x = []
        self.holes_y = []
        for h in holes:
            if Utility.is_poly_cw(h['geometry']['coordinates'][0][0]):
                self.holes_list.append(Utility.reformat_polygon(h['geometry']['coordinates'][0][0], True))
            else:
                self.holes_list.append(Utility.reformat_polygon(h['geometry']['coordinates'][0][0], False))
            h_x, h_y = Utility.save_print_geojson(h['geometry']['coordinates'][0][0])
            self.holes_x.append(h_x)
            self.holes_y.append(h_y)
            self.holes_shape.append(shape(h['geometry']).geoms[0])

        self.env = Utility.create_env(self.space_poly, self.holes_list)
        print('Container environment is valid: {}'.format(self.env.is_valid()))

        self.door_points = []
        for d in doors:
            self.door_points.append(Utility.reformat_point(d['geometry']['coordinates']))
        self.door_idx = len(self.door_points)
        for d in dpoints:
            self.door_points.append(Utility.reformat_point(d['geometry']['coordinates']))
        self.landmarks_points = []
        for l in landmarks:
            self.landmarks_points.append(Utility.reformat_point(l['geometry']['coordinates']))

        self.isovists = []
        self.door_props = {}
        idx = 0
        for door in self.door_points:
            door.snap_to_boundary_of(self.env, Parameters.epsilon)
            door.snap_to_vertices_of(self.env, Parameters.epsilon)
            isovist = vis.Visibility_Polygon(door, self.env, Parameters.epsilon)
            self.isovists.append(isovist)
            self.door_props[idx] = {'id': idx, 'y': door.y(), 'x': door.x()}
            if idx < self.door_idx:
                self.door_props[idx]['type'] = 'gateway'
            else:
                self.door_props[idx]['type'] = 'dt'
            idx += 1

        self.isovists_x_y = []
        self.shapes = {}
        idx = 0
        test_case = None
        test_idx = None
        for isovist in self.isovists:
            iso_x, iso_y = Utility.save_print(isovist)
            geojson_polygon = Utility.to_polygon_geojson(iso_x, iso_y)
            shp = Utility.to_polygon_shape(geojson_polygon, self.space_shp, self.holes, clip=True)
            if not shp.is_valid:
                print('invalid')
            if isinstance(shp, MultiPolygon):
                test_case = shp
                shps = list(shp)
                areas = [s.area for s in shps]
                max_area = max(areas)
                idxs = areas.index(max_area)
                shp = shps[idxs]
                test_idx = idx
            self.shapes[idx] = {'shape': shp, 'props': self.door_props[idx]}

            self.isovists_x_y.append([iso_x, iso_y])
            idx += 1

    def isovist_calc(self, x, y):
        door = vis.Point(x, y)
        door.snap_to_boundary_of(self.env, Parameters.epsilon)
        door.snap_to_vertices_of(self.env, Parameters.epsilon)
        isovist = vis.Visibility_Polygon(door, self.env, Parameters.epsilon)
        iso_x, iso_y = Utility.save_print(isovist)
        geojson_polygon = Utility.to_polygon_geojson(iso_x, iso_y)
        shp = Utility.to_polygon_shape(geojson_polygon, self.space_shp, self.holes, clip=True)
        if not shp.is_valid:
            print('invalid')
        if isinstance(shp, MultiPolygon):
            test_case = shp
            shps = list(shp)
            areas = [s.area for s in shps]
            max_area = max(areas)
            idxs = areas.index(max_area)
            shp = shps[idxs]
        return shp

    def view_intersects_holes(self, view_ls):
        for hole in self.holes_shape:
            if hole.intersects(view_ls):
                return True
        return False

    def view_intersects_boundary(self, view_ls):
        if self.space_shp.intersects(view_ls):
            if self.space_shp.contains(view_ls):
                return False
            elif self.space_shp.touches(view_ls):
                intersection = self.space_shp.intersection(view_ls)
                if isinstance(intersection, MultiLineString) or isinstance(intersection, GeometryCollection):
                    return True
                return False
            return True
        return False
