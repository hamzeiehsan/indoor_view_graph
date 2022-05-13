import math

import geojson
import visilibity as vis
from geojson import Polygon
import numpy as np
from numpy import arctan2, sin, cos, degrees
from shapely.geometry import shape, Point
import skgeom as sg


class Utility:
    @staticmethod
    def read_geojson(address, file):
        with open(address + file, encoding='utf-8') as fp:
            file = geojson.load(fp)
        return file

    @staticmethod
    def reformat_point(record):
        return vis.Point(record[0], record[1])

    @staticmethod
    def signed_area(pr2):
        xs, ys = map(list, zip(*pr2))
        xs.append(xs[1])
        ys.append(ys[1])
        return sum(xs[i] * (ys[i + 1] - ys[i - 1]) for i in range(1, len(pr2))) / 2.0

    @staticmethod
    def reformat_polygon(records_original, is_hole=True):
        points = []
        records_rev = records_original.copy()
        if records_rev[-1] == records_rev[0]:
            records_rev.pop()

        if is_hole:
            records = records_rev
        else:
            records = list(reversed(records_rev))
        for record in records:
            points.append(Utility.reformat_point(record))
        return vis.Polygon(points)

    @staticmethod
    def create_env(space_polygon, holes_polygons):
        env_list = [space_polygon]
        for hole in holes_polygons:
            env_list.append(hole)
        return vis.Environment(env_list)

    @staticmethod
    def save_print(polygon):
        end_pos_x = []
        end_pos_y = []
        for i in range(polygon.n()):
            x = polygon[i].x()
            y = polygon[i].y()

            end_pos_x.append(x)
            end_pos_y.append(y)
        return end_pos_x, end_pos_y

    @staticmethod
    def save_print_geojson(polygon):
        x = []
        y = []
        for p_x_y in polygon:
            x.append(p_x_y[0])
            y.append(p_x_y[1])
        return x, y

    @staticmethod
    def to_polygon_geojson(x_list, y_list):
        formatted_list = [(x_list[i], y_list[i]) for i in range(len(x_list))]
        return Polygon([formatted_list])

    @staticmethod
    def to_polygon_shape(polygon_geojson, space_shp, holes, clip=False):
        shp = shape(polygon_geojson)
        if not shp.is_valid:
            print('not valid: {}'.format(polygon_geojson))
        elif clip:
            shp = shp.intersection(space_shp)
            for hole in holes:
                shp = shp.difference(shape(hole['geometry']))
        return shp

    @staticmethod
    def calculate_distance(d1, d2):
        return math.sqrt(math.pow(d1.x - d2.x, 2)+math.pow(d1.y-d2.y, 2))*10000

    @staticmethod
    def calculate_bearing(v):
        lat1 = v[0].y
        lat2 = v[1].y
        lon1 = v[0].x
        lon2 = v[1].x
        dL = lon2 - lon1
        X = cos(lat2) * sin(dL)
        Y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
        return (degrees(arctan2(X, Y)) + 360) % 360

    @staticmethod
    def calculate_coordinates(v, angle, d):
        bearing = Utility.calculate_bearing(v)
        nbearing = bearing + angle
        x = v[0].x + d * sin(np.deg2rad(nbearing))
        y = v[1].y + d * cos(np.deg2rad(nbearing))
        return Point(x, y)

    @staticmethod
    def slope(x1, y1, x2, y2):  # Line slope given two points:
        return (y2 - y1) / (x2 - x1)

    @staticmethod
    def angle(s1, s2):
        return math.degrees(math.atan((s2 - s1) / (1 + (s2 * s1))))

    @staticmethod
    def calculate_angle(p1, p2, p3):
        slope1 = Utility.slope(p1[0], p1[1], p2[0], p2[1])
        slope2 = Utility.slope(p1[0], p1[1], p3[0], p3[1])
        return Utility.angle(slope1, slope2)

    @staticmethod
    def print_instructions(instructions):
        print('************Verbal Description**************')
        for instruction in instructions:
            print('\t{}'.format(instruction))
        print('********************END*********************\n')

    @staticmethod
    def generate_sg_polygon(bound, holes):
        bound_coordinates = bound['features'][0]['geometry']['coordinates'][0][0]
        bound_coordinates.reverse()
        poly = sg.Polygon([sg.Point2(k[0], k[1]) for k in bound_coordinates[:-1]])
        holes_polys = []
        for h in holes:
            h_coordinates = h['geometry']['coordinates'][0][0]
            # h_coordinates.reverse()
            h_poly = sg.Polygon([sg.Point2(k[0], k[1]) for k in h_coordinates[:-1]])
            holes_polys.append(h_poly)
        polygon = sg.PolygonWithHoles(poly, holes_polys)
        return polygon

    @staticmethod
    def generate_skeleton(bound, holes):
        polygon = Utility.generate_sg_polygon(bound, holes)
        return sg.skeleton.create_interior_straight_skeleton(polygon)