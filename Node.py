__author__ = 'babak_khorrami'

import numpy as np

class Node(object):
    node_count = 0

    def __init__(self, id, lat, lon, type):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.type = type
        Node.node_count += 1

    @property
    def id(self):
        return self.id

    @property
    def lat(self):
        return self.lat

    @property
    def lon(self):
        return self.lon


    @property
    def type(self):
        return self.type

    @id.setter
    def id(self,id):
        self.id = id

    @lat.setter
    def lat(self,lat):
        self.lat = lat


    @lon.setter
    def lon(self,lon):
        self.lon = lon

    @type.setter
    def type(self,type):
        self.type = type

    @classmethod
    def get_node_count(cls):
        return cls.node_count

    def __str__(self):
        return '{} , {} , {} , {}'.format(self.id,self.lat,self.lon,self.type)