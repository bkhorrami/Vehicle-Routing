__author__ = 'babak_khorrami'

import numpy as np


class Graph(object):
    def __init__(self, nodes=None, edges=None):
        """
        nodes : Nodes of Graph
        edges : Edges of Graph
        """
        self.nodes = nodes
        self.edges = edges

    @property
    def nodes(self):
        """
        returns nodes
        """
        return self.nodes

    @property
    def edges(self):
        """
        returns edges
        """
        return self.edges

    @nodes.setter
    def nodes(self, nodes):
        self.nodes = nodes

    @edges.setter
    def edges(self,edges):
        self.edges = edges

    def find_trip_time(self,trip):
        trip = np.array(trip)
        idx=np.nonzero(np.all(self.edges[:, 0:2] == trip[:, np.newaxis], axis=2))[1]
        return self.edges[idx, 1:]