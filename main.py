__author__ = 'babak_khorrami'

import heapq
import pandas as pd
import numpy as np
from itertools import combinations
import numpy_indexed as npi
from math import sin, cos, sqrt, atan2, radians, asin
from collections import defaultdict
from itertools import permutations
from itertools import accumulate

from Node import *
from Graph import *
from Shopper import *
# from Test_Run import *
import matplotlib.pyplot as plt


class VRPTW(object):
    """
    This is a class to create an instance of the VRPTW problem and test run the  model/algorithm.
    """
    def __init__(self,start_time,start_store,customerFile,storeFile):
        """
        Initiates an instance of the VRP problem
        :param start_time: time at which the problem instance starts ('2014-03-13 15:00:00')
        :param start_store: store # 49
        :param customerFile: location of the file containing the delivery/customer data
        :param storeFile: location of the file containing the stores data
        :return: creates attributes of the problem instance : (graph/network , customers, stores, nodes)
        """

        # ** Use pandas to read data-sets:
        start_time = pd.to_datetime('2014-03-13 15:00:00')
        try:
            dt = pd.read_table(customerFile,sep='\t')
            stores = pd.read_table(storeFile,sep='\t')
        except FileNotFoundError as e:
            print(e)

        # **** Instantiating Delivery/Store(Node) Objects  ****:
        nodes = []
        # ** Delivery/Customer Nodes:
        for i in range(dt.shape[0]):
            nodes.append(Node(dt.delivery_id[i],dt.latitude[i],dt.longitude[i],1))

        # ** Store Nodes:
        store_list = []  # List of nodes containing stores
        for i in range(stores.shape[0]):
            nodes.append(Node(stores.store_id[i],stores.latitude[i],stores.longitude[i],0))
            store_list.append(Node(stores.store_id[i],stores.latitude[i],stores.longitude[i],0))

        # **** Converting due_dates to pandas datetime
        # **** this code sets the start date/time as ZERO and transforms other due dates as
        # **** minutes elapsed from start-time
        dt['Time'] = pd.to_datetime(dt.due_at)
        tm = dt['Time'] - start_time
        tmd = (tm / np.timedelta64(1,'s'))/60.0
        dt['Time']=tmd

        # ** Customer List and instantiating all the Customer/Delivery objects:
        customers = []
        for i in range(dt.shape[0]):
            customers.append(Customer(dt.delivery_id[i],dt.latitude[i],\
                                      dt.longitude[i],dt.Time[i],dt.items_count[i]))

        # ***** Generating Vertex-List and Edge-List to create Graph object for a problem instance:
        # ***** The underlying graph is a complete-graph having connections between every pair of nodes.
        ver_num=np.r_[np.c_[dt.delivery_id,np.ones((dt.delivery_id.shape[0],1))],\
                np.c_[stores.store_id,np.zeros((stores.store_id.shape[0],1))]]
        ver_lat = np.r_[dt.latitude,stores.latitude]
        ver_lon = np.r_[dt.longitude,stores.longitude]
        vertices = np.c_[ver_num[:,0],ver_lat,ver_lon,ver_num[:,1]]

        ed = np.array(list(combinations(vertices[:,0],2))) #creating connections between every pair

        ind0 = npi.indices(vertices[:,0],ed[:,0])
        ind1 = npi.indices(vertices[:,0],ed[:,1])
        lat0 = vertices[ind0,1]
        lon0 = vertices[ind0,2]
        lat1 = vertices[ind1,1]
        lon1 = vertices[ind1,2]
        tmp1=np.c_[lat0,lon0]
        tmp2=np.c_[lat1,lon1]
        tmp3=np.c_[tmp1,tmp2]
        edge=np.c_[ed,tmp3]
        dist=np.apply_along_axis(VRPTW.haversine,1,edge[:,2:6])
        drive_time = 5*dist
        edges_tmp=np.c_[edge[:,0:2],drive_time]
        edges_tmp_2 = np.c_[edges_tmp[:,1],edges_tmp[:,0],edges_tmp[:,2]]
        edges = np.r_[edges_tmp,edges_tmp_2]

        #clearup the memory:
        del(tmp1,tmp2,tmp3,ind0,ind1,lat0,lat1,lon1,lon0,edge,dist,drive_time,edges_tmp,edges_tmp_2)

        self.graph = Graph(vertices,edges)
        self.customers = customers
        self.stores = store_list
        self.nodes = nodes

    def get_graph(self):
        return self.graph

    def get_nodes(self):
        return self.nodes

    def get_customers(self):
        return self.customers

    def get_stores(self):
        return self.stores

    @staticmethod
    def haversine(points):
        """
        static method using Haversine formula to calculate the great circle (GC) distance
        between two points given (lat,lon)

        :param points: lat,lon of two points.
        :return: GC distance between two points in miles
        """
        p=np.array(points)
        dlat = radians(p[2]-p[0])
        dlon = radians(p[3]-p[1])
        lat1 = radians(p[0])
        lat2 = radians(p[2])
        a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
        c = 2 * asin(sqrt(a))
        miles = 3963 * c
        return miles


def main():
    start_time = '2014-03-13 15:00:00'
    start_store = 49
    customerFile = "/Users/babak_khorrami/Documents/Babak/instacart_challenge/deliveries.csv"
    storeFile = "/Users/babak_khorrami/Documents/Babak/instacart_challenge/stores.csv"

    #*** An instance of the VRPTW problem:
    vrp = VRPTW(start_time,start_store,customerFile,storeFile)

    #*** 10 shoppers :
    shoppers=[]
    for i in range(0,10):
        shoppers.append(Shopper(i,start_store,curr_time=0,trips=None))


    customers = vrp.get_customers()
    stores_list = vrp.get_stores()
    stores = np.array([s.get_id() for s in stores_list])
    graph = vrp.get_graph()

    cust_dt = np.array([[c.get_visit_time()] for c in customers])
    unvisited = list(cust_dt[cust_dt[:]==0])
    while len(unvisited)!=0: #** For all customers
        for i in range(10):  #** For all shoppers
            # r=np.random.uniform(0,1,1)
            # if r < 0.6:
            #     trip_size=3
            # else:
            #     trip_size=2
            top_ten = shoppers[i].generate_trips(customers,stores,graph,trip_size=3,customer_pick='best')
            customers = shoppers[i].trip_assessment(top_ten,customers,graph,stores,trip_size=3)

        cust_dt = np.array([[c.get_visit_time()] for c in customers])
        unvisited = list(cust_dt[cust_dt[:]==0])



    print("-------------------- RESULTS --------------------")
    # for c in customers:
    #     c.print_customer()
    print("delivery_id,trip_id,shopper_id,trip_started_at,trip_ended_at,store_id,delivered_at")
    st_time = '2014-03-13 15:00:00'

    trip_count = 0
    all_trip_list=[]
    for s in shoppers:
        for t in s.get_trips():
            trip_count += 1
            t.set_trip_id(trip_count)
            t.print_trip(pd.to_datetime(st_time))
            curr_trips=t.trip_string(pd.to_datetime(st_time))
            all_trip_list.append(curr_trips)
    all_trips = [y for x in all_trip_list for y in x]
    with open('./results.csv', 'w') as file:
        file.write("delivery_id,trip_id,shopper_id,trip_started_at,trip_ended_at,store_id,delivered_at")
        file.write('\n')
        for line in all_trips:
            file.write(line)
            file.write('\n')
    print("-------------------------------------------------")

    total_violation = [abs(s.get_due_time() - s.get_visit_time()) for s in customers]
    print("Total Violation = ",sum(total_violation))
    print("Total Violations = ",total_violation)
    viols = [(c.get_id(),c.get_shopper_id(),c.get_signed_violation())  for c in customers]
    n_viols = [v for v in viols if v[2]<0 ]
    p_viols = [v for v in viols if v[2]>=0]

    s_p_viol = sorted(p_viols, key=lambda tup: tup[2])
    s_n_viol = sorted(n_viols, key=lambda tup: tup[2])

    s_p_viol_list = [t[2] for t in s_p_viol]
    s_n_viol_list = [t[2] for t in s_n_viol]

    print(s_p_viol_list)
    print(s_n_viol_list)

    plt.hist(total_violation, bins = 20)
    plt.show()


if __name__ == '__main__':
    main()
