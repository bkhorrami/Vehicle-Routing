__author__ = 'babak_khorrami'

import numpy as np
import pandas as pd


class Trip(object):
    """
    Trip class is used to record (1,2 or 3) - customer trips of each shopper
    """
    trip_count = 0

    def __init__(self,shopper_id=0,nodes=None,arrv_time=None,violations=None):
        """

        :param shopper_id: shopper_id of person doing the trip
        :param nodes: list of nodes/CUSTOMERS in the trip
        :param arrv_time: list of delivery times for the customers/nodes and arrival time at store
        :param violations: list of violations from the due_times
        :return: Trip Object
        """

        if nodes == None:
            nodes = []
        else:
            if not isinstance(nodes,list):
                nodes = list(nodes)
        if arrv_time == None:
            arrv_time = []
        else:
            if not isinstance(arrv_time,list):
                arrv_time = list(arrv_time)

        if violations==None:
            violations = []
        else:
            if not isinstance(violations,list):
                violations = list(violations)

        assert(len(nodes)==len(arrv_time))
        assert(len(nodes)==len(violations))
        self.customers = nodes
        self.visit_time = arrv_time
        self.violations = violations
        self.shopper_id = shopper_id
        self.total_violation = sum([abs(v) for v in self.violations])
        self.trip_id = 0 # this attribute is for presentation purpose and is set at run-time
        Trip.trip_count += 1

    @property
    def trip_id(self):
        return self.trip_id

    @trip_id.setter
    def trip_id(self, trip_id):
        self.trip_id = trip_id

    @property
    def customers(self):
        return self.customers

    @property
    def visit_time(self):
        return self.visit_time

    @property
    def violations(self):
        return self.violations

    @property
    def total_violations(self):
        return self.total_violation

    @property
    def shopper_id(self):
        return self.shopper_id

    @customers.setter
    def customers(self, nodes):
        self.customers = nodes

    @visit_time.setter
    def visit_time(self, arrv_time):
        if isinstance(arrv_time, list):
            self.visit_time = arrv_time
        else:
            self.visit_time = list(arrv_time)

    @violations.setter
    def violations(self, violations):
        self.violations = violations

    @shopper_id.setter
    def shopper_id(self, shopper_id):
        self.shopper_id = shopper_id

    @trip_id.setter
    def trip_id(self, trip_id):
        self.trip_id = 0 # this attribute is for presentation purpose and is set at run-time

    def get_start_time(self):
        return self.visit_time[0]

    def get_end_time(self):
        return self.visit_time[-1]

    def get_store(self):
        return self.customers[0]

    @classmethod
    def get_trip_count(cls):
        return cls.trip_count

    def print_trip(self,st_time):
        """
        :param st_time: start time of the experiment
        :return: prints the trips on the screen
        """
        if len(self.customers) == 0 or len(self.visit_time)==0:
            print("Trip is Empty")
            return None

        start_time = pd.to_datetime(self.get_start_time() * np.timedelta64(1,'s') * 60.0 + np.datetime64(st_time))
        end_time = pd.to_datetime(self.get_end_time() * np.timedelta64(1,'s') * 60.0 + np.datetime64(st_time))

        for i in range(1,len(self.customers)):
            visit_time = pd.to_datetime(self.visit_time[i] * np.timedelta64(1,'s') * 60.0 + np.datetime64(st_time))
            print(self.customers[i],",",self.trip_id,",",self.shopper_id,",",start_time,",",end_time,",",
                  int(self.customers[0]),",",visit_time)

    def trip_string(self,st_time):
        """
        :param st_time: start time of the experiment
        :return: a list of strings describing the trip for printing in a file
        """
        if len(self.customers) == 0 or len(self.visit_time)==0:
            print("Trip is Empty")
            return None

        trip_string_list=[]
        start_time = pd.to_datetime(self.get_start_time() * np.timedelta64(1,'s') * 60.0 + np.datetime64(st_time))
        end_time = pd.to_datetime(self.get_end_time() * np.timedelta64(1,'s') * 60.0 + np.datetime64(st_time))

        for i in range(1,len(self.customers)):
            visit_time = pd.to_datetime(self.visit_time[i] * np.timedelta64(1,'s') * 60.0 + np.datetime64(st_time))
            t_str="{},{},{},{},{},{},{}".format(self.customers[i],self.trip_id,self.shopper_id,start_time,end_time,
                  int(self.customers[0]),visit_time)
            trip_string_list.append(t_str)

        return trip_string_list

    def nodes_to_switch(self,other):
        """
        :param other: the other Trip
        :return: The index(s) of the corresponding nodes of two trips that can be exchanged
        """
        self_visits = self.get_visit_time()
        other_visits = other.get_visit_time()
        lst1 = list(np.array(self_visits[0:len(self_visits) - 1]) - np.array(other_visits[1:len(other_visits)]))
        lst2 = list(np.array(other_visits[0:len(other_visits) - 1]) - np.array(self_visits[1:len(self_visits)]))
        ind1 = np.where(np.array(lst1))
        ind2 = np.where(np.array(lst2))
        intersect_idx = np.intersect1d(ind1,ind2)
        if len(intersect_idx) > 0:
            return intersect_idx + 1
        else:
            return None
        
    def customer_switch(self,other):
        pass

    def __lt__(self, other):
        return self.total_violation < other.total_violation

    def __gt__(self, other):
        return self.total_violation > other.total_violation

    def __le__(self, other):
        return self.total_violation <= other.total_violation

    def __ge__(self, other):
        return self.total_violation >= other.total_violation



