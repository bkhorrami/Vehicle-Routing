__author__ = 'babak_khorrami'

import numpy as np
from itertools import combinations
from collections import defaultdict
from itertools import permutations
from itertools import accumulate
from Node import *
from Trip import *
import numpy_indexed as npi
import random
import copy


class Shopper(object):
    """
    Class representing a Shopper attributes and actions
    """
    shopper_count = 0  # No. of Shoppers in the system

    def __init__(self,id,curr_location,curr_time=0,trips=None):
        self.shopper_id = id
        if trips:
            self.trips = []
        else:
            self.trips = trips

        self.current_time = curr_time
        self.current_location = curr_location
        Shopper.shopper_count += 1

    @property
    def shopper_id(self):
        return self.shopper_id

    @shopper_id.setter
    def shopper_id(self, shopper_id):
        self.shopper_id = shopper_id

    @property
    def current_time(self):
        return self.current_time

    @current_time.setter
    def current_time(self, current_time):
        self.current_time = current_time

    @property
    def trips(self):
        return self.trips

    @trips.setter
    def trips(self, trips):
        self.trips = trips

    @property
    def current_location(self):
        return self.current_location

    @current_location.setter
    def current_location(self, current_location):
        self.current_location = current_location

    def drop_last_trip(self):
        #*** ADD CODE HERE
        if len(self.trips) == 0:
            print("No Trips")
            return
        last_trip = self.trips[-1]
        store_loc = last_trip.get_store() #get the starting store of the last trip
        start_time = last_trip.get_start_time() #get the starting time of the last trip
        #*** Drop the last trip and set the shopper's location
        # at the store and push back the time to arriv at same store
        self.trips=self.trips[:-1]
        self.set_current_time(start_time)
        self.set_current_location(store_loc)

    def add_trips(self,trip=None):
        if trip==None:
            self.trips.append(Trip(shopper_id=self.shopper_id))
        else:
            self.trips.append(trip)

    def go_to_store(self,stores,graph,distance='shortest'):
        """
        This method is run at the end of every trip to send the shopper to a nearby store and update
        the location and time of the shopper
        :param stores: List of Stores (np array)
        :param graph: graph object containing vertices and edges of the graph (np arrays)
        :distance: if 'shortest' the shopper is sent to the shortest store otherwise to a random(2nd/3rd shortest)
        :return: None
        """
        curr_loc = self.get_current_location()
        if np.sum(np.isin(stores,curr_loc)) != 0:
            print("Shopper ",self.get_shopper_id(),"is already in a store!")
            return
        e = graph.get_edges() #graph edges

        #edge set from current location to all stores:
        potential_stores = e[(e[:,0]==curr_loc) & (np.isin(e[:,1],stores)),:]

        #*** Allows to either go to the nearest store in terms of distance to one of the top three closest stores:
        if distance == 'shortest':
            closest_store = potential_stores[np.argsort(potential_stores[:,2]),1][0]
            time_to_store = potential_stores[np.argsort(potential_stores[:,2]),2][0]
        elif distance == 'randomize':
            r = list(np.random.randint(1,3,1))
            closest_store = potential_stores[np.argsort(potential_stores[:,2]),1][r[0]]
            time_to_store = potential_stores[np.argsort(potential_stores[:,2]),2][r[0]]

        #update the shopper's location and time after going to store:
        self.set_current_location(closest_store)
        old_time = self.get_current_time()
        self.set_current_time(old_time + time_to_store)

    def three_node_trip(self,curr_store , customers , stores , graph,trip_size = 3):
        """
        ******* Using Shortest Distance Method to build the trips **************
        This function builds a three-customer trip based on a greedy type algorithm.
        The closest 10 customers are selected and the total violations for all the combinations are calculated
        and the best trip is selected.
        :param curr_store: Store in which the shopper is located
        :param customers:  customers (np array)
        :param stores: store list (np array)
        :param graph: graph object containing vertices and edges of the graph (np arrays)
        :return: Updated List of Customers
        """
        curr_store = self.get_current_location()
        stores = np.array(stores)
        visited_customers = Customer.find_visited_customers(customers)
        #**--- If there is no customer to serve -----
        if visited_customers.size == len(customers):
            print("All Customers have been visited")
            return customers
        if visited_customers.size > len(customers) - 3 :
            print("Less than three customers")
            return
        #** -----------------------------------------

        e=graph.get_edges() #graph edges
        potentials=e[(e[:,0] == curr_store) & (~np.isin(e[:,1],stores)) & (~np.isin(e[:,1],visited_customers)),:]
        if potentials.shape[0] >= 10:
            top_ten=potentials[np.argsort(potentials[:,2]),1][0:10] #
        elif potentials.shape[0] < 10 and potentials.shape[0] >= 3:
            top_ten=potentials[np.argsort(potentials[:,2]),1][0:potentials.shape[0]] #
        else:
            return

        all_trp=list(combinations(top_ten,trip_size))
        all_trp2=[list(c) for c in all_trp]
        all_trp2_2 = [list(set(permutations(c))) for c in all_trp2]
        all_trp2_2_1=[y for x in all_trp2_2 for y in x]
        all_trp22=[list(t) for t in all_trp2_2_1]
        all_trp3=[np.array([curr_store]+c) for c in all_trp22]
        if trip_size == 3:
            all_trp4 = list(map(lambda x:np.array([[x[0],x[1]],[x[1],x[2]],[x[2],x[3]]]),all_trp3))
        elif trip_size==2:
            all_trp4 = list(map(lambda x:np.array([[x[0],x[1]],[x[1],x[2]]]),all_trp3))
        else:
            print("Trip Size Should be either 2 or 3 !")
            return
        del(all_trp,all_trp2,all_trp22,all_trp3)
        travel_time = [graph.find_trip_time(c) for c in all_trp4]
        customer_table = defaultdict(int)
        for c in customers:
            customer_table[c.get_id()] = c
        cust_list=[list(c[:,0]) for c in travel_time]
        items=[[customer_table[c].get_item_count() for c in t] for t in cust_list]
        node_due_time=[[customer_table[c].get_due_time() for c in t] for t in cust_list]
        store_time = [sum(c)+5 for c in items]
        trv_time = [list(c[:,1]) for c in travel_time] #travel time to each individul node from the store
        assert(len(store_time) == len(trv_time))
        node_arr_time=[]
        for i in range(len(store_time)):
            node_arr_time.append([store_time[i]+s for s in trv_time[i]])
        cust_arrival_time=[accumulate(s) for s in node_arr_time]
        node_arrival_time = [[self.current_time+s for s in ss] for ss in cust_arrival_time]
        arrv_vec = np.array(node_arrival_time)
        due_vec = np.array(node_due_time)
        violations = [list(s) for s in list(abs(np.subtract(arrv_vec,due_vec)))] #violations of each node in a trip
        total_viol=[sum(s) for s in violations] #total violations of each trip
        min_idx=np.argmin(np.array(total_viol)) #index of the minimum violations
        trip=cust_list[min_idx]

        self.add_trips([self.current_location]+trip) # Add trip to the list of shoppers trips
        # Update the visit customers data:
        for i in range(len(trip)):
            customer_table[trip[i]].set_visit_time(node_arrival_time[min_idx][i])
            customer_table[trip[i]].set_shopper(self.get_shopper_id())

        self.set_current_time(node_arrival_time[min_idx][-1]) # Set the current time for shopper
        self.set_current_location(trip[-1])

        return customers

    def generate_trips(self,customers,stores,graph,trip_size=3,customer_pick='best'):
        """

        :param customers: list of customers
        :param stores: list of stores
        :param graph: underlying network
        :param trip_size: trips of size 1,2,3
        :param customer_pick: method of picking the customers to visiti, either 'best' (closest in terms of due date)
                              or 'randomize' pick among the best 20
        :return: returns top ten {or less if less than 10 unvisited customers remain} based on the best time criteria
        """

        curr_loc = self.get_current_location()
        stores = np.array(stores)
        # cust_data = Customer.due_date_list(customers) # An np array  [id , due , items]

        cust_dt = np.array([[c.get_id(),c.get_due_time(),c.get_item_count(),c.get_visit_time()] for c in customers])
        cust_data = cust_dt[cust_dt[:,3]==0] #** Select unvisited customers (visit_time = 0)

        # Check whether current location is a Store, if not, send the shopper to a nearby store.
        if np.sum(np.isin(stores,curr_loc)) == 0:
            # Randomly choose a method of sending the shopper to a nearby store:
            p=np.random.uniform(0,1,1)
            if p<=0.5:
                self.go_to_store(stores,graph,distance='randomize')
            else:
                self.go_to_store(stores,graph,distance='shortest')
            curr_loc = self.get_current_location()
        visited_customers = Customer.find_visited_customers(customers)
        if visited_customers.size == len(customers):
            print("All Customers have been visited")
            return customers
        if visited_customers.size > len(customers) - trip_size :
            trip_size = len(customers) - visited_customers.size
        # -----------------------------------------
        e = graph.get_edges()  # graph edges

        potentials=e[(e[:,0]==curr_loc) & (~np.isin(e[:,1],stores)) & (~np.isin(e[:,1],visited_customers)),:]
        potential_cust = potentials[:,1]

        # ************ Using Due Times Method to build the trips **************
        id = npi.indices(potential_cust, cust_data[:, 0], missing='ignore')
        idx = np.array(list(set(id)))
        potentials = cust_data[idx, :]
        # Adjust the due dates by the current shoppers' time
        potentials[:, 1] -= self.get_current_time()
        potentials[:, 1] = abs(potentials[:, 1])  # ** Absolute difference between current time & due times
        customer_count=potentials.shape[0]

        if customer_pick == 'best':
            if customer_count >= 10:
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][0:10]  # ** top 10 closest due dates
            elif customer_count < 10 and customer_count >= trip_size:
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][0:customer_count]  # ** top closest due dates
            elif customer_count < trip_size and customer_count > 0:
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][0:customer_count]  #
            else:
                print("******* ZERO CUSTOMERS LEFT ; SOMETHING WENT WRONG *******")
                print("SIZE OF POTENTIALS = ",potentials.shape[0])
                return
        elif customer_pick == 'randomize':
            if customer_count >= 20:
                picks = random.sample(range(0, 20), 10) #** 10 random numbers 0 - 19
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][picks]  #
            elif customer_count < 20 and customer_count >= 10:
                picks = random.sample(range(0, customer_count), 10) #** 10 random numbers 1 - 20
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][picks]  #
            elif customer_count < 10 and customer_count >= trip_size:
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][0:customer_count]  #
            elif customer_count < trip_size and customer_count > 0:
                top_ten = potentials[np.argsort(potentials[:, 1]), 0][0:customer_count]  #
            else:
                print("$$$$$$ ZERO CUSTOMERS LEFT $$$$$$")
                return
        else:
            print("customer_pick must be either 'best' or 'randomize' ")
            return
        return top_ten

    def trip_assessment(self,top_ten,customers,graph,stores,trip_size=3):
        """

        :top_ten: top ten target customers for generating the trips
        :customers: list of all customers with their data attributes
        :graph: underlying network structure
        :stores: list of all stores
        :trip_size: 1,2 or 3
        return: list of customers with updated data
        """

        curr_store = self.get_current_location()
        stores = np.array(stores)

        # ** Check whether current location is a Store, if not, send the shopper to a nearby store.
        if np.sum(np.isin(stores,curr_store)) > 0:
            self.go_to_store(stores,graph,distance='randomize')
        visited_customers = Customer.find_visited_customers(customers)
        if visited_customers.size == len(customers):
            print("All Customers have been visited")
            return customers
        if visited_customers.size > len(customers) - trip_size :
            trip_size = len(customers) - visited_customers.size
        all_trp=list(combinations(top_ten,trip_size))
        all_trp2=[list(c) for c in all_trp]
        all_trp2_2 = [list(set(permutations(c))) for c in all_trp2]
        all_trp2_2_1=[y for x in all_trp2_2 for y in x]
        all_trp22=[list(t) for t in all_trp2_2_1]
        all_trp3=[np.array([curr_store]+c) for c in all_trp22]
        if trip_size == 3:
            all_trp4 = list(map(lambda x:np.array([[x[0],x[1]],[x[1],x[2]],[x[2],x[3]]]),all_trp3))
        elif trip_size==2:
            all_trp4 = list(map(lambda x:np.array([[x[0],x[1]],[x[1],x[2]]]),all_trp3))
        elif trip_size == 1:
            all_trp4 = list(map(lambda x:np.array([[x[0],x[1]]]),all_trp3))
        else:
            print("Trip size must be either 1,2 or 3!")
            return
        del(all_trp,all_trp2,all_trp22,all_trp3) # clear up memory
        travel_time = [graph.find_trip_time(c) for c in all_trp4]
        customer_table = defaultdict(int)
        for c in customers:
            customer_table[c.get_id()] = c
        cust_list=[list(c[:,0]) for c in travel_time]
        items=[[customer_table[c].get_item_count() for c in t] for t in cust_list]
        node_due_time=[[customer_table[c].get_due_time() for c in t] for t in cust_list]
        store_time = [sum(c)+5 for c in items]
        trv_time = [list(c[:,1]) for c in travel_time]  # travel time to each individual node from the store
        assert(len(store_time) == len(trv_time))

        node_arr_time = [accumulate(s) for s in trv_time]
        node_arrival_time = []
        for i in range(len(store_time)):
            node_arrival_time.append([store_time[i]+self.current_time+s for s in node_arr_time[i]])

        arrv_vec = np.array(node_arrival_time)
        due_vec = np.array(node_due_time)
        violations = [list(s) for s in list(abs(np.subtract(arrv_vec,due_vec)))] #violations of each node in a trip
        total_viol=[sum(s) for s in violations]  # total violations of each trip
        min_idx=np.argmin(np.array(total_viol))  # index of the minimum violations
        trip=cust_list[min_idx]
        trip_nodes = [self.get_current_location()]+trip  # store + customers
        trip_arr_times = [self.get_current_time()]+node_arrival_time[min_idx]  # current time & arrv times of customers
        trip_violations = [0]+violations[min_idx]
        curr_trip = Trip(self.get_shopper_id(),trip_nodes,trip_arr_times,trip_violations)
        self.add_trips(curr_trip)
        # ** Update the visit customers data:
        for i in range(len(trip)):
            customer_table[trip[i]].set_visit_time(node_arrival_time[min_idx][i])
            customer_table[trip[i]].set_shopper(self.get_shopper_id())

        self.set_current_time(node_arrival_time[min_idx][-1])  # Set the current time for shopper
        self.set_current_location(trip[-1])

        return customers

    def print_shopper(self):
        if len(self.trips) == 0:
            print("No Trips!")
            return None
        for t in self.trips:
            t.print_trip()
            print("-----------------")

    def exchange_customer(self,other):
        pass

    def exchange_route(self,other,customers,graph,stores):
        """
        """

        if not isinstance(other,Shopper):
            print("Argument must be a Shopper Object!")
            return

        customer_table = defaultdict(int)
        for c in customers:
            customer_table[c.get_id()] = c

        # ** Get the list of trips for both shoppers:
        self_trips = self.get_trips()
        other_trips = other.get_trips()

        # ** Get the last trip generated for both shoppers:
        self_trip = self_trips[-1]
        other_trip = other_trips[-1]

        self_t = self_trip.get_customers()
        other_t = other_trip.get_customers()

        # *** Get the total violations of each trip before swap:
        curr_total_viol = self_trip.get_total_violations() + other_trip.get_total_violations()

        switch_nodes_idx = self_trip.nodes_to_switch(other_trip)  # indices of the nodes that can be switched!
        if len(switch_nodes_idx) == 0:
            return
        elif len(switch_nodes_idx) == 1:
            n = switch_nodes_idx[0]  # pick the first node (for both trips) to be exchanged
        elif len(switch_nodes_idx) > 1:
            n = switch_nodes_idx[1]

        # *** new trips with exchanged routes:
        new_self_t = self_t[0:n] + other_t[n:len(other_t)]
        new_other_t = other_t[0:n] + self_t[n:len(self_t)]
        new_self_trip = copy.deepcopy(new_self_t)
        new_other_trip = copy.deepcopy(new_other_t)
        new_trip_list=[]
        new_trip_list.extend((new_self_trip,new_other_trip))

        # ** Start-Time at stores:
        self_start = self_trip.get_visit_time()[0]
        other_start = other_trip.get_visit_time()[0]

        start_time_list=[]
        start_time_list.extend((self_start,other_start))
        new_total_viol,new_trip_nodes,new_trip_arrv_times,new_total_viol = \
            Shopper.evaluate_trips(new_trip_list,start_time_list,customers,graph,stores)

        # *** If violations didn't improve stop
        if sum(new_total_viol) >= curr_total_viol:
            return customers

        print(" $$$$$-----   USEFUL SWAP -----$$$$$")

        # *** If violations improved Update customers and shoppers data:
        self_trip_new = Trip(self.get_shopper_id(),new_trip_nodes[0],new_trip_arrv_times[0],list(new_total_viol[0]))
        other_trip_new = Trip(other.get_shopper_id(),new_trip_nodes[1],new_trip_arrv_times[1],list(new_total_viol[1]))
        # ** Drop the last trips of both shoppers to be able to swap some customers
        self.drop_last_trip()
        other.drop_last_trip()
        # *** Add new trips:
        self.add_trips(self_trip_new)
        other.add_trips(other_trip_new)

        # #** Update the visit customers data:

        # ** self shopper:
        for i in range(len(new_self_trip)):
            customer_table[new_self_trip[i]].set_visit_time(new_trip_arrv_times[0][i])
            customer_table[new_self_trip[i]].set_shopper(self.get_shopper_id())

        # ** other shopper:
        for i in range(len(new_other_trip)):
            customer_table[new_other_trip[i]].set_visit_time(new_trip_arrv_times[1][i])
            customer_table[new_other_trip[i]].set_shopper(other.get_shopper_id())
        #
        self.set_current_time(new_trip_arrv_times[0][-1]) # Set the current time for shopper
        self.set_current_location(new_self_trip[-1])

        other.set_current_time(new_trip_arrv_times[1][-1]) # Set the current time for shopper
        other.set_current_location(new_other_trip[-1])

        return customers

    @staticmethod
    def evaluate_trips(trip_list,start_time_list,customers,graph,stores):
        all_trips = []
        for i in range(0,len(trip_list)):
            trip_size = len(trip_list[i]) - 1
            if trip_size == 3:
                trip_g = list(np.array([[trip_list[i][0],trip_list[i][1]],[trip_list[i][1],trip_list[i][2]],[trip_list[i][2],trip_list[i][3]]]))
                all_trips.append((trip_g,start_time_list[i]))
            elif trip_size == 2:
                trip_g = list(np.array([[trip_list[i][0],trip_list[i][1]],[trip_list[i][1],trip_list[i][2]]]))
                all_trips.append((trip_g,start_time_list[i]))
            elif trip_size == 1:
                trip_g = list(np.array([[trip_list[i][0],trip_list[i][1]]]))
                all_trips.append((trip_g,start_time_list[i]))
            else:
                print("Trip size must be either 1,2 or 3!")
                return

        travel_time = [graph.find_trip_time(c[0]) for c in all_trips] # find travel times of the trips
        customer_table = defaultdict(int)
        for c in customers:
            customer_table[c.get_id()] = c
        cust_list=[list(c[:,0]) for c in travel_time]
        items=[[customer_table[c].get_item_count() for c in t] for t in cust_list]
        node_due_time=[[customer_table[c].get_due_time() for c in t] for t in cust_list]
        store_time = [sum(c)+5 for c in items]
        trv_time = [list(c[:,1]) for c in travel_time]  # travel time to each individual node from the store
        assert(len(store_time) == len(trv_time))

        node_arr_time = [accumulate(s) for s in trv_time]
        node_arrival_time = []
        for i in range(len(store_time)):
            node_arrival_time.append([store_time[i] + all_trips[i][1] + s for s in node_arr_time[i]])

        arrv_vec = np.array(node_arrival_time)
        due_vec = np.array(node_due_time)
        violations = [list(s) for s in list(abs(np.subtract(arrv_vec,due_vec)))]  # violations of each node in a trip
        total_viol=[sum(s) for s in violations]  # total violations of each trip

        trips = cust_list
        self_store = trip_list[0][0]
        other_store = trip_list[1][0]
        self_trip_nodes = [self_store] + trips[0] # store + customers
        other_trip_nodes = [other_store] + trips[1]
        self_trip_arr_times = [start_time_list[0]]+node_arrival_time[0]  # current time & arrv times of customers
        other_trip_arr_times = [start_time_list[1]]+node_arrival_time[1]
        self_trip_violations = [0]+violations[0]
        other_trip_violations = [0]+violations[1]

        new_trip_nodes = []
        new_trip_arrv_times = []
        new_trip_violations  =[]

        new_trip_nodes.extend((list(self_trip_nodes),list(other_trip_nodes)))
        new_trip_arrv_times.extend((list(self_trip_arr_times),list(other_trip_arr_times)))
        new_trip_violations.extend((self_trip_violations,other_trip_violations))
        new_total_viol = sum(total_viol)

        print(total_viol)

        return new_total_viol,new_trip_nodes,new_trip_arrv_times,total_viol

    @classmethod
    def shopper_number(cls):
        return cls.shopper_count
