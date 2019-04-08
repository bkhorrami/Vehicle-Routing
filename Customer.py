class Customer(Node):
    customer_count = 0

    def __init__(self, id, lat, lon, time, demand, type=0, visit=0, shopper=0):
        super().__init__(id, lat, lon, type)
        self.due_time = time
        self.visit_time = visit
        self.shopper_id = shopper
        self.item_count = demand
        self.violation = 0
        self.signed_violation = 0
        Customer.customer_count += 1 # add 1 to customer count

    @classmethod
    def get_customer_count(cls):
        return customer_count

    def get_violation(self):
        if self.visit_time == 0:
            self.violation = None
        elif self.visit_time > 0:
            self.violation = abs(self.due_time - self.visit_time)
        return self.violation

    def get_signed_violation(self):
        if self.visit_time == 0:
            self.signed_violation = None
        elif self.visit_time > 0:
            self.signed_violation = self.due_time - self.visit_time
        return self.signed_violation

    @property
    def due_time(self):
        return self.due_time

    @property
    def visit_time(self):
        return self.visit_time

    @property
    def shopper_id(self):
        return self.shopper_id

    @property
    def item_count(self):
        return self.item_count

    @due_time.setter
    def due_time(self, dt):
        self.due_time = dt

    @visit_time.setter
    def visit_time(self,vtime):
        self.visit_time = vtime

    @shopper_id.setter
    def shopper_id(self,shp):
        self.shopper_id = shp

    @item_count.setter
    def item_count(self,demand):
        self.item_count = demand

    @staticmethod
    def due_date_list(cust):
        """
        :param cust: List of customer Objects
        :return: an np array containing [customer_id , due_times , item_counts]
        """
        customers=list(cust)
        if len(customers) == 0:
            print("No Customers!")
            return
        return np.array([[c.get_id(),c.get_due_time(), c.get_item_count()] for c in customers])

    @staticmethod
    def find_visited_customers(customers):
        """
        Return an array containing ID's of visited customers
        :param customers: List of customers
        """
        shoppers=np.array([[c.get_visit_time(),c.get_id()] for c in customers])
        return shoppers[np.nonzero(shoppers[:,0]),1]

    def print_customer(self):
        if self.visit_time == 0:
            violation = 0
        else:
            violation = abs(self.due_time - self.visit_time)

        print(self.id,",DUE = ",self.due_time,", VISIT = ",self.visit_time,", SHP = ",self.shopper_id,
              "violation = ",violation)

    def __eq__(self, other):
        return abs(self.due_time - self.visit_time) == abs(other.due_time - other.visit_time)

    def __lt__(self, other):
        return abs(self.due_time - self.visit_time) < abs(other.due_time - other.visit_time)

    def __gt__(self, other):
        return abs(self.due_time - self.visit_time) > abs(other.due_time - other.visit_time)

    def __le__(self, other):
        return abs(self.due_time - self.visit_time) <= abs(other.due_time - other.visit_time)

    def __ge__(self, other):
        return abs(self.due_time - self.visit_time) >= abs(other.due_time - other.visit_time)
