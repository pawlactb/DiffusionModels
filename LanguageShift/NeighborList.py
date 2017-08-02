import heapq
import itertools
import pickle

class heap(object):
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def heapify(self):
        heapq.heapify(self.pq)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the highest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def get_smallest(self, n):
        self.heapify()
        return heapq.nsmallest(n, self.pq)[:]



class NeighborList(object):
    """ Grid where adjacency is stored, rather calculated. Used for fast neighbor lookups.


    Performance:
        Let n be number of Agents

        Obtain agent by ID:
            O(n) (worst case)
        Obtain neighborhood of agent:
            O(n) (worst case)
        Calculate neighborhood (performed once at beginning):
            O((n**2)log_2(n)) (worst case)
            O(n**2 + n) (average case for small neighborhoods)


    Methods:
        get_neighbors: Returns the objects surrounding a given cell.
    """

    def __init__(self, distance_fn, neighborhood_size, loadpickle=None):
        self.agent_list = {}
        if loadpickle is not None:
            self.agent_neighbors = pickle.load(open(loadpickle, 'rb'))
        else:
            self.agent_neighbors = {}

        self.get_distance = distance_fn
        self.neighborhood_size = neighborhood_size

    def calc_neighbors(self):
        '''
        After adding all agents to the model, calculate the neighbors


        '''

        # Store other agents in a priority queue by distance (lesser distance == higher priority)
        # We pop a tuple of type (int, agent). The first element is the agent id.
        print('Calculating Neighbors')
        # iterate over all agents
        for a_id, a in self.agent_list.items():
            # create a priority queue
            neighbors_pq = heap()
            # print('Agent #' + str(a.unique_id))

            #iterate through all agents to calculate the distance to all other agents
            for b_id, b in self.agent_list.items():
                # print('\tadded: #' + str(b.unique_id))
                #add the agent's id to the priority queue, with the distance as the priority
                neighbors_pq.add_task(b_id, priority=self.get_distance(a, b))
            neighbors_pq.heapify()
            #
            neighbors = [i[2] for i in neighbors_pq.get_smallest(self.neighborhood_size + 1)]
            # print('\t\tadding ' + str(neighbors))
            self.agent_neighbors.update({int(a_id): neighbors})
            del neighbors_pq, neighbors

    def add_agent(self, pos, agent):
        x, y = pos
        agent.pos = pos
        self.agent_list.update({agent.unique_id: agent})

    def get_agent(self, id):
        return self.agent_list[id]

    def get_neighbors_by_agent(self, agent, include_self=False):
        # print(self.agent_neighbors.keys())
        return [self.get_agent(a_id) for a_id in self.agent_neighbors[agent.unique_id]]
