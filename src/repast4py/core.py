import itertools
import collections

from ._core import Agent
from .random import default_rng as rng

from mpi4py import MPI


class SharedContext:
    """Encapsulates a population of agents on a single process rank.

    A SharedContext may have one or more projections associated with it to 
    impose a relational structure on the agents in the context. It also
    provides functionality for synchroizes agents across processes, moving 
    agents from one processe to another and managing any ghosting strategy.    
    """
    def __init__(self, comm):
        """Initializes this SharedContext with the specified communicator.

        Args:
            comm (mpi4py communicator): the communicator uses to communicate among SharedContexts in the distributed model
        """
        self._local_agents = collections.OrderedDict()
        self._agents_by_type = {}
        self.projections = {}
        self.projection_id = 0
        self.rank = comm.Get_rank()
        self.comm = comm
        self.buffered_projs = []

    def add(self, agent: Agent):
        """Adds the specified agent to this SharedContext.

        The agent will also be added to any projections currently in this SharedContext

        Args:
            agent: the agent to add
        """
        self._local_agents[agent.uid] = agent
        self._agents_by_type.setdefault(agent.uid[1], collections.OrderedDict())[agent.uid] = agent
        for proj in self.projections.values():
            proj.add(agent)
    
    def add_projection(self, projection):
        """Adds the specified projection to this SharedContext.

        If the projection has `clear_buffer` and `synchronize_buffer` attributes,
        then clear_buffer will be called prior to synchornization, and synchronize buffer
        durring synchronization in the SharedProjection.synchronize method. 
        
        Any agents currently in this context will be added to the projection.

        Args:
            projection: the projection add
        """
        for prj in self.projections.values():
            if projection.name == prj.name:
                raise ValueError('Context already contains the named projection "{}"'.format(prj.name))

        self.projections[self.projection_id] = projection
        self.projection_id += 1
        for a in self._local_agents.values():
            projection.add(a)

        if hasattr(projection, 'clear_buffer') and hasattr(projection, 'synchronize_buffer'):
            self.buffered_projs.append(projection)

    def remove(self, agent: Agent):
        """Removes the specified agent from this SharedContext

        This agent is also removed from any projections associated with this SharedContext.
        """
        del self._local_agents[agent.uid]
        del self._agents_by_type[agent.uid[1]][agent.uid]
        for proj in self.projections.values():
            proj.remove(agent)

    def _fill_send_data(self):
        send_data = [[] for i in range(self.comm.size)]
        removed_agents = {}
        # gather agents to send from the out of bounds (oob) sequence 
        for pid, proj in self.projections.items():
            for agent_id, ngh_rank, pt in proj._get_oob():
                agent = self._local_agents.pop(agent_id, None)
                if agent == None:
                    data = ((agent_id),)
                    # TODO use id
                    proj.remove(removed_agents[agent_id])
                else:
                    removed_agents[agent_id] = agent
                    del self._agents_by_type[agent_id[1]][agent_id]
                    data = agent.save()
                    proj.remove(agent)
               
                # send to ngh rank a list of lists - inner list 
                # is [(agent_id, agent_state), (projection id, and projection data)]
                send_data[ngh_rank].append([data, (pid, pt)])

            proj._clear_oob()

        removed_agents.clear()
        
        return send_data
    
    def _process_recv_data(self, recv_data, create_agent):
        for data_list in recv_data:
            for data in data_list:
                # agent_data: tuple - agent id tuple, and agent state
                agent_data = data[0]
                proj_data = data[1]

                if agent_data[0] in self._local_agents:
                    agent = self._local_agents[agent_data[0]]
                else:
                    #print("New agent", agent_data)
                    agent = create_agent(agent_data)
                    # print("Adding", agent.id)
                    self.add(agent)

                    # add to the projection
                self.projections[proj_data[0]]._synch_move(agent, proj_data[1])

    
    def synchronize(self, create_agent, sync_buffer=True):
        if sync_buffer:
            for proj in self.buffered_projs:
                proj.clear_buffer()

        send_data = self._fill_send_data()
        # print('{}: send data - {}'.format(self.rank, send_data))
        recv_data = self.comm.alltoall(send_data)
        # print('{}: recv data - {}'.format(self.rank, recv_data))
        self._process_recv_data(recv_data, create_agent)

        if sync_buffer:
            for proj in self.buffered_projs:
                proj.synchronize_buffer(create_agent)


    # TODO implement the shuffle -- values to list, and then shuffle that?
    def agents(self, agent_type: int=None, shuffle: bool=False):
        """Gets the agents in SharedContext, optionally of the specified type.

        Args:
            agent_type: the type id of the agent, defaults to None.
            shuffle: whether or not the iteration order is shuffled. if true, 
            the order is shuffled. If false, the iteration order is the order of 
            insertion.

        Returns:
            iterable: An iterable over all the agents in the context.

            If the agent_type is not None then an iterable over agents of that type will be 
            returned.

        """
        if shuffle:
            if agent_type == None:
                l = list(self._local_agents.values())
                rng.shuffle(l)
                return l
            else:
                l = list(self._agents_by_type[agent_type].values())
                rng.shuffle(l)
                return l
        else:
            if agent_type == None:
                return self._local_agents.values().__iter__()
            else:
                return self._agents_by_type[agent_type].values().__iter__()
        
    def count(self, agent_type_ids:list=None)-> dict:
        """Gets a count of agents in this SharedContext by type.

        Args:
            agent_type_ids: a list of the agent type ids identifying the agents types to count. If
            this is None then all the agents will be counted.

        Returns:
            A dictionary containing the counts (the dict values) by type (the dict keys).
        """
        counts = {}
        if agent_type_ids:
            for i in agent_type_ids:
                counts[i] = len(self._agents_by_type[i])
        else:
            for k, v in self._agents_by_type.items():
                counts[k] = len(v)

        return counts
            

        
    