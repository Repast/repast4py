import itertools

from ._core import Agent

from mpi4py import MPI

def printf(msg):
    import sys
    print(msg)
    sys.stdout.flush()

def create_agent_predicate(agent_type):
    def pred(agent):
        return agent.uid[1] != agent_type
    return pred

class SharedContext:

    def __init__(self, comm):
        self._local_agents = {}
        self.projections = {}
        self.projection_id = 0
        self.rank = comm.Get_rank()
        self.comm = comm
        self.buffered_projs = []

    def add(self, agent):
        self._local_agents[agent.uid] = agent
        for proj in self.projections.values():
            proj.add(agent)
    
    def add_projection(self, projection):
        # TODO check already added
        self.projections[self.projection_id] = projection
        self.projection_id += 1
        for a in self._local_agents.values():
            projection.add(a)

        if hasattr(projection, 'clear_buffer') and hasattr(projection, 'synchronize_buffer'):
            self.buffered_projs.append(projection)

    def remove(self, agent):
        del self._local_agents[agent.uid]
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
                # agent_data tuple is agent id tuple, and agent state
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


    def agents(self, agent_type=None):
        if agent_type == None:
            return self._local_agents.values().__iter__()
        else:
            pred = create_agent_predicate(agent_type)
            return itertools.filterfalse(pred, self._local_agents.values())
    
    def get_counts(self, counts):
        for uid in self._local_agents:
            counts[uid[1]] += 1
    