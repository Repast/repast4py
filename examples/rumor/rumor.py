import networkx as nx
from typing import Dict
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py.network import write_network, read_network
from repast4py import context as ctx
from repast4py import core, random, schedule, logging, parameters


def generate_network_file(fname: str, n_ranks: int, n_agents: int):
    """Generates a network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = nx.connected_watts_strogatz_graph(n_agents, 2, 0.25)
    try:
        import nxmetis
        write_network(g, 'rumor_network', fname, n_ranks, partition_method='metis')
    except ImportError:
        write_network(g, 'rumor_network', fname, n_ranks)


model = None


class RumorAgent(core.Agent):

    def __init__(self, nid: int, agent_type: int, rank: int, received_rumor=False):
        super().__init__(nid, agent_type, rank)
        self.received_rumor = received_rumor

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data (self.received_rumor).

        Returns:
            The agent's state
        """
        return (self.uid, self.received_rumor)

    def update(self, data: bool):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state (received_rumor)
        """
        if not self.received_rumor and data:
            # only update if the received rumor state
            # has changed from false to true
            model.rumor_spreaders.append(self)
            self.received_rumor = data


def create_rumor_agent(nid, agent_type, rank, **kwargs):
    return RumorAgent(nid, agent_type, rank)


def restore_agent(agent_data):
    uid = agent_data[0]
    return RumorAgent(uid[0], uid[1], uid[2], agent_data[1])


@dataclass
class RumorCounts:
    total_rumor_spreaders: int
    new_rumor_spreaders: int


class Model:

    def __init__(self, comm, params):
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['network_file']
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_rumor_agent, restore_agent)
        self.net = self.context.get_projection('rumor_network')

        self.rumor_spreaders = []
        self.rank = comm.Get_rank()
        self._seed_rumor(params['initial_rumor_count'], comm)

        rumored_count = len(self.rumor_spreaders)
        self.counts = RumorCounts(rumored_count, rumored_count)
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['counts_file'])
        self.data_set.log(0)

        self.rumor_prob = params['rumor_probability']

    def _seed_rumor(self, init_rumor_count: int, comm):
        world_size = comm.Get_size()
        # np array of world size, the value of i'th element of the array
        # is the number of rumors to seed on rank i.
        rumor_counts = np.zeros(world_size, np.int32)
        if (self.rank == 0):
            for _ in range(init_rumor_count):
                idx = random.default_rng.integers(0, high=world_size)
                rumor_counts[idx] += 1

        rumor_count = np.empty(1, dtype=np.int32)
        comm.Scatter(rumor_counts, rumor_count, root=0)

        for agent in self.context.agents(count=rumor_count[0], shuffle=True):
            agent.received_rumor = True
            self.rumor_spreaders.append(agent)

    def at_end(self):
        self.data_set.close()

    def step(self):
        new_rumor_spreaders = []
        rng = random.default_rng
        for agent in self.rumor_spreaders:
            for ngh in self.net.graph.neighbors(agent):
                # only update agents local to this rank
                if not ngh.received_rumor and ngh.local_rank == self.rank and rng.uniform() <= self.rumor_prob:
                    ngh.received_rumor = True
                    new_rumor_spreaders.append(ngh)

        self.rumor_spreaders += new_rumor_spreaders
        self.counts.total_rumor_spreaders = len(self.rumor_spreaders)
        self.counts.new_rumor_spreaders = len(new_rumor_spreaders)
        self.data_set.log(self.runner.schedule.tick)

        self.context.synchronize(restore_agent)

    def start(self):
        self.runner.execute()


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
