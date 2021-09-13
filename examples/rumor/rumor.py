from os import write
import networkx as nx
from typing import Dict
from mpi4py import MPI

from repast4py.network import write_network
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
        options = nxmetis.types.MetisOptions(dbglvl=nxmetis.enums.MetisDbgLvl.info)
        options.dbglvl = 3
        write_network(g, 'rumor_network', fname, n_ranks, partition_method='metis', options=options)
    except ImportError:
        write_network(g, 'rumor_network', fname, n_ranks)


class Model:

    def __init__(self) -> None:
        pass


def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)

