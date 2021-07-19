from typing import List, Dict
from pathlib import Path
import yaml
import json

from repast4py import random


# class Timer:

#     def __init__(self):
#         self.times = {}

#     def start_timer(self, name):
#         if not name in self.times:
#             self.times[name] = [time.time(), 0, []]
#         else:
#             self.times[name][0] = time.time()

#     def stop_timer(self, name):
#         t = time.time()
#         data = self.times[name]
#         data[1] = t
#         data[2].append(data[1] - data[0])

#     def print_times(self):
#         comm = MPI.COMM_WORLD
#         all_timings = comm.gather(self.times)
#         rank = MPI.COMM_WORLD.Get_rank()
#         if rank == 0:
#             print('{:<6s}{:<16s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.
#                 format('rank', 'timer_name', 'sum','min', 'max', 'mean', 'std'))

#             for r, timings in enumerate(all_timings):
#                 for k, v in timings.items():
#                     mean = np.mean(v[2])
#                     sm = np.sum(v[2])
#                     mn = np.min(v[2])
#                     mx = np.max(v[2])
#                     std = np.std(v[2])
#                     print('{:<6d}{:<16s}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}'.
#                         format(r, k,
#                         sm, mn, mx, mean, std))


def is_empty(lst: List[List]) -> bool:
    """Returns whether or not the specified list of lists
    is empty.

    Args:
        lst: the list of lists
    Returns:
        True if the list is empty or all of its nested lists are empty, otherwise False.
    """
    for nl in lst:
        if len(nl) > 0:
            return False
    return True


def find_free_filename(file_path: str) -> Path:
    """Given a file path, check if that file exists,
    and if so, repeatedly add a numeric infix to that
    file path until the file does not exist.

    For example, if output/counts.csv, exists check
    if counts_1.csv, counts_2.csv, and so exists until
    finding one that doesn't exist.

    Args:
        file_path: the path to the file to check
    Return:
        the path to the unused file
    """
    op = Path(file_path)
    p = Path(file_path)
    suffix = p.suffix
    infix = 1
    while (p.exists()):
        p = op.with_name(f'{op.stem}_{infix}{suffix}')
        infix += 1

    return p


def parse_params(parameters_file: str, parameters: str) -> Dict:
    """Parses model input parameters.

    Parameter parsing reads the parameters file, overrides
    any of those properties with those in the parameters string,
    and then executes the code that creates the derived parameters.
    Args:
        parameters_file: yaml format file containing model parameters
        parameters: json format string that overrides those in the file
    Returns:
        A dictionary containing the final model parameters.
    """
    params = {}
    with open(parameters_file) as f_in:
        params = yaml.load(f_in, Loader=yaml.SafeLoader)
    if parameters != '':
        params.update(json.loads(parameters))

    # Set seed from params, but before derived params are evaluated.
    if 'random.seed' in params:
        random.init(int(params['random.seed']))

    return params
