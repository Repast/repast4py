import yaml
import json
import argparse
from typing import Dict

from repast4py import random


params = {}
"""
Dict: once init
"""


def create_args_parser():
    """Creates an argparse parser with two arguments for
    accepting a yaml format file containing model parameter input,
    and an optional json dictionary string that can override that input.

    The two added arguments are:
    1. parameters_file: a yaml format file
    2. parameters: a json dictionary string that can override the
    parameters specified in the yaml format.

    This function is intended to work in concert with the :func:`repast4py.util.parse_params`
    function where the results of the argparse argument parsing are passed as arguments to
    that function.

    Examples:
        >>> parser = create_args_parser()
        >>> args = parser.parse_args()
        >>> params = init_params(args.parameters_file, args.parameters)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters_file", help="parameters file (yaml format)")
    parser.add_argument("parameters", nargs="?", default="{}", help="json parameters string")

    return parser


def init_params(parameters_file: str, parameters: str) -> Dict:
    """Parses model input parameters.

    Parameter parsing reads the parameters file, overrides
    any of those properties with those in the parameters string,
    and then executes the code that creates the derived parameters.
    This will automatically set the random number generator's seed
    if the parameters file or the parameters string contain a
    'random.seed' parameter.

    Args:
        parameters_file: yaml format file containing model parameters
        parameters: json format string that overrides those in the file
    Returns:
        A dictionary containing the final model parameters.
    """
    global params
    with open(parameters_file) as f_in:
        params = yaml.load(f_in, Loader=yaml.SafeLoader)
    if parameters != '':
        params.update(json.loads(parameters))

    if 'random.seed' in params:
        random.init(params['random.seed'])

    return params
