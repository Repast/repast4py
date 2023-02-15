# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
The parameters module contains functions for working with model input parameters.
"""

import yaml
import json
import argparse
from typing import Dict

from . import random
from . import util


params = {}
"""
Dict: After calling :func:`repast4py.parameters.init_params`, this dictionary
will contain the model parameters.
"""


def create_args_parser():
    """Creates an argparse parser with two arguments: 1)
    a yaml format file containing model parameter input,
    and 2) an optional json dictionary string that can override that input.

    This function is intended to work in concert with the :func:`repast4py.parameters.init_params`
    function where the results of the argparse argument parsing are passed as arguments to
    that function.

    Examples:
        >>> parser = create_args_parser()
        ...
        >>> args = parser.parse_args()
        >>> params = init_params(args.parameters_file, args.parameters)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters_file", help="parameters file (yaml format)")
    parser.add_argument("parameters", nargs="?", default="{}", help="json parameters string")

    return parser


def init_params(parameters_file: str, parameters: str, dump_file: str = None) -> Dict:
    """Initializes the :attr:`repast4py.parameters.params` dictionary with
    the model input parameters.

    This reads the parameters file, overrides
    any of those parameter values with those in the parameters string.
    This will automatically set the random number generator's seed
    if the parameters file or the parameters string contain a
    'random.seed' parameter.

    Args:
        parameters_file: yaml format file containing model parameters as key value pairs.
        parameters: json map format string that overrides those in the file.
        dump_file: optional file name to dump the resolved parameters to.
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

    if dump_file is not None:
        fname = util.find_free_filename(dump_file)
        with open(fname, 'w') as f_out:
            yaml.dump(params, f_out, indent=2, Dumper=yaml.SafeDumper)

    return params
