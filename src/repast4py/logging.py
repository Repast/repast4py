# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
The Logging module contains classes and functions for logging data produced by a
repast4py simulation to a file.
"""

import numpy as np
from mpi4py import MPI
import os

from typing import List, Dict
import csv

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

from dataclasses import dataclass, fields

from .util import find_free_filename


class DataSource(Protocol):
    """Protocol class for objects that can be used during logging as a
    source of data. Such objects must have a name, a type, and be able return
    the data to log via a "value" property.
    """

    @property
    def name(self) -> str:
        """Gets the name of this DataSource.

        Returns:
            The name of this DataSource.
        """
        pass

    @property
    # int is acceptable when float is type
    # https://stackoverflow.com/questions/50928592/mypy-type-hint-unionfloat-int-is-there-a-number-type
    def value(self) -> float:
        """Gets the value (i.e., the data to log) of this DataSource.

        Returns:
            The value (i.e., the data to log) of this DataSource.
        """
        pass

    @property
    def dtype(self) -> np.dtype:
        """Gets the numpy dtype of this DataSource.

        Returns:
            The numpy dtype of this DataSource.
        """
        pass


class ReducingDataLogger:

    _ARRAY_SIZE = 500

    def __init__(self, data_source: DataSource, op, rank: int):
        """Creates a ReducingDataRecorder that gets its data from the
        specified source, reduces that data using the specified op and runs on
        the specified rank.

        Args:
            data_source: the source of the data to reduce.
            op: an mpi reduction operator, e.g. MPI.SUM
            rank: the rank of this ReducingDataLogger
        """
        self._data = np.zeros(ReducingDataLogger._ARRAY_SIZE, dtype=data_source.dtype)
        self._data_source = data_source
        self._op = op
        self._rank = rank
        self._idx = 0

    @property
    def name(self) -> str:
        """Gets the name of this ReducingDataLogger.

        This is forwarded from the data source.

        Returns:
            The name of this ReducingDataLogger.
        """
        return self._data_source.name

    @property
    def size(self) -> int:
        """Gets the number of values this logger currently holds. This is
        set back to 0 on a reduce.

        Returns:
            The number of values this logger currently holds.
        """
        return self._idx

    @property
    def dtype(self):
        """Gets the numpy dtype of this ReducingDataLogger. This
        is forwarded from the data source.

        Returns:
            The numpy dtype of this ReducingDataLogger.
        """
        return self._data_source.dtype

    def log(self):
        """Logs the current value of this ReducingDataRecorder's
        data source.
        """
        if self._idx == self._data.shape[0]:
            self._data = np.concatenate((self._data, np.zeros(ReducingDataLogger._ARRAY_SIZE,
                                                              dtype=self._data_source.dtype)))
        self._data[self._idx] = self._data_source.value
        self._idx += 1

    def reduce(self, comm: MPI.Intracomm) -> np.array:
        """Reduces the values on all processes in the specified
        communicator to single values using the op specified in the
        constructor. The reduction is performed on each logged value
        at which point, the logged values are discarded.

        Args:
            comm: the communicator over whose ranks the reduction is performed.

        Returns:
            A numpy array containing the reduced values.
        """
        recv_buf = None
        if self._rank == 0:
            recv_buf = np.empty(self._idx, dtype=self._data_source.dtype)

        comm.Reduce(self._data[:self._idx], recv_buf, self._op, 0)
        self._idx = 0
        return recv_buf


# creates a datasource from a dataclass field
class DCDataSource:
    """A DCDataSource implements the :class:`repast4py.logging.DataSource` protocol for Python
    :py:obj:`dataclasses.dataclass` objects. Each DCDataSource gets its data to log
    from a dataclass field.
    """

    def __init__(self, data_class: dataclass, field_name: str, ds_name: str = None):
        """The constructor creates a DCDataSource that will log the specified field of the
        specified dataclass. By default, the field name will be used as the data source
        name, but an optional data source name can be supplied. The data source
        name will become the column header in the logged tabular data.

        Args:
            data_class: the dataclass containing the values to log
            field_name: the name of the field in the dataclass to log
            ds_name: an optional name that will be used as the column
                header if supplied, otherwise the field_name will be
                used.
        """
        self.data_class = data_class
        self.field_name = field_name
        if ds_name is None:
            self.ds_name = self.field_name
        else:
            self.ds_name = ds_name

        self._dtype = None

        for f in fields(self.data_class):
            if f.name == field_name:
                self._dtype = np.dtype(f.type)
                break

        if self._dtype is None:
            raise ValueError('Field "{}" not found in dataclass {}'.format(field_name, data_class))

    @property
    def name(self) -> str:
        """Gets the name of this DCDataSource.

        Returns:
            The name of this DataSource.
        """
        return self.ds_name

    @property
    # int is acceptable when float is type
    # https://stackoverflow.com/questions/50928592/mypy-type-hint-unionfloat-int-is-there-a-number-type
    def value(self) -> float:
        """Gets the value of this DCDataSource.

        Returns:
            The value of this DataSource.
        """
        return getattr(self.data_class, self.field_name)

    @property
    def dtype(self):
        """Gets the numpy dtype of this DCDataSource.

        Returns:
            The numpy dtype of this DCDataSource.
        """
        return self._dtype


def create_loggers(data_class: dataclass, op, rank: int, names: Dict[str, str] = None) -> List[ReducingDataLogger]:
    """Creates ReducingDataLogger-s from the fields in a :py:obj:`dataclasses.dataclass`, optionally
    constraining the loggers to log only from specified fields. By default the
    names argument is None and all the dataclass fields will be logged.

    Args:
        data_class: the dataclass providing the values to log
        op: an mpi reduction operator (e.g., MPI.SUM)
        rank: the rank of this process
        names: a Python dict where the keys are the names of the dataclass fields to log, and the values
            are the names of the column header in the output tabular data. If
            value is None, then the dataclass field name will be used as the data source
            column name.

    Returns:
        A list of ReducingDataLoggers that can be added to a ReducingDataSet for logging.

    Examples:
        Creating multiple different loggers from the same data source, assigning each a different reduction
        operation. We append the new loggers to the original list with :samp:`+=`.

        >>> meet_log = MeetLog()
        >>> meet_log
        MeetLog(total_meets=0, min_meets=0, max_meets=0)
        >>> loggers = logging.create_loggers(meet_log, op=MPI.SUM, names={'total_meets': 'total'}, rank=rank)
        >>> loggers += logging.create_loggers(meet_log, op=MPI.MIN, names={'min_meets': 'min'}, rank=rank)
        >>> loggers += logging.create_loggers(meet_log, op=MPI.MAX, names={'max_meets': 'max'}, rank=rank)
    """
    loggers = []
    if names is None:
        for f in fields(data_class):
            source = DCDataSource(data_class, f.name)
            loggers.append(ReducingDataLogger(source, op, rank))
    else:
        for f in fields(data_class):
            if f.name in names:
                ds_name = f.name if names[f.name] is None else names[f.name]
                source = DCDataSource(data_class, f.name, ds_name)
                loggers.append(ReducingDataLogger(source, op, rank))

        if len(loggers) == 0:
            raise ValueError('Unable to create any loggers: names dictionary keys do not match any dataclass fields.')

    return loggers


class ReducingDataSet:
    """A ReducingDataSet represents a tabular data set where each column
    is produced by a ReducingDataLogger and where the name of each logger is
    the name of the column. The produced tabular data is periodically
    written to a file.
    """

    def __init__(self, data_loggers: List[ReducingDataLogger], comm: MPI.Comm,
                 fpath: str, delimiter: str = ',', buffer_size: int = 1000):
        """The constructor creates a ReducingDataSet whose columns are produced from
        the specified data loggers which are reduced across the specified
        communicator. The data_loggers can be created using the :func:`repast4py.logging.create_loggers`
        function.

        Args:
            data_loggers: a list of ReducingDataLoggers that will produce the tabular data,
                one data_logger per column.
            comm: the communicator to reduce over
            fpath: the file to write the data to
            delimiter: the delimiter to use to separate the column values
            buffer_size: the number of values to log before writing to a file.
        """
        self._data_loggers = data_loggers
        self._delimiter = delimiter
        self._comm = comm
        self._rank = comm.Get_rank()
        self._buffer_size = buffer_size
        self.fpath = find_free_filename(fpath)

        if self._rank == 0:
            parent = os.path.dirname(fpath)
            if not os.path.exists(parent):
                os.makedirs(parent)
            self.ticks = []

            with open(self.fpath, 'w') as f_out:
                f_out.write('tick')
                for dl in data_loggers:
                    f_out.write(self._delimiter)
                    f_out.write(dl.name)
                f_out.write('\n')

    def close(self):
        """Closes this ReducingDataSet, writing any remaining
        data to the file.
        """
        self.write()

    def log(self, tick: float):
        """Logs the data for the specified tick, by calling
        log on each data logger contained in this ReducingDataSet.

        Args:
            tick: the tick (timestamp) at which the data is logged.
        """
        if self._rank == 0:
            self.ticks.append(tick)

        for logger in self._data_loggers:
            logger.log()

        if self._data_loggers[0].size >= self._buffer_size:
            self.write()

    def write(self):
        """Writes any currently logged, but not yet reduced
        data to this ReducingDataSet's file, by reducing and then writing the
        resulting data to the file.
        """
        val_arrays = [x.reduce(self._comm) for x in self._data_loggers]
        if self._rank == 0:
            nrows = len(self.ticks)
            val_arrays = [self.ticks] + val_arrays
            with open(self.fpath, 'a') as f_out:
                for i in range(nrows):
                    first = True
                    for vals in val_arrays:
                        if first:
                            f_out.write(str(vals[i]))
                            first = False
                        else:
                            f_out.write(self._delimiter)
                            f_out.write(str(vals[i]))
                    f_out.write('\n')

            self.ticks.clear()


class TabularLogger:
    """Logs arbitrary values by row in a delimited tabular format. All the rows
    logged by each rank are concatenated on a write into multiple rows.

    Args:
        comm: the communicator to reduce over
        fpath: the file to write the data to
        headers: the header values for each column
        delimiter: the seperator to use to seperate the column values
    """
    def __init__(self, comm: MPI.Comm, fpath: str, headers: List[str], delimiter: str = ','):
        self._delimiter = delimiter
        self._comm = comm
        self._rank = comm.Get_rank()
        self._rows = []

        if self._rank == 0:
            self._fpath = find_free_filename(fpath)
            parent = os.path.dirname(self._fpath)
            if not os.path.exists(parent):
                os.makedirs(parent)

            with open(self._fpath, 'w') as fout:
                writer = csv.writer(fout, delimiter=self._delimiter)
                writer.writerow(headers)

    def log_row(self, *args):
        """Logs the specified values as a row in the tabular output.

        Each value in the argument list is written to a column in the
        order they appear in the argument list.

        Args:
            args: variable length argument list containing the values to log.
                The order of the values should correspond to the headers argument
                in the constructor.

        Examples:
            The following will log the value of the tick, person_id variables, and 12, and 24
            as a row in the tabular data.

            >>> logger.log_row(tick, person_id, 12, 24)
        """
        self._rows.append(args)

    def write(self):
        """Writes all the currently logged rows to a file by gathering all the
        rows from all ranks, concatenating them, and writing the total collection of rows to
        the file specified in the constructor. This is a collective
        operation and must be called by all ranks in the communicator.
        """
        all_items = self._comm.gather(self._rows, root=0)
        if self._rank == 0:
            with open(self._fpath, 'a') as fout:
                writer = csv.writer(fout, delimiter=self._delimiter)
                for items in all_items:
                    writer.writerows(items)

        self._rows.clear()

    def close(self):
        """Closes this TabularLogger, writing any rows of
        data to the file.
        """
        self.write()
