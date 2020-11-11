import numpy as np
from mpi4py import MPI
import os

from typing import List, Dict

from dataclasses import dataclass, fields


class ReducingDataLogger:

    ARRAY_SIZE = 500

    def __init__(self, data_source, op, rank: int):
        """Creates a ReducingDataRecorder that gets its data from the
        specified source, reduces using the specified op and running on
        the specified rank.

        Args:
            op: an mpi reduction operator, e.g. MPI.SUM

        """
        self._data = np.zeros(ReducingDataLogger.ARRAY_SIZE, dtype=data_source.dtype)
        self._data_source = data_source
        self._op = op
        self._rank = rank
        self._idx = 0

    @property
    def name(self) -> str:
        return self._data_source.name

    @property
    def size(self) -> int:
        return self._idx

    @property
    def dtype(self):
        return self._data_source.dtype

    def log(self):
        """Logs the current value of this ReducingDataRecorder's
        data source.
        """
        if self._idx == self._data.shape[0]:
            self._data = np.concatenate(self._data, np.zeros(ReducingDataLogger.ARRAY_SIZE, 
                                                             dtype=self._data_source.dtype))
        self._data[self._idx] = self._data_source.value
        self._idx += 1

    def reduce(self, comm: MPI.Comm) -> np.array:
        recv_buf = None
        if self._rank == 0:
            recv_buf = np.empty(self._idx, dtype=self._data_source.dtype)

        comm.Reduce(self._data[:self._idx], recv_buf, self._op, 0)
        self._idx = 0
        return recv_buf


# creates a datasource from a dataclass field
class DCDataSource:

    def __init__(self, data_class, field_name: str, ds_name: str=None):
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
    def name(self):
        return self.ds_name

    @property
    def value(self):
        return getattr(self.data_class, self.field_name)

    @property
    def dtype(self):
        return self._dtype


def create_loggers(data_class: dataclass, op, rank: int, names: Dict[str, str]=None) -> List[ReducingDataLogger]:
    """Creates ReducingDataLogger-s from a dataclasses.dataclass, optionally
    constrainingthe loggers to log only from specified fields. By default the
    names arg is None and all the dataclass fields will be logged. 

    Args:
        data_class: the dataclass to log
        op: an mpi reduction operator (e.g., MPI.SUM)
        rank: the rank of this process
        names: dict where the keys are the names of the dataclass fields to log, and the values
        are the names of the datasource (i.e., the column header in the output tabular data). If
        value is None, then the dataclass field name will be used as the data source name.

    Returns:
        A list of ReducingDataLoggers that can be added to a ReducingDataSet for logging.
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

    return loggers


class ReducingDataSet:

    def __init__(self, data_loggers: List[ReducingDataLogger], comm,
                 fpath: str, sep: str=',', buffer_size: int=1000):
        self._data_loggers = data_loggers
        self._sep = sep
        self._comm = comm
        self._rank = comm.Get_rank()
        self._buf = None
        self._buffer_size = buffer_size
        self.fpath = fpath

        if self._rank == 0:
            parent = os.path.dirname(fpath)
            if not os.path.exists(parent):
                os.makedirs(parent)

            self._buf = np.zeros(2)
            self.ticks = []

            with open(self.fpath, 'w') as f_out:
                f_out.write('tick')
                for dl in data_loggers:
                    f_out.write(self._sep)
                    f_out.write(dl.name)
                f_out.write('\n')

    def close(self):
        self.write()

    def log(self, tick: float):
        if self._rank == 0:
            self.ticks.append(tick)

        for logger in self._data_loggers:
            logger.log()

        if self._data_loggers[0].size >= self._buffer_size:
            self.write()

    def write(self):
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
                            f_out.write(self._sep)
                            f_out.write(str(vals[i]))
                    f_out.write('\n')
