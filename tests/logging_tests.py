import sys
import unittest
import os
import csv

from dataclasses import dataclass
from mpi4py import MPI

sys.path.append("{}/../src".format(os.path.dirname(os.path.abspath(__file__))))

from repast4py import logging


@dataclass
class Counts:
    a: int = 0
    b: float = 0.0
    c: float = 0.0


class DSLoggingTests(unittest.TestCase):

    def update(self, counts, data_set, rank):
        for i in range(10):
            counts.a = i * rank
            counts.b = (i + 0.1) * rank
            counts.c = (i + 0.2) * rank
            data_set.log(float(i))

    def test_logging1(self):
        c = Counts()
        rank = MPI.COMM_WORLD.Get_rank()

        fpath = './test_out/test_log.csv'
        if rank == 0 and os.path.exists(fpath):
            os.remove(fpath)

        loggers = logging.create_loggers(c, names={'a': 'A', 'b': 'BValue', 'c': None}, op=MPI.SUM, rank=rank)
        data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, fpath=fpath)

        self.update(c, data_set, rank)
        data_set.close()

        if rank == 0:
            expected = [['tick', 'A', 'BValue', 'c']]
            w_size = MPI.COMM_WORLD.Get_size()

            for j in range(10):
                a, b, c = 0, 0, 0
                for i in range(w_size):
                    a += j * i
                    b += (j + 0.1) * i
                    c += (j + 0.2) * i
                expected.append([j, round(a, 1), round(b, 1), round(c, 1)])

            row_count = 0
            with open(fpath) as f_in:
                reader = csv.reader(f_in)
                for i, row in enumerate(reader):
                    row_count += 1
                    exp = expected[i]
                    if i > 0:
                        row = [float(row[0]), int(row[1]), round(float(row[2]), 1), round(float(row[3]), 1)]
                    self.assertEqual(exp, row)
            self.assertEqual(11, row_count)

    def test_logging2(self):
        c = Counts()
        rank = MPI.COMM_WORLD.Get_rank()

        fpath = './test_out/test_log.csv'
        if rank == 0 and os.path.exists(fpath):
            os.remove(fpath)
        # names is none so all field names, with ds name as field name
        loggers = logging.create_loggers(c, MPI.SUM, rank)
        data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, fpath=fpath)

        self.update(c, data_set, rank)
        data_set.close()

        if rank == 0:
            expected = [['tick', 'a', 'b', 'c']]
            w_size = MPI.COMM_WORLD.Get_size()

            for j in range(10):
                a, b, c = 0, 0, 0
                for i in range(w_size):
                    a += j * i
                    b += (j + 0.1) * i
                    c += (j + 0.2) * i
                expected.append([j, round(a, 1), round(b, 1), round(c, 1)])

            row_count = 0
            with open(fpath) as f_in:
                reader = csv.reader(f_in)
                for i, row in enumerate(reader):
                    row_count += 1
                    exp = expected[i]
                    if i > 0:
                        row = [float(row[0]), int(row[1]), round(float(row[2]), 1), round(float(row[3]), 1)]
                    self.assertEqual(exp, row)
            self.assertEqual(11, row_count)

    def test_logging3(self):
        c = Counts()
        rank = MPI.COMM_WORLD.Get_rank()

        fpath = './test_out/test_log.csv'
        if rank == 0 and os.path.exists(fpath):
            os.remove(fpath)
        # only log a
        loggers = logging.create_loggers(c, MPI.SUM, rank, names={'a': 'A'})
        data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, fpath=fpath)

        self.update(c, data_set, rank)
        data_set.close()

        if rank == 0:
            expected = [['tick', 'A']]
            w_size = MPI.COMM_WORLD.Get_size()

            for j in range(10):
                a = 0
                for i in range(w_size):
                    a += j * i
                expected.append([j, round(a, 1)])

            row_count = 0
            with open(fpath) as f_in:
                reader = csv.reader(f_in)
                for i, row in enumerate(reader):
                    row_count += 1
                    exp = expected[i]
                    if i > 0:
                        row = [float(row[0]), int(row[1])]
                    self.assertEqual(exp, row)
            self.assertEqual(11, row_count)

    def test_logging4(self):
        c = Counts()
        rank = MPI.COMM_WORLD.Get_rank()

        fpath = './test_out/test_log.csv'
        if rank == 0 and os.path.exists(fpath):
            os.remove(fpath)
        loggers = logging.create_loggers(c, names={'a': 'A', 'b': 'BValue', 'c': None}, op=MPI.SUM, rank=rank)
        data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD, fpath=fpath, buffer_size=1)

        self.update(c, data_set, rank)
        data_set.close()

        if rank == 0:
            expected = [['tick', 'A', 'BValue', 'c']]
            w_size = MPI.COMM_WORLD.Get_size()

            for j in range(10):
                a, b, c = 0, 0, 0
                for i in range(w_size):
                    a += j * i
                    b += (j + 0.1) * i
                    c += (j + 0.2) * i
                expected.append([j, round(a, 1), round(b, 1), round(c, 1)])

            row_count = 0
            with open(fpath) as f_in:
                reader = csv.reader(f_in)
                for i, row in enumerate(reader):
                    row_count += 1
                    exp = expected[i]
                    if i > 0:
                        row = [float(row[0]), int(row[1]), round(float(row[2]), 1), round(float(row[3]), 1)]
                    self.assertEqual(exp, row)
            self.assertEqual(11, row_count)


class TabularLoggingTests(unittest.TestCase):

    def test_logging(self):
        fpath = './test_out/test_log.csv'
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0 and os.path.exists(fpath):
            os.remove(fpath)

        logger = logging.TabularLogger(MPI.COMM_WORLD, fpath, ['A', 'B', 'C'])
        for i in range(1, 10):
            logger.log_row(rank * i, rank * i + 2, str(rank))
        logger.write()

        if rank == 0:
            expected = [['A', 'B', 'C']]
            for r in range(4):
                for i in range(1, 10):
                    row = [str(r * i), str(r * i + 2), str(r)]
                    expected.append(row)

            row_count = 0
            with open(fpath) as f_in:
                reader = csv.reader(f_in)
                for i, row in enumerate(reader):
                    row_count += 1
                    self.assertEqual(expected[i], row)

            self.assertEqual(len(expected), row_count)
