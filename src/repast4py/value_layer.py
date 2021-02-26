import torch
import numpy as np

from typing import Callable

from . import geometry

from .space import BorderType, GridStickyBorders, GridPeriodicBorders, CartesianTopology, BoundingBox
from .space import DiscretePoint as dpt

from .core import AgentManager


class Impl1D:

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand((bounds.xextent,), dtype=dtype)
        else:
            self.grid = torch.full((bounds.xextent,), init_value, dtype=dtype)
        self.ngh_translation = 0 - bounds.xmin
        self.translation = np.array([self.ngh_translation, 0, 0])
        self.ngh_finder = geometry.find_1d_nghs_sticky if borders == BorderType.Sticky else geometry.find_1d_nghs_periodic
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1])

    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.translation
        return self.grid[npt[0]]

    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.translation
        self.grid[npt[0]] = val

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max)
        grid_idxs = nghs + self.ngh_translation
        return (self.grid[grid_idxs], nghs)

    def _compute_slice(self, key):
        if isinstance(key, tuple):
            slc = key[0]
            step = 1 if slc.step is None else slc.step
            start = 0 if slc.start is None else (slc.start + self.translation[0])
            stop = self.grid.shape[1] if slc.stop is None else (slc.stop + self.translation[0])
            return (slice(start, stop, step),)

        elif isinstance(key, slice):
            step = 1 if key.step is None else key.step
            start = 0 if key.start is None else (key.start + self.translation[0])
            stop = self.grid.shape[1] if key.stop is None else (key.stop + self.translation[0])
            return (slice(start, stop, step),)

        else:
            #return self.grid[:, key + self.pt_translation[0]]
            return (slice(key + self.translation[0]),)

    def __getitem__(self, key):
        return self.grid[self._compute_slice(key)]

    def __setitem__(self, key, val):
        self.grid[self._compute_slice(key)] = val


class Impl2D:

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        self.idx = np.array([[0], [0]])
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand(
                (bounds.yextent, bounds.xextent), dtype=dtype)
        else:
            self.grid = torch.full((bounds.yextent, bounds.xextent), init_value, dtype=dtype)

        self.pt_translation = np.array([0 - bounds.xmin, 0 - bounds.ymin, 0])
        self.ngh_translation = np.array([[0 - bounds.ymin], [0 - bounds.xmin]])
        self.ngh_finder = geometry.find_2d_nghs_sticky if borders == BorderType.Sticky else geometry.find_2d_nghs_periodic
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1, bounds.ymin, bounds.ymin + bounds.yextent - 1])

    def _update_idx(self, pt):
        self.idx[0, 0] = pt[1]
        self.idx[1, 0] = pt[0]

        return self.idx


    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        return self.grid[self._update_idx(npt)]

    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        self.grid[self._update_idx(npt)] = val

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max, row_major=True)
        grid_idxs = nghs + self.ngh_translation
        # swap to remove the row major
        nghs[[0, 1]] = nghs[[1, 0]]
        return (self.grid[grid_idxs], nghs)

    def _compute_slice(self, key):
        if isinstance(key, tuple):
            if len(key) == 1:
                slc = key[0]
                step = 1 if slc.step is None else slc.step
                start = 0 if slc.start is None else (slc.start + self.pt_translation[0])
                stop = self.grid.shape[1] if slc.stop is None else (slc.stop + self.pt_translation[0])
                return (slice(None, None), slice(start, stop, step))
                
            else:
                cslice = key[0]
                cstep = 1 if cslice.step is None else cslice.step
                cstart = 0 if cslice.start is None else (cslice.start + self.pt_translation[0])
                cstop = self.grid.shape[1] if cslice.stop is None else (cslice.stop + self.pt_translation[0])

                rslice = key[1]
                rstep = 1 if rslice.step is None else rslice.step
                rstart = 0 if rslice.start is None else (rslice.start + self.pt_translation[1])
                rstop = self.grid.shape[0] if rslice.stop is None else (rslice.stop + self.pt_translation[1])

                #return self.grid[rstart : rstop : rstep, cstart : cstop : cstep]
                return (slice(rstart, rstop, rstep), slice(cstart, cstop, cstep))


        elif isinstance(key, slice):
            step = 1 if key.step is None else key.step
            start = 0 if key.start is None else (key.start + self.pt_translation[0])
            stop = self.grid.shape[1] if key.stop is None else (key.stop + self.pt_translation[0])
            #return self.grid[:, start : stop : step]
            return (slice(None, None), slice(start, stop, step))

        else:
            #return self.grid[:, key + self.pt_translation[0]]
            return (slice(None, None), slice(key + self.pt_translation[0], None))

    def __getitem__(self, key):
        return self.grid[self._compute_slice(key)]
        
    def __setitem__(self, key, val):
        self.grid[self._compute_slice(key)] = val


class Impl3D:

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        self.idx = np.array([[0], [0], [0]])
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand(
                (bounds.yextent, bounds.xextent, bounds.zextent), dtype=dtype)
        else:
            self.grid = torch.full(
                (bounds.yextent, bounds.xextent, bounds.zextent), init_value, dtype=dtype)

        # row major
        self.pt_translation = np.array([0 - bounds.xmin, 0 - bounds.ymin, 0 - bounds.zmin])
        self.ngh_translation = np.array([[self.pt_translation[1]], [self.pt_translation[0]], [self.pt_translation[2]]])
        self.ngh_finder = None #geometry.find_3d_nghs_sticky if borders == BorderType.Sticky else geometry.find_3d_nghs_periodic,
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1,
             bounds.ymin, bounds.ymin + bounds.xextent - 1,
             bounds.zmin, bounds.zmin + bounds.zextent - 1])

    def _update_idx(self, pt):
        self.idx[0, 0] = pt[1]
        self.idx[1, 0] = pt[0]
        self.idx[2, 0] = pt[2]
        
        return self.idx

    def get(self, pt):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        return self.grid[self._update_idx(npt)]

    def set(self, pt, val):
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        self.grid[self._update_idx(npt)] = val

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max, row_major=True) 
        grid_idxs = nghs + self.ngh_translation
        # swap to remove the row major
        nghs[[0, 1, 2]] = nghs[[1, 0, 2]]
        return (self.grid[grid_idxs], nghs)

    def _compute_slice(self, key):
        if isinstance(key, tuple):
            if len(key) == 1:
                slc = key[0]
                step = 1 if slc.step is None else slc.step
                start = 0 if slc.start is None else (slc.start + self.pt_translation[0])
                stop = self.grid.shape[1] if slc.stop is None else (slc.stop + self.pt_translation[0])
                return (slice(None, None), slice(start, stop, step), slice(None, None))
                
            elif (len(key) == 2):
                cslice = key[0]
                cstep = 1 if cslice.step is None else cslice.step
                cstart = 0 if cslice.start is None else (cslice.start + self.pt_translation[0])
                cstop = self.grid.shape[1] if cslice.stop is None else (cslice.stop + self.pt_translation[0])

                rslice = key[1]
                rstep = 1 if rslice.step is None else rslice.step
                rstart = 0 if rslice.start is None else (rslice.start + self.pt_translation[1])
                rstop = self.grid.shape[0] if rslice.stop is None else (rslice.stop + self.pt_translation[1])

                #return self.grid[rstart : rstop : rstep, cstart : cstop : cstep]
                return (slice(rstart, rstop, rstep), slice(cstart, cstop, cstep), slice(None, None))

            else:
                cslice = key[0]
                cstep = 1 if cslice.step is None else cslice.step
                cstart = 0 if cslice.start is None else (cslice.start + self.pt_translation[0])
                cstop = self.grid.shape[1] if cslice.stop is None else (cslice.stop + self.pt_translation[0])

                rslice = key[1]
                rstep = 1 if rslice.step is None else rslice.step
                rstart = 0 if rslice.start is None else (rslice.start + self.pt_translation[1])
                rstop = self.grid.shape[0] if rslice.stop is None else (rslice.stop + self.pt_translation[1])

                zslice = key[2]
                zstep = 1 if zslice.step is None else zslice.step
                zstart = 0 if zslice.start is None else (zslice.start + self.pt_translation[2])
                zstop = self.grid.shape[2] if zslice.stop is None else (zslice.stop + self.pt_translation[2])

                #return self.grid[rstart : rstop : rstep, cstart : cstop : cstep]
                return (slice(rstart, rstop, rstep), slice(cstart, cstop, cstep), slice(zstart, zstop, zstep))


        elif isinstance(key, slice):
            step = 1 if key.step is None else key.step
            start = 0 if key.start is None else (key.start + self.pt_translation[0])
            stop = self.grid.shape[1] if key.stop is None else (key.stop + self.pt_translation[0])
            #return self.grid[:, start : stop : step]
            return (slice(None, None), slice(start, stop, step), slice(None, None))

        else:
            #return self.grid[:, key + self.pt_translation[0]]
            return (slice(None, None), slice(key + self.pt_translation[0], None), slice(None, None))

    def __getitem__(self, key):
        return self.grid[self._compute_slice(key)]

    def __setitem__(self, key, val):
        self.grid[self._compute_slice(key)] = val
    

class ValueLayer:

    def __init__(self, bounds, borders, init_value, dtype=torch.float64):

        if bounds.yextent == 0:
            self.impl = Impl1D(bounds, borders, init_value, dtype)
        elif bounds.zextent == 0:
            self.impl = Impl2D(bounds, borders, init_value, dtype)
        else:
            self.impl = Impl3D(bounds, borders, init_value, dtype)
        self.bbounds = bounds

    @property
    def grid(self):
        return self.impl.grid

    @property
    def bounds(self):
        return self.bbounds
            
    def get(self, pt):
        return self.impl.get(pt)
        
    def set(self, pt, val):
        self.impl.set(pt, val)

    def __getitem__(self, key):
        return self.impl.__getitem__(key)
    
    def __setitem__(self, key, val):
        self.impl.__setitem__(key, val)

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        return self.impl.get_nghs(pt, extent)
        

class SharedValueLayer(ValueLayer):

    def __init__(self, comm, bounds, borders, buffer_size, init_value, dtype=torch.float64):
        periodic = borders == BorderType.Periodic
        topo = CartesianTopology(comm, bounds, periodic)
        self.cart_comm = topo.comm
        self.rank = self.cart_comm.Get_rank()
        self.coords = topo.coordinates
        self.buffer_size = buffer_size

        # add buffer to local bounds
        self.local_bounds = topo.local_bounds
        if comm.Get_size() > 1:
            self._init_buffered_bounds(bounds, buffer_size, periodic)
        else:
            self.buffered_bounds = self.local_bounds
            self.non_buff_grid_offsets = np.zeros(6, dtype=np.int32)

        super().__init__(self.buffered_bounds, borders, init_value, dtype)

        if comm.Get_size() > 1:
            self._init_sync_data(topo, buffer_size)

    def _init_buffered_bounds(self, bounds, buffer_size, periodic):
        xmin, xextent = 0, 0
        ymin, yextent = 0, 0
        zmin, zextent = 0, 0
        # offsets from buffered grid into the local grid data
        self.non_buff_grid_offsets = np.zeros(6, dtype=np.int32)

        if periodic:
            self.non_buff_grid_offsets[0:2] = buffer_size
            if self.local_bounds.xmin > bounds.xmin:
                xmin = self.local_bounds.xmin - buffer_size

            # extent is x2 to create buffer on both sides
            xextent = self.local_bounds.xextent + (buffer_size * 2)
            if bounds.yextent > 0:
                self.non_buff_grid_offsets[2:4] = buffer_size
                if self.local_bounds.ymin > bounds.ymin:
                    ymin = self.local_bounds.ymin - buffer_size
                # extent is x2 to create buffer on both sides
                yextent = self.local_bounds.yextent + (buffer_size * 2)
            if bounds.zextent > 0:
                self.non_buff_grid_offsets[4:6] = buffer_size
                if self.local_bounds.zmin > bounds.zmin:
                    zmin = self.local_bounds.zmin - buffer_size
                # extent is x2 to create buffer on both sides
                zextent = self.local_bounds.zextent + (buffer_size * 2)

        else:
            xmax = bounds.xmin + bounds.xextent
            if self.local_bounds.xmin > bounds.xmin:
                xmin = self.local_bounds.xmin - buffer_size
                self.non_buff_grid_offsets[0] = buffer_size
                if self.local_bounds.xmin + self.local_bounds.xextent < xmax:
                    # min is buffer_size less so need to increase extent to cover the space
                    xextent = self.local_bounds.xextent + (buffer_size * 2)
                    self.non_buff_grid_offsets[1] = buffer_size
                else:
                    xextent = self.local_bounds.xextent + buffer_size
            else:
                if self.local_bounds.xmin + self.local_bounds.xextent < xmax:
                    xextent = self.local_bounds.xextent + buffer_size
                    self.non_buff_grid_offsets[1] = buffer_size

            if bounds.yextent > 0:
                ymax = bounds.ymin + bounds.yextent
                if self.local_bounds.ymin > bounds.ymin:
                    ymin = self.local_bounds.ymin - buffer_size
                    self.non_buff_grid_offsets[2] = buffer_size
                    if self.local_bounds.ymin + self.local_bounds.yextent < ymax:
                        # min is buffer_size less so need to increase extent to cover the space
                        yextent = self.local_bounds.yextent + (buffer_size * 2)
                        self.non_buff_grid_offsets[3] = buffer_size
                    else:
                        yextent = self.local_bounds.yextent + buffer_size
                elif self.local_bounds.ymin + self.local_bounds.yextent < ymax:
                        yextent = self.local_bounds.yextent + buffer_size
                        self.non_buff_grid_offsets[3] = buffer_size
                else:
                    # topology is 1D
                    ymin = self.local_bounds.ymin
                    yextent = self.local_bounds.yextent

            if bounds.zextent > 0:
                zmax = bounds.zmin + bounds.zextent
                if self.local_bounds.zmin > bounds.zmin:
                    zmin = self.local_bounds.zmin - buffer_size
                    self.non_buff_grid_offsets[4] = buffer_size
                    if self.local_bounds.zmin + self.local_bounds.zextent < zmax:
                        # min is buffer_size less so need to increase extent to cover the space
                        zextent = self.local_bounds.zextent + (buffer_size * 2)
                        self.non_buff_grid_offsets[5] = buffer_size
                    else:
                        zextent = self.local_bounds.zextent + buffer_size
                else:
                    if self.local_bounds.zmin + self.local_bounds.zextent < zmax:
                        zextent = self.local_bounds.zextent + buffer_size
                        self.non_buff_grid_offsets[5] = buffer_size

        self.buffered_bounds = BoundingBox(xmin, xextent, ymin, yextent, zmin, zextent)

    def _init_sync_data_1d(self, topo, buffer_size):
        ngh_buffers = []
        # create send_counts, send_displs, and send_buf
        # buf_ngh: (rank, (ranges)) e.g., (1, (8, 10, 40, 42, 0, 0))
        r_trans = self.impl.pt_translation[0]
        for ngh_rank, bounds in topo.compute_buffer_nghs(buffer_size):
            # convert bounds into grid slices, row major'ing and
            # applying the pt translations.
            row = bounds[0] + r_trans
            # origin + extent
            shape = (bounds[1] - bounds[0],)
            stop_row = row + shape[0]
            ngh_buffers.append(
                (ngh_rank, shape[0], shape, bounds, (slice(row, stop_row),)))

        ngh_buffers.sort(key=lambda data: data[0])
        buf_dtype = self.grid[ngh_buffers[0][4]].numpy().dtype

        num_procs = self.cart_comm.Get_size()
        meta_data_send = np.zeros((num_procs, 7), dtype=np.int32)
        self.ngh_meta_data_recv = np.zeros((num_procs, 7), dtype=np.int32)

        for ngh_rank, _, buf_shape, bounds, _ in ngh_buffers:
            # send bounds in row major
            meta_data_send[ngh_rank, :] = [buf_shape[0]] + \
                [bounds[2], bounds[3], bounds[0], bounds[1], bounds[4], bounds[5]]

        self.cart_comm.Alltoall(meta_data_send, self.ngh_meta_data_recv)

        # shape element in first colum is the size
        send_counts = meta_data_send[:, 0]
        send_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(send_counts, dtype=np.int32)[:-1]))
        send_buf = np.empty(np.sum(send_counts), dtype=buf_dtype)
        self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)

        # shape element in first col
        recv_counts = self.ngh_meta_data_recv[:, 0]
        recv_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(recv_counts, dtype=np.int32)[:-1]))
        recv_buf = np.empty(np.sum(recv_counts), dtype=buf_dtype)
        recv_buf_data = []

        for ngh_rank, data in enumerate(self.ngh_meta_data_recv):
            # data: shape, bounds [ 2   0 30 0  0  0  0]
            if data[0] != 0:
                row = data[1] + r_trans
                # origin + extent
                stop_row = row + (data[2] - data[1])
                recv_buf_data.append((ngh_rank, data[0], data[0], (slice(row, stop_row),)))

        self.recv_data = (
            (recv_buf, (recv_counts, recv_displs)), recv_buf_data)

    def _init_sync_data_2d(self, topo, buffer_size):
        ngh_buffers = []
        # create send_counts, send_displs, and send_buf
        # buf_ngh: (rank, (ranges)) e.g., (1, (8, 10, 40, 42, 0, 0))
        c_trans = self.impl.pt_translation[0]
        r_trans = self.impl.pt_translation[1]
        for ngh_rank, bounds in topo.compute_buffer_nghs(buffer_size):
            # convert bounds into grid slices, row major'ing and
            # applying the pt translations.
            row = bounds[2] + r_trans
            col = bounds[0] + c_trans
            # origin + extent
            shape = (bounds[3] - bounds[2], bounds[1] - bounds[0])
            stop_row = row + shape[0]
            stop_col = col + shape[1]
            ngh_buffers.append(
                (ngh_rank, np.prod(shape), shape, bounds, (slice(row, stop_row), slice(col, stop_col))))

        ngh_buffers.sort(key=lambda data: data[0])
        buf_dtype = self.grid[ngh_buffers[0][4]].numpy().dtype

        num_procs = self.cart_comm.Get_size()
        meta_data_send = np.zeros((num_procs, 8), dtype=np.int32)
        self.ngh_meta_data_recv = np.zeros((num_procs, 8), dtype=np.int32)

        for ngh_rank, _, buf_shape, bounds, _ in ngh_buffers:
            # send bounds in row major
            meta_data_send[ngh_rank, : ] = [buf_shape[0], buf_shape[1]] + [bounds[2], bounds[3], bounds[0], bounds[1], bounds[4], bounds[5]]

        self.cart_comm.Alltoall(meta_data_send, self.ngh_meta_data_recv)
        
        # x the shapes to get count for each rank
        send_counts = np.apply_along_axis(lambda row: np.prod(row[:2]), 1, meta_data_send)
        send_displs = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(send_counts, dtype=np.int32)[:-1]))
        send_buf = np.empty(np.sum(send_counts), dtype=buf_dtype)
        self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)

        recv_counts = np.apply_along_axis(lambda row: row[0] * row[1], 1, self.ngh_meta_data_recv)
        recv_displs = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(recv_counts, dtype=np.int32)[:-1]))
        recv_buf = np.empty(np.sum(recv_counts), dtype=buf_dtype)
        recv_buf_data = []

        for ngh_rank, data in enumerate(self.ngh_meta_data_recv):
            # data: shape, row_major bounds [ 2 30  0 30 80 82  0  0]
            if data[0] != 0:
                row = data[2] + r_trans
                col = data[4] + c_trans
                # origin + extent
                stop_row = row + (data[3] - data[2])
                stop_col = col + (data[5] - data[4])
                recv_buf_data.append((ngh_rank, np.prod(data[:2]), data[:2], (slice(row, stop_row), slice(col, stop_col)))) #buf))
            
        self.recv_data = ((recv_buf, (recv_counts, recv_displs)), recv_buf_data)

    def _init_sync_data_3d(self, topo, buffer_size):
        ngh_buffers = []
        # create send_counts, send_displs, and send_buf
        # buf_ngh: (rank, (ranges)) e.g., (1, (8, 10, 40, 42, 0, 0))
        c_trans = self.impl.pt_translation[0]
        r_trans = self.impl.pt_translation[1]
        z_trans = self.impl.pt_translation[2]
        for ngh_rank, bounds in topo.compute_buffer_nghs(buffer_size):
            # convert bounds into grid slices, row major'ing and
            # applying the pt translations.
            row = bounds[2] + r_trans
            col = bounds[0] + c_trans
            z = bounds[4] + z_trans
            # origin + extent
            shape = (bounds[3] - bounds[2], bounds[1] - bounds[0], bounds[5] - bounds[4])
            stop_row = row + shape[0]
            stop_col = col + shape[1]
            stop_z = z + shape[2]
            ngh_buffers.append(
                (ngh_rank, np.prod(shape), shape, bounds, (slice(row, stop_row), slice(col, stop_col), slice(z, stop_z))))

        ngh_buffers.sort(key=lambda data: data[0])
        buf_dtype = self.grid[ngh_buffers[0][4]].numpy().dtype

        num_procs = self.cart_comm.Get_size()
        meta_data_send = np.zeros((num_procs, 9), dtype=np.int32)
        self.ngh_meta_data_recv = np.zeros((num_procs, 9), dtype=np.int32)

        for ngh_rank, _, buf_shape, bounds, _ in ngh_buffers:
            # send bounds in row major
            meta_data_send[ngh_rank, :] = [buf_shape[0], buf_shape[1], buf_shape[2]] + \
                [bounds[2], bounds[3], bounds[0], bounds[1], bounds[4], bounds[5]]

        self.cart_comm.Alltoall(meta_data_send, self.ngh_meta_data_recv)

        # x the shapes to get count for each rank
        send_counts = np.apply_along_axis(
            lambda row: np.prod(row[0:3]), 1, meta_data_send)
        send_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(send_counts, dtype=np.int32)[:-1]))
        send_buf = np.empty(np.sum(send_counts), dtype=buf_dtype)
        self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)

        recv_counts = np.apply_along_axis(
            lambda row: np.prod(row[0:3]), 1, self.ngh_meta_data_recv)
        recv_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(recv_counts, dtype=np.int32)[:-1]))
        recv_buf = np.empty(np.sum(recv_counts), dtype=buf_dtype)
        recv_buf_data = []

        for ngh_rank, data in enumerate(self.ngh_meta_data_recv):
            # data: shape, row_major bounds [ 2 30  0 30 80 82  0  0]
            if data[0] != 0:
                row = data[2] + r_trans
                col = data[4] + c_trans
                z = data[6] + z_trans
                # origin + extent
                stop_row = row + (data[3] - data[2])
                stop_col = col + (data[5] - data[4])
                stop_z = z + (data[7] - data[6])
                recv_buf_data.append((ngh_rank, np.prod(data[:3]), data[:3], (slice(
                    row, stop_row), slice(col, stop_col), slice(z, stop_z))))

        self.recv_data = (
            (recv_buf, (recv_counts, recv_displs)), recv_buf_data)

    def _init_sync_data(self, topo, buffer_size):
        dims = 1 if self.local_bounds.yextent == 0 else (3 if self.local_bounds.zextent > 0 else 2)
        if dims == 1:
            self._init_sync_data_1d(topo, buffer_size)
        elif dims == 2:
            self._init_sync_data_2d(topo, buffer_size)
        else:
            self._init_sync_data_3d(topo, buffer_size)

    def _pre_synch_ghosts(self, agent_manager: AgentManager):
        """Called prior to synchronizing ghosts and before any cross-rank movement
        synchronization.

        This is a null op on a value layer.

        Args:
            agent_manager: this rank's AgentManager
        """
        pass

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent: Callable):
        """Synchronizes the ghosted part of this projection

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        # self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)
        # self.recv_data = ((recv_buf, (recv_counts, recv_displs)), recv_buf_data)
        # ngh_buffers: (ngh_rank, size, shape, bounds, slice_tuple)
        # recv_buf_data: (ngh_rank, shape, slice_tuple)

        # fill send buffer from grid view
        send_msg, ngh_buffers = self.send_data
        send_displs = send_msg[1][1]
        send_buf = send_msg[0]
        for ngh_rank, size, _, _, tslice in ngh_buffers:
            disp = send_displs[ngh_rank]
            send_buf[disp: disp + size] = self.grid[tslice].numpy().reshape(size)

        self.cart_comm.Alltoallv(self.send_data[0], self.recv_data[0])

        # update grid view from recv_buffer
        recv_msg, recv_buf_data = self.recv_data
        recv_displs = recv_msg[1][1]
        recv_buf = recv_msg[0]
        for ngh_rank, size, shape, tslice in recv_buf_data:
            disp = recv_displs[ngh_rank]
            self.grid[tslice] = torch.as_tensor(
                recv_buf[disp: disp + size].reshape(shape))


class ReadWriteValueLayer:

    def __init__(self, comm, bounds, borders, buffer_size, init_value, dtype=torch.float64):
        self.read_layer = SharedValueLayer(comm, bounds, borders, buffer_size, init_value, dtype)
        self.write_layer = SharedValueLayer(comm, bounds, borders, buffer_size, init_value, dtype)

    def swap_layers(self):
        self.read_layer, self.write_layer = self.write_layer, self.read_layer

    def grid(self):
        return self.impl.grid

    @property
    def bounds(self):
        return self.read.bbounds

    def get(self, pt):
        return self.read.get(pt)

    def set(self, pt, val):
        self.write.set(pt, val)

    def __getitem__(self, key):
        return self.read.__getitem__(key)

    def __setitem__(self, key, val):
        self.write.__setitem__(key, val)

    def get_nghs(self, pt, extent=1):
        """ Gets the neighboring values and locations around the specified point.
        :param pt: the point whose neighbors to get
        :param extent: the extent of the neighborhood
        """
        return self.read.get_nghs(pt, extent)

    def _pre_synch_ghosts(self, agent_manager: AgentManager):
        """Called prior to synchronizing ghosts and before any cross-rank movement
        synchronization.

        Args:
            agent_manager: this rank's AgentManager
        """
        pass

    def _synch_ghosts(self, agent_manager: AgentManager, create_agent: Callable):
        """Synchronizes the ghosted part of this projection

        Args:
            agent_manager: this rank's AgentManager
            create_agent: a callable that can create an agent instance from
            an agent id and data.
        """
        self.write_layer._synch_ghosts()
        self.read_layer._synch_ghosts()

    def apply(self, func):
        func(self)
