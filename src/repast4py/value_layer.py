import torch
import numpy as np

from typing import Callable

from . import geometry

from .space import BorderType, GridStickyBorders, GridPeriodicBorders, CartesianTopology, BoundingBox
from .space import DiscretePoint as dpt
from .space import DiscretePoint

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
        self.pt_translation = np.array([0 - bounds.xmin, 0, 0])
        self.ngh_translation = 0 - bounds.xmin
        self.translation = np.array([self.ngh_translation, 0, 0])
        self.ngh_finder = geometry.find_1d_nghs_sticky if borders == BorderType.Sticky else geometry.find_1d_nghs_periodic
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1])

    def get(self, pt: DiscretePoint):
        """Gets the value at the specified point

        Args:
            pt: the location to get the value of
        """
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.translation
        return self.grid[npt[0]]

    def set(self, pt: DiscretePoint, val):
        """Sets the value at the specified location

        Args:
            pt: the location to set the value of
            val: the value to set the location to
        """
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.translation
        self.grid[npt[0]] = val

    def get_nghs(self, pt: DiscretePoint):
        """Gets the neighboring values and locations around the specified point.

        Args:
            pt: the point whose neighbors to get
        Returns:
            The value at the specicifed location.
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max)
        grid_idxs = nghs + self.ngh_translation
        return (self.grid[grid_idxs], nghs)

    def _compute_slice(self, key):
        if isinstance(key, tuple):
            slc = key[0]
            print(slc)
            step = 1 if slc.step is None else slc.step
            start = 0 if slc.start is None else (slc.start + self.translation[0])
            stop = self.grid.shape[0] if slc.stop is None else (slc.stop + self.translation[0])
            return (slice(start, stop, step),)

        elif isinstance(key, slice):
            print(key)
            step = 1 if key.step is None else key.step
            start = 0 if key.start is None else (key.start + self.translation[0])
            stop = self.grid.shape[0] if key.stop is None else (key.stop + self.translation[0])
            return (slice(start, stop, step),)

        else:
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
        """Updates in the internal indexing array from the specified point,
        converting from x,y,z order to pytorch order (z,y,x)
        """
        self.idx[0, 0] = pt[1]
        self.idx[1, 0] = pt[0]

        return self.idx

    def get(self, pt: DiscretePoint):
        """Gets the value at the specified point

        Args:
            pt: the location to get the value of
        """
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        return self.grid[self._update_idx(npt)]

    def set(self, pt: DiscretePoint, val):
        """Sets the value at the specified location

        Args:
            pt: the location to set the value of
            val: the value to set the location to
        Returns:
            The value at the specicifed location.
        """
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        self.grid[self._update_idx(npt)] = val

    def get_nghs(self, pt: DiscretePoint):
        """ Gets the neighboring values and locations around the specified point.

        Args:
            pt: the point whose neighbors to get
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max, pytorch_order=True)
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

                return (slice(rstart, rstop, rstep), slice(cstart, cstop, cstep))

        elif isinstance(key, slice):
            step = 1 if key.step is None else key.step
            start = 0 if key.start is None else (key.start + self.pt_translation[0])
            stop = self.grid.shape[1] if key.stop is None else (key.stop + self.pt_translation[0])
            return (slice(None, None), slice(start, stop, step))

        else:
            return (slice(None, None), slice(key + self.pt_translation[0], None))

    def __getitem__(self, key):
        return self.grid[self._compute_slice(key)]

    def __setitem__(self, key, val):
        self.grid[self._compute_slice(key)] = val


class Impl3D:
    # Torch 3D tensor of shape (3, 4, 2)
    # is 3 arrays of 4 rows and 2 columns:
    # thus (z, y, x)

    def __init__(self, bounds, borders, init_value, dtype):
        self.tpt = dpt(0, 0, 0)
        self.idx = np.array([[0], [0], [0]])
        if borders == BorderType.Sticky:
            self.borders = GridStickyBorders(bounds)
        elif borders == BorderType.Periodic:
            self.borders = GridPeriodicBorders(bounds)

        if init_value == 'random':
            self.grid = torch.rand(
                (bounds.zextent, bounds.yextent, bounds.xextent), dtype=dtype)
        else:
            self.grid = torch.full(
                (bounds.zextent, bounds.yextent, bounds.xextent), init_value, dtype=dtype)

        self.pt_translation = np.array([0 - bounds.xmin, 0 - bounds.ymin, 0 - bounds.zmin])
        # pytorch z,y,z order
        self.ngh_translation = np.array([[self.pt_translation[2]], [self.pt_translation[1]], [self.pt_translation[0]]])
        self.ngh_finder = geometry.find_3d_nghs_sticky if borders == BorderType.Sticky else geometry.find_3d_nghs_periodic
        self.min_max = np.array(
            [bounds.xmin, bounds.xmin + bounds.xextent - 1,
             bounds.ymin, bounds.ymin + bounds.yextent - 1,
             bounds.zmin, bounds.zmin + bounds.zextent - 1])

    def _update_idx(self, pt):
        """Updates in the internal indexing array from the specified point,
        converting from x,y,z order to pytorch order (z,y,x)
        """
        # Torch 3D tensor of shape (3, 4, 2)
        # is 3 arrays of 4 rows and 2 columns:
        # thus (z, y, x)
        self.idx[0, 0] = pt[2]
        self.idx[1, 0] = pt[1]
        self.idx[2, 0] = pt[0]

        return self.idx

    def get(self, pt: DiscretePoint):
        """Gets the value at the specified point

        Args:
            pt: the location to get the value of
        """
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        return self.grid[self._update_idx(npt)]

    def set(self, pt: DiscretePoint, val):
        """Sets the value at the specified location

        Args:
            pt: the location to set the value of
            val: the value to set the location to
        """
        self.borders._transform(pt, self.tpt)
        # npt is a np array
        npt = self.tpt.coordinates + self.pt_translation
        self.grid[self._update_idx(npt)] = val

    def get_nghs(self, pt: DiscretePoint):
        """Gets the neighboring values and locations around the specified point.

        Args:
            pt: the point whose neighbors to get
        Returns:
            The value at the specicifed location.
        """
        nghs = self.ngh_finder(pt.coordinates, self.min_max, pytorch_order=True)
        grid_idxs = nghs + self.ngh_translation
        # swap to reverse tensor (z,y,x) order
        nghs[[0, 1, 2]] = nghs[[2, 1, 0]]
        return (self.grid[grid_idxs], nghs)

    def _compute_slice(self, key):
        if isinstance(key, tuple):
            if len(key) == 1:
                slc = key[0]
                step = 1 if slc.step is None else slc.step
                start = 0 if slc.start is None else (slc.start + self.pt_translation[0])
                stop = self.grid.shape[1] if slc.stop is None else (slc.stop + self.pt_translation[0])
                return (slice(None, None), slice(None, None), slice(start, stop, step))

            elif (len(key) == 2):
                cslice = key[0]
                cstep = 1 if cslice.step is None else cslice.step
                cstart = 0 if cslice.start is None else (cslice.start + self.pt_translation[0])
                cstop = self.grid.shape[1] if cslice.stop is None else (cslice.stop + self.pt_translation[0])

                rslice = key[1]
                rstep = 1 if rslice.step is None else rslice.step
                rstart = 0 if rslice.start is None else (rslice.start + self.pt_translation[1])
                rstop = self.grid.shape[0] if rslice.stop is None else (rslice.stop + self.pt_translation[1])

                return (slice(None, None), slice(rstart, rstop, rstep), slice(cstart, cstop, cstep))

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

                return (slice(zstart, zstop, zstep), slice(rstart, rstop, rstep), slice(cstart, cstop, cstep))

        elif isinstance(key, slice):
            step = 1 if key.step is None else key.step
            start = 0 if key.start is None else (key.start + self.pt_translation[0])
            stop = self.grid.shape[1] if key.stop is None else (key.stop + self.pt_translation[0])
            return (slice(None, None), slice(None, None), slice(start, stop, step))

        else:
            return (slice(None, None), slice(None, None), slice(key + self.pt_translation[0], None))

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

    def get(self, pt: DiscretePoint):
        """Gets the value at the specified point

        Args:
            pt: the location to get the value of
        """
        return self.impl.get(pt)

    def set(self, pt: DiscretePoint, val):
        """Sets the value at the specified location

        Args:
            pt: the location to set the value of
            val: the value to set the location to
        """
        self.impl.set(pt, val)

    # def __getitem__(self, key):
    #     return self.impl.__getitem__(key)

    # def __setitem__(self, key, val):
    #     self.impl.__setitem__(key, val)

    def get_nghs(self, pt: DiscretePoint):
        """Gets the neighboring values and locations around the specified point.

        Args:
            pt: the point whose neighbors to get
        """
        return self.impl.get_nghs(pt)


def _compute_meta_data_counts(meta_data, num_dims, offset, num_slices):
    # np.apply_along_axis(lambda row: np.prod(row[:2]) + np.prod(row[6:8]), 1, meta_data_send)
    if num_slices == 1:
        return np.apply_along_axis(lambda row: np.prod(row[:num_dims]), 1, meta_data)
    elif num_slices == 2:
        # return meta_data[:, 0] + meta_data[:, offset]
        return np.apply_along_axis(
            lambda row: np.prod(row[:num_dims]) + np.prod(row[offset: offset + num_dims]), 1, meta_data)
    elif num_slices == 3:
        return np.apply_along_axis(
            lambda row: np.prod(row[:num_dims]) +
            np.prod(row[offset: offset + num_dims]) +
            np.prod(row[offset * 2: offset * 2 + num_dims]),
            1, meta_data)
    elif num_slices == 4:
        return np.apply_along_axis(
            lambda row: np.prod(row[:num_dims]) +
            np.prod(row[offset: offset + num_dims]) +
            np.prod(row[offset * 2: offset * 2 + num_dims]) +
            np.prod(row[offset * 3: offset * 3 + num_dims]),
            1, meta_data)


class SharedValueLayer(ValueLayer):

    def __init__(self, comm, bounds, borders, buffer_size, init_value, dtype=torch.float64):
        self.periodic = borders == BorderType.Periodic
        topo = CartesianTopology(comm, bounds, self.periodic)
        self.cart_comm = topo.comm
        self.rank = self.cart_comm.Get_rank()
        self.coords = topo.coordinates
        self.buffer_size = buffer_size
        self.full_bounds = bounds

        # add buffer to local bounds
        self.local_bounds = topo.local_bounds
        if comm.Get_size() > 1:
            self._init_buffered_bounds(bounds, buffer_size, self.periodic)
        else:
            self.buffered_bounds = self.local_bounds
            self.non_buff_grid_offsets = np.zeros(6, dtype=np.int32)

        super().__init__(self.buffered_bounds, borders, 'random', dtype)

        nd = 1 if bounds.yextent == 0 else (3 if bounds.zextent > 0 else 2)
        mins = [bounds.xmin, bounds.ymin, bounds.zmin]
        if self.periodic:
            for i in range(0, nd):
                if self.impl.pt_translation[i] == 0:
                    self.impl.pt_translation[i] = buffer_size - mins[i]
        self._init_value(init_value, nd)

        if comm.Get_size() > 1:
            self._init_sync_data(nd, topo, buffer_size)

    def _init_value(self, init_value, nd: int):
        if init_value != 'random':
            lb = self.local_bounds
            x1 = lb.xmin + self.impl.pt_translation[0]
            x2 = lb.xmin + lb.xextent + self.impl.pt_translation[0]

            if nd == 1:
                self.grid[x1: x2] = init_value
            else:
                y1 = lb.ymin + self.impl.pt_translation[1]
                y2 = lb.ymin + lb.yextent + self.impl.pt_translation[1]
                if nd == 2:
                    self.grid[y1: y2, x1: x2] = init_value
                else:
                    z1 = lb.zmin + self.impl.pt_translation[2]
                    z2 = lb.zmin + lb.zextent + self.impl.pt_translation[2]
                    self.grid[z1: z2, y1: y2, x1: x2] = init_value

    def _wrap_slice_vals(self, start_val, end_val, extent):
        if start_val < 0:
            start_val += extent
            end_val += extent
        elif start_val >= extent:
            start_val -= extent
            end_val -= extent

        return (start_val, end_val)

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

        # bounds including the buffer if necessary
        self.buffered_bounds = BoundingBox(xmin, xextent, ymin, yextent, zmin, zextent)

    def _init_sync_data_1d(self, topo, buffer_size):
        ngh_buffers = []
        # create send_counts, send_displs, and send_buf
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

        data_size = 3 if topo.procs_per_dim[0] != 2 and self.periodic else 6

        num_procs = self.cart_comm.Get_size()
        meta_data_send = np.zeros((num_procs, data_size), dtype=np.int32)
        self.ngh_meta_data_recv = np.zeros((num_procs, data_size), dtype=np.int32)

        offset = 3
        num_sent_slices = int(data_size / offset)
        offsets = [0] * num_procs
        for ngh_rank, _, buf_shape, bounds, _ in ngh_buffers:
            s_idx = offsets[ngh_rank]
            meta_data_send[ngh_rank, s_idx: s_idx + offset] = [buf_shape[0]] + [bounds[0], bounds[1]]
            offsets[ngh_rank] += offset

        self.cart_comm.Alltoall(meta_data_send, self.ngh_meta_data_recv)

        # shape element in first and 3rd column is the size
        # send_counts = meta_data_send[:, 0] + meta_data_send[:, 3]
        send_counts = _compute_meta_data_counts(meta_data_send, 1, 3, num_sent_slices)
        send_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(send_counts, dtype=np.int32)[:-1]))
        send_buf = np.empty(np.sum(send_counts), dtype=buf_dtype)
        self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)

        # shape element in first col
        # recv_counts = self.ngh_meta_data_recv[:, 0] + self.ngh_meta_data_recv[:, 3]
        recv_counts = _compute_meta_data_counts(self.ngh_meta_data_recv, 1, 3, num_sent_slices)
        recv_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(recv_counts, dtype=np.int32)[:-1]))
        recv_buf = np.empty(np.sum(recv_counts), dtype=buf_dtype)
        recv_buf_data = []

        for ngh_rank, data in enumerate(self.ngh_meta_data_recv):
            slice_data = []
            if data[0] != 0:
                row = data[1] + r_trans
                # origin + extent
                stop_row = row + (data[2] - data[1])
                row, stop_row = self._wrap_slice_vals(row, stop_row, self.full_bounds.xextent)
                slice_data.append((ngh_rank, data[0], data[0], (slice(row, stop_row),)))

                for i in range(1, num_sent_slices):
                    idx = offset * i
                    if data[idx] != 0:
                        row = data[idx + 1] + r_trans
                        # origin + extent
                        stop_row = row + (data[idx + 2] - data[idx + 1])
                        row, stop_row = self._wrap_slice_vals(row, stop_row, self.full_bounds.xextent)
                        slice_data.append((ngh_rank, data[idx], data[idx], (slice(row, stop_row),)))
                    else:
                        break
            if len(slice_data) > 0:
                recv_buf_data.append(slice_data)

        self.recv_data = (
            (recv_buf, (recv_counts, recv_displs)), recv_buf_data)

    def _init_sync_data_2d(self, topo, buffer_size):
        ngh_buffers = []
        c_trans = self.impl.pt_translation[0]
        r_trans = self.impl.pt_translation[1]
        # Compute what to send to which neighbor
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

        # sort by rank of receiving neighbor
        ngh_buffers.sort(key=lambda data: data[0])
        buf_dtype = self.grid[ngh_buffers[0][4]].numpy().dtype
        num_procs = self.cart_comm.Get_size()

        offset = 6
        data_size = offset
        if self.periodic:
            if topo.procs_per_dim[0] == 2 and topo.procs_per_dim[1] == 2:
                data_size = 24
            elif topo.procs_per_dim[0] == 2 or topo.procs_per_dim[1] == 2:
                data_size = 12

        # send buffer for meta data (bounds etc.)
        meta_data_send = np.zeros((num_procs, data_size), dtype=np.int32)
        # recv buffer for meta
        self.ngh_meta_data_recv = np.zeros((num_procs, data_size), dtype=np.int32)

        offsets = [0] * num_procs
        for ngh_rank, _, buf_shape, bounds, _ in ngh_buffers:
            s_idx = offsets[ngh_rank]
            # send bounds in row major
            meta_data_send[ngh_rank, s_idx: s_idx + offset] = [buf_shape[0], buf_shape[1]] + [bounds[2], bounds[3], bounds[0], bounds[1]]
            offsets[ngh_rank] += offset

        self.cart_comm.Alltoall(meta_data_send, self.ngh_meta_data_recv)

        # * the shapes to get count for each rank
        num_sent_slices = int(data_size / offset)
        send_counts = _compute_meta_data_counts(meta_data_send, 2, offset, num_sent_slices)
        # send_counts = np.apply_along_axis(lambda row: np.prod(row[:2]) + np.prod(row[6:8]), 1, meta_data_send)
        send_displs = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(send_counts, dtype=np.int32)[:-1]))
        send_buf = np.empty(np.sum(send_counts), dtype=buf_dtype)
        self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)

        recv_counts = _compute_meta_data_counts(self.ngh_meta_data_recv, 2, offset, num_sent_slices)
        # np.apply_along_axis(lambda row: row[0] * row[1] + row[6] * row[7], 1, self.ngh_meta_data_recv)
        recv_displs = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(recv_counts, dtype=np.int32)[:-1]))
        recv_buf = np.empty(np.sum(recv_counts), dtype=buf_dtype)
        recv_buf_data = []

        for ngh_rank, data in enumerate(self.ngh_meta_data_recv):
            # data: shape, pytorch_ordered bounds [ 2 30  0 30 80 82  0  0]
            slice_data = []
            if data[0] != 0:
                row = data[2] + r_trans
                col = data[4] + c_trans
                # origin + extent
                stop_row = row + (data[3] - data[2])
                stop_col = col + (data[5] - data[4])

                row, stop_row = self._wrap_slice_vals(row, stop_row, self.full_bounds.yextent)
                col, stop_col = self._wrap_slice_vals(col, stop_col, self.full_bounds.xextent)
                slice_data.append((ngh_rank, np.prod(data[:2]), data[:2], (slice(row, stop_row), slice(col, stop_col))))

                for i in range(1, num_sent_slices):
                    idx = offset * i
                    if data[idx] != 0:
                        row = data[idx + 2] + r_trans
                        col = data[idx + 4] + c_trans
                        # origin + extent
                        stop_row = row + (data[idx + 3] - data[idx + 2])
                        stop_col = col + (data[idx + 5] - data[idx + 4])

                        row, stop_row = self._wrap_slice_vals(row, stop_row, self.full_bounds.yextent)
                        col, stop_col = self._wrap_slice_vals(col, stop_col, self.full_bounds.xextent)
                        slice_data.append((ngh_rank, np.prod(data[idx: idx + 2]), data[idx: idx + 2],
                                          (slice(row, stop_row), slice(col, stop_col))))
                    else:
                        break

            if len(slice_data) > 0:
                recv_buf_data.append(slice_data)

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
        meta_data_send = np.zeros((num_procs, 18), dtype=np.int32)
        self.ngh_meta_data_recv = np.zeros((num_procs, 18), dtype=np.int32)

        for ngh_rank, _, buf_shape, bounds, _ in ngh_buffers:
            s_idx = 0 if meta_data_send[ngh_rank, 0] == 0 else 9
            # send bounds in row major
            meta_data_send[ngh_rank, s_idx: s_idx + 9] = [buf_shape[0], buf_shape[1], buf_shape[2]] + \
                bounds[4], bounds[5], [bounds[2], bounds[3], bounds[0], bounds[1]]

        self.cart_comm.Alltoall(meta_data_send, self.ngh_meta_data_recv)

        # x the shapes to get count for each rank
        send_counts = np.apply_along_axis(
            lambda row: np.prod(row[0:3] + np.prod(row[9:12])), 1, meta_data_send)
        send_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(send_counts, dtype=np.int32)[:-1]))
        send_buf = np.empty(np.sum(send_counts), dtype=buf_dtype)
        self.send_data = ((send_buf, (send_counts, send_displs)), ngh_buffers)

        recv_counts = np.apply_along_axis(
            lambda row: np.prod(row[0:3] + np.prod(row[9:12])), 1, self.ngh_meta_data_recv)
        recv_displs = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(recv_counts, dtype=np.int32)[:-1]))
        recv_buf = np.empty(np.sum(recv_counts), dtype=buf_dtype)
        recv_buf_data = []

        for ngh_rank, data in enumerate(self.ngh_meta_data_recv):
            # data: shape, pytorch_order bounds [ 2 30  0 30 80 82  0  0]
            # TODO check order -- is this z,y,x?
            slice_data = []
            if data[0] != 0:
                z = data[2] + z_trans
                row = data[4] + r_trans
                col = data[6] + c_trans

                # origin + extent
                stop_z = z + (data[3] - data[2])
                stop_row = row + (data[5] - data[4])
                stop_col = col + (data[7] - data[6])

                row, stop_row = self._wrap_slice_vals(row, stop_row, self.full_bounds.yextent)
                col, stop_col = self._wrap_slice_vals(col, stop_col, self.full_bounds.xextent)
                z, stop_z = self._wrap_slice_vals(z, stop_z, self.full_bounds.zextent)
                slice_data.append((ngh_rank, np.prod(data[:3]), data[:3], (slice(
                    row, stop_row), slice(col, stop_col), slice(z, stop_z))))

                if data[9] != 0:
                    z = data[11] + z_trans
                    row = data[13] + r_trans
                    col = data[15] + c_trans

                    # origin + extent
                    stop_z = z + (data[12] - data[11])
                    stop_row = row + (data[14] - data[13])
                    stop_col = col + (data[16] - data[15])

                    row, stop_row = self._wrap_slice_vals(row, stop_row, self.full_bounds.yextent)
                    col, stop_col = self._wrap_slice_vals(col, stop_col, self.full_bounds.xextent)
                    z, stop_z = self._wrap_slice_vals(z, stop_z, self.full_bounds.zextent)
                    slice_data.append((ngh_rank, np.prod(data[9:12]), data[9:12], (slice(
                        row, stop_row), slice(col, stop_col), slice(z, stop_z))))

            if len(slice_data) > 0:
                recv_buf_data.append(slice_data)

        self.recv_data = (
            (recv_buf, (recv_counts, recv_displs)), recv_buf_data)

    def _init_sync_data(self, num_dims, topo, buffer_size):
        if num_dims == 1:
            self._init_sync_data_1d(topo, buffer_size)
        elif num_dims == 2:
            self._init_sync_data_2d(topo, buffer_size)
        else:
            self._init_sync_data_3d(topo, buffer_size)

    def _synch_ghosts(self):
        """Synchronizes the ghosted buffer part of this value layer.
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
        for items in recv_buf_data:
            for ngh_rank, size, shape, tslice in items:
                try:
                    disp = recv_displs[ngh_rank]
                    self.grid[tslice] = torch.as_tensor(
                        recv_buf[disp: disp + size].reshape(shape))

                except Exception:
                    print(self.rank, ngh_rank, size, shape, tslice)
                    raise


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
