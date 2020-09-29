
from __future__ import division, print_function
from collections import namedtuple
import os
import shutil
import tempfile
import uuid

from concorde._concorde import _CCutil_gettsplib, _CCtsp_solve_dat, _CCtsp_solve_sparse
from concorde.util import write_tsp_file, EDGE_WEIGHT_TYPES

ComputedTour = namedtuple('ComputedTour', [
    'tour', 'optimal_value', 'success', 'found_tour', 'hit_timebound'])


class TSPSolver(object):

    def __init__(self):
        self._data = None
        self._dis = None
        self._ncount = -1

    @classmethod
    def from_tspfile(cls, fname):
        ncount, data = _CCutil_gettsplib(fname)
        if data is None:
            raise RuntimeError("Error in loading {}".format(fname))
        self = cls()
        self._ncount = ncount
        self._data = data
        return self

    @classmethod
    def from_data(cls, xs, ys, norm, name=None):
        """ Construct data group from given data.
        This routine writes the given data to a temporary file, and then uses
        Concorde's file parser to read from file and do the initialization.
        """
        if norm not in EDGE_WEIGHT_TYPES:
            raise ValueError("norm must be one of {} but got {!r}".format(
                             ', '.join(EDGE_WEIGHT_TYPES), norm))

        # TODO: properly figure out Concorde's CCdatagroup format and
        # initialize this object directly instead of going via file.
        if name is None:
            name = uuid.uuid4().hex
        try:
            ccdir = tempfile.mkdtemp()
            ccfile = os.path.join(ccdir, 'data.tsp')
            with open(ccfile, 'w') as fp:
                write_tsp_file(fp, xs, ys, norm, name)
            return cls.from_tspfile(ccfile)
        finally:
            shutil.rmtree(ccdir)

    @property
    def x(self):
        return self._data.x

    @property
    def y(self):
        return self._data.y

    @property
    def z(self):
        return self._data.z

    def __str__(self):
        if self._data is None:
            return "Uninitialized TSPSolver"
        else:
            return "TSPSolver with {} nodes".format(self._ncount)

    @classmethod
    def from_matrix(cls, dis):
        self = cls()
        self._ncount = int(dis.shape[0])
        self._dis = dis
        return self

    def solve(self,time_bound=-1, verbose=True, random_seed=0):
        if self._dis is not None:
            res = _CCtsp_solve_sparse(self._ncount, self._dis,"name", time_bound)
        else:
            res = _CCtsp_solve_dat(
                self._ncount, self._data, "TSP_Concorde",
                time_bound, not verbose, random_seed)

        return ComputedTour(*res)
