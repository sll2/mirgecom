"""MPI helper functionality.

.. autofunction:: mpi_entry_point
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from functools import wraps
import os
import sys


def mpi_entry_point(func):
    """
    Return a decorator that designates a function as the "main" function for MPI.

    Declares that all MPI code that will be executed on the current process is
    contained within *func*. Calls `MPI_Init()`/`MPI_Init_thread()` and sets up a
    hook to call `MPI_Finalize()` on exit.
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if "mpi4py.run" not in sys.modules:
            raise RuntimeError("Must run MPI scripts via mpi4py (i.e., 'python -m "
                        "mpi4py <args>').")

        if "mpi4py.MPI" in sys.modules:
            raise RuntimeError("mpi4py.MPI imported before designated MPI entry "
                        "point. Check for prior imports.")

        # Avoid hwloc version conflicts by forcing pocl to load before mpi4py
        # (don't ask). See https://github.com/illinois-ceesd/mirgecom/pull/169
        # for details.
        import pyopencl as cl
        cl.get_platforms()

        # Avoid https://github.com/illinois-ceesd/mirgecom/issues/132 on
        # some MPI runtimes.
        import mpi4py
        mpi4py.rc.recv_mprobe = False

        # Runs MPI_Init()/MPI_Init_thread() and sets up a hook for MPI_Finalize() on
        # exit
        from mpi4py import MPI

        # This code warns the user of potentially slow startups due to file system
        # locking when running with large numbers of ranks. See
        # https://mirgecom.readthedocs.io/en/latest/running.html#running-with-large-numbers-of-ranks-and-nodes
        # for more details
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        if size > 1 and rank == 0 and "XDG_CACHE_HOME" not in os.environ:
            from warnings import warn
            warn("Please set the XDG_CACHE_HOME variable in your job script to "
                 "avoid file system overheads when running on large numbers of "
                 "ranks. See https://mirgecom.readthedocs.io/en/latest/running.html#running-with-large-numbers-of-ranks-and-nodes"  # noqa: E501
                 " for more information.")

        func(*args, **kwargs)

    return wrapped_func

class MPI_Info:
    """
    Holds relevant information for MPI communication including
    - MPI communicator
    - MPI datatype being sent
    - Flag for using CudaAware MPI
    """

    def __init__(self, comm=None, cflag=False):
        """
        """
        from mpi4py import MPI
        self.comm = comm
        if comm is None:
            self.comm = MPI.COMM_WORLD
        self.d_type = MPI.DOUBLE
        self.cuda_flag = cflag
    

class CommunicationProfile:
    """
    Holds communication profiling information
    """

    def __init__(self):
        """
        init_t : holds the amount of time spent in initializing sends and receives
        finish_t : holds the amount of time spent in waits and receiving data
        dev_copy_t : holds the amount of time spent copying data to and from the device for naive communication
        """
        self.init_t = 0.0
        self.finish_t = 0.0
        self.dev_copy_t = 0.0

        self.init_m = 0
        self.finish_m = 0
        self.dev_copy_m = 0
        
        self.init_avg = 0.0
        self.finish_avg = 0.0
        self.dev_copy_avg = 0.0

    def init_start(self):
        from mpi4py import MPI
        self.init_t -= MPI.Wtime()
        self.init_m += 1
    
    def init_stop(self):
        from mpi4py import MPI
        self.init_t += MPI.Wtime()
    
    def finish_start(self):
        from mpi4py import MPI
        self.finish_t -= MPI.Wtime()
        self.finish_m += 1

    def finish_stop(self):
        from mpi4py import MPI
        self.finish_t += MPI.Wtime()
    
    def dev_copy_start(self):
        from mpi4py import MPI
        self.dev_copy_t -= MPI.Wtime()
        self.dev_copy_m += 1
    
    def dev_copy_stop(self):
        from mpi4py import MPI
        self.dev_copy_t += MPI.Wtime()

    def average_profile(self):
        self.init_avg = self.init_t / self.init_m
        self.finish_avg = self.finish_t / self.finish_m
        self.dev_copy_avg = self.dev_copy_t / self.dev_copy_m

    def finalize(self):
        """
        Returns the entire comunication profile in 3 separate tuples
          totals = (total time spent in initializing messages,
                    total time spent in waiting and receiving messages,
                    total time spent moving data to and from the device for naive communication)
        messages = (total number of messages initialized,
                    total number of messages received,
                    total number of copies to or from the device)
        averages = (average time to initialize a message,
                    average time waiting and receiving messages,
                    average time copying data to and from device) 
        """
        totals = (self.init_t, self.finish_t, self.dev_copy_t)
        messages = (self.init_m, self.finish_m, self.dev_copy_m)
        averages = (self.init_avg, self.finish_avg, self.dev_copy_avg)
        return  totals, messages, averages 
