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

class Communicator(self, comm=None, cflag=False, profile=False):
    """
    Communication class holds relevant information for MPI communication
    - MPI communicator - defaults to MPI.COMM_WORLD
    - MPI datatype being sent
    - Flag for using CUDAAware MPI
    - Profiling flag
    """
    def __init__(self, comm=None, cflag=False):
        """
        """
        from mpi4py import MPI
        self.mpi_communicator = comm
        if comm is None:
            self.mpi_communicator = MPI.COMM_WORLD

        self.d_type    = MPI.DOUBLE
        self.cuda_flag = cflag
        self.profiling = profile
        self.isend = _isend_cpu
        self.irecv = _irecv_cpu
        self.send_wait = _send_wait_cpu
        self.recv_wait = _recv_wait_gpu

        if self.cuda_flag:
            self.isend = _isend_gpu
            self.irecv = _irecv_gpu
            self.send_wait = _send_wait_cpu
            self.recv_wait = _recv_wait_cpu

    def ISend(self, actx, gpu_ary, data_ary_size, receiver_rank, Tag):
        return self.isend(actx, gpu_ary, data_ary_size, receiver_rank, Tag)

    def IRecv(self, actx, gpu_ary, data_ary_size, sender_rank, Tag):
        return self.irecv(actx, gpu_ary, data_ary_size, receiver_rank, Tag)

    def _isend_cpu(self, actx, data_ary, data_ary_size, receiver_rank, Tag):
        local_data = actx.to_numpy(data_ary)
        return self.mpi_communicator.Isend(local_data, receiver_rank, tag=Tag)
    
    def _irecv_cpu(self, actx, data_ary, data_ary_size, receiver_rank, Tag):
        local_data = actx.to_numpy(data_ary)
        return self.mpi_communicator.Isend(local_data, receiver_rank, tag=Tag)
    
    def _isend_gpu(self, actx, data_ary, data_ary_size, receiver_rank, Tag):
        bdata = data_ary.base_data
        cl_mem = bdata.int_ptr
        bytes_size = data_ary_size * 8
        buf = cacl.as_buffer(cl_mem, bytes_size, 0)

        return self.mpi_communicator.Isend([buf, self.d_type], receiver_rank, tag=Tag)
    
    def _irecv_gpu(self, actx, data_ary, data_ary_size, sender_rank, Tag):
        bdata = data_ary.base_data
        cl_mem = bdata.int_ptr
        bytes_size = data_ary_size * 8
        buf = cacl.as_buffer(cl_mem, bytes_size, 0)

        return self.mpi_communicator.Irecv([buf, self.d_type], sender_rank, tag=Tag)
        
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

        self.init_msg_sizes = []
        self.finish_msg_sizes = []

    def init_start(self, msg_size=None):
        from mpi4py import MPI
        self.init_t -= MPI.Wtime()
        self.init_m += 1
        if msg_size:
            self.init_msg_sizes.append(msg_size)
    
    def init_stop(self):
        from mpi4py import MPI
        self.init_t += MPI.Wtime()
    
    def finish_start(self, msg_size=None):
        from mpi4py import MPI
        self.finish_t -= MPI.Wtime()
        self.finish_m += 1
        if msg_size:
            self.finish_msg_sizes.append(msg_size)

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

    def print_profile(self):
        """
        Formatted print of profiling totals
        """
