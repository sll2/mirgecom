from mpi4py import MPI

################################################
#
# CPU helper functions for sending and receiving
#
################################################
def _isend_cpu(mpi_communicator, actx, data_ary, data_ary_size, receiver_rank, Tag, profiler):
    """
    Returns :
        MPI send request
    Inputs :
             actx : meshmode array_context
         data_ary : data to be communicated -- in a format that the array context understands
    data_ary_size : number of values in data_ary to be communicated (not bytes)
    receiver_rank : MPI rank receiving data
              Tag : MPI communication tag 
    """
    if profiler:
        profiler.dev_copy_start()
    local_data = actx.to_numpy(data_ary)
    if profiler:
        profiler.dev_copy_stop()

    if profiler:
        profiler.init_start()
    Return_Request = mpi_communicator.Isend(local_data, receiver_rank, tag=Tag)
    if profiler:
        profiler.init_stop()
    
    return Return_Request 
    
def _irecv_cpu(mpi_communicator, actx, data_ary, data_ary_size, sender_rank, Tag, profiler):
    """
    Returns mpi recv request
    """
    if profiler:
        profiler.init_start()
    Return_Request = mpi_communicator.Irecv(data_ary, sender_rank, tag=Tag)
    if profiler:
        profiler.init_stop()

    return Return_Request 

def _wait_cpu(mpi_req, actx, data_ary, profiler):
    """
    Returns the data received in the data_ary position in a form that actx understands
        or
    Returns a None object if it's a send request being waited on
    """
    if profiler:
        profiler.finish_start()
    mpi_req.Wait()
    if profiler:
        profiler.finish_stop()

    if actx:
        if profiler:
            profiler.dev_copy_start()
        array = actx.from_numpy(data_ary)
        if profiler:
            profiler.dev_copy_stop()
        return array
    else:
        return None 

################################################
#
# GPU helper functions for sending and receiving
#
################################################
def _isend_gpu(mpi_communicator, actx, data_ary, data_ary_size, receiver_rank, Tag, profiler):
    """
    Returns mpi send request
    """
    if profiler:
        profiler.init_start()
    bdata = data_ary.base_data
    cl_mem = bdata.int_ptr
    bytes_size = data_ary_size * 8
    buf = utils.as_buffer(cl_mem, bytes_size, 0)
    
    Return_Request = mpi_communicator.Isend([buf, self.d_type], receiver_rank, tag=Tag)
    
    if profiler:
        profiler.init_stop()

    return Return_Request 

def _irecv_gpu(mpi_communicator, actx, data_ary, data_ary_size, sender_rank, Tag, profiler):
    """
    Returns mpi recv request
    """
    if profiler:
        profiler.init_start()
    bdata = data_ary.base_data
    cl_mem = bdata.int_ptr
    bytes_size = data_ary_size * 8
    buf = utils.as_buffer(cl_mem, bytes_size, 0)
    
    Return_Request = mpi_communicator.Irecv([buf, self.d_type], sender_rank, tag=Tag)
    
    if profiler:
        profiler.init_stop()

    return Return_Request 

def _wait_gpu(mpi_req, actx, data_ary, profiler):
    """
    Returns the data received in the data_ary position in a form that actx understands
        or
    Returns a None object if it's a send request being waited on
    """
    if profiler:
        profiler.finish_start()
    mpi_req.Wait()
    if profiler:
        profiler.finish_stop()

    if actx:
        return data_ary
    else:
        return None 

################################################
#
# Profiling object to contain all communication
# data
#
################################################
class CommunicationProfile:
    """
    Holds all communication profile data
    """

    def __init__(self):
        """
        Initializes communication profile object
        """
        # Variables to hold total times for sends and receives
        # as well as the data transfer costs encurred for communication
        self.init_t    = 0.0 
        self.finish_t  = 0.0 
        self.dev_cpy_t = 0.0

        # Variables to hold numbers of messages initialized and received
        # as well as the number of data copies to and from device
        self.init_m    = 0.0
        self.finish_m  = 0.0 
        self.dev_cpy_m = 0.0

        # Lists to contain ALL messages sizes for initialized and
        # received messages per rank
        self.init_msg_sizes   = []
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
        self.dev_cpy_t -= MPI.Wtime()
        self.dev_cpy_m += 1
    
    def dev_copy_stop(self):
        from mpi4py import MPI
        self.dev_cpy_t += MPI.Wtime()

    def average(self):
        """
        Returns profiling data averages in a tuple of the form:
        ( init_avg, finish_avg, dev_avg )
        
          init_avg : initialization time average, 
        finish_avg : finishing communication time average,
           dev_avg : average amount of time spent copying data to/from device
        """

        init_avg     = self.init_t / self.init_m
        finish_avg   = self.finish_t / self.finish_m
        dev_copy_avg = self.dev_cpy_t / self.dev_cpy_m

        return (init_avg, finish_avg, dev_copy_avg)

    def finalize(self):
        """
        Finalizes profiling data
        """
        return

    def print_profile(self):
        """
        Formatted print of profiling data
        """
        from mpi4py import MPI
        rank = MPI.Get_rank()
        print(F'------------------Process {rank:4d}------------')
        print(F'Init Total Time {self.init_t:.5f}')
        print(F'Init Messages {self.init_m:4d}')
        print(F'Finish Total Time {self.finish_t:.5f}')
        print(F'Finish Messages {self.finish_m:4d}')
        print(F'Device Copy Total Time {self.dev_cpy_t:.5f}')
        print(F'Device Copies {self.dev_cpy_m:4d}')

################################################
#
# Communicator object to hold which sending
# and receiving functions to use 
#
################################################
class Communicator:
    """
    Communication class
    actx : meshmode array_context
    """

    def __init__(self, comm=None, cflag=False, profile=True):
        """
        Initialization function
        """
        self.mpi_communicator = comm
        if comm is None:
            self.mpi_communicator = MPI.COMM_WORLD

        self.d_type       = MPI.DOUBLE # The MPI datatype being communicated
        self.cuda_flag    = cflag      # Whether the MPI is CUDA-Aware and running on Nvidia GPU
        self.comm_profile = None       # Communication profile is not initialized unless profile
                                       # flag is set
        
        # Initialize communication routines to be performed via CPU communication
        self.isend = _isend_cpu
        self.irecv = _irecv_cpu
        self.wait  = _wait_cpu

        # If cuda flag set, then update communicatoin routines to use
        # CUDA-Aware MPI calls
        if self.cuda_flag:
            self.isend = _isend_gpu
            self.irecv = _irecv_gpu
            self.wait  = _wait_gpu

        # Create CommunicationProfile object
        if profile:
            self.comm_profile = CommunicationProfile()

    def Isend(self, actx, data_ary, data_ary_size, receiver_rank, Tag):
        """
                 actx : meshmode array_context
             data_ary : data to be communicated -- in a format that the array context understands
        data_ary_size : number of values in data_ary to be sent (not bytes)
        receiver_rank : MPI rank receiving data
                  Tag : MPI communication tag 
        """
        return self.isend(self.mpi_communicator, actx, data_ary, data_ary_size, receiver_rank, Tag, self.comm_profile)

    def Irecv(self, actx, data_ary, data_ary_size, sender_rank, Tag):
        """
                 actx : meshmode array_context
             data_ary : aray for data to be received into -- in a format that the array context understands
        data_ary_size : number of values in data_ary to be received (not bytes)
          sender_rank : MPI rank sending data
                  Tag : MPI communication tag 
        """
        return self.irecv(self.mpi_communicator, actx, data_ary, data_ary_size, sender_rank, Tag, self.comm_profile)

    def Wait(self, mpi_req, actx=None, data_ary=None):
        # If it's a recv req, return recv_req, if it's a send req, then return send req
        return self.wait(mpi_req, actx, data_ary, self.comm_profile)
