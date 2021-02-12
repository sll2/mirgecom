"""Demonstrate wave-eager MPI example."""

__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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


import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl

from pytools.obj_array import flat_obj_array

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.mpi import mpi_entry_point
from mirgecom.mpi import Communicator 
from mirgecom.mpi import CommunicationProfile 
from mirgecom.integrators import rk4_step
from mirgecom.wave import wave_operator
from mirgecom.profiling import PyOpenCLProfilingArrayContext

import pyopencl.tools as cl_tools
import pyopencl as pycl


def bump(actx, discr, t=0):
    """Create a bump."""
    source_center = np.array([0.2, 0.35, 0.1])[:discr.dim]
    source_width = 0.05
    source_omega = 3

    nodes = thaw(actx, discr.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


@mpi_entry_point
def main():
    """Drive the example."""
    profiling = False 
    cl_ctx = cl.create_some_context()
    if profiling:
        queue = cl.CommandQueue(cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()
    CommProf = CommunicationProfile(comm)
    Comm = Communicator(comm)
    print("%d num procs" % num_parts)
    print("Device ", pycl.Device.hashable_model_and_version_identifier)

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(Comm.mpi_communicator)

    dim = 2
    nel_1d = 16

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim)

        print("%d elements" % mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=Comm)

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    fields = flat_obj_array(
        bump(actx, discr),
        [discr.zeros(actx) for i in range(discr.dim)])

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    rank = comm.Get_rank()

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        #if istep == 10:
        #    from pyinstrument import Profiler
        #    profiler = Profiler()
        #    profiler.start()
        #    if profiling:
        #        ignore = actx.tabulate_profiling_data()

        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            print(istep, t, discr.norm(fields[0]))

        #if istep == 19:
        #    profiler.stop()
        #    print(profiler.output_text(unicode=True, color=True, show_all=True))
        #    if profiling:
        #        print(actx.tabulate_profiling_data())

        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
