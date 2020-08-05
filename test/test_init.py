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

import logging
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clrandom
import pyopencl.clmath

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
from mirgecom.initializers import Uniform
from mirgecom.initializers import SodShock1D

from mirgecom.euler import split_conserved
from mirgecom.eos import IdealSingleGas

from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
import pytest


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_lump_init(ctx_factory, dim):
    """
    Simple test to check that Lump initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)
    logger = logging.getLogger(__name__)

    #    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )
    #    mesh = generate_regular_rect_mesh(
    #        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    #    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    for vdim in range(dim):
        # Init soln with Vortex
        center = np.zeros(shape=(dim,))
        velocity = np.zeros(shape=(dim,))
        velocity[vdim] = 1.0
        lump = Lump(center=center, velocity=velocity)
        lump_soln = lump(0, nodes)

        qs = split_conserved(dim, lump_soln)
        mass = qs.mass
        energy = qs.energy
        mom = qs.momentum
        p = 0.4 * (energy - 0.5 * np.dot(mom, mom) / mass)
        exp_p = 1.0
        errmax = discr.norm(p - exp_p, np.inf)

        logger.info(f"lump_soln = {lump_soln}")
        logger.info(f"pressure = {p}")

        assert errmax < 1e-15


def test_vortex_init(ctx_factory):
    """
    Simple test to check that Vortex2D initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    logger = logging.getLogger(__name__)

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    vortex = Vortex2D()
    vortex_soln = vortex(0, nodes)
    gamma = 1.4
    qs = split_conserved(dim, vortex_soln)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum
    p = 0.4 * (energy - 0.5 * np.dot(mom, mom) / mass)
    exp_p = mass ** gamma
    errmax = discr.norm(p - exp_p, np.inf)

    logger.info(f"vortex_soln = {vortex_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15


@pytest.mark.parametrize("dim", [2, 3])
def test_shock_init(ctx_factory, dim):
    """
    Simple test to check that Shock1D initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 10
    #    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    #    for dim in [1, 2, 3]:
    mesh = generate_regular_rect_mesh(
        a=(-1.0,) * dim, b=(1.0,) * dim, n=(nel_1d,) * dim
    )

    order = 1
    print(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    print(f"nodes={nodes}")

    xpl = 1.0
    xpr = 0.1
    tol = 1e-15
    eos = IdealSingleGas()

    for xdir in range(dim):
        x0 = 0.0
        initr = SodShock1D(dim=dim, xdir=xdir, x0=x0)
        initsoln = initr(t=0.0, x_vec=nodes)
        print(f"Sod Soln(dim={dim},xdir={xdir}): {initsoln}")
        x_rel = nodes[xdir]
        print(f"x_rel = {x_rel}")

        #        from meshmode.dof_array import flatten
        #        nodes_x = flatten(x_rel)
        #        new_actx = nodes_x.array_context

        #        print(f"nodes_x = {nodes_x}")

        #        p = flatten(eos.pressure(initsoln))
        #        print(f"Pressure={p}")
        #        new_actx = p.array_context
        #        testyesno = nodes_x < x0
        #        print(f"testyesno = {testyesno}")
        #        thing = actx.np.where(nodes_x < x0, p-xpl, p-xpr)
        #        print(f"thing = {thing}")
        #        new_actx2 = thing.array_context

        #        assert discr.norm(thing, np.inf) < tol
    assert(False)

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_uniform(ctx_factory, dim):
    """
    Simple test to check that Uniform initializer
    creates the expected solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 10
    #    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    #    for dim in [1, 2, 3]:
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )
    #    mesh = generate_regular_rect_mesh(
    #        a=[(0.0,), (1.0,)], b=[(-0.5,), (0.5,)], n=(nel_1d,) * dim
    #    )

    order = 3
    print(f"Number of elements: {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    initr = Uniform(numdim=dim)
    initsoln = initr(t=0.0, x_vec=nodes)
    print("Uniform Soln:", initsoln)
    tol = 1e-15
    eos = IdealSingleGas()
    p = eos.pressure(initsoln)

    assert discr.norm(p - 1.0, np.inf) < tol
