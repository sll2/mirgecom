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

import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)
import pyopencl.clmath as clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.symbolic.primitives import TracePair
from mirgecom.eos import IdealSingleGas


class Vortex2D:
    """Implements the isentropic vortex after
        - Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        - JSH/TW Nodal DG Methods, p. 209

    A call to this object after creation/init creates
    the isentropic vortex solution at a given time (t)
    relative to the configured origin (center) and
    background flow velocity (velocity).

    This object also functions as a boundary condition
    by providing the "get_boundary_flux" routine to
    prescribe exact field values on the given boundary.
    """

    def __init__(
        self,
        beta=5,
        center=np.zeros(shape=(2,)),
        velocity=np.zeros(shape=(2,)),
    ):
        self._beta = beta
        self._center = center  # np.array([5, 0])
        self._velocity = velocity  # np.array([0, 0])

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        # Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        # also JSH/TW Nodal DG Methods, p. 209
        vortex_loc = self._center + t * self._velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        gamma = eos.gamma()
        r = clmath.sqrt(x_rel ** 2 + y_rel ** 2)
        expterm = self._beta * clmath.exp(1 - r ** 2)
        u = self._velocity[0] - expterm * y_rel / (2 * np.pi)
        v = self._velocity[1] + expterm * x_rel / (2 * np.pi)
        mass = (
            1 - (gamma - 1) / (16 * gamma * np.pi ** 2) * expterm ** 2
        ) ** (1 / (gamma - 1))
        p = mass ** gamma

        e = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

        return flat_obj_array(mass, e, mass * u, mass * v)

    def get_boundary_flux(
        self, discr, w, t=0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        queue = w[0].queue

        # help - how to make it just the boundary nodes?
        nodes = discr.nodes().with_queue(queue)
        vortex_soln = self.__call__(t, nodes)
        dir_bc = discr.interp("vol", btag, vortex_soln)
        dir_soln = discr.interp("vol", btag, w)
        from mirgecom.euler import _facial_flux  # hrm

        return _facial_flux(
            discr, w_tpair=TracePair(btag, dir_soln, dir_bc), eos=eos,
        )


class Lump:
    """Implements a 1,2,3-dimensional Gaussian lump of mass:

    rho(r) = rho0 + rhoamp*e(1-r*r)

    A call to this object after creation/init creates
    the lump solution at a given time (t)
    relative to the configured origin (center) and
    background flow velocity (velocity).

    This object also functions as a boundary condition
    by providing the "get_boundary_flux" method to
    prescribe exact field values on the given boundary.

    This object also supplies the exact expected RHS
    terms from the analytic expression in the
    "expected_rhs" method.
    """

    def __init__(
        self,
        numdim=1,
        rho0=1.0,
        rhoamp=1.0,
        p0=1.0,
        center=[0],
        velocity=[0],
    ):
        if len(center) == numdim:
            self._center = center
        elif len(center) > numdim:
            numdim = len(center)
            self._center = center
        else:
            self._center = np.zeros(shape=(numdim,))

        if len(velocity) == numdim:
            self._velocity = velocity
        elif len(velocity) > numdim:
            numdim = len(velocity)
            self._velocity = velocity
            new_center = np.zeros(shape=(numdim,))
            for i in range(len(self._center)):
                new_center[i] = self._center[i]
            self._center = new_center
        else:
            self._velocity = np.zeros(shape=(numdim,))

        assert len(self._velocity) == numdim
        assert len(self._center) == numdim

        self._p0 = p0
        self._rho0 = rho0
        self._rhoamp = rhoamp
        self._dim = numdim

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        lump_loc = self._center + t * self._velocity
        assert len(x_vec) == self._dim
        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - lump_loc[i] for i in range(self._dim)]
        )
        r = clmath.sqrt(np.dot(rel_center, rel_center))

        gamma = eos.gamma()
        expterm = self._rhoamp * clmath.exp(1 - r ** 2)
        mass = expterm + self._rho0
        mom = self._velocity * make_obj_array([mass])
        energy = (self._p0 / (gamma - 1.0)) + np.dot(mom, mom) / (
            2.0 * mass
        )

        return flat_obj_array(mass, energy, mom)

    def exact_rhs(self, discr, w, t=0.0):
        queue = w[0].queue
        nodes = discr.nodes().with_queue(queue)
        lump_loc = self._center + t * self._velocity
        # coordinates relative to lump center
        rel_center = make_obj_array(
            [nodes[i] - lump_loc[i] for i in range(self._dim)]
        )
        r = clmath.sqrt(np.dot(rel_center, rel_center))

        # The expected rhs is:
        # rhorhs  = -2*rho*(r.dot.v)
        # rhoerhs = -rho*v^2*(r.dot.v)
        # rhovrhs = -2*rho*(r.dot.v)*v
        expterm = self._rhoamp * clmath.exp(1 - r ** 2)
        mass = expterm + self._rho0

        v = self._velocity * make_obj_array([1.0 / mass])
        v2 = np.dot(v, v)
        rdotv = np.dot(rel_center, v)
        massrhs = -2 * rdotv * mass
        energyrhs = -v2 * rdotv * mass
        momrhs = v * make_obj_array([-2 * mass * rdotv])

        return flat_obj_array(massrhs, energyrhs, momrhs)

    def get_boundary_flux(
        self, discr, w, t=0.0, btag=BTAG_ALL, eos=IdealSingleGas()
    ):
        queue = w[0].queue

        # help - how to make it just the boundary nodes?
        nodes = discr.nodes().with_queue(queue)
        mysoln = self.__call__(t, nodes)
        dir_bc = discr.interp("vol", btag, mysoln)
        dir_soln = discr.interp("vol", btag, w)
        from mirgecom.euler import _facial_flux  # hrm

        return _facial_flux(
            discr, w_tpair=TracePair(btag, dir_soln, dir_bc), eos=eos,
        )