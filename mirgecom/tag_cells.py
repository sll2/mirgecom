r""":mod:`mirgecom.tag_cells` Computes smoothness indicator

Perssons smoothness indicator:

.. math::

    S_e = \frac{\langle u_{N_p} - u_{N_{p-1}}, u_{N_p} - u_{N_{p-1}}\rangle_e}{\langle u_{N_p}, u_{N_p} \rangle_e}

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

import numpy as np
import loopy as lp
from grudge import sym
from meshmode.dof_array import DOFArray
from modepy import vandermonde
from pytools import memoize_in
from pytools.obj_array import make_obj_array


def linear_operator_kernel():
    """Apply linear operator to all elements."""
    from meshmode.array_context import make_loopy_program
    knl = make_loopy_program(
        """{[iel,idof,j]:
        0<=iel<nelements and
        0<=idof<ndiscr_nodes_out and
        0<=j<ndiscr_nodes_in}""",
        "result[iel,idof] = sum(j, mat[idof, j] * vec[iel, j])",
        name="modal_decomp")
    knl = lp.tag_array_axes(knl, "mat", "stride:auto,stride:auto")
    return knl

def compute_smoothness_indicator():
    """Compute the smoothness indicator for all elements."""
    from meshmode.array_context import make_loopy_program
    knl = make_loopy_program(
        """{[iel,idof,j,k]:
        0<=iel<nelements and
        0<=idof<ndiscr_nodes_out and
        0<=j<ndiscr_nodes_in and
        0<=k<ndiscr_nodes_in}""",
        "result[iel,idof] = sum(k,vec[iel,k]*vec[iel,k]*modes[k])/sum(j, vec[iel,j]*vec[iel,j]+1.0e-12/ndiscr_nodes_in)",
        name="smooth_comp")
    #knl = lp.tag_array_axes(knl, "vec", "stride:auto,stride:auto")
    return knl

def smoothness_indicator(u,discr):
    
    assert isinstance(u,DOFArray)

    # #@memoize_in(u.array_context, (smoothness_indicator, "get_kernel"))
    def get_kernel():
        return linear_operator_kernel()

    # #@memoize_in(u.array_context, (smoothness_indicator, "get_indicator"))
    def get_indicator():
        return compute_smoothness_indicator()
    
    #Convert to modal solution representation
    actx = u.array_context
    uhat = discr.empty(actx, dtype=u.entry_dtype)
    for group in discr.discr_from_dd("vol").groups:
        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        actx.call_loopy(
            get_kernel(),
            mat=actx.from_numpy(vanderm1),
            result=uhat[group.index],
            vec=u[group.index])

    
    #Compute smoothness indicator value
    indicator = discr.empty(actx, dtype=u.entry_dtype)
    for group in discr.discr_from_dd("vol").groups:
        mode_ids = group.mode_ids()
        modes = len(mode_ids) 
        order = group.order
        highest_mode = np.ones(modes)
        for mode_index,mode_id in enumerate(mode_ids):
            highest_mode[mode_index] = highest_mode[mode_index] * (sum(mode_id)==order)
        
        actx.call_loopy(
            get_indicator(),
            result=indicator[group.index],
            vec=uhat[group.index],
            modes=actx.from_numpy(highest_mode))

    #Take log10 of indicator
    indicator = actx.np.log10(indicator+1.0e-12)

    #No special meaning to these values
    #Should be exposed as tuning parameters
    # for third order?
    #kappa=0.5
    #so = -7.0
    # for first order?
    kappa=0.5
    so = -1.0

    #Compute artificail viscosity percentage based on idicator and set parameters
    yesnol = indicator > (so-kappa)
    yesnou = indicator > (so+kappa)
    sin_indicator = actx.np.where(yesnol,0.5*(1.0+actx.np.sin(np.pi *(indicator - so)/(2.0*kappa))),0.0*indicator)
    indicator = actx.np.where(yesnou,1.0+0.0*indicator,sin_indicator)

    return indicator
     
