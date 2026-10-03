"""Basis and field operators, as functions and as operator objects.

Mirrors the operators of ``treams``: the functions ``rotate``, ``translate``,
``expand``, ``expandlattice``, ``permute``, ``changepoltype`` and ``efield`` to
``ffield``, and the matching classes ``Rotate`` to ``FField``.

Each function returns its matrix as a plain NumPy array; treams returns a
PhysicsArray. The keyword-only arguments describe the coefficients the matrix
acts on: ``basis``, ``k0``, ``material``, ``poltype``, ``modetype``, and
``lattice`` and ``kpar`` for periodic arrays. An operator object such as
``Rotate(phi)`` stores the other arguments, and ``Rotate(phi) @ array`` reads
the keyword-only ones from the attributes of ``array``. Physics objects keep
their metadata through their own methods (``in_basis``, ``rotate``, ...).

``PhysicsArray`` lives here because it exists for these operators: it holds the
attributes they read, and its attributes ``rotate``, ``expand``, ... apply them::

    import numpy as np
    import treams_rs as tr

    basis = tr.SphericalBasis.default(1)
    array = tr.operators.PhysicsArray(np.eye(len(basis)), basis=basis)
    rotation = tr.operators.rotate(0.3, basis=basis)
    assert np.allclose(tr.operators.Rotate(0.3) @ array, rotation)
    assert np.allclose(array.rotate.apply_left(0.3), rotation)
"""

from ._array import PhysicsArray
from ._operator_objects import (
    BField,
    ChangePoltype,
    DField,
    EField,
    Expand,
    ExpandLattice,
    FField,
    FieldOperator,
    GField,
    HField,
    Operator,
    OperatorAttribute,
    Permute,
    Rotate,
    Translate,
)
from ._operators import (
    bfield,
    changepoltype,
    dfield,
    efield,
    expand,
    expandlattice,
    ffield,
    gfield,
    hfield,
    permute,
    rotate,
    translate,
)

__all__ = [
    "BField",
    "ChangePoltype",
    "DField",
    "EField",
    "Expand",
    "ExpandLattice",
    "FField",
    "FieldOperator",
    "GField",
    "HField",
    "Operator",
    "OperatorAttribute",
    "Permute",
    "PhysicsArray",
    "Rotate",
    "Translate",
    "bfield",
    "changepoltype",
    "dfield",
    "efield",
    "expand",
    "expandlattice",
    "ffield",
    "gfield",
    "hfield",
    "permute",
    "rotate",
    "translate",
]
