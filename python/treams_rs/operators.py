"""Explicit numerical field and basis operators.

These functions return arrays; use wave and scattering-object methods when the
result should retain its physical metadata. Matrix builders accept explicit
basis, frequency, medium and polarization conventions.
"""

from ._array import PhysicsArray
from ._operators import (
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
