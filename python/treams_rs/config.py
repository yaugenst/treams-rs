"""Defaults for newly evaluated operations; existing objects retain their convention."""

POLTYPE: str = "helicity"
"""Default polarization convention, either 'helicity' or 'parity'."""


def _resolve_poltype(value: str | None) -> str:
    value = POLTYPE if value is None else value
    if value not in ("helicity", "parity"):
        raise ValueError("polarization type must be helicity or parity")
    return value
