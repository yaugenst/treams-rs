"""Polarization validation shared by the numerical and physics APIs.

Defaults are deterministic: omitted polarization means helicity. Pass parity
explicitly; there is no process-global setting that changes later calculations.
"""


def _resolve_poltype(value: str | None) -> str:
    value = "helicity" if value is None else value
    if value not in ("helicity", "parity"):
        raise ValueError("polarization type must be helicity or parity")
    return value
