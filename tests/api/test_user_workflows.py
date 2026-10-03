"""Three public workflows: an iterative against a dense cluster, an HDF5 round
trip of a T-matrix, and the material constructors with the operator namespace."""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import io

pytestmark = pytest.mark.workflows


def test_iterative_physical_wave_matches_dense_cluster():
    radii = [0.15, 0.2]
    positions = [[0, 0, 0], [0.8, 0.2, 0]]
    particles = [
        tr.sphere_tmatrix(k0=1.3, lmax=2, radius=r, material=e)
        for r, e in zip(radii, [2.0, 3.0], strict=True)
    ]
    incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
    dense = tr.Cluster(particles, positions=positions).scatter(incident)
    iterative = tr.iterative.SphereCluster(2, 1.3, radii, [2.0, 3.0], positions)
    result = iterative.scatter(incident, rtol=1e-13)
    assert len(result.convergence) == 1
    assert result.convergence[0].residual_norm <= 1e-13 * result.convergence[0].rhs_norm
    assert_allclose(result.wave.coefficients, dense.coefficients, rtol=1e-9, atol=1e-13)
    points = [[0.3, 0.4, 1.2], [0.1, -0.2, 0.9]]
    assert_allclose(
        result.wave.efield(points), dense.efield(points), rtol=1e-8, atol=1e-13
    )


def test_hdf5_loaded_response_remains_a_physical_object():
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, sphere)
        loaded = io.load_hdf5(handle)
    assert isinstance(loaded, tr.TMatrix)
    assert_allclose(
        loaded.scatter(incident).coefficients, sphere.scatter(incident).coefficients
    )
    assert_allclose(loaded.cross_sections(incident), sphere.cross_sections(incident))


def test_explicit_material_constructors_and_operator_namespace():
    medium = tr.Material.from_refractive_index(1.5, impedance=2 / 3)
    assert_allclose(medium.epsilon, 2.25)
    assert_allclose(medium.mu, 1)
    chiral = tr.Material.from_helicity_indices((1.4, 1.6))
    assert_allclose(chiral.nmp, [1.4, 1.6])
    basis = tr.SphericalBasis.default(2)
    rotation = tr.operators.rotate(0.2, 0.4, 0.1, basis=basis)
    assert_allclose(rotation @ rotation.conj().T, np.eye(len(basis)), atol=1e-14)
    ports = tr.PlaneWavePorts.default([0.1, 0.2])
    layer = tr.interface(basis=ports, k0=1.3, negative_medium=1, positive_medium=4)
    assert isinstance(layer, tr.SMatrix)
    block = layer.block(outgoing="positive", incoming="negative")
    assert isinstance(block, tr.ScatteringBlock)
    assert_allclose(block.array, layer.array[0, 0])
    assert block.material == (layer.positive_medium, layer.negative_medium)
    assert not hasattr(tr, "SMatrices")
    assert not hasattr(tr, "efield")
    assert not hasattr(tr.TMatrix, "cluster")
    assert not hasattr(tr.SMatrix, "from_array")
