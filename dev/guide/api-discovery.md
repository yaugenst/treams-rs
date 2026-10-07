# API discovery

The installed package lists its functions and classes without network access
or imports of JAX, PyTorch or Advect. Use the command line, catalog or
documentation index below.

## The command line

```sh
python -m treams_rs                         # a runnable example and the main entry points
python -m treams_rs sphere_tmatrix          # one function
python -m treams_rs advect.Cluster          # one class of a framework adapter
python -m treams_rs --search 'cross|power'  # public names containing either word
python -m treams_rs --format json > catalog.json
python -m treams_rs --format markdown > catalog.md
```

`help()` and `inspect.signature()` work on every Python function. The NumPy
ufuncs of `special` and `lattice` show their positional arguments in `help()`;
`inspect.signature()` shows only the generic ufunc arguments.

## The catalog

`treams_rs.support_catalog()` returns the same data as `--format json`: every
public function and class with its signature, docstring and source location,
the gradient contexts returned by its records, the treams names, and installed
versions of optional packages. The
[Python reference](../reference/python/index.md) is generated from it.

```python
import treams_rs as tr

catalog = tr.support_catalog()
assert catalog["backends"]["cpu"]["compiled"]
rows = {row["path"]: row for row in catalog["api"]}
cluster = rows["treams_rs.diff.sphere_cluster"]
print(cluster["signature"])
assert [p["context"] for p in cluster["pullbacks"]] == ["SphereClusterContext"]
assert rows["treams_rs.diff.mie"]["pullbacks"][0]["context"] == "MieContext"
assert rows["treams_rs.diff.solve"]["pullbacks"][0]["context"] == "SolveContext"
```

A record such as `diff.sphere_cluster` returns a value and a context;
`context.pullback(g)` converts the gradient `g` with respect to the value into
gradients with respect to the inputs. The catalog row lists the context class
and the signature of its pullback. [Differentiation](../differentiation/index.md)
explains records and contexts.

The rules that hold for every record are in `catalog["rules"]`:

```python
import treams_rs as tr

rules = tr.support_catalog()["rules"]
assert {"derivative_order", "context_lifetime", "static_parameters"} <= set(rules)
for name, text in rules.items():
    print(f"{name}: {text}")
```

## llms.txt

[`llms.txt`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/llms.txt) at the repository root lists every documentation
page with a one-line summary, in the order of the site navigation. It is for
programs and language models that read the documentation as Markdown.

## Two things to know

- Lengths and `1 / k0` share one unit, and numbers passed as materials are
  relative permittivities. See [Conventions](../coming-from-treams/conventions.md).
- The native `diff` records for spheres, `diff.sphere` and `diff.mie`, take
  their materials as arrays ordered from the innermost layer outwards, followed
  by the surrounding medium. Functions such as `multilayer_sphere_tmatrix`
  take the particle `material` and surrounding `medium` as separate arguments.

```python
import numpy as np
import treams_rs as tr
from treams_rs import diff

# A two-layer sphere: radii 0.1 and 0.2, epsilon 4 inside, 2 outside, vacuum around.
array, context = diff.sphere(2, 1.3, [0.1, 0.2], [4, 2, 1])
assert type(context).__name__ == "SphereContext"
sphere = tr.multilayer_sphere_tmatrix(
    k0=1.3, lmax=2, radii=[0.1, 0.2], materials=[4, 2], medium=1
)
np.testing.assert_allclose(array, sphere.array, atol=1e-15)
```
