# Numerical support

These internal helpers serve all calculation layers and contain no physics.

- [jet.rs](jet.rs) carries a value and its first derivatives through the same
  arithmetic, applying the chain rule.
- [broadcast.rs](broadcast.rs) evaluates scalar-or-array inputs and adds
  gradients for repeated scalar inputs.
- [parallel.rs](parallel.rs) chooses serial or parallel work, preserves output
  order, and adds shared gradients in an order independent of thread count.
- [memory.rs](memory.rs) provides allocation helpers that return an error when
  memory cannot be reserved.
- [mod.rs](mod.rs) holds shared complex arithmetic and finite-value checks.

Parallel helpers use [threads.rs](../threads.rs), which owns the thread pool
and its budget. Numerical formulas and their physical input checks belong in
the modules that use these helpers.
