---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tracing

To use `jax.jit` effectively, it is useful to understand how it works. Let's put a few `print()` statements within a JIT-compiled function and then call the function:

```{code-cell}
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
f(x, y)
```

Notice that the print statements execute, but rather than printing the data we passed to the function, though, it prints *tracer* objects that stand-in for them.

These tracer objects are what `jax.jit` uses to extract the sequence of operations specified by the function. Basic tracers are stand-ins that encode the **shape** and **dtype** of the arrays, but are agnostic to the values. This recorded sequence of computations can then be efficiently applied within XLA to new inputs with the same shape and dtype, without having to re-execute the Python code.

When we call the compiled function again on matching inputs, no re-compilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python:

```{code-cell}
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

The extracted sequence of operations is encoded in a JAX expression, or [*jaxpr*](https://docs.jax.dev/en/latest/key-concepts.html#jaxprs) for short. You can view the jaxpr using the `jax.make_jaxpr` transformation:

```{code-cell}
from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

Note one consequence of this: because JIT compilation is done *without* information on the content of the array, control flow statements in the function cannot depend on traced values. For example, this fails:

```{code-cell}
:tags: [raises-exception]

@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

If there are variables that you would not like to be traced, they can be marked as static for the purposes of JIT compilation:

```{code-cell}
from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

Note that calling a JIT-compiled function with a different static argument results in re-compilation, so the function still works as expected:

```{code-cell}
f(1, False)
```

Understanding which values and operations will be static and which will be traced is a key part of using `jax.jit` effectively.

(faq-different-kinds-of-jax-values)=
## Different kinds of JAX values

In the process of transforming functions, JAX replaces some function
arguments with special tracer values.

You could see this if you use a `print` statement:

```{code-cell}
import jax

def func(x):
  print(x)
  return jnp.cos(x)

res = jax.jit(func)(0.)

print(res)
```

The above code does return the correct value `1.` but it also prints
`Traced<ShapedArray(float32[])>` for the value of `x`. Normally, JAX
handles these tracer values internally in a transparent way, e.g.,
in the numeric JAX primitives that are used to implement the
`jax.numpy` functions. This is why `jnp.cos` works in the example above.

More precisely, a **tracer** value is introduced for the argument of
a JAX-transformed function, except the arguments identified by special
parameters such as `static_argnums` for {func}`jax.jit` or
`static_broadcasted_argnums` for {func}`jax.pmap`. Typically, computations
that involve at least a tracer value will produce a tracer value. Besides tracer
values, there are **regular** Python values: values that are computed outside JAX
transformations, or arise from above-mentioned static arguments of certain JAX
transformations, or computed solely from other regular Python values.
These are the values that are used everywhere in absence of JAX transformations.

A tracer value carries an **abstract** value, e.g., `ShapedArray` with information
about the shape and dtype of an array. We will refer here to such tracers as
**abstract tracers**. Some tracers, e.g., those that are
introduced for arguments of autodiff transformations, carry `ConcreteArray`
abstract values that actually include the regular array data, and are used,
e.g., for resolving conditionals. We will refer here to such tracers
as **concrete tracers**. Tracer values computed from these concrete tracers,
perhaps in combination with regular values, result in concrete tracers.
A **concrete value** is either a regular value or a concrete tracer.

Most often values computed from tracer values are themselves tracer values.
There are very few exceptions, when a computation can be entirely done
using the abstract value carried by a tracer, in which case the result
can be a regular value. For example, getting the shape of a tracer
with `ShapedArray` abstract value. Another example is when explicitly
casting a concrete tracer value to a regular type, e.g., `int(x)` or
`x.astype(float)`.
Another such situation is for `bool(x)`, which produces a Python bool when
concreteness makes it possible. That case is especially salient because
of how often it arises in control flow.

Here is how the transformations introduce abstract or concrete tracers:

* {func}`jax.jit`: introduces **abstract tracers** for all positional arguments
  except those denoted by `static_argnums`, which remain regular
  values.
* {func}`jax.pmap`: introduces **abstract tracers** for all positional arguments
  except those denoted by `static_broadcasted_argnums`.
* {func}`jax.vmap`, {func}`jax.make_jaxpr`, {func}`xla_computation`:
  introduce **abstract tracers** for all positional arguments.
* {func}`jax.jvp` and {func}`jax.grad` introduce **concrete tracers**
  for all positional arguments. An exception is when these transformations
  are within an outer transformation and the actual arguments are
  themselves abstract tracers; in that case, the tracers introduced
  by the autodiff transformations are also abstract tracers.
* All higher-order control-flow primitives ({func}`lax.cond`, {func}`lax.while_loop`,
  {func}`lax.fori_loop`, {func}`lax.scan`) when they process the functionals
  introduce **abstract tracers**, whether or not there is a JAX transformation
  in progress.

All of this is relevant when you have code that can operate
only on regular Python values, such as code that has conditional
control-flow based on data:

```python
def divide(x, y):
  return x / y if y >= 1. else 0.
```

If we want to apply {func}`jax.jit`, we must ensure to specify `static_argnums=1`
to ensure `y` stays a regular value. This is due to the boolean expression
`y >= 1.`, which requires concrete values (regular or tracers). The
same would happen if we write explicitly `bool(y >= 1.)`, or `int(y)`,
or `float(y)`.

Interestingly, `jax.grad(divide)(3., 2.)`, works because {func}`jax.grad`
uses concrete tracers, and resolves the conditional using the concrete
value of `y`.
