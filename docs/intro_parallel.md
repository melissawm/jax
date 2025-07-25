(faq-data-placement)=
# Introduction to Parallel Programming with JAX

Let's look at the principles of data and computation placement in JAX.

In JAX, the computation follows data placement. JAX arrays
have two placement properties: 1) the device where the data resides;
and 2) whether it is **committed** to the device or not (the data is sometimes
referred to as being *sticky* to the device).

By default, JAX arrays are placed uncommitted on the default device
(`jax.devices()[0]`), which is the first GPU or TPU by default. If no GPU or
TPU is present, `jax.devices()[0]` is the CPU. The default device can
be temporarily overridden with the {func}`jax.default_device` context manager, or
set for the whole process by setting the environment variable `JAX_PLATFORMS`
or the absl flag `--jax_platforms` to "cpu", "gpu", or "tpu"
(`JAX_PLATFORMS` can also be a list of platforms, which determines which
platforms are available in priority order).

```python
>>> from jax import numpy as jnp
>>> print(jnp.ones(3).devices())  # doctest: +SKIP
{CudaDevice(id=0)}
```

Computations involving uncommitted data are performed on the default
device and the results are uncommitted on the default device.

Data can also be placed explicitly on a device using {func}`jax.device_put`
with a `device` parameter, in which case the data becomes **committed** to the device:

```python
>>> import jax
>>> from jax import device_put
>>> arr = device_put(1, jax.devices()[2])  # doctest: +SKIP
>>> print(arr.devices())  # doctest: +SKIP
{CudaDevice(id=2)}
```

Computations involving some committed inputs will happen on the
committed device and the result will be committed on the
same device. Invoking an operation on arguments that are committed
to more than one device will raise an error.

You can also use {func}`jax.device_put` without a `device` parameter. If the data
is already on a device (committed or not), it's left as-is. If the data isn't on any
device—that is, it's a regular Python or NumPy value—it's placed uncommitted on the default
device.

Jitted functions behave like any other primitive operations—they will follow the
data and will show errors if invoked on data committed on more than one device.

(Before [PR #6002](https://github.com/jax-ml/jax/pull/6002) in March 2021
there was some laziness in creation of array constants, so that
`jax.device_put(jnp.zeros(...), jax.devices()[1])` or similar would actually
create the array of zeros on `jax.devices()[1]`, instead of creating the
array on the default device then moving it. But this optimization was removed
so as to simplify the implementation.)

(As of April 2020, {func}`jax.jit` has a `device` parameter that affects the device
placement. That parameter is experimental, is likely to be removed or changed,
and its use is not recommended.)

For a worked-out example, we recommend reading through
`test_computation_follows_data` in
[multi_device_test.py](https://github.com/jax-ml/jax/blob/main/tests/multi_device_test.py).
