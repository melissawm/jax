# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for calling pallas functions from JAX."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import enum
from functools import partial, reduce
import types
from typing import Any

import jax
from jax import lax
from jax._src import ad_util
from jax._src import api_util
from jax._src import checkify
from jax._src import config
from jax._src import core as jax_core
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas import helpers as pallas_helpers
from jax._src.pallas import hlo_interpreter
from jax._src.pallas import primitives
from jax._src.state import discharge as state_discharge
from jax._src.state import types as state_types
from jax._src.util import (
    safe_map,
    safe_zip,
    split_list,
    tuple_insert,
    unzip2,
)
import jax.numpy as jnp

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = pallas_core.Grid
TupleGrid = pallas_core.TupleGrid
GridSpec = pallas_core.GridSpec
BlockMapping = pallas_core.BlockMapping
GridMapping = pallas_core.GridMapping
BlockSpec = pallas_core.BlockSpec
BlockSpecTree = pallas_core.BlockSpecTree
NoBlockSpec = pallas_core.NoBlockSpec
no_block_spec = pallas_core.no_block_spec
ScratchShapeTree = pallas_core.ScratchShapeTree
CostEstimate = pallas_core.CostEstimate
Backend = pallas_core.Backend
CompilerParams = pallas_core.CompilerParams

# See the docstring for GridMapping for the calling convention
pallas_call_p = jax_core.Primitive('pallas_call')
pallas_call_p.multiple_results = True


def _pallas_call_impl(*args, **params):
  # Call the lowering path
  @partial(jax.jit, inline=True)
  def _jit_run(*args):
    return pallas_call_p.bind(*args, **params)

  with config.disable_jit(False):
    return _jit_run(*args)

pallas_call_p.def_impl(_pallas_call_impl)


def _pallas_call_abstract_eval(
    *avals,
    out_avals: tuple[jax_core.AbstractValue, ...],
    interpret,
    backend,
    **params
):
  del avals

  if isinstance(interpret, mosaic_tpu_interpret.InterpretParams):
    # Report effects that will be introduced when running/lowering
    # mosaic_tpu_interpret.mosaic_tpu_interpret.interpret_pallas_call .
    effs = mosaic_tpu_interpret.get_interpret_effects()
  elif getattr(params.get('compiler_params', None), 'has_side_effects', False):
    effs = jax_core.GenericEffect(pallas_call_p)
  else:
    effs = jax_core.no_effects

  # Make sure we don't return ShapedArrayWithMemorySpace to the outside world.
  return [
      jax_core.ShapedArray(a.shape, a.dtype, a.weak_type)
      if isinstance(a, pallas_core.ShapedArrayWithMemorySpace)
      else a
      for a in out_avals
  ], effs


pallas_call_p.def_effectful_abstract_eval(_pallas_call_abstract_eval)


def _pallas_call_jvp_rule(
    primals,
    tangents,
    *,
    jaxpr: jax_core.Jaxpr,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    debug: bool,
    interpret: Any,
    compiler_params: Any,
    cost_estimate: CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    backend: Backend | None,
    metadata: FrozenDict[str, str] | None,
):
  debug_info = jaxpr.debug_info
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError("interpret with dynamic grid bounds unsupported")
  if grid_mapping.num_index_operands:
    raise NotImplementedError
  if input_output_aliases:
    raise NotImplementedError("JVP with aliasing not supported.")
  if mesh is not None:
    raise NotImplementedError("pallas_call with a mesh does not support JVP")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  nonzero_tangents_with_outputs = nonzero_tangents + [True] * grid_mapping.num_outputs
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, nonzero_tangents_with_outputs, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  # `pallas_call` takes in inputs and returns outputs but its jaxpr *does not*.
  # `pallas_call` takes in a stateful jaxpr, meaning the jaxpr accepts input
  # `Ref`s that are read from followed by output `Ref`s that are written to.
  # This means that when we do `jvp_jaxpr` on the `jaxpr`, we get out a new
  # jaxpr that has tangents following primals. In order for this jaxpr to be
  # compatible w/ `pallas_call` (inputs then outputs), we need to shuffle around
  # the jaxpr's invars.
  primal_refs, primal_out_refs, tangent_refs, tangent_out_refs = split_list(
      jvp_jaxpr.invars, [len(primals), grid_mapping.num_outputs, len(tangents)]
  )
  invars = (*primal_refs, *tangent_refs, *primal_out_refs, *tangent_out_refs)
  effs = []
  for eff in jvp_jaxpr.effects:
    if isinstance(eff, effects.JaxprInputEffect):
      eff = eff.replace(
          input_index=invars.index(jvp_jaxpr.invars[eff.input_index])
      )
    effs.append(eff)
  jvp_jaxpr = jvp_jaxpr.replace(invars=invars, effects=effs)
  if debug:
    print(f"\nThe jaxpr for the jvp of pallas_call {debug_info.func_src_info}:")
    print(jvp_jaxpr)
  in_bms, out_bms = split_list(grid_mapping.block_mappings, [len(primals)])
  jvp_bms = (*in_bms, *in_bms, *out_bms, *out_bms)
  jvp_grid_mapping = grid_mapping.replace(
      block_mappings=jvp_bms,
      num_inputs=grid_mapping.num_inputs * 2,
      num_outputs=grid_mapping.num_outputs * 2,
  )
  if cost_estimate is not None:
    jvp_cost_estimate = CostEstimate(
        flops=2 * cost_estimate.flops,
        bytes_accessed=2 * cost_estimate.bytes_accessed,
        transcendentals=2 * cost_estimate.transcendentals,
    )
  else:
    jvp_cost_estimate = None
  out_flat = pallas_call_p.bind(
      *primals,
      *tangents,
      jaxpr=jvp_jaxpr,
      grid_mapping=jvp_grid_mapping,
      mesh=mesh,
      interpret=interpret,
      debug=debug,
      input_output_aliases=(),
      compiler_params=compiler_params,
      cost_estimate=jvp_cost_estimate,
      out_avals=(*out_avals, *out_avals),
      backend=backend,
      metadata=metadata,
  )
  out_primals, out_tangents = split_list(out_flat, [len(out_flat) // 2])
  return out_primals, out_tangents


ad.primitive_jvps[pallas_call_p] = _pallas_call_jvp_rule


def _batch_block_mapping(
    grid_mapping: GridMapping,
    axis_size: int,
    for_ragged: bool,
    aval: jax_core.ShapedArray,
    dim: int | batching.NotMapped,
    block_mapping: BlockMapping,
    ragged_axis_values,
) -> BlockMapping:
  def _block_map_function(new_idx, *args):
    if for_ragged:
      drop_last_args = args[:-1]
    else:
      drop_last_args = args

    indices = jax_core.eval_jaxpr(
        block_mapping.index_map_jaxpr.jaxpr,
        block_mapping.index_map_jaxpr.consts,
        *drop_last_args,
    )
    unflat_indices = tree_util.tree_unflatten(
        block_mapping.index_map_out_tree, indices)
    if not isinstance(unflat_indices, tuple):
      unflat_indices = (unflat_indices,)
    unflat_indices = list(unflat_indices)
    if dim is not batching.not_mapped:
      if isinstance(dim, batching.RaggedAxis):
        assert for_ragged, "Ragged axis not supported for non-ragged batching."
        stacked_axis = dim.stacked_axis
        unflat_indices.insert(stacked_axis, new_idx)
      else:
        unflat_indices.insert(dim, new_idx)
    return tuple(unflat_indices)
  idx_avals = [pallas_core.index_map_grid_aval, *block_mapping.index_map_jaxpr.in_avals]

  if for_ragged:
    if isinstance(dim, batching.RaggedAxis):
      assert for_ragged, "Ragged axis not supported for non-ragged batching."
      _, _, _, lengths_aval = ragged_axis_values
      idx_avals = [*idx_avals, lengths_aval]
    else:
      i32_aval_memref = state.AbstractRef(
          jax_core.ShapedArray(([axis_size]), jnp.int32),
          pallas_core.MemorySpace.INDEX,
      )
      idx_avals = [*idx_avals, i32_aval_memref]

  block_mapping_flat_fn, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(_block_map_function,
                   debug_info=block_mapping.index_map_jaxpr.jaxpr.debug_info),
      tree_util.tree_structure(idx_avals))
  with grid_mapping.trace_env():
    block_mapping_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        block_mapping_flat_fn,
        idx_avals)
  new_index_map_out_tree = out_tree_thunk()
  shape = block_mapping.block_shape
  if dim is batching.not_mapped:
    new_block_shape = shape
    new_array_shape_dtype = block_mapping.array_shape_dtype
  else:
    if isinstance(dim, batching.RaggedAxis):
      assert for_ragged, "Ragged axis not supported for non-ragged batching."
      new_block_shape = shape
      stacked_axis = dim.stacked_axis
      new_block_shape = tuple_insert(
          new_block_shape, stacked_axis, pallas_core.squeezed
      )
    else:
      new_block_shape = tuple_insert(shape, dim, pallas_core.squeezed)

    array_shape = block_mapping.array_shape_dtype.shape
    if isinstance(dim, batching.RaggedAxis):
      assert for_ragged, "Ragged axis not supported for non-ragged batching."
      stacked_axis = dim.stacked_axis
      array_shape = tuple_insert(array_shape, stacked_axis, axis_size)
    else:
      array_shape = tuple_insert(array_shape, dim, axis_size)

    new_array_shape_dtype = jax.ShapeDtypeStruct(
        array_shape, block_mapping.array_shape_dtype.dtype
    )

  jaxpr = jax_core.ClosedJaxpr(block_mapping_jaxpr, consts)
  return block_mapping.replace(block_shape=new_block_shape,
                               array_shape_dtype=new_array_shape_dtype,
                               index_map_jaxpr=jaxpr,
                               index_map_out_tree=new_index_map_out_tree)


def _broadcast_input_output_aliases(
    args: Sequence[jax.Array],
    dims: Sequence[int | batching.NotMapped],
    *,
    input_output_aliases: tuple[tuple[int, int], ...],
    axis_size: int,
) -> tuple[tuple[jax.Array, ...], tuple[int | batching.NotMapped, ...]]:
  """Broadcast input/output operands.

  When we have input/output aliasing, since the output will be mapped, we need
  to make sure to broadcast the input across that dimension if it is not
  mapped. If the input is mapped, but on a different axis, we transpose the input
  to match the output.
  """

  args_ = list(args)
  dims_ = list(dims)
  for input_index, _ in input_output_aliases:
    dim = dims_[input_index]
    dims_[input_index] = 0
    if isinstance(dim, batching.RaggedAxis):
      stacked_axis = dim.stacked_axis
      if stacked_axis != 0:
        raise NotImplementedError("Ragged aliasing on non 0 dim NYI")
      return tuple(args_), tuple(dims_)

    if dim is batching.not_mapped:
      args_[input_index] = batching.broadcast(args_[input_index], axis_size, 0)
    elif dim != 0:
      # TODO(cjfj): Change output batching axis instead?
      args_[input_index] = jnp.moveaxis(args[input_index], dim, 0)

  return tuple(args_), tuple(dims_)


def _batch_with_explicit_loop(
    args: Sequence[jax.Array],
    dims: Sequence[int | batching.NotMapped],
    *,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: Any,
    compiler_params: Any,
    cost_estimate: CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    backend: Backend | None,
    metadata: FrozenDict[str, str] | None,
):
  """Batch the pallas_call by calling it in loop over the batch size.

  This function provides a fallback implementation of batching a pallas_call
  for the cases in which adding a batch dimension to the pallas grid is not
  supported. This is currently the case when the batched dimension corresponds
  to a dynamic axis or a scalar prefetch argument.

  This implementation builds a HLO loop that dynamic_slices the inputs according
  to the current iteration index and dynamic_updates an (initially empty) output
  allocation.
  """
  if not dims:
    raise NotImplementedError("vmapping pallas_call with no arguments.")

  (axis_size,) = {
      arg.shape[dim]
      for arg, dim in zip(args, dims)
      if dim is not batching.not_mapped
  }

  args, dims = _broadcast_input_output_aliases(
      args,
      dims,
      input_output_aliases=input_output_aliases,
      axis_size=axis_size,
  )

  # The output arrays are completely overwritten, so we can just initialize
  # empty arrays.
  initial_state = [
      jnp.empty(tuple_insert(bm.array_shape_dtype.shape, 0, axis_size),
                dtype=bm.array_shape_dtype.dtype)
      for bm in grid_mapping.block_mappings_output
  ]

  def body(batch_index: jax.Array, state: list[jax.Array]) -> list[jax.Array]:
    batch_args = []

    for arg, dim in zip(args, dims):
      # If the argument is mapped, extract a slice of size 1 in the mapped
      # dimension at the current index.
      if dim is batching.not_mapped:
        batch_args.append(arg)
      else:
        batch_args.append(
            jnp.squeeze(
                jax.lax.dynamic_slice_in_dim(
                    operand=arg,
                    start_index=batch_index,
                    slice_size=1,
                    axis=dim,
                ),
                axis=dim,
            )
        )
    batch_out = pallas_call_p.bind(
        *batch_args,
        jaxpr=jaxpr,
        grid_mapping=grid_mapping,
        mesh=mesh,
        input_output_aliases=input_output_aliases,
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
        out_avals=out_avals,
        backend=backend,
        metadata=metadata,
    )
    for i, batch_out_array in enumerate(batch_out):
      state[i] = jax.lax.dynamic_update_index_in_dim(
          state[i],
          batch_out_array,
          batch_index,
          axis=0,
      )

    return state

  result = jax.lax.fori_loop(0, axis_size, body, initial_state, unroll=False)

  return result, (0,) * len(result)


def _pallas_call_batching_rule(
    args,
    dims,
    *,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: Any,
    compiler_params: Any,
    cost_estimate: CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    backend: Backend | None,
    metadata: FrozenDict[str, str] | None = None,
):
  if mesh is not None:
    raise NotImplementedError(
        "pallas_call with a mesh does not support batching"
    )

  def _maybe_squeeze_out_bdim(
      x: jax.Array, bdim: int | batching.NotMapped
  ) -> jax.Array:
    if bdim is batching.not_mapped:
      return x
    return jnp.squeeze(x, axis=bdim)

  def get_size(i, x, d):
    if not isinstance(d, batching.RaggedAxis):
      return x.shape[d]
    return x.aval.shape[d.stacked_axis]

  (axis_size,) = {
      get_size(i=i, x=x, d=d)
      for i, (x, d) in enumerate(zip(args, dims))
      if d is not batching.not_mapped
  }
  if axis_size == 1:
    # Why are we even vmapping?
    args = map(_maybe_squeeze_out_bdim, args, dims)
    out = pallas_call_p.bind(
        *args,
        jaxpr=jaxpr,
        grid_mapping=grid_mapping,
        mesh=mesh,
        input_output_aliases=input_output_aliases,
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
        out_avals=out_avals,
        backend=backend,
        metadata=metadata,
    )
    return [jnp.expand_dims(x, 0) for x in out], (0,) * len(out)

  # The first num_dynamic_grid_bounds arguments are size-1 arrays that store
  # the size of the dynamic bounds.
  dynamic_grid_args, args = split_list(
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  dynamic_grid_dims, dims = split_list(
      dims, [grid_mapping.num_dynamic_grid_bounds]
  )
  if all(
      bdim is batching.not_mapped or arg.shape[bdim] == 1
      for arg, bdim in zip(dynamic_grid_args, dynamic_grid_dims)
  ):
    dynamic_grid_args = safe_map(
        _maybe_squeeze_out_bdim, dynamic_grid_args, dynamic_grid_dims
    )
  elif any(bdim is not batching.not_mapped for bdim in dynamic_grid_dims):
    # TODO(amagni, sharadmv): Explore possibility of batching dynamic grid
    # bounds.
    return _batch_with_explicit_loop(
        args=dynamic_grid_args + args,
        dims=dynamic_grid_dims + dims,
        jaxpr=jaxpr,
        grid_mapping=grid_mapping,
        mesh=mesh,
        input_output_aliases=input_output_aliases,
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
        out_avals=out_avals,
        backend=backend,
        metadata=metadata,
    )
  else:
    pass  # No dynamic grid dimensions
  del dynamic_grid_dims
  if grid_mapping.num_index_operands:
    scalar_args, args = split_list(args, [grid_mapping.num_index_operands])
    scalar_bdims, bdims = split_list(dims, [grid_mapping.num_index_operands])
    # Ordinarily, adding support for scalar prefetch in vmap would involve
    # modifying the block specs in a nontrivial way. However, if we are only
    # vmapping over 1-sized dimensions, we can just get rid of the dimensions
    # and pretend we were never vmapped over them at all.
    if all(
        bdim is batching.not_mapped or arg.shape[bdim] == 1
        for arg, bdim in zip(scalar_args, scalar_bdims)
    ):
      scalar_args = safe_map(_maybe_squeeze_out_bdim, scalar_args, scalar_bdims)
      scalar_bdims = [batching.not_mapped] * len(scalar_args)
      args = (*scalar_args, *args)
      dims = (*scalar_bdims, *bdims)
    else:
      # TODO(amagni,sharadmv,apaszke): enable efficient batching over
      # prefetched scalar args.
      return _batch_with_explicit_loop(
          args=scalar_args + args,
          dims=scalar_bdims + bdims,
          jaxpr=jaxpr,
          grid_mapping=grid_mapping,
          mesh=mesh,
          input_output_aliases=input_output_aliases,
          debug=debug,
          interpret=interpret,
          compiler_params=compiler_params,
          cost_estimate=cost_estimate,
          out_avals=out_avals,
          backend=backend,
          metadata=metadata,
      )

  if not dims:
    raise NotImplementedError("vmapping pallas_call with no arguments.")
  block_mappings = grid_mapping.block_mappings
  avals = [v.aval for v in jaxpr.invars]
  # How should we pick output dimensions? This actually matters because XLA
  # can't optimize our pallas kernels, and this layout impacts performance. For
  # now, because `vmap` doesn't really offer a way of inferring good output
  # dimensions. For now, we just use 0.
  # TODO(sharadmv): explore inferring better output dimensions via a heuristic
  # TODO(sharadmv): explore a long term solution to output dim inference

  args, dims = _broadcast_input_output_aliases(
      args, dims, input_output_aliases=input_output_aliases, axis_size=axis_size
  )

  # Each dim either has data about its ragged axis, or None
  ragged_axis_values = []
  for d in dims:
    if isinstance(d, batching.RaggedAxis):
      stacked_axis, ragged_axis_dim, ragged_axis_length = (
          batching._ragged_axis_parts(d)
      )
      aval = jax_core.get_aval(ragged_axis_length).update(dtype=jnp.int32)
      if isinstance(aval, jax_core.DShapedArray):
        aval = jax_core.ShapedArray(aval.shape, aval.dtype, aval.weak_type)
      lengths_aval = state.AbstractRef(
          aval,
          pallas_core.MemorySpace.INDEX,
      )
      # TODO(mvoz): Give this its own type
      ragged_axis_values.append(
          (stacked_axis, ragged_axis_dim, ragged_axis_length, lengths_aval)
      )
    else:
      ragged_axis_values.append(None)  # type: ignore[arg-type]

  all_dims = list(dims) + [0] * grid_mapping.num_outputs
  ragged_axis_values = ragged_axis_values + [None] * grid_mapping.num_outputs

  num_index_operands = grid_mapping.num_index_operands
  num_scratch_operands = grid_mapping.num_scratch_operands

  # Only add a batch dimension for the avals that actually have a grid mapping.
  # This excludes scalar prefetch inputs (the first in the list) and scratch
  # operands (the last in the list).
  avals_to_batch = avals[num_index_operands:(len(avals) - num_scratch_operands)]

  batched_block_mappings = map(
      partial(
          _batch_block_mapping,
          grid_mapping,
          axis_size,
          any(ragged_axis_values),
      ),
      avals_to_batch,
      all_dims[num_index_operands:],
      block_mappings,
      ragged_axis_values[num_index_operands:],
  )

  index_map_tree_args, index_map_tree_kwargs = grid_mapping.index_map_tree.unflatten(
      grid_mapping.index_map_avals)
  assert not index_map_tree_kwargs
  batched_index_map_args = (pallas_core.index_map_grid_aval,) + index_map_tree_args

  lengths_aval = None  # type: ignore[assignment]

  # Check all the ragged axis values, ensure their raggedness pattern
  # is identical (consider moving this check up!)
  for rav in ragged_axis_values:
    if rav is not None:
      if lengths_aval is None:
        lengths_aval = rav[3]
      else:
        assert lengths_aval == rav[3], "NYI - different lengths in ragged batch"

  if lengths_aval:
    batched_index_map_args = batched_index_map_args + (lengths_aval,)
    num_index_operands += 1

  batched_index_map_avals, batched_index_map_tree = tree_util.tree_flatten(
      (batched_index_map_args, {}))

  batched_grid_mapping = grid_mapping.replace(
      grid=(axis_size, *grid_mapping.grid),
      block_mappings=tuple(batched_block_mappings),
      index_map_avals=tuple(batched_index_map_avals),
      index_map_tree=batched_index_map_tree,
      num_index_operands=num_index_operands,
      vmapped_dims=(0,) + tuple(a + 1 for a in grid_mapping.vmapped_dims),
  )

  if cost_estimate is not None:
    batched_cost_estimate = CostEstimate(
        flops=cost_estimate.flops * axis_size,
        bytes_accessed=cost_estimate.bytes_accessed * axis_size,
        transcendentals=cost_estimate.transcendentals * axis_size,
    )
  else:
    batched_cost_estimate = None

  # Start the ragged handling code
  # Here, we:
  #  - Rewrite the indexer to save memory (skip indices outside the ragged bounds)
  #  - Rewrite the kernel to save compute (skip elements outside the ragged bounds)
  #  - Update various internal structures/metadata to account for the new
  #    block spec.
  #  - Set the hacky flag of ragged_originating on the mapping, to signal to
  #    the lowering code to treat mapped dimensions as part of the user grid.
  if lengths_aval:
    batched_grid_mapping = batched_grid_mapping.replace(
        get_grid_indices=lambda indices, maybe_include_mapped_dims: indices,
        local_grid_env=lambda loop_idx, grid: tuple(
            pallas_core.GridAxis(idx, b) for (idx, b) in zip(loop_idx, grid)
        ),
    )

    # Note - on zero filling counterfactuals
    # A debug util to produce a counterfactual version of the when
    # gating, where for all values that don't pass the @when check,
    # we write 0s. This is useful for debugging, as certain lowering paths
    # like mosaic will write the last data as passthrough, leading to
    # potentially confusing results.
    block_mapped_dim_idxs = []
    for block_mapping in batched_grid_mapping.block_mappings:
      mapped_dim_idxs = []
      for i, d in enumerate(block_mapping.block_shape):
        if isinstance(d, pallas_core.Squeezed):
          mapped_dim_idxs.append(i)
        else:
          mapped_dim_idxs.append(None)  # type: ignore[arg-type]
      block_mapped_dim_idxs.append(mapped_dim_idxs)

    mapped_dim_idx = None
    for rav, mapped_dim_idxs in zip(ragged_axis_values, block_mapped_dim_idxs):
      if rav is not None:
        stacked_axis = rav[0]
        if mapped_dim_idx is None:
          mapped_dim_idx = mapped_dim_idxs[stacked_axis]
          if mapped_dim_idxs[stacked_axis] is None:
            raise ValueError(
                f"Expected mapped dim to be {stacked_axis}, but got"
                f" {mapped_dim_idxs[stacked_axis]}"
            )
        else:
          assert mapped_dim_idx == mapped_dim_idxs[stacked_axis], (
              f"Different mapped dims - expected {mapped_dim_idx}, but got"
              f" {mapped_dim_idxs[stacked_axis]}"
          )

    # This is the blockspec size of the dimension
    block_shapes = [b.block_shape for b in batched_grid_mapping.block_mappings]

    # Parse out the operations from the jaxpr to determine how to mask the output
    # NOTE! while this *could* be a default dict of None, and None is sound, as
    # it denotes that there is no raggedness for the given var, we explicitly
    # do not do this, so as to get better signal on implementation of rules
    # A misimplemented rule that does not account for new vars being introduced
    # will result in an error on the next op using the new var. The benefit of
    # of forcing implementers to account for all outputs and intermediaries is
    # a very nice one.

    var_to_raggedness = {}
    for invar, rav in zip(jaxpr.invars, ragged_axis_values):
      var_to_raggedness[invar] = rav

    for eqn in jaxpr.eqns:
      prim = eqn.primitive
      if prim not in batching.ragged_prop_rules:
        raise NotImplementedError(f"Not implemented - ragged prop for {prim}")
      rule = batching.ragged_prop_rules[prim]

      invar_raggedness = [
          (
              var_to_raggedness.get(invar, None)
              if isinstance(invar, jax_core.Var)
              else None
          )
          for invar in eqn.invars
      ]
      try:
        invar_raggedness, outvar_raggedness = rule(
            eqn.params, invar_raggedness, eqn.outvars  # type: ignore[arg-type]
        )
      except Exception as e:
        raise RuntimeError(
            f"Failed to run rule for {prim}. invars: {eqn.invars}, outvars:"
            f" {eqn.outvars}. Underlying reason: {e}"
        ) from e

      for invar, rav in zip(eqn.invars, invar_raggedness):  # type: ignore[assignment]
        if isinstance(invar, jax_core.Var):
          var_to_raggedness[invar] = rav
      for outvar, rav in zip(eqn.outvars, outvar_raggedness):
        if isinstance(outvar, jax_core.Var):
          var_to_raggedness[outvar] = rav

    for pos, invar in enumerate(jaxpr.invars):
      ragged_axis_values[pos] = var_to_raggedness[invar]

    per_input_ragged_axis_dim: list[int | None] = []
    for rav in ragged_axis_values:
      if rav is not None:
        per_input_ragged_axis_dim.append(rav[1])
      else:
        per_input_ragged_axis_dim.append(None)

    def when_wrapped_kernel(lengths_ref, *args, **kwargs):
      b_idx = primitives.program_id(mapped_dim_idx)

      b_len = lengths_ref[b_idx]
      run_kernel = jnp.array(True)
      for i, _ in enumerate(args):
        ragged_axis_dim = per_input_ragged_axis_dim[i]
        if ragged_axis_dim is None:
          continue
        arg_i_idx = (
            primitives.program_id(ragged_axis_dim)
            * pallas_core._get_block_dim_size(block_shapes[i][ragged_axis_dim])
        )
        run_kernel = jnp.logical_and(run_kernel, arg_i_idx < b_len)

      # TODO(mvoz): Unimplemented primitive in pallas
      # b_len_mod = jnp.equal(jnp.mod(b_len, val_at_ragged_dim), 0)
      # checkify.check(b_len_mod, "b_len % val_at_ragged_dim != 0")

      @pallas_helpers.when(run_kernel)
      def f():
        # Important! This allows us to trace the inner kernel with the correct
        # grid to preserve user program_id semantics. Ex: program_id(0) will
        # always be analogous to program_id(1) in the outer kernel.
        with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
          jax_core.eval_jaxpr(jaxpr, (), *args, **kwargs)

    kernel_avals = [lengths_aval] + [v.aval for v in jaxpr.invars]
    flat_kernel_avals, kernel_in_tree = tree_util.tree_flatten(
        list(kernel_avals)
    )

    def _rewrite_index_jaxpr(enumerate_batched_block_mapping):
      arg_pos, batched_block_mapping = enumerate_batched_block_mapping
      indexer_avals = [
          v.aval for v in batched_block_mapping.index_map_jaxpr.jaxpr.invars
      ]
      flat_indexer_avals, indexer_in_tree = tree_util.tree_flatten(
          list(indexer_avals)
      )

      def index_rewrite_kernel(*indexer_args):
        ragged_axis_dim = per_input_ragged_axis_dim[arg_pos]

        # the problem here seems to be that we are rnning this for all inputs, per input, because they each have an indexer - which means
        # that the indexer for output isn't getting written - before, it always was

        lengths_ref = indexer_args[-1]
        rest_indexer_args = indexer_args[:-1]
        # Lengths are always the last argument of the indexer.
        # lengths_ref = args[-1]
        # Invariant: Stacked axis is enforced to be the mapped axis above.
        b_idx = indexer_args[mapped_dim_idx]

        nargs = list(rest_indexer_args)

        if ragged_axis_dim is not None:
          val_at_ragged_dim = pallas_core._get_block_dim_size(
              batched_block_mapping.block_shape[ragged_axis_dim])

          # The current index into the ragged dimension.
          # Invariant: There is only one ragged dimension, enforced above.
          i_idx = indexer_args[ragged_axis_dim]

          # grid space -> element space
          i_len = i_idx * val_at_ragged_dim

          # The length of the current batch.
          b_len = lengths_ref[b_idx]

          # Have we reached the end of the current batch?
          not_done = i_len < b_len

          am_last_batch = b_idx == axis_size - 1
          last_good_block = lax.div(b_len, val_at_ragged_dim) - 1

          # The logic below can be thought of as:
          #  if index_oob_ragged:
          #    if not last_batch:
          #      batch_idx += 1
          #      ragged_idx = 0
          #    else:
          #      ragged_idx = last_good_block
          #
          # wherein we find the next good block by incrementing the batch index
          # and setting the ragged index to 0 if we are not in the last batch.
          # Otherwise, we set the ragged index to the last good block.
          b_next = jnp.where(
              not_done, b_idx, jnp.where(am_last_batch, b_idx, b_idx + 1)
          )
          i_next = jnp.where(
              not_done, i_idx, jnp.where(am_last_batch, last_good_block, 0)
          )
          nargs[ragged_axis_dim] = i_next
          nargs[mapped_dim_idx] = b_next

        nargs = nargs + [lengths_ref]
        return jax_core.eval_jaxpr(
            batched_block_mapping.index_map_jaxpr.jaxpr,
            batched_block_mapping.index_map_jaxpr.consts,
            *nargs,
        )
      index_jaxpr, _ = _trace_kernel_to_jaxpr(
          index_rewrite_kernel,
          batched_block_mapping.index_map_jaxpr.jaxpr.debug_info,
          batched_grid_mapping,
          tuple(flat_indexer_avals),
          indexer_in_tree,
          tuple(() for _ in flat_indexer_avals),
          indexer=True,
      )

      batched_block_mapping = batched_block_mapping.replace(
          index_map_jaxpr=pe.close_jaxpr(index_jaxpr)
      )
      return batched_block_mapping

    # Important! This allows us to trace the outer kernel with the correct grid
    # to enable accessing the batch program_id.
    with pallas_core.tracing_grid_env(batched_grid_mapping.grid, ()):
      batched_block_mappings = map(
          _rewrite_index_jaxpr, enumerate(batched_block_mappings)
      )

      batched_grid_mapping = batched_grid_mapping.replace(
          block_mappings=tuple(batched_block_mappings),
      )

      jaxpr, consts = _trace_kernel_to_jaxpr(
          when_wrapped_kernel,
          jaxpr.debug_info,
          batched_grid_mapping,
          tuple(flat_kernel_avals),
          kernel_in_tree,
          tuple(() for _ in flat_kernel_avals),
      )
      if consts:
        raise NotImplementedError("consts not supported in pallas_call")

    # We need to rewrite the input_output_aliases here, the initial call
    # to broadcast is done, and we have inserted a new input (lengths), so
    # there's an off-by-one here now.
    new_input_output_aliases = []
    for k, v in input_output_aliases:
      new_input_output_aliases.append((k + 1, v))
    input_output_aliases = tuple(new_input_output_aliases)

    # assert ragged_axis_length is not None
    args = (ragged_axis_length, *args)
  assert all(isinstance(aval, jax_core.ShapedArray) for aval in out_avals)

  batched_out_avals = []
  for aval in out_avals:
    sharding = aval.sharding.update(spec=tuple_insert(aval.sharding.spec, 0, None))
    shape = tuple_insert(aval.shape, 0, axis_size)
    batched_out_avals.append(aval.update(shape=shape, sharding=sharding))
  batched_out_avals = tuple(batched_out_avals)

  out = pallas_call_p.bind(
      *dynamic_grid_args,
      *args,
      jaxpr=jaxpr,
      grid_mapping=batched_grid_mapping,
      mesh=mesh,
      input_output_aliases=input_output_aliases,
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
      cost_estimate=batched_cost_estimate,
      out_avals=batched_out_avals,
      backend=backend,
      metadata=metadata,
  )
  return out, (0,) * len(out)


batching.primitive_batchers[pallas_call_p] = _pallas_call_batching_rule


def checkify_pallas_kernel_body_jaxpr(
    body_jaxpr: jax_core.ClosedJaxpr,
    enabled_errors,
    error: checkify.Error,
    grid_mapping: GridMapping) -> tuple[
        jax_core.ClosedJaxpr, tree_util.PyTreeDef, set[checkify.ErrorEffect]]:
  err_vals, err_tree = tree_util.tree_flatten(error)
  err_vals = map(jax_core.get_aval, err_vals)
  flat_err_and_in_vals = [*err_vals, *body_jaxpr.in_avals]

  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    checked_jaxpr, out_tree, error_effects = checkify.jaxpr_to_checkify_jaxpr(
        body_jaxpr, enabled_errors, err_tree, *flat_err_and_in_vals)
  return checked_jaxpr, out_tree, error_effects

def pallas_call_checkify_oob_grid(error: checkify.Error,
                                  enabled_errors,
                                  args: jax_core.Value,
                                  grid_mapping: GridMapping,
                                  input_output_aliases) -> checkify.Error:
  if checkify.OOBError not in enabled_errors:
    return error
  dynamic_grid_args, args = split_list(
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  output_args = hlo_interpreter._initialize_output_vals(grid_mapping.block_mappings_output,
                                    args, input_output_aliases)
  scalars, input_args, _ = split_list(
      args, [grid_mapping.num_index_operands,
             grid_mapping.num_inputs],
  )
  dynamic_grid_args_iter = iter(dynamic_grid_args)
  grid = tuple(
      a if a is not pallas_core.dynamic_grid_dim
      else next(dynamic_grid_args_iter)
      for a in grid_mapping.grid
  )
  grid_start_indices = (jnp.int32(0),) * len(grid)
  if grid:
    num_iterations = reduce(jnp.multiply, grid)  # type: ignore[arg-type]
  else:
    # Base case is always one iteration when grid is ()
    num_iterations = 1

  is_indexing_dim = [
      tuple(isinstance(b, pallas_core.Squeezed) for b in bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  block_shapes = [
      pallas_core._get_block_shape(bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  # The scan carry: (i, loop_idx, *consts, *ins, *outs, *scratch)
  # i:int32 is the iteration index
  # loop_idx: tuple[int32] are the program ids for each grid axis
  def cond(carry):
    i, *_ = carry
    return i < num_iterations
  def body(carry):
    i, loop_idx, blocks = carry
    if grid_mapping.local_grid_env is not None:
      local_grid_env = grid_mapping.local_grid_env(loop_idx, grid)
    else:
      local_grid_env = tuple(
          pallas_core.GridAxis(idx, b)
          for dim, (idx, b) in enumerate(zip(loop_idx, grid))
          if dim not in grid_mapping.vmapped_dims
      )
    with pallas_core.grid_env(local_grid_env):
      start_indices = [
          None if bm is None else bm.compute_start_indices_interpret(loop_idx, *scalars)
          for bm in grid_mapping.block_mappings]
    # We perform a dynamic slice on the i/o blocks, which will be checked by
    # checkify for OOB accesses.
    blocks = map(hlo_interpreter._dynamic_slice, start_indices, block_shapes,
        [*input_args, *output_args], is_indexing_dim)
    return (i + 1, hlo_interpreter._get_next_indices(grid, loop_idx), blocks)
  def f(_):
    return lax.while_loop(
        cond, body, (jnp.int32(0), grid_start_indices, [jnp.zeros(shape) for shape in block_shapes])
    )
  flat_args, jaxpr_in_tree = jax.tree_util.tree_flatten((jnp.int32(0),))
  wrapped_loop, _ = api_util.flatten_fun_nokwargs(
      lu.wrap_init(f,
                   debug_info=api_util.debug_info("checkify oob_grid_access",
                                                  f, (0,), {})),
      jaxpr_in_tree)
  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    avals_in = map(jax_core.get_aval, flat_args)
    traced_loop, _, consts = pe.trace_to_jaxpr_dynamic(
        wrapped_loop, list(avals_in))
    traced_loop = jax_core.ClosedJaxpr(traced_loop, consts)
  out_error, _ = checkify.checkify_jaxpr(
      traced_loop, checkify.index_checks, error, flat_args)
  return out_error

def pallas_call_checkify_rule(error: checkify.Error,
                              enabled_errors,
                              *args: jax_core.Value,
                              jaxpr: jax_core.Jaxpr,
                              interpret: Any,
                              input_output_aliases: tuple[tuple[int, int], ...],
                              grid_mapping: GridMapping,
                              out_avals: tuple[jax_core.AbstractValue, ...],
                              **kwargs):
  # Check for OOB accesses in the grid.
  error = pallas_call_checkify_oob_grid(error, enabled_errors,
                                        args, grid_mapping,
                                        input_output_aliases)
  # We implement the checkify rule in 4 steps:
  # 1) First, trace the kernel body to get the expected error shapes.
  # 2) Checkify the kernel body to obtain a jaxpr with errors as inputs
  #   and outputs.
  # 3) Create a new kernel which stores the errors in output memrefs instead of
  #   returning them, since pallas kernels do not return outputs.
  # 4) Create block specs for the error state and call pallas_call with
  #   the new kernel.
  dynamic_grid_bounds, scalars, args = split_list(
      args, [grid_mapping.num_dynamic_grid_bounds,
             grid_mapping.num_index_operands]
  )
  num_scalars = len(scalars)
  num_kernel_inputs = len(args)
  num_kernel_outputs = grid_mapping.num_outputs

  # Trace the jaxpr to get an initial error value so the kernel jaxpr has all of
  # the required inputs.
  closed_jaxpr = pe.close_jaxpr(jaxpr)
  _jaxpr, _, error_effects = checkify_pallas_kernel_body_jaxpr(
      closed_jaxpr, enabled_errors, error, grid_mapping)
  error = error._add_placeholder_effects(error_effects)
  err_vals, err_in_tree = jax.tree.flatten(error)
  shaped_err_avals = map(jax_core.get_aval, err_vals)

  # Trace the kernel jaxpr to get a checkified jaxpr. This jaxpr will have
  # all enabled errors removed, but have the error as inputs and return values.
  input_avals = [v.aval for v in jaxpr.invars]
  num_err_vals = len(err_vals)
  shaped_input_avals = tuple(input_avals)
  checkify_in_avals = [*shaped_err_avals,
                       *shaped_input_avals]
  closed_kernel_jaxpr = pe.close_jaxpr(jaxpr)
  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    checked_jaxpr, error_out_tree, _ = checkify.jaxpr_to_checkify_jaxpr(
        closed_kernel_jaxpr, enabled_errors, err_in_tree, *checkify_in_avals)

  # Create a new kernel to remove the error as an return value and instead
  # write them to a memref. This is because pallas kernels are expected
  # to have no return values but instead write their outputs to a ref.
  def checked_kernel_fn(*args):
    (scalars, in_error_refs, inputs, out_error_refs, outputs, scratch
     ) = split_list(
        args,
        [num_scalars, num_err_vals,
         num_kernel_inputs, num_err_vals, num_kernel_outputs])
    # TODO(b/350593266): Remove zero-indexing once we support ()-shaped scalars.
    input_error_vals = [err_ref[0, 0] for err_ref in in_error_refs]
    # We need to re-order the inputs here. A checkified jaxpr always expects
    # errors before other arguments.
    jaxpr_args = [*input_error_vals, *scalars, *inputs, *outputs, *scratch]
    assert len(checked_jaxpr.jaxpr.invars) == len(jaxpr_args)
    result_flat = jax_core.eval_jaxpr(
        checked_jaxpr.jaxpr, checked_jaxpr.consts, *jaxpr_args)
    output_errors, _ = split_list(result_flat, [num_err_vals])
    # Store new errors back in the error refs.
    for in_ref, out_ref, error in zip(
        in_error_refs, out_error_refs, output_errors):
      in_ref[0, 0] = error
      out_ref[0, 0] = error
    return []

  # Trace the new checked_kernel_fn with Memref inputs so that
  # we can replace the old kernel jaxpr with the new checked jaxpr in
  # pallas_call.

  # ensure_2d_shape is only necessary because pallas does not support
  # ()-shaped Memrefs.
  # TODO(b/350593266): Remove once we support ()-shaped scalars.
  def _ensure_2d_error_shape(arg):
    if isinstance(arg, jax_core.ShapedArray):
      dtype = arg.dtype
      return jax_core.ShapedArray((1, 1) + arg.shape, dtype=dtype,
                                  weak_type=arg.weak_type)
    elif isinstance(arg, jax.Array):
      return jnp.reshape(arg, (1, 1) + arg.shape)
    else:
      return jnp.array([[arg]])
  shaped_err_avals = map(_ensure_2d_error_shape, shaped_err_avals)
  err_vals = map(_ensure_2d_error_shape, err_vals)

  error_memref_aval = [state.AbstractRef(
      err_val, pallas_core.MemorySpace.ERROR) for err_val in shaped_err_avals]
  shaped_scalar_avals, input_aval, output_aval, scratch_aval = split_list(
      shaped_input_avals, [num_scalars, num_kernel_inputs, num_kernel_outputs])
  retrace_in_avals = [*shaped_scalar_avals, *error_memref_aval, *input_aval,
                      *error_memref_aval, *output_aval, *scratch_aval]
  jaxpr_flat_avals, jaxpr_in_tree = tree_util.tree_flatten(retrace_in_avals)
  debug_info = api_util.debug_info("checkify_pallas", checked_kernel_fn,
                              retrace_in_avals, {})
  wrapped_kernel_with_err, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(checked_kernel_fn, debug_info=debug_info), jaxpr_in_tree)

  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    final_jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(
        wrapped_kernel_with_err, jaxpr_flat_avals)

  # Prepare pallas_call inputs. We need to create new block specs
  # for the new error inputs and outputs.
  error_block_specs = [pallas_core.BlockSpec(None, None)] * len(shaped_err_avals)
  error_paths, _ = unzip2(tree_util.tree_flatten_with_path(error_block_specs)[0])
  error_origins = tuple(f"errors[{tree_util.keystr(p)}" for p in error_paths)
  error_block_mappings = map(
      partial(
          pallas_core._convert_block_spec_to_block_mapping,
          index_map_avals=grid_mapping.index_map_avals,
          index_map_tree=grid_mapping.index_map_tree,
          grid=grid_mapping.grid,
          mapped_dims=grid_mapping.vmapped_dims,
          debug=True,
      ),
      error_block_specs,
      error_origins,
      shaped_err_avals,
  )
  input_block_mappings, output_block_mappings = split_list(
      grid_mapping.block_mappings, [num_kernel_inputs,])
  grid_mapping_with_error = grid_mapping.replace(
      block_mappings=(*error_block_mappings, *input_block_mappings,
                      *error_block_mappings, *output_block_mappings),
      num_inputs=grid_mapping.num_inputs + len(error_block_mappings),
      num_outputs=grid_mapping.num_outputs + len(error_block_mappings)
  )
  # Bump all input_output_aliases by num_err_vals to make room for error
  # TODO(justinfu): Don't bump scalars here.
  input_output_aliases = tuple(
      (i+num_err_vals, o+num_err_vals) for (i, o) in input_output_aliases)
  input_output_aliases_with_error = tuple(
      (i+num_scalars, i) for i in range(num_err_vals)) + input_output_aliases

  new_vals_in = [*scalars, *err_vals, *args]
  new_out_avals = (*shaped_err_avals, *out_avals)
  result = pallas_call_p.bind(*dynamic_grid_bounds, *new_vals_in,
    jaxpr=final_jaxpr,
    interpret=interpret,
    grid_mapping=grid_mapping_with_error,
    input_output_aliases=input_output_aliases_with_error,
    out_avals=new_out_avals,
    **kwargs)
  errors, results = split_list(result, [num_err_vals])
  # TODO(b/350593266): Remove line below once we support ()-shaped scalars.
  errors = [err_val[0, 0] for err_val in errors]
  new_error, _ = jax.tree.unflatten(error_out_tree, errors)
  return new_error, results
checkify.error_checks[pallas_call_p] = pallas_call_checkify_rule


def _trace_kernel_to_jaxpr(
    fun: Callable,
    debug_info: jax_core.DebugInfo,
    grid_mapping: GridMapping,
    kernel_avals: tuple[pallas_core.AbstractMemRef, ...],
    kernel_in_tree: tree_util.PyTreeDef,
    kernel_in_transforms: tuple[tuple[pallas_core.Transform, ...], ...],
    indexer: bool = False,
) -> tuple[jax_core.Jaxpr, tuple[jax.Array, ...]]:
  wrapped_kernel_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun, debug_info=debug_info), kernel_in_tree)
  wrapped_kernel_fun = primitives.wrap_with_transforms(
      wrapped_kernel_fun, kernel_in_transforms
  )
  with grid_mapping.trace_env(), config._check_vma(False):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_kernel_fun,
                                                     kernel_avals)
    if consts:
      consts_avals = [jax_core.get_aval(c) for c in consts]
      if any(not isinstance(aval, state.AbstractRef) for aval in consts_avals):
        raise ValueError(
            f"The kernel function in the pallas_call {debug_info.func_src_info} "
            f"captures constants {consts_avals}. "
            "You should pass them as inputs")

  kernel_out_tree = out_tree_thunk()
  if not indexer and kernel_out_tree != tree_util.tree_structure(None):
    raise ValueError(
        f"The kernel function in the pallas_call {debug_info.func_src_info} "
        f"should return None. It returns a PyTree: {kernel_out_tree}")
  return jaxpr, tuple(consts)


_PALLAS_USE_MOSAIC_GPU = config.bool_state(
    "jax_pallas_use_mosaic_gpu",
    default=config.bool_env("JAX_PALLAS_USE_MOSAIC_GPU", False),
    help=(
        "If True, lower Pallas kernels to the experimental Mosaic GPU"
        " dialect, instead of Triton IR."
    ),
)


_PALLAS_VERBOSE_ERRORS = config.bool_flag(
    "jax_pallas_verbose_errors",
    default=config.bool_env("JAX_PALLAS_VERBOSE_ERRORS", False),
    help=(
        "If True, print verbose error messages for Pallas kernels."
    ),
)


def _verbose_errors_enabled() -> bool:
  return _PALLAS_VERBOSE_ERRORS.value


def _unsupported_lowering_error(platform: str) -> Exception:
  return ValueError(
      f"Cannot lower pallas_call on platform: {platform}. To use Pallas on GPU,"
      " install jaxlib GPU 0.4.24 or newer. To use Pallas on TPU, install"
      " jaxlib TPU and libtpu. See"
      " https://docs.jax.dev/en/latest/installation.html."
  )


def _pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    interpret: Any,
    backend: Backend | None,
    **params,
):
  if params['jaxpr'].constvars:
    raise ValueError('Cannot lower a pallas_call with constants.')
  if interpret:
    if isinstance(interpret, mosaic_tpu_interpret.InterpretParams):
      impl = partial(mosaic_tpu_interpret.interpret_pallas_call,
                     interpret_params=interpret,
                     **params)
    else:
      impl = partial(hlo_interpreter.pallas_call_hlo_interpret,
                     backend=backend,
                     **params)
    return mlir.lower_fun(impl, multiple_results=True)(ctx, *in_nodes)

  def cpu_lowering(ctx: mlir.LoweringRuleContext,
                   *in_nodes: mlir.ir.Value | Sequence[mlir.ir.Value],
                   **params):
    raise ValueError("Only interpret mode is supported on CPU backend.")

  def tpu_lowering(ctx: mlir.LoweringRuleContext,
                   *in_nodes: mlir.ir.Value | Sequence[mlir.ir.Value],
                   **params):
    if backend and backend != "mosaic_tpu":
      raise ValueError("Only mosaic backend supported for TPU")
    if mosaic_tpu_backend is None:
      raise _unsupported_lowering_error("tpu")
    return mosaic_tpu_backend.pallas_call_tpu_lowering_rule(
        ctx, *in_nodes, **params
    )

  def gpu_lowering(ctx: mlir.LoweringRuleContext,
                   *in_nodes: mlir.ir.Value | Sequence[mlir.ir.Value],
                   **params):
    try:
      match backend:
        case "mosaic_gpu":
          from jax._src.pallas.mosaic_gpu import pallas_call_registration
        case "triton":
          from jax._src.pallas.triton import pallas_call_registration  # type: ignore
        case None:
          if _PALLAS_USE_MOSAIC_GPU.value:
            from jax._src.pallas.mosaic_gpu import pallas_call_registration
          else:
            from jax._src.pallas.triton import pallas_call_registration  # type: ignore
        case _:
          raise ValueError(f"Unsupported backend: {backend}")
    except ImportError as e:
      raise _unsupported_lowering_error("gpu")

    return pallas_call_registration.pallas_call_lowering(
        ctx, *in_nodes, **params
    )

  return mlir.lower_per_platform(ctx, "pallas_call",
                                 dict(cpu=cpu_lowering,
                                      tpu=tpu_lowering,
                                      cuda=gpu_lowering,
                                      rocm=gpu_lowering),
                                 None,  # default_rule
                                 effects.no_effects,
                                 *in_nodes,
                                 interpret=interpret,
                                 **params)


mlir.register_lowering(pallas_call_p, _pallas_call_lowering)


def _pallas_custom_str_eqn_compact(
    prim: jax_core.Primitive, params: dict[Any, Any]
) -> str:
  del prim, params
  # Hide most info from compact str representation
  return "pallas_call"
jax_core.custom_str_eqn_compact_rules[pallas_call_p] = (
    _pallas_custom_str_eqn_compact
)

def _pallas_call_typecheck_rule(*in_avals, grid_mapping, **params):
  with grid_mapping.trace_env():
    return pallas_call_p.abstract_eval(
        *in_avals, grid_mapping=grid_mapping, **params
    )
jax_core.custom_typechecks[pallas_call_p] = _pallas_call_typecheck_rule

def _convert_out_shape_to_aval(out_shape: Any) -> jax_core.AbstractValue:
  match out_shape:
    case jax.ShapeDtypeStruct():
      if config._check_vma.value:
        if out_shape.vma is None:
          raise ValueError(
              "When `check_vma=True` on `jax.shard_map`, `vma` on"
              " `jax.ShapeDtypeStruct` must not be `None`. Please specify how the"
              " output should be varying across mesh axes using the `vma`"
              " argument of `jax.ShapeDtypeStruct` or set `check_vma=False` on"
              " `jax.shard_map`.")
        return jax_core.ShapedArray(
            shape=out_shape.shape, dtype=out_shape.dtype, vma=out_shape.vma)
      return jax_core.ShapedArray(shape=out_shape.shape, dtype=out_shape.dtype)
    case pallas_core.MemoryRef():
      return out_shape.get_array_aval()
    case _:
      if type(out_shape) in pallas_core._out_shape_to_aval_mapping:
        return pallas_core._out_shape_to_aval_mapping[type(out_shape)](
            out_shape
        )
      if not (hasattr(out_shape, "shape") and hasattr(out_shape, "dtype")):
        raise ValueError(f"Invalid out_shape type: {type(out_shape)}")
      return jax_core.ShapedArray(shape=out_shape.shape, dtype=out_shape.dtype)


@state_discharge.register_discharge_rule(pallas_call_p)
def _pallas_call_state_discharge_rule(
    avals_in,
    avals_out,
    *args,
    jaxpr: jax_core.Jaxpr,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    debug: bool,
    interpret: Any,
    compiler_params: Any,
    cost_estimate: CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    backend: Backend | None,
    metadata: FrozenDict[str, str] | None,
):
  del avals_out
  assert all(isinstance(v.aval, state.AbstractRef) for v in jaxpr.constvars)
  num_refs = len(jaxpr.constvars)
  ref_avals, rest_in_avals = split_list(avals_in, [num_refs])
  assert all(isinstance(ref_aval, state.AbstractRef) for ref_aval in ref_avals)
  ref_avals = [
      state.AbstractRef(
          ref_aval.inner_aval, pallas_core.MemorySpace.ANY
      )
      for ref_aval in ref_avals
  ]
  ref_block_specs = [
      pallas_core.BlockSpec(memory_space=pallas_core.MemorySpace.ANY)
  ] * num_refs
  ref_block_mappings = [
      block_spec.to_block_mapping(
          origin="",  # TODO(sharadmv): enable origins for refs
          array_aval=ref_aval.inner_aval,  # type: ignore[arg-type]
          index_map_avals=grid_mapping.index_map_avals,
          index_map_tree=grid_mapping.index_map_tree,
          grid=grid_mapping.grid,
          mapped_dims=grid_mapping.mapped_dims,
          debug=debug,
      )
      for ref_aval, block_spec in zip(ref_avals, ref_block_specs)
  ]
  in_block_mappings, out_block_mappings = split_list(
      grid_mapping.block_mappings, [grid_mapping.num_inputs]
  )
  new_block_mappings = (
      *ref_block_mappings,
      *in_block_mappings,
      *ref_block_mappings,
      *out_block_mappings,
  )
  new_grid_mapping = grid_mapping.replace(
      block_mappings=new_block_mappings,
      num_inputs=grid_mapping.num_inputs + num_refs,
      num_outputs=grid_mapping.num_outputs + num_refs)
  new_input_output_aliases = [
      (i + grid_mapping.num_index_operands, i) for i in range(num_refs)
  ]
  for i, o in input_output_aliases:
    new_input_output_aliases.append((i + num_refs, o + num_refs))
  ref_out_avals = [ref_aval.inner_aval for ref_aval in ref_avals]
  new_out_avals = (*ref_out_avals, *out_avals)
  ref_args, dynamic_grid_bounds, index_operands, rest_args = split_list(
      args,
      [
          num_refs,
          grid_mapping.num_dynamic_grid_bounds,
          grid_mapping.num_index_operands,
      ],
  )
  def _rewritten_body(*args):
    index_args, in_args, out_args, rest_args = split_list(
        args, [new_grid_mapping.num_index_operands, new_grid_mapping.num_inputs,
               new_grid_mapping.num_outputs])
    ref_in_args, in_args = split_list(in_args, [num_refs])
    ref_out_args, out_args = split_list(out_args, [num_refs])
    # We don't care about ref_out_args because they are aliased to ref_in_args
    del ref_out_args
    jax_core.eval_jaxpr(
        jaxpr, ref_in_args, *index_args, *in_args, *out_args, *rest_args
    )
    return []
  index_map_avals, jaxpr_in_avals, jaxpr_out_avals, jaxpr_rest_avals = (
      split_list(
          [v.aval for v in jaxpr.invars],
          [
              grid_mapping.num_index_operands,
              grid_mapping.num_inputs,
              grid_mapping.num_outputs,
          ],
      )
  )
  new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_rewritten_body, debug_info=jaxpr.debug_info),
      [
          *index_map_avals,
          *ref_avals,
          *jaxpr_in_avals,
          *ref_avals,
          *jaxpr_out_avals,
          *jaxpr_rest_avals,
      ],
  )
  out_flat = pallas_call_p.bind(
      *consts,
      *dynamic_grid_bounds,
      *index_operands,
      *ref_args,
      *rest_args,
      jaxpr=new_jaxpr,
      input_output_aliases=tuple(new_input_output_aliases),
      grid_mapping=new_grid_mapping,
      mesh=mesh,
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
      cost_estimate=cost_estimate,
      out_avals=new_out_avals,
      backend=backend,
      metadata=metadata,
  )
  refs_out, rest = split_list(out_flat, [num_refs])
  updated_vals_in = refs_out + [None] * len(rest_in_avals)
  return updated_vals_in, rest


def pallas_call(
    kernel: Callable[..., None],
    out_shape: Any,
    *,
    grid_spec: GridSpec | None = None,
    grid: TupleGrid = (),
    in_specs: BlockSpecTree = no_block_spec,
    out_specs: BlockSpecTree = no_block_spec,
    scratch_shapes: ScratchShapeTree = (),
    input_output_aliases: Mapping[int, int] = {},
    debug: bool = False,
    interpret: Any = False,
    name: str | None = None,
    compiler_params: (
        Mapping[Backend, pallas_core.CompilerParams] | pallas_core.CompilerParams | None
    ) = None,
    cost_estimate: CostEstimate | None = None,
    backend: Backend | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[..., Any]:
  """Invokes a Pallas kernel on some inputs.

  See `Pallas Quickstart <https://docs.jax.dev/en/latest/pallas/quickstart.html>`_.

  Args:
    kernel: the kernel function, that receives a Ref for each input and output.
      The shape of the Refs are given by the ``block_shape`` in the
      corresponding ``in_specs`` and ``out_specs``.
    out_shape: a PyTree of :class:`jax.ShapeDtypeStruct` describing the shape
      and dtypes of the outputs.
    grid_spec: An alternative way to specify ``grid``, ``in_specs``,
      ``out_specs`` and ``scratch_shapes``. If given, those other parameters
      must not be also given.
    grid: the iteration space, as a tuple of integers. The kernel is executed
      as many times as ``prod(grid)``.
      See details at :ref:`pallas_grid`.
    in_specs: a PyTree of :class:`jax.experimental.pallas.BlockSpec` with
      a structure matching that of the positional arguments.
      The default value for ``in_specs`` specifies the whole array for all
      inputs, e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.
      See details at :ref:`pallas_blockspec`.
    out_specs: a PyTree of :class:`jax.experimental.pallas.BlockSpec` with
      a structure matching that of the outputs.
      The default value for ``out_specs`` specifies the whole array,
      e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.
      See details at :ref:`pallas_blockspec`.
    scratch_shapes: a PyTree of backend-specific temporary objects required
      by the kernel, such as temporary buffers, synchronization primitives,
      etc.
    input_output_aliases: a dictionary mapping the index of some inputs to
      the index of the output that aliases them. These indices are in the
      flattened inputs and outputs.
    debug: if True, Pallas prints various intermediate forms of the kernel
      as it is being processed.
    interpret: runs the ``pallas_call`` as a ``jax.jit`` of a scan over the
      grid whose body is the kernel lowered as a JAX function. This does not
      require a TPU or a GPU, and is the only way to run Pallas kernels on CPU.
      This is useful for debugging.
    name: if present, specifies the name to use for this kernel call in
      debugging and error messages. To this name we append the file and line
      where the kernel function is defined, .e.g: `{name} for kernel function
      {kernel_name} at {file}:{line}`. If missing, then we use `{kernel_name} at
      {file}:{line}`.
    compiler_params: Optional compiler parameters. The value should either be a
      backend-specific dataclass
      (:class:`jax.experimental.pallas.tpu.CompilerParams`,
      :class:`jax.experimental.pallas.triton.CompilerParams`,
      :class:`jax.experimental.pallas.mosaic_gpu.CompilerParams`) or a dict
      mapping backend name to the corresponding platform-specific dataclass.
    backend: Optional string literal one of  ``"mosaic_tpu"``, ``"triton"`` or
      ``"mosaic_gpu"`` determining the backend to be used. None means let Pallas
      decide.
    metadata: Optional dictionary of information about the kernel that will be
      serialized as JSON in the HLO. Can be used for debugging and analysis.

  Returns:
    A function that can be called on a number of positional array arguments to
    invoke the Pallas kernel.
  """
  if grid_spec is None:
    grid_spec = GridSpec(grid, in_specs, out_specs, scratch_shapes)
  else:
    if grid:
      raise ValueError(
          "If `grid_spec` is specified, then `grid` must "
          f"be `()`. It is {grid}")
    if in_specs is not no_block_spec:
      raise ValueError(
          "If `grid_spec` is specified, then `in_specs` must "
          f"be `no_block_spec`. It is {in_specs}")
    if out_specs is not no_block_spec:
      raise ValueError(
          "If `grid_spec` is specified, then `out_specs` must "
          f"be `no_block_spec`. It is {out_specs}")
    if scratch_shapes:
      raise ValueError(
          "If `grid_spec` is specified, then `scratch_shapes` must "
          f"be `()`. It is {scratch_shapes}")
  del grid, in_specs, out_specs
  return _pallas_call(
      kernel,
      out_shape,
      grid_spec=grid_spec,
      input_output_aliases=input_output_aliases,
      debug=debug,
      interpret=interpret,
      name=name,
      compiler_params=compiler_params,
      cost_estimate=cost_estimate,
      backend=backend,
      metadata=metadata,
  )



def _normalize_compiler_params(
    compiler_params: Mapping[Backend, pallas_core.CompilerParams] | pallas_core.CompilerParams | None,
) -> Mapping[Backend, pallas_core.CompilerParams]:
  if compiler_params is None:
    return FrozenDict({})
  if isinstance(compiler_params, CompilerParams):
    compiler_params = {compiler_params.BACKEND: compiler_params}
  assert isinstance(compiler_params, Mapping)
  for backend, params in compiler_params.items():
    if backend not in ["mosaic_tpu", "mosaic_gpu", "triton"]:
      raise ValueError(f"Unknown backend in compiler_params: {backend}")
    if not isinstance(params, CompilerParams):
      raise ValueError(
          f"Unexpected compiler_params for backend {backend}: {params}"
      )
    if params.BACKEND != backend:
      raise ValueError(
          f"Inconsistent backend in compiler_params: {params.BACKEND} !="
          f" {backend}"
      )
  if not isinstance(compiler_params, FrozenDict):
    compiler_params = FrozenDict(compiler_params)
  return compiler_params


def _pallas_call(
    kernel: Callable[..., None],
    out_shape: Any,
    *,
    grid_spec: GridSpec,
    mesh: pallas_core.Mesh | None = None,
    input_output_aliases: Mapping[int, int] = {},
    debug: bool = False,
    interpret: Any = False,
    name: str | None = None,
    compiler_params: (
        Mapping[Backend, CompilerParams] | CompilerParams | None
    ) = None,
    cost_estimate: CostEstimate | None = None,
    backend: Backend | None = None,
    metadata: dict[str, str] | None = None,
):
  interpret = (
      config.pallas_tpu_interpret_mode_context_manager.value or interpret)
  compiler_params = _normalize_compiler_params(compiler_params)

  if mesh is not None:
    if tuple(mesh.shape.values()) != grid_spec.grid:
      raise ValueError(
          f"Mesh shape {tuple(mesh.shape.values())} does not match grid "
          f"shape {grid_spec.grid}."
      )
    if backend is not None:
      raise ValueError("If `mesh` is specified, then `backend` must be `None`.")
    backend = mesh.backend

  grid_spec, dynamic_grid_bounds = pallas_core.unzip_dynamic_grid_bounds(grid_spec)
  # TODO(necula): this canonicalization may be convenient for some usage
  # but it is lossy, because it prevents expressing functions that return
  # lists.
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  flat_out_shapes_with_paths, out_tree = tree_util.tree_flatten_with_path(out_shape)
  out_paths, flat_out_shapes = unzip2(flat_out_shapes_with_paths)

  @partial(jax.jit, inline=True)
  def wrapped(*args):
    flat_args_with_paths, in_tree = tree_util.tree_flatten_with_path(args)
    in_paths, flat_args = unzip2(flat_args_with_paths)
    flat_in_avals = tuple(jax_core.get_aval(a) for a in flat_args)

    flat_out_avals = tuple(_convert_out_shape_to_aval(v)
                           for v in flat_out_shapes)

    in_origins = tuple(f"args{tree_util.keystr(p)}" for p in in_paths)
    out_origins = tuple(f"outputs{tree_util.keystr(p)}" for p in out_paths)
    # TODO(necula): check that input_output_aliases is well-formed: no duplicates, etc.
    kernel_args, grid_mapping = pallas_core.get_grid_mapping(
        grid_spec,
        flat_in_avals,
        in_tree,
        in_origins,
        flat_out_avals,
        out_tree,
        out_origins,
        debug,
    )
    flat_kernel_args, kernel_in_tree = tree_util.tree_flatten(kernel_args)
    flat_kernel_avals = tuple(
        x.ref if isinstance(x, state_types.TransformedRef) else x
        for x in flat_kernel_args
    )
    if config._check_vma.value:
      flat_kernel_avals = tuple(a.update_vma(frozenset())
                                for a in flat_kernel_avals)
    # Note that only a subset of all transforms can be found here, and they are
    # never expected to contain any arrays.
    kernel_arg_transforms = tuple(
        x.transforms if isinstance(x, state_types.TransformedRef) else ()
        for x in flat_kernel_args
    )
    kernel_dbg = api_util.debug_info("pallas_call kernel", kernel,
                                      kernel_args, {})
    if name is not None:
      kernel_dbg = kernel_dbg.replace_func_name(mlir.sanitize_name(name))
    jaxpr, consts = _trace_kernel_to_jaxpr(
        kernel, kernel_dbg, grid_mapping, flat_kernel_avals,
        kernel_in_tree, kernel_arg_transforms)
    for i_idx, o_idx in input_output_aliases.items():
      if i_idx not in range(len(flat_in_avals)):
        raise ValueError(
            f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' with "
            f"input index {i_idx} outside the range "
            f"[0, {len(flat_in_avals)})")
      if o_idx not in range(len(flat_out_avals)):
        raise ValueError(
            f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' with "
            f"output index {o_idx} outside the range "
            f"[0, {len(flat_out_avals)})")
      in_aval = flat_in_avals[i_idx]
      out_aval = flat_out_avals[o_idx]
      if isinstance(in_aval, jax_core.DShapedArray):
        new_shape = []
        for d in in_aval.shape:
          if isinstance(d, int):
            new_shape.append(d)
          else:
            new_shape.append(d.dtype.bound)

        in_aval = jax_core.ShapedArray(
            tuple(new_shape), in_aval.dtype, in_aval.weak_type
        )

      if in_aval.shape != out_aval.shape or in_aval.dtype != out_aval.dtype:
        raise ValueError(
            f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
            f"referring to input{tree_util.keystr(in_paths[i_idx])} with "
            f"abstract value {in_aval} "
            f"and to output{tree_util.keystr(out_paths[o_idx])} with "
            f"a different abstract value {out_aval}.")

    index_args, rest_args = split_list(flat_args, [grid_mapping.num_index_operands])
    out_flat = pallas_call_p.bind(
        *consts,
        *dynamic_grid_bounds,
        *index_args,
        *rest_args,
        out_avals=flat_out_avals,
        jaxpr=jaxpr,
        debug=debug,
        interpret=interpret,
        grid_mapping=grid_mapping,
        mesh=mesh,
        input_output_aliases=tuple(input_output_aliases.items()),
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
        backend=backend,
        metadata=FrozenDict(metadata) if metadata is not None else None,
    )
    out = tree_util.tree_unflatten(out_tree, out_flat)
    return out
  return wrapped


def in_path_to_input_origin(
    in_path: tree_util.KeyPath, arg_names: tuple[str, ...] | None
) -> pallas_core.OriginStr:
  """Converts `args[k]<rest>` into `arg_k_name<rest>`."""
  if arg_names is None:
    return f"args{tree_util.keystr(in_path)}"
  if len(in_path) == 0:
    return "args"
  arg_idx, *rest_path = in_path
  if isinstance(arg_idx, tree_util.SequenceKey) and arg_idx.idx < len(
      arg_names
  ):
    if arg_names[arg_idx.idx] is None:
      # TODO(necula): when is this needed?
      # Repro: pallas_test:test_with_input_output_aliasing
      return f"args{tree_util.keystr(in_path)}"
    return arg_names[arg_idx.idx] + tree_util.keystr(tuple(rest_path))
  else:
    return f"args{tree_util.keystr(tuple(in_path))}"


# We import the TPU backend at the top level because it defines flags. Note that
# we can only do that at the bottom of this file, because it also depends on
# this module already being initialized.

try:
  from jax._src.pallas.mosaic import pallas_call_registration as mosaic_tpu_backend
except ImportError:
  mosaic_tpu_backend = None  # type: ignore

try:
  from jax._src.pallas.mosaic import interpret as mosaic_tpu_interpret
except ImportError:
  mosaic_tpu_interpret = types.SimpleNamespace(  # type: ignore
      InterpretParams=types.new_class('_NoInstances', (enum.Enum,)),
  )
