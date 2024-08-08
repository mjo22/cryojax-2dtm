"""Grid search method that produces a scaled version of the of cross-correlation."""

from typing import cast
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from cryojax.inference import (
    AbstractGridSearchMethod,
    MinimumSearchMethod,
    MinimumSolution,
    MinimumState,
)


class ScaledMinimumState(eqx.Module):

    minimum_state: MinimumState
    current_sum: Array
    current_sum_of_squares: Array


class ScaledMinimumSolution(eqx.Module):

    scaled_minimum_eval: Array
    minimum_solution: MinimumSolution
    state: ScaledMinimumState


class ScaledMinimumSearchMethod(AbstractGridSearchMethod, strict=True):

    minimum_search_method: MinimumSearchMethod

    def __init__(self, minimum_search_method: MinimumSearchMethod):
        self.minimum_search_method = eqx.tree_at(
            lambda x: x.store_current_eval, minimum_search_method, True
        )

    @override
    def init(self, tree_grid, f_struct, *, is_leaf=None) -> ScaledMinimumState:
        state = ScaledMinimumState(
            minimum_state=self.minimum_search_method.init(
                tree_grid, f_struct, is_leaf=is_leaf
            ),
            current_sum=jnp.full(f_struct.shape, 0.0, dtype=float),
            current_sum_of_squares=jnp.full(f_struct.shape, 0.0, dtype=float),
        )

        return state

    @override
    def update(
        self, fn, tree_grid_point, args, state: ScaledMinimumState, raveled_grid_index
    ) -> ScaledMinimumState:
        current_minimum_state = self.minimum_search_method.update(
            fn, tree_grid_point, args, state.minimum_state, raveled_grid_index
        )
        current_eval = cast(Array, current_minimum_state.current_eval)
        return ScaledMinimumState(
            minimum_state=current_minimum_state,
            current_sum=current_eval + state.current_sum,
            current_sum_of_squares=current_eval**2 + state.current_sum_of_squares,
        )

    @override
    def batch_update(
        self,
        fn,
        tree_grid_point_batch,
        args,
        state: ScaledMinimumState,
        raveled_grid_index_batch,
    ) -> ScaledMinimumState:
        current_minimum_state = self.minimum_search_method.update(
            fn, tree_grid_point_batch, args, state.minimum_state, raveled_grid_index_batch
        )
        current_eval = cast(Array, current_minimum_state.current_eval)
        return ScaledMinimumState(
            minimum_state=current_minimum_state,
            current_sum=jnp.sum(current_eval, axis=0) + state.current_sum,
            current_sum_of_squares=jnp.sum(current_eval**2, axis=0)
            + state.current_sum_of_squares,
        )

    @override
    def postprocess(
        self, tree_grid, final_state: ScaledMinimumState, f_struct, *, is_leaf=None
    ) -> ScaledMinimumSolution:
        # ... postprocessing in the `MinimumSearchMethod`
        minimum_solution = self.minimum_search_method.postprocess(
            tree_grid, final_state.minimum_state, f_struct, is_leaf=is_leaf
        )
        # ... compute the average and standard deviation of evaluations over all grid
        # points
        grid_size = minimum_solution.stats["grid_size"]
        average = final_state.current_sum / grid_size
        std = jnp.sqrt(final_state.current_sum_of_squares / grid_size - average**2)
        # ... scale the function evaluations by the average and std
        return ScaledMinimumSolution(
            scaled_minimum_eval=(final_state.minimum_state.current_minimum_eval - average)
            / std,
            minimum_solution=minimum_solution,
            state=final_state,
        )
