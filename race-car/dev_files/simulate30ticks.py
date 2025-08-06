import time
import os
import sys
print(os.getcwd())

# Add the parent directory to the path so we can import from models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.PredictionModel import *

@jit
def _update_lane_prob(x_resolution: float, vel_resolution_coef: int, vel_res_coef_half: int, ego_velocity: float, lane_vel_bin_means: jnp.ndarray, joint_prob: jnp.ndarray) -> jnp.ndarray:
    # TODO doesn't remove cars o
    x_shift = jnp.round((lane_vel_bin_means - ego_velocity) / x_resolution)
    X, V = joint_prob.shape

    # Compute base probabilities: shape (X, V)
    p = joint_prob / vel_resolution_coef

    # Generate base index grids
    x = jnp.arange(X)[:, None]  # shape (X, 1)
    v = jnp.arange(V)[None, :]  # shape (1, V)

    # Compute destination x indices using broadcasted x_shift[v]
    x_dest = x + x_shift[v].astype(int)  # shape (X, V)
    x_dest = jnp.clip(x_dest, 0, X - 1)

    # Compute vertical offsets v ± 3
    offsets = jnp.arange(-vel_res_coef_half, vel_res_coef_half+1)[:, None, None]  # shape (vel_resolution_coef, 1, 1)
    v_dest = jnp.broadcast_to(v, (vel_resolution_coef, X, V)) + offsets  # shape (vel_resolution_coef, X, V)
    v_dest = jnp.clip(v_dest, 0, V - 1)

    # Broadcast x_dest and p to shape (vel_resolution_coef, X, V)
    x_dest_b = jnp.broadcast_to(x_dest[None, :, :], (vel_resolution_coef, X, V))  # shape (vel_resolution_coef, X, V)
    p_b = jnp.broadcast_to(p[None, :, :], (vel_resolution_coef, X, V))            # shape (vel_resolution_coef, X, V)

    # Flatten everything
    x_idx = x_dest_b.reshape(-1)  # shape (vel_resolution_coef*X*V,)
    v_idx = v_dest.reshape(-1)    # shape (vel_resolution_coef*X*V,)
    p_vals = p_b.reshape(-1)      # shape (vel_resolution_coef*X*V,)

    # Flatten (x, v) index into single index: idx = x * V + v
    flat_idx = x_idx * V + v_idx
    flat_size = X * V

    # Segment sum to aggregate values at flat indices
    flat_result = segment_sum(p_vals, flat_idx, num_segments=flat_size)

    # Reshape back to (X, V)
    new_joint_prob = flat_result.reshape((X, V))

    return new_joint_prob

# print the time it takes to update the model
start_time = time.time()
state = {
        "elapsed_ticks": 0,
        "velocity": {"x": 10, "y": 0},
        "sensors": {
            "front": 100,
            "right_front": 100,
            "right_side": 100,
            "right_back": 100,
            "back": 100,
            "left_back": 100,
            "left_side": 100,
            "left_front": 100,
            "front_left_front": 100,
            "front_right_front": 100,
            "right_side_front": 100,
            "right_side_back": 100,
            "back_right_back": 100,
            "back_left_back": 100,
            "left_side_back": 100,
            "left_side_front": 100,
        }
    }

for i in range(30):
    state["elapsed_ticks"] = i
    model._update_lane_prob(state)
end_time = time.time()
print("Time taken: ", end_time-start_time)