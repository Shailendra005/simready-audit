import coacd
import numpy as np


def decompose(mesh, threshold=0.05, seed=0, max_hulls=-1):
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)),
        threshold=threshold,
        max_convex_hull=max_hulls,
        seed=seed,
        preprocess_mode="auto",
        resolution=2000,
        mcts_nodes=20,
        mcts_iterations=150,
        mcts_max_depth=3,
    )
    return [(np.asarray(v, np.float64), np.asarray(f, np.int64)) for v, f in parts]
