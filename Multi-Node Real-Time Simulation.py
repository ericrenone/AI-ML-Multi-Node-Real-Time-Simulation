#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical AI/ML Multi-Node Simulation
3D Real-Time Dynamic Forces Visualization with Attention

This module simulates dynamic force distribution across multiple nodes,
utilizing a pseudo-attention mechanism to normalize importance weights.
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import List, Tuple, Generator

# ---------------- Configuration ----------------
class Config:
    NODE_COUNT = 6
    STEPS = 50
    NODE_LOAD = [50, 60, 40, 55, 45, 65]
    CONNECTION_ANGLE = [math.pi/4, math.pi/3, math.pi/6, math.pi/4, math.pi/5, math.pi/3]
    ELASTICITY = [2e11, 5e9, 5e9, 2e10, 1e10, 5e9]
    EPSILON = 1e-12
    SAFETY_FACTOR = 5
    MAX_FORCE = 100.0  # for color normalization

# ---------------- Mechanics ----------------
def calculate_node_force(load: float, angle: float, connection: float) -> float:
    sin_conn = math.sin(connection)
    if abs(sin_conn) < Config.EPSILON:
        sin_conn = Config.EPSILON
    return load * math.sin(angle) / sin_conn

def dynamic_adjustment(elasticity: float, max_deflection: float, load: float) -> float:
    stiffness = elasticity * 0.01 / max_deflection
    adjusted = load * (1 - min(1.0, stiffness / max(load, Config.EPSILON)))
    return max(adjusted, 0.0)

# ---------------- Simulation Generator ----------------
def simulate_nodes() -> Generator[Tuple[int, List[List[float]], List[float]], None, None]:
    attention = [1.0 / Config.NODE_COUNT] * Config.NODE_COUNT
    dynamic_history = [[] for _ in range(Config.NODE_COUNT)]

    for step in range(Config.STEPS):
        forces = [
            dynamic_adjustment(
                Config.ELASTICITY[i],
                1.0,
                calculate_node_force(Config.NODE_LOAD[i], math.pi/6, Config.CONNECTION_ANGLE[i])
            ) for i in range(Config.NODE_COUNT)
        ]
        total = sum(forces)
        if total < Config.EPSILON:
            attention = [1.0 / Config.NODE_COUNT] * Config.NODE_COUNT
        else:
            attention = [f / total for f in forces]

        for i, f in enumerate(forces):
            dynamic_history[i].append(f)

        yield step, dynamic_history, attention

# ---------------- Real-Time Optimized 3D Visualization ----------------
def plot_3d_realtime():
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Timestep")
    ax.set_zlabel("Dynamic Force")
    ax.set_title("Real-Time 3D Node Forces Simulation")
    
    ax.set_xlim(1, Config.NODE_COUNT)
    ax.set_ylim(0, Config.STEPS)
    ax.set_zlim(0, max(Config.NODE_LOAD) * 1.5)

    node_indices = list(range(1, Config.NODE_COUNT + 1))
    colors_map = cm.get_cmap('viridis')

    # Initialize scatter objects for all nodes
    scatters = [
        ax.scatter([node_indices[i]], [0], [0], s=100, c=[colors_map(0)], label=f'Node {i+1}')
        for i in range(Config.NODE_COUNT)
    ]
    ax.legend(loc='upper left', fontsize='small')

    sim = simulate_nodes()
    final_state = None

    for step, forces_history, attention in sim:
        for i in range(Config.NODE_COUNT):
            xs = [node_indices[i]] * len(forces_history[i])
            ys = list(range(len(forces_history[i])))
            zs = forces_history[i]
            
            scatters[i]._offsets3d = (xs, ys, zs)
            # Update color based on latest force
            color_value = min(zs[-1] / Config.MAX_FORCE, 1.0)
            scatters[i].set_color([colors_map(color_value)])
            # Update size based on attention
            scatters[i]._sizes = [100 + 300 * attention[i]]

        plt.pause(0.01)
        final_state = (forces_history, attention)

    plt.ioff()
    plt.show()

    # ---------------- Print Final Summary ----------------
    if final_state:
        history, final_attention = final_state
        final_forces = [history[i][-1] for i in range(Config.NODE_COUNT)]
        print("\n" + "="*40)
        print(" FINAL SIMULATION SUMMARY ")
        print("="*40)
        for i in range(Config.NODE_COUNT):
            print(f"Node {i+1} | Force: {final_forces[i]:6.2f} | Attn: {final_attention[i]:.3f}")
        print(f"\nSafety Factor: x{Config.SAFETY_FACTOR}")
        print("="*40 + "\n")

# ---------------- Main ----------------
if __name__ == "__main__":
    plot_3d_realtime()
