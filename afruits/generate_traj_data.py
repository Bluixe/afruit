import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from afruits.utils.RuleBasedTrajGenerator import RuleBasedTrajGenerator


def plot_2d(traj, save_dir='visualizations', filename='rule_traj_2d.png'):
    states = traj['states']
    own_x, own_y = states[:, 0], states[:, 1]
    bdt_x, bdt_y = states[:, 6], states[:, 7]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.plot(own_x, own_y, label='Ownship', color='C0')
    ax.plot(bdt_x, bdt_y, label='Bandit', color='C1')

    # 起点/终点标记
    ax.scatter(own_x[0], own_y[0], c='C0', marker='o', s=40)
    ax.scatter(own_x[-1], own_y[-1], c='C0', marker='x', s=60)
    ax.scatter(bdt_x[0], bdt_y[0], c='C1', marker='o', s=40)
    ax.scatter(bdt_x[-1], bdt_y[-1], c='C1', marker='x', s=60)

    ax.set_title('Rule-based Air Combat Trajectory (2D XY)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, ls='--', alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    plt.tight_layout()
    fig.savefig(path)
    return fig, path


def plot_3d(traj, save_dir='visualizations', filename='rule_traj_3d.png'):
    states = traj['states']
    own_x, own_y, own_z = states[:, 0], states[:, 1], states[:, 2]
    bdt_x, bdt_y, bdt_z = states[:, 6], states[:, 7], states[:, 8]

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(own_x, own_y, own_z, label='Ownship', color='C0')
    ax.plot(bdt_x, bdt_y, bdt_z, label='Bandit', color='C1')

    # 起点/终点标记
    ax.scatter(own_x[0], own_y[0], own_z[0], c='C0', marker='o', s=30)
    ax.scatter(own_x[-1], own_y[-1], own_z[-1], c='C0', marker='x', s=50)
    ax.scatter(bdt_x[0], bdt_y[0], bdt_z[0], c='C1', marker='o', s=30)
    ax.scatter(bdt_x[-1], bdt_y[-1], bdt_z[-1], c='C1', marker='x', s=50)

    ax.set_title('Rule-based Air Combat Trajectory (3D)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Alt Z (m)')
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    plt.tight_layout()
    fig.savefig(path)
    return fig, path


if __name__ == '__main__':
    gen = RuleBasedTrajGenerator()
    num_trajectories = 100
    horizon = 80
    data = gen.generate_trajectories(num_trajectories=num_trajectories, horizon=horizon, with_reward=True)
    print(data["state_dim"], data["action_dim"], len(data["trajectories"]))
    
    # print(len(data['trajectories']))
    # 保存轨迹
    np.save("data/trajectory.npy", np.array(data["trajectories"]))
    meta_data = {
        "state_dim": data["state_dim"],
        "action_dim": data["action_dim"],
        "traj_len": horizon
    }
    import json
    with open("data/trajectory_meta_data.json", "w") as f:
        json.dump(meta_data, f)

    # 仅对第1条轨迹做可视化示例
    traj0 = data['trajectories'][0]
    fig2d, path2d = plot_2d(traj0)
    fig3d, path3d = plot_3d(traj0)

    print(f"2D图已保存至: {path2d}")
    print(f"3D图已保存至: {path3d}")