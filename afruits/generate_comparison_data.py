"""
读取对比数据脚本
用于读取两种不同格式、不同维度、不同场景的博弈轨迹数据
并提供对比可视化方法
"""

import os
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体，避免警告
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
# 抑制matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from current font.*')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Dict, List, Any

from afruits.utils.RuleBasedTrajGenerator import RuleBasedTrajGenerator, PhysicsParams, RuleParams, InitialSpawnParams


class ComparisonDataGenerator:
    """对比数据读取器"""
    
    def __init__(self):
        self.output_dir = "data/comparison"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_dataset_1(self, num_trajectories: int = 50, horizon: int = 60) -> Dict:
        """
        读取数据集1：标准空战场景
        - 格式：JSON格式
        - 维度：12维状态空间（标准空战状态）
        - 场景：空战追击/规避场景
        - 特点：少量样本，中等长度
        """
        print("=" * 60)
        print("读取数据集1（JSON格式，12维，少量样本）")
        print("=" * 60)
        
        gen = RuleBasedTrajGenerator()
        data = gen.generate_trajectories(
            num_trajectories=num_trajectories,
            horizon=horizon,
            with_reward=True
        )
        
        # 转换为JSON格式
        json_data = {
            "data": [],
            "state_dim": data["state_dim"],
            "action_dim": data["action_dim"],
            "traj_length": horizon,
            "format": "json",
            "scenario": "空战场景",
            "description": "标准空战轨迹数据，包含追击、规避、巡航等典型场景"
        }
        
        for traj in data["trajectories"]:
            json_traj = {
                "states": traj["states"].tolist(),
                "actions": traj["actions"].tolist(),
                "rewards": traj.get("rewards", []).tolist() if "rewards" in traj else [],
                "next_states": traj.get("next_states", []).tolist() if "next_states" in traj else [],
                "dones": traj.get("dones", []).tolist() if "dones" in traj else [],
                "opponent_actions": traj.get("opponent_actions", []).tolist() if "opponent_actions" in traj else []
            }
            json_data["data"].append(json_traj)
        
        # 保存JSON格式
        json_path = os.path.join(self.output_dir, "dataset1_air_combat.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据集1已读取，保存至: {json_path}")
        print(f"  - 轨迹数量: {num_trajectories}")
        print(f"  - 轨迹长度: {horizon}")
        print(f"  - 状态维度: {data['state_dim']}")
        print(f"  - 动作维度: {data['action_dim']}")
        print(f"  - 格式: JSON")
        
        return json_data
    
    def generate_dataset_2(self, num_trajectories: int = 200, horizon: int = 100) -> Dict:
        """
        读取数据集2：扩展场景
        - 格式：NPY格式（NumPy数组）
        - 维度：18维状态空间（扩展状态，包含更多信息）
        - 场景：多机协同空战场景
        - 特点：大批量样本，较长轨迹
        """
        print("\n" + "=" * 60)
        print("读取数据集2（NPY格式，18维，大批量样本）")
        print("=" * 60)
        
        gen = RuleBasedTrajGenerator()
        # 读取基础轨迹
        base_data = gen.generate_trajectories(
            num_trajectories=num_trajectories,
            horizon=horizon,
            with_reward=True
        )
        
        # 扩展状态维度：从12维扩展到18维
        # 添加：相对速度、相对距离、角度差、能量状态等
        expanded_trajectories = []
        for traj in base_data["trajectories"]:
            states = traj["states"]  # [T, 12]
            T = states.shape[0]
            
            # 计算扩展特征
            own_pos = states[:, :3]  # [T, 3] x, y, z
            own_vel = states[:, 3:6]  # [T, 3] v, psi, vz
            opp_pos = states[:, 6:9]  # [T, 3]
            opp_vel = states[:, 9:12]  # [T, 3]
            
            # 相对位置和速度
            rel_pos = opp_pos - own_pos  # [T, 3]
            rel_vel = opp_vel - own_vel  # [T, 3]
            
            # 相对距离
            rel_dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)  # [T, 1]
            
            # 相对速度大小
            rel_speed = np.linalg.norm(rel_vel, axis=1, keepdims=True)  # [T, 1]
            
            # 角度差（航向差）
            angle_diff = (own_vel[:, 1:2] - opp_vel[:, 1:2])  # [T, 1] psi差
            
            # 能量状态（速度的平方，作为能量指标）
            energy_own = (own_vel[:, 0:1] ** 2)  # [T, 1]
            energy_opp = (opp_vel[:, 0:1] ** 2)  # [T, 1]
            
            # 高度差
            alt_diff = (opp_pos[:, 2:3] - own_pos[:, 2:3])  # [T, 1]
            
            # 组合扩展状态 [T, 18]
            expanded_states = np.concatenate([
                states,  # [T, 12] 原始状态
                rel_dist,  # [T, 1]
                rel_speed,  # [T, 1]
                angle_diff,  # [T, 1]
                energy_own,  # [T, 1]
                energy_opp,  # [T, 1]
                alt_diff,  # [T, 1]
            ], axis=1)
            
            expanded_traj = {
                "states": expanded_states,
                "actions": traj["actions"],
                "rewards": traj.get("rewards", np.zeros(T)),
                "next_states": expanded_states[1:] if expanded_states.shape[0] > 1 else expanded_states,
                "dones": traj.get("dones", np.zeros(T, dtype=bool)),
                "opponent_actions": traj.get("opponent_actions", np.zeros(T, dtype=int))
            }
            expanded_trajectories.append(expanded_traj)
        
        # 保存为NPY格式
        npy_path = os.path.join(self.output_dir, "dataset2_extended_combat.npy")
        np.save(npy_path, np.array(expanded_trajectories, dtype=object))
        
        # 保存元数据
        meta_data = {
            "state_dim": (18,),
            "action_dim": base_data["action_dim"],
            "traj_length": horizon,
            "format": "npy",
            "scenario": "扩展多机协同空战场景",
            "description": "扩展状态空间的空战轨迹数据，包含相对位置、速度、能量等扩展特征，适用于多机协同场景"
        }
        meta_path = os.path.join(self.output_dir, "dataset2_meta_data.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据集2已读取")
        print(f"元数据已读取")
        print(f"  - 轨迹数量: {num_trajectories}")
        print(f"  - 轨迹长度: {horizon}")
        print(f"  - 状态维度: (18,)")
        print(f"  - 动作维度: {base_data['action_dim']}")
        print(f"  - 格式: NPY")
        
        return {
            "trajectories": expanded_trajectories,
            "state_dim": (18,),
            "action_dim": base_data["action_dim"],
            "meta": meta_data
        }
    
    def create_comparison_visualization(self, dataset1: Dict, dataset2: Dict):
        """
        创建对比可视化，展示两种数据集的区别
        """
        print("\n" + "=" * 60)
        print("创建对比可视化图表")
        print("=" * 60)
        
        vis_dir = os.path.join(self.output_dir, "comparison_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. 数据规模对比（轨迹数量、长度）
        self._plot_data_scale_comparison(dataset1, dataset2, vis_dir)
        
        # 2. 状态维度对比
        self._plot_dimension_comparison(dataset1, dataset2, vis_dir)
        
        # 3. 轨迹可视化对比（2D和3D）
        self._plot_trajectory_comparison(dataset1, dataset2, vis_dir)
        
        # 4. 数据分布对比
        self._plot_distribution_comparison(dataset1, dataset2, vis_dir)
        
        # 5. 格式和场景信息对比表
        self._create_comparison_table(dataset1, dataset2, vis_dir)
        
        print(f"\n所有对比图表已保存至: {vis_dir}")
    
    def _plot_data_scale_comparison(self, dataset1: Dict, dataset2: Dict, save_dir: str):
        """数据规模对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
        
        # 轨迹数量对比
        num_traj1 = len(dataset1["data"])
        num_traj2 = len(dataset2["trajectories"])
        
        axes[0].bar(["数据集1\n(少量样本)", "数据集2\n(大批量样本)"], 
                   [num_traj1, num_traj2], 
                   color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[0].set_ylabel("轨迹数量", fontsize=12)
        axes[0].set_title("轨迹数量对比", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].text(0, num_traj1, str(num_traj1), ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[0].text(1, num_traj2, str(num_traj2), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 轨迹长度对比
        traj_len1 = dataset1.get("traj_length", 60)
        traj_len2 = dataset2.get("meta", {}).get("traj_length", 100)
        
        axes[1].bar(["数据集1\n(中等长度)", "数据集2\n(较长轨迹)"], 
                   [traj_len1, traj_len2], 
                   color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[1].set_ylabel("轨迹长度（时间步）", fontsize=12)
        axes[1].set_title("轨迹长度对比", fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].text(0, traj_len1, str(traj_len1), ha='center', va='bottom', fontsize=11, fontweight='bold')
        axes[1].text(1, traj_len2, str(traj_len2), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        path = os.path.join(save_dir, "1_data_scale_comparison.png")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
            plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 数据规模对比图: {path}")
    
    def _plot_dimension_comparison(self, dataset1: Dict, dataset2: Dict, save_dir: str):
        """状态维度对比图"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        dim1 = dataset1["state_dim"][0] if isinstance(dataset1["state_dim"], tuple) else dataset1["state_dim"]
        dim2 = dataset2["state_dim"][0] if isinstance(dataset2["state_dim"], tuple) else dataset2["state_dim"]
        
        bars = ax.bar(["数据集1\n(12维状态空间)", "数据集2\n(18维扩展状态空间)"], 
                     [dim1, dim2], 
                     color=['#3498db', '#e74c3c'], alpha=0.7, width=0.6)
        ax.set_ylabel("状态维度", fontsize=12)
        ax.set_title("状态空间维度对比", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, dim in zip(bars, [dim1, dim2]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dim}维',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加说明文字
        ax.text(0.5, 0.95, 
               f"数据集1: 标准12维状态（位置、速度、航向等）\n数据集2: 扩展18维状态（增加相对距离、相对速度、能量状态等）",
               transform=ax.transAxes, fontsize=10, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        path = os.path.join(save_dir, "2_dimension_comparison.png")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
            plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 维度对比图: {path}")
    
    def _plot_trajectory_comparison(self, dataset1: Dict, dataset2: Dict, save_dir: str):
        """轨迹可视化对比（2D和3D）"""
        # 2D对比
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120)
        
        # 数据集1的轨迹（2D）
        traj1 = dataset1["data"][0]
        states1 = np.array(traj1["states"])
        own_x1, own_y1 = states1[:, 0], states1[:, 1]
        bdt_x1, bdt_y1 = states1[:, 6], states1[:, 7]
        
        axes[0].plot(own_x1, own_y1, label='我方', color='#3498db', linewidth=2)
        axes[0].plot(bdt_x1, bdt_y1, label='对手', color='#e74c3c', linewidth=2)
        axes[0].scatter(own_x1[0], own_y1[0], c='#3498db', marker='o', s=80, zorder=5, label='起点')
        axes[0].scatter(own_x1[-1], own_y1[-1], c='#3498db', marker='x', s=100, zorder=5, label='终点')
        axes[0].scatter(bdt_x1[0], bdt_y1[0], c='#e74c3c', marker='o', s=80, zorder=5)
        axes[0].scatter(bdt_x1[-1], bdt_y1[-1], c='#e74c3c', marker='x', s=100, zorder=5)
        axes[0].set_title("数据集1：标准空战场景（12维）", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("X (m)", fontsize=11)
        axes[0].set_ylabel("Y (m)", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_aspect('equal', 'box')
        
        # 数据集2的轨迹（2D）
        traj2 = dataset2["trajectories"][0]
        states2 = traj2["states"]
        own_x2, own_y2 = states2[:, 0], states2[:, 1]
        bdt_x2, bdt_y2 = states2[:, 6], states2[:, 7]
        
        axes[1].plot(own_x2, own_y2, label='我方', color='#3498db', linewidth=2)
        axes[1].plot(bdt_x2, bdt_y2, label='对手', color='#e74c3c', linewidth=2)
        axes[1].scatter(own_x2[0], own_y2[0], c='#3498db', marker='o', s=80, zorder=5, label='起点')
        axes[1].scatter(own_x2[-1], own_y2[-1], c='#3498db', marker='x', s=100, zorder=5, label='终点')
        axes[1].scatter(bdt_x2[0], bdt_y2[0], c='#e74c3c', marker='o', s=80, zorder=5)
        axes[1].scatter(bdt_x2[-1], bdt_y2[-1], c='#e74c3c', marker='x', s=100, zorder=5)
        axes[1].set_title("数据集2：扩展多机协同场景（18维）", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("X (m)", fontsize=11)
        axes[1].set_ylabel("Y (m)", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_aspect('equal', 'box')
        
        plt.tight_layout()
        path = os.path.join(save_dir, "3_trajectory_2d_comparison.png")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
            plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 轨迹2D对比图: {path}")
        
        # 3D对比
        fig = plt.figure(figsize=(16, 6), dpi=120)
        
        # 数据集1的3D轨迹
        ax1 = fig.add_subplot(121, projection='3d')
        own_z1 = states1[:, 2]
        bdt_z1 = states1[:, 8]
        ax1.plot(own_x1, own_y1, own_z1, label='我方', color='#3498db', linewidth=2)
        ax1.plot(bdt_x1, bdt_y1, bdt_z1, label='对手', color='#e74c3c', linewidth=2)
        ax1.scatter(own_x1[0], own_y1[0], own_z1[0], c='#3498db', marker='o', s=50)
        ax1.scatter(own_x1[-1], own_y1[-1], own_z1[-1], c='#3498db', marker='x', s=80)
        ax1.set_title("数据集1：标准空战场景（12维）", fontsize=13, fontweight='bold')
        ax1.set_xlabel("X (m)", fontsize=10)
        ax1.set_ylabel("Y (m)", fontsize=10)
        ax1.set_zlabel("Z (m)", fontsize=10)
        ax1.legend()
        
        # 数据集2的3D轨迹
        ax2 = fig.add_subplot(122, projection='3d')
        own_z2 = states2[:, 2]
        bdt_z2 = states2[:, 8]
        ax2.plot(own_x2, own_y2, own_z2, label='我方', color='#3498db', linewidth=2)
        ax2.plot(bdt_x2, bdt_y2, bdt_z2, label='对手', color='#e74c3c', linewidth=2)
        ax2.scatter(own_x2[0], own_y2[0], own_z2[0], c='#3498db', marker='o', s=50)
        ax2.scatter(own_x2[-1], own_y2[-1], own_z2[-1], c='#3498db', marker='x', s=80)
        ax2.set_title("数据集2：扩展多机协同场景（18维）", fontsize=13, fontweight='bold')
        ax2.set_xlabel("X (m)", fontsize=10)
        ax2.set_ylabel("Y (m)", fontsize=10)
        ax2.set_zlabel("Z (m)", fontsize=10)
        ax2.legend()
        
        plt.tight_layout()
        path = os.path.join(save_dir, "4_trajectory_3d_comparison.png")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
            plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 轨迹3D对比图: {path}")
    
    def _plot_distribution_comparison(self, dataset1: Dict, dataset2: Dict, save_dir: str):
        """数据分布对比"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
        
        # 提取速度分布
        speeds1 = []
        speeds2 = []
        for traj in dataset1["data"][:10]:  # 采样前10条
            states = np.array(traj["states"])
            speeds1.extend(states[:, 3].tolist())  # 我方速度
        
        for traj in dataset2["trajectories"][:10]:  # 采样前10条
            states = traj["states"]
            speeds2.extend(states[:, 3].tolist())  # 我方速度
        
        axes[0, 0].hist(speeds1, bins=30, alpha=0.7, color='#3498db', label='数据集1')
        axes[0, 0].set_title("速度分布对比（数据集1）", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("速度 (m/s)", fontsize=10)
        axes[0, 0].set_ylabel("频数", fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(speeds2, bins=30, alpha=0.7, color='#e74c3c', label='数据集2')
        axes[0, 1].set_title("速度分布对比（数据集2）", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("速度 (m/s)", fontsize=10)
        axes[0, 1].set_ylabel("频数", fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 动作分布对比
        actions1 = []
        actions2 = []
        for traj in dataset1["data"][:10]:
            actions1.extend(traj["actions"])
        for traj in dataset2["trajectories"][:10]:
            actions2.extend(traj["actions"])
        
        unique_actions1, counts1 = np.unique(actions1, return_counts=True)
        unique_actions2, counts2 = np.unique(actions2, return_counts=True)
        
        axes[1, 0].bar(unique_actions1, counts1, alpha=0.7, color='#3498db')
        axes[1, 0].set_title("动作分布对比（数据集1）", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("动作索引", fontsize=10)
        axes[1, 0].set_ylabel("频数", fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        axes[1, 1].bar(unique_actions2, counts2, alpha=0.7, color='#e74c3c')
        axes[1, 1].set_title("动作分布对比（数据集2）", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("动作索引", fontsize=10)
        axes[1, 1].set_ylabel("频数", fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = os.path.join(save_dir, "5_distribution_comparison.png")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
            plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 数据分布对比图: {path}")
    
    def _create_comparison_table(self, dataset1: Dict, dataset2: Dict, save_dir: str):
        """创建对比表格"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        table_data = [
            ["属性", "数据集1", "数据集2"],
            ["数据格式", "JSON", "NPY"],
            ["状态维度", f"{dataset1['state_dim'][0] if isinstance(dataset1['state_dim'], tuple) else dataset1['state_dim']}维", 
             f"{dataset2['state_dim'][0] if isinstance(dataset2['state_dim'], tuple) else dataset2['state_dim']}维"],
            ["轨迹数量", str(len(dataset1["data"])), str(len(dataset2["trajectories"]))],
            ["轨迹长度", str(dataset1.get("traj_length", 60)), str(dataset2.get("meta", {}).get("traj_length", 100))],
            ["场景类型", dataset1.get("scenario", "标准空战场景"), dataset2.get("meta", {}).get("scenario", "扩展多机协同场景")],
            ["特点", "少量样本，中等长度", "大批量样本，较长轨迹"],
            ["适用场景", "基础算法验证", "复杂场景建模"]
        ]
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(table_data)):
            table[(i, 0)].set_facecolor('#E8F5E9')
            table[(i, 1)].set_facecolor('#E3F2FD')
            table[(i, 2)].set_facecolor('#FFEBEE')
        
        plt.title("数据集对比表", fontsize=16, fontweight='bold', pad=20)
        
        path = os.path.join(save_dir, "6_comparison_table.png")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
            plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 对比表格: {path}")
    
    def generate_large_scale_dataset(self, num_trajectories: int = 1000, min_horizon: int = 80, max_horizon: int = 100):
        """
        生成大批量、覆盖多种场景的博弈轨迹数据
        - 轨迹数量：1000条
        - 轨迹长度：80-100之间随机分布
        - 场景：通过不同参数组合创建多种场景
        """
        print("\n" + "=" * 60)
        print(f"生成大批量多场景数据集（{num_trajectories}条，长度{min_horizon}-{max_horizon}）")
        print("=" * 60)
        
        # 定义多种场景配置
        scenario_configs = [
            {
                "name": "近距离追击场景",
                "spawn": InitialSpawnParams(min_range=2000.0, max_range=5000.0, alt_mean=5000.0, alt_std=500.0),
                "rules": RuleParams(engage_dist=2000.0, pursue_angle=np.deg2rad(60)),
                "weight": 0.25  # 25%的数据
            },
            {
                "name": "远距离巡航场景",
                "spawn": InitialSpawnParams(min_range=8000.0, max_range=12000.0, alt_mean=8000.0, alt_std=1000.0),
                "rules": RuleParams(engage_dist=4000.0, cruise_speed=300.0),
                "weight": 0.25  # 25%的数据
            },
            {
                "name": "高空高速场景",
                "spawn": InitialSpawnParams(min_range=4000.0, max_range=8000.0, alt_mean=12000.0, alt_std=1500.0),
                "rules": RuleParams(cruise_speed=350.0, target_closure_speed=150.0),
                "weight": 0.20  # 20%的数据
            },
            {
                "name": "低空近距格斗场景",
                "spawn": InitialSpawnParams(min_range=1500.0, max_range=4000.0, alt_mean=3000.0, alt_std=400.0),
                "rules": RuleParams(engage_dist=1500.0, hard_evade_dist=800.0, evade_angle=np.deg2rad(45)),
                "weight": 0.15  # 15%的数据
            },
            {
                "name": "中距离对抗场景",
                "spawn": InitialSpawnParams(min_range=5000.0, max_range=9000.0, alt_mean=6000.0, alt_std=800.0),
                "rules": RuleParams(engage_dist=3000.0, pursue_angle=np.deg2rad(50)),
                "weight": 0.15  # 15%的数据
            }
        ]
        
        # 计算每个场景的轨迹数量
        scenario_counts = []
        total_assigned = 0
        for i, config in enumerate(scenario_configs):
            if i == len(scenario_configs) - 1:
                # 最后一个场景分配剩余的所有轨迹
                count = num_trajectories - total_assigned
            else:
                count = int(num_trajectories * config["weight"])
            scenario_counts.append(count)
            total_assigned += count
        
        # 生成轨迹
        all_trajectories = []
        scenario_labels = []
        horizon_distribution = []
        
        print("\读取待增强的轨迹数据...")
        for scenario_idx, (config, count) in enumerate(zip(scenario_configs, scenario_counts)):
            print(f"  场景 {scenario_idx + 1}/{len(scenario_configs)}: {config['name']} ({count}条)")
            
            # 创建该场景的生成器
            gen = RuleBasedTrajGenerator(
                physics=PhysicsParams(),
                rules=config["rules"],
                spawn=config["spawn"],
                seed=42 + scenario_idx  # 不同场景使用不同种子
            )
            
            # 为每条轨迹随机生成长度
            for _ in range(count):
                horizon = np.random.randint(min_horizon, max_horizon + 1)
                horizon_distribution.append(horizon)
                
                # 生成单条轨迹
                data = gen.generate_trajectories(
                    num_trajectories=1,
                    horizon=horizon,
                    with_reward=True
                )
                
                if data["trajectories"]:
                    all_trajectories.append(data["trajectories"][0])
                    scenario_labels.append(config["name"])
        
        print(f"\n轨迹生成完成，共 {len(all_trajectories)} 条")
        
        # 构建数据字典
        data = {
            "trajectories": all_trajectories,
            "state_dim": (12,),
            "action_dim": all_trajectories[0]["actions"].max() + 1 if all_trajectories else 5
        }
        
        # 计算轨迹长度统计信息
        avg_horizon = np.mean(horizon_distribution)
        min_horizon_actual = min(horizon_distribution)
        max_horizon_actual = max(horizon_distribution)
        
        # 打印基本信息（类似main()中的格式）
        print(f"\n数据集基本信息:")
        print(f"  - 轨迹数量: {len(all_trajectories)}")
        print(f"  - 状态维度: {data['state_dim']}")
        print(f"  - 动作维度: {data['action_dim']}")
        print(f"  - 轨迹平均长度: {avg_horizon:.1f}")
        print(f"  - 轨迹最短长度: {min_horizon_actual}")
        print(f"  - 轨迹最长长度: {max_horizon_actual}")
        
        # 保存数据
        output_path = os.path.join(self.output_dir, "large_scale_dataset.npy")
        np.save(output_path, np.array(all_trajectories, dtype=object))
        
        # 保存元数据
        meta_data = {
            "num_trajectories": len(all_trajectories),
            "state_dim": (12,),
            "action_dim": data["action_dim"],
            "horizon_range": [min_horizon_actual, max_horizon_actual],
            "avg_horizon": float(avg_horizon),
            "scenarios": {config["name"]: count for config, count in zip(scenario_configs, scenario_counts)}
        }
        meta_path = os.path.join(self.output_dir, "large_scale_dataset_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据已保存至: {output_path}")
        print(f"元数据已保存至: {meta_path}")
        
        return {
            "trajectories": all_trajectories,
            "scenario_labels": scenario_labels,
            "horizon_distribution": horizon_distribution,
            "meta": meta_data
        }
    
def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("开始读取对比数据集")
    print("=" * 60 + "\n")
    
    generator = ComparisonDataGenerator()
    
    # 读取数据集1：少量样本，JSON格式，12维
    dataset1 = generator.generate_dataset_1(num_trajectories=50, horizon=60)
    
    # 读取数据集2：大批量样本，NPY格式，18维
    dataset2 = generator.generate_dataset_2(num_trajectories=200, horizon=100)
    
    # 创建对比可视化
    generator.create_comparison_visualization(dataset1, dataset2)
    
    print("\n" + "=" * 60)
    print("所有数据读取和可视化完成！")
    print("=" * 60)
    print(f"\n数据文件保存在: {generator.output_dir}")
    print(f"可视化图表保存在: {os.path.join(generator.output_dir, 'comparison_visualizations')}")


def generate_large_scale_main():
    """生成大批量多场景数据集的主函数"""
    generator = ComparisonDataGenerator()
    
    # 生成1000条轨迹，长度在80-100之间
    result = generator.generate_large_scale_dataset(
        num_trajectories=1000,
        min_horizon=80,
        max_horizon=100
    )
    
    print("\n" + "=" * 60)
    print("大批量数据集生成完成！")
    print("=" * 60)
    print(f"\n数据文件保存在: {generator.output_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--large-scale':
        # 生成大批量数据集
        generate_large_scale_main()
    else:
        # 默认生成对比数据集
        main()

