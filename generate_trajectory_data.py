import json
import numpy as np
import os

def generate_trajectory_data(num_trajectories=20, traj_length=50, state_dim=4, action_dim=2):
    """生成示例轨迹数据"""
    trajectories = []
    
    for i in range(num_trajectories):
        # 生成随机状态序列
        states = np.random.rand(traj_length, state_dim).tolist()
        
        # 生成随机动作序列
        actions = np.random.rand(traj_length, action_dim).tolist()
        
        # 生成随机奖励序列
        rewards = np.random.rand(traj_length).tolist()
        
        # 生成随机下一状态序列
        next_states = np.random.rand(traj_length, state_dim).tolist()
        
        # 生成随机终止标志序列
        dones = [0] * (traj_length - 1) + [1]  # 最后一步终止
        
        # 生成随机对手动作序列
        opponent_actions = np.random.rand(traj_length, action_dim).tolist()
        
        # 生成额外信息
        infos = [{"step": j} for j in range(traj_length)]
        
        # 添加到轨迹列表
        trajectories.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "infos": infos,
            "opponent_actions": opponent_actions
        })
    
    return {
        "trajectories": trajectories,
        "state_dim": (state_dim,),
        "action_dim": action_dim
    }

if __name__ == "__main__":
    # 生成20条轨迹，每条50步，状态维度4，动作维度2
    data = generate_trajectory_data()
    
    # 保存到文件
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_trajectories.json")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"已生成示例轨迹数据并保存至: {output_path}")
    print(f"轨迹数量: {len(data['trajectories'])}")
    print(f"每条轨迹步数: {len(data['trajectories'][0]['states'])}")
    print(f"状态维度: {data['state_dim']}")
    print(f"动作维度: {data['action_dim']}")