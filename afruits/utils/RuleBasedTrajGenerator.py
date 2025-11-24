import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class PhysicsParams:
    # 基础物理参数
    dt: float = 0.5                       # 仿真步长(s)
    g: float = 9.81                        # 重力加速度(m/s^2)
    drag_coeff: float = 0.02               # 线性阻力系数(简化)
    a_max: float = 30.0                    # 最大加速度(由推力/油门提供)(m/s^2)
    max_bank_deg: float = 70.0             # 最大持续倾角(度)
    max_climb_rate: float = 60.0           # 最大爬升/下降速率(m/s)
    min_speed: float = 70.0                # 失速速度(m/s)
    max_speed: float = 380.0               # 最大速度(m/s)
    min_alt: float = 0.0                   # 海拔下限(m)
    max_alt: float = 18000.0               # 海拔上限(m)
    separation_min: float = 50.0           # 安全最小间隔(m)，小于此视为接触/终止
    climb_energy_penalty: float = 0.08     # 爬升耦合能量代价：|vz|带来的纵向加速度损失比例


@dataclass
class RuleParams:
    # 规则参数(角度以弧度给出)
    pursue_angle: float = np.deg2rad(45)   # 视场内转入追击
    evade_angle: float = np.deg2rad(60)    # 尾追角触发规避
    engage_dist: float = 2500.0            # 进入交战的距离阈值(m)
    hard_evade_dist: float = 1200.0        # 强规避距离(m)
    target_closure_speed: float = 120.0    # 目标闭合速度(m/s)用于追击时节流
    cruise_speed: float = 250.0            # 巡航速度(m/s)
    climb_match_gain: float = 0.3          # 高度差匹配比例
    turn_gain: float = 1.0                 # 转弯跟随比例(0~1)
    noise_std_action: float = 0.05         # 动作噪声(相对幅值)，提升多样性


@dataclass
class InitialSpawnParams:
    # 初始生成参数
    min_range: float = 4000.0              # 初始距离下限(m)
    max_range: float = 12000.0             # 初始距离上限(m)
    alt_mean: float = 6000.0               # 初始高度均值(m)
    alt_std: float = 800.0                 # 初始高度标准差(m)
    v_mean: float = 250.0                  # 初始速度均值(m/s)
    v_std: float = 20.0                    # 初始速度标准差(m/s)


class RuleBasedTrajGenerator:
    """
    规则驱动+物理约束的空战轨迹生成器

    状态定义(12维):
        ownship:  [x, y, z, v, psi, vz]
        bandit:   [x, y, z, v, psi, vz]
        单位: x,y(m), z(m, 海拔), v(m/s), psi(航向弧度, 0朝x正向, 左手系逆时针), vz(m/s)

    动作定义(离散一维, 针对我方与对手):
        a ∈ {0, 1, ..., K-1}，为宏动作编号。每个编号映射到一组连续控制(u_macro = [omega_cmd, throttle, vz_cmd])。
        训练/数据加载阶段可对a进行one-hot编码；动力学推进时使用其对应的u_macro。

    规则策略:
        - 追击(Pursue): 目标在前方视场且距离进入交战区, 转向对准目标, 速度向目标闭合速度调节, 高度差缓慢匹配
        - 规避(Evade): 目标在后方且接近, 进行大坡度横向规避(反向转向), 全油门加速, 快速爬升/俯冲脱离
        - 巡航(Cruise): 距离较远时保持巡航, 稳定高度与速度

    物理约束:
        - 最大持续转弯率: omega_max = g * tan(bank_max) / v
        - 速度积分: dv = dt * (throttle * a_max - drag_coeff * v) - climb_energy_penalty * |vz|
        - 航向更新: dpsi = clip(omega_cmd, -omega_max, +omega_max) * dt
        - 高度/速度/倾角/间隔等硬约束裁剪

    返回数据格式与项目内示例保持一致:
        {
          "trajectories": [
             {
               "states": [T, 12],
               "actions": [T],                  # ownship离散动作索引
               "rewards": [T],                 # 简单密集奖励(可选)
               "next_states": [T, 12],
               "dones": [T],
               "opponent_actions": [T]         # bandit离散动作索引
             }, ...
          ],
          "state_dim": (12,),
          "action_dim": K                      # 离散动作数量
        }
    """

    def __init__(
        self,
        physics: PhysicsParams = PhysicsParams(),
        rules: RuleParams = RuleParams(),
        spawn: InitialSpawnParams = InitialSpawnParams(),
        seed: Optional[int] = None
    ):
        self.physics = physics
        self.rules = rules
        self.spawn = spawn
        self._rng = np.random.default_rng(seed if seed is not None else None)
        # 离散动作集合与维度
        self._action_catalog = self._build_action_catalog()
        self.action_dim = int(self._action_catalog.shape[0])

    # ========== 公共接口 ==========

    def generate_trajectories(
        self,
        num_trajectories: int = 20,
        horizon: int = 50,
        with_reward: bool = True
    ) -> Dict[str, Any]:
        """
        生成多条空战轨迹

        参数:
            num_trajectories: 轨迹数量
            horizon: 每条轨迹步数
            with_reward: 是否计算简单密集奖励

        返回:
            数据字典(见类文档)
        """
        trajs: List[Dict[str, Any]] = []

        for _ in range(num_trajectories):
            s_own, s_bdt = self._random_spawn()
            states = []
            next_states = []
            actions = []
            opp_actions = []
            rewards = []
            dones = []

            done = False
            for t in range(horizon):
                # 规则连续动作
                u_cont_own = self._rule_policy_ownship(s_own, s_bdt)
                u_cont_bdt = self._rule_policy_bandit(s_bdt, s_own)

                # 连续 -> 离散宏动作
                idx_own, u_macro_own = self._quantize_action(u_cont_own)
                idx_bdt, u_macro_bdt = self._quantize_action(u_cont_bdt)

                # 物理推进（使用宏动作）
                s_next_own = self._step_dynamics(s_own, u_macro_own)
                s_next_bdt = self._step_dynamics(s_bdt, u_macro_bdt)

                # 约束裁剪/终止判断
                done = self._terminal_check(s_next_own, s_next_bdt)

                # 记录
                state_all = np.concatenate([s_own, s_bdt], axis=0)
                next_state_all = np.concatenate([s_next_own, s_next_bdt], axis=0)

                states.append(state_all)
                next_states.append(next_state_all)
                actions.append(int(idx_own))
                opp_actions.append(int(idx_bdt))
                dones.append(1 if done else 0)

                if with_reward:
                    r = self._dense_reward(s_own, s_bdt, u_macro_own, done)
                    rewards.append(r)

                # 步进
                s_own, s_bdt = s_next_own, s_next_bdt

                if done:
                    # 填充剩余步(保持长度一致)
                    for _pad in range(t + 1, horizon):
                        states.append(state_all.copy())
                        next_states.append(next_state_all.copy())
                        actions.append(0)
                        opp_actions.append(0)
                        dones.append(1)
                        if with_reward:
                            rewards.append(0.0)
                    break

            # numpy化
            trajs.append({
                "states": np.stack(states, axis=0).astype(np.float32),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32) if with_reward else np.zeros((horizon,), dtype=np.float32),
                "next_states": np.stack(next_states, axis=0).astype(np.float32),
                "dones": np.array(dones, dtype=np.int32),
                "opponent_actions": np.array(opp_actions, dtype=np.int64)
            })

        return {
            "trajectories": trajs,
            "state_dim": (12,),
            "action_dim": int(self.action_dim)
        }

    # ========== 规则策略 ==========

    def _rule_policy_ownship(self, s_own: np.ndarray, s_bdt: np.ndarray) -> np.ndarray:
        # 计算相对量
        rel = self._relative_state(s_own, s_bdt)
        rel_bearing = self._wrap_angle(rel["bearing"] - s_own[4])  # 相对航向误差
        dist = rel["range"]
        alt_err = s_bdt[2] - s_own[2]

        # 后方尾追角(敌对相对航向相对我机)
        tail_angle = self._wrap_angle(s_own[4] - rel["bearing"])

        # 触发条件
        pursue = (abs(rel_bearing) <= self.rules.pursue_angle) and (dist <= max(self.rules.engage_dist, 0.5 * self.rules.engage_dist + 3000))
        hard_evade = (abs(tail_angle) <= self.rules.evade_angle) and (dist <= self.rules.hard_evade_dist)
        far = dist > (self.rules.engage_dist * 1.3)

        # 基本指令
        omega_cmd = 0.0
        throttle = 0.0
        vz_cmd = 0.0

        if hard_evade:
            # 强规避: 反向转向(远离敌机), 全油门, 快速改变高度
            evade_dir = -np.sign(rel_bearing) if rel_bearing != 0 else 1.0
            omega_cmd = 1.2 * evade_dir  # 高速大转向指令, 之后会被omega_max裁剪
            throttle = 1.0
            vz_cmd = np.clip(-np.sign(alt_err) * self.physics.max_climb_rate, -self.physics.max_climb_rate, self.physics.max_climb_rate)
        elif pursue:
            # 追击: 指向目标, 速度调至目标闭合速度, 高度差缓慢匹配
            omega_cmd = self.rules.turn_gain * np.clip(rel_bearing, -1.5, 1.5)  # 角速度指令(简化为比例控制)
            # 调节速度: 高于目标闭合速度则收油, 低于则加油
            v = s_own[3]
            v_ref = self.rules.target_closure_speed + 0.5 * s_bdt[3]
            throttle = np.clip((v_ref - v) / max(v_ref, 1e-3), -1.0, 1.0)
            vz_cmd = np.clip(self.rules.climb_match_gain * alt_err, -self.physics.max_climb_rate, self.physics.max_climb_rate)
        elif far:
            # 巡航: 稳定航向(小幅对准敌机), 保持巡航速度并缓慢对高度
            omega_cmd = 0.5 * np.clip(rel_bearing, -1.0, 1.0)
            v = s_own[3]
            v_ref = self.rules.cruise_speed
            throttle = np.clip((v_ref - v) / max(v_ref, 1e-3), -1.0, 1.0)
            vz_cmd = np.clip(0.2 * self.rules.climb_match_gain * alt_err, -0.5 * self.physics.max_climb_rate, 0.5 * self.physics.max_climb_rate)
        else:
            # 交战但未强规避也未完全追击: 持续逼近
            omega_cmd = self.rules.turn_gain * np.clip(rel_bearing, -1.2, 1.2)
            v = s_own[3]
            v_ref = 0.5 * (self.rules.cruise_speed + s_bdt[3])
            throttle = np.clip((v_ref - v) / max(v_ref, 1e-3), -1.0, 1.0)
            vz_cmd = np.clip(self.rules.climb_match_gain * alt_err, -self.physics.max_climb_rate, self.physics.max_climb_rate)

        return np.array([omega_cmd, throttle, vz_cmd], dtype=np.float32)

    def _rule_policy_bandit(self, s_bdt: np.ndarray, s_own: np.ndarray) -> np.ndarray:
        # 敌机策略: 简化为“防守-机会反打”
        rel = self._relative_state(s_bdt, s_own)
        rel_bearing = self._wrap_angle(rel["bearing"] - s_bdt[4])
        dist = rel["range"]
        alt_err = s_own[2] - s_bdt[2]

        # 如果被尾追且很近则强规避，否则尝试侧向机动保持距离
        tail_angle = self._wrap_angle(s_bdt[4] - rel["bearing"])
        hard_evade = (abs(tail_angle) <= self.rules.evade_angle) and (dist <= self.rules.hard_evade_dist)

        if hard_evade:
            omega_cmd = -np.sign(rel_bearing) * 1.1
            throttle = 1.0
            vz_cmd = np.clip(np.sign(alt_err) * self.physics.max_climb_rate, -self.physics.max_climb_rate, self.physics.max_climb_rate)
        else:
            # 侧向绕行, 试图保持距离, 速度维持在巡航附近
            side = 1.0 if rel_bearing >= 0 else -1.0
            omega_cmd = 0.6 * side
            v = s_bdt[3]
            v_ref = self.rules.cruise_speed
            throttle = np.clip((v_ref - v) / max(v_ref, 1e-3), -1.0, 1.0)
            vz_cmd = np.clip(0.2 * (-alt_err), -self.physics.max_climb_rate, self.physics.max_climb_rate)

        return np.array([omega_cmd, throttle, vz_cmd], dtype=np.float32)

    # ========== 物理与工具函数 ==========

    def _build_action_catalog(self) -> np.ndarray:
        """
        构建离散宏动作集合: 大小为K×3，每行对应[omega_cmd, throttle, vz_cmd]
        """
        base_omega = 1.0  # rad/s
        vz_scale = 0.6 * self.physics.max_climb_rate
        omega_mults = [-1.2, -0.6, 0.0, 0.6, 1.2]
        throttle_levels = [-1.0, 0.0, 1.0]
        vz_levels = [-1.0, 0.0, 1.0]
        catalog = []
        for om in omega_mults:
            for th in throttle_levels:
                for vzl in vz_levels:
                    catalog.append([om * base_omega, th, vzl * vz_scale])
        return np.array(catalog, dtype=np.float32)

    def _quantize_action(self, u_cont: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        将连续动作映射到最近的离散宏动作，返回(动作索引, 宏动作向量)
        """
        base_omega = 1.0
        vz_scale = 0.6 * self.physics.max_climb_rate

        # 归一化到相近尺度后做最近邻
        def norm(u):
            return np.array([u[0] / base_omega, u[1], u[2] / max(vz_scale, 1e-6)], dtype=np.float32)

        u_n = norm(u_cont)
        cat_n = np.stack([norm(a) for a in self._action_catalog], axis=0)
        dists = np.linalg.norm(cat_n - u_n[None, :], axis=1)
        idx = int(np.argmin(dists))
        u_macro = self._action_catalog[idx]
        return idx, u_macro

    def _apply_action_noise(self, u: np.ndarray) -> np.ndarray:
        # 对动作加入小幅高斯噪声(相对幅度)
        std = self.rules.noise_std_action
        noisy = u.copy()
        noisy[0] += self._rng.normal(0.0, std)                 # omega
        noisy[1] += self._rng.normal(0.0, std)                 # throttle
        noisy[2] += self._rng.normal(0.0, std * self.physics.max_climb_rate)  # vz
        noisy[1] = np.clip(noisy[1], -1.0, 1.0)
        noisy[2] = np.clip(noisy[2], -self.physics.max_climb_rate, self.physics.max_climb_rate)
        return noisy

    def _step_dynamics(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        # 解包
        x, y, z, v, psi, vz = s.astype(np.float64)
        omega_cmd, throttle, vz_cmd = u.astype(np.float64)

        # 物理约束: 最大持续转弯率
        bank_max = np.deg2rad(self.physics.max_bank_deg)
        omega_max = max(self.physics.g * np.tan(bank_max) / max(v, 1.0), 0.0)  # rad/s
        omega = np.clip(omega_cmd, -omega_max, omega_max)

        # 纵向速度/高度
        vz_target = np.clip(vz_cmd, -self.physics.max_climb_rate, self.physics.max_climb_rate)
        # 一阶滞后逼近, 使垂直速度变化平滑
        vz = vz + 0.6 * (vz_target - vz)

        # 速度积分(含阻力与爬升能量耦合损耗)
        a_long = throttle * self.physics.a_max - self.physics.drag_coeff * v
        a_long -= self.physics.climb_energy_penalty * abs(vz)
        v = np.clip(v + a_long * self.physics.dt, self.physics.min_speed, self.physics.max_speed)

        # 航向更新
        psi = self._wrap_angle(psi + omega * self.physics.dt)

        # 位移积分(地速投影)
        x = x + v * np.cos(psi) * self.physics.dt
        y = y + v * np.sin(psi) * self.physics.dt
        z = np.clip(z + vz * self.physics.dt, self.physics.min_alt, self.physics.max_alt)

        return np.array([x, y, z, v, psi, vz], dtype=np.float32)

    def _relative_state(self, s_src: np.ndarray, s_tgt: np.ndarray) -> Dict[str, float]:
        dx = s_tgt[0] - s_src[0]
        dy = s_tgt[1] - s_src[1]
        dz = s_tgt[2] - s_src[2]
        rng = float(np.hypot(dx, dy))
        bearing = float(np.arctan2(dy, dx))  # 从src指向tgt的方位(弧度)
        return {"range": rng, "bearing": bearing, "dz": float(dz)}

    def _terminal_check(self, s_own: np.ndarray, s_bdt: np.ndarray) -> bool:
        # 碰撞/接触
        rel = self._relative_state(s_own, s_bdt)
        if rel["range"] <= self.physics.separation_min and abs(rel["dz"]) <= 30.0:
            return True
        # 高度超界(软裁剪后仍可继续, 但这里设为终止更利于收敛)
        if (s_own[2] <= self.physics.min_alt + 1.0) or (s_own[2] >= self.physics.max_alt - 1.0):
            return True
        if (s_bdt[2] <= self.physics.min_alt + 1.0) or (s_bdt[2] >= self.physics.max_alt - 1.0):
            return True
        # 速度极小(失速)或异常
        if (s_own[3] <= self.physics.min_speed + 1.0) or (s_bdt[3] <= self.physics.min_speed + 1.0):
            return True
        return False

    def _dense_reward(self, s_own: np.ndarray, s_bdt: np.ndarray, u_own: np.ndarray, done: bool) -> float:
        # 奖励示例: 距离惩罚 + 视场对准奖励 - 大动作惩罚 + 高度差惩罚
        rel = self._relative_state(s_own, s_bdt)
        dist = rel["range"]
        bearing_err = abs(self._wrap_angle(rel["bearing"] - s_own[4]))
        alt_err = abs(rel["dz"])

        r_dist = -0.001 * dist
        r_align = -0.2 * bearing_err
        r_alt = -0.0005 * alt_err
        r_ctrl = -0.02 * (abs(u_own[0]) + 0.5 * abs(u_own[1]) + 0.002 * abs(u_own[2]))

        r = r_dist + r_align + r_alt + r_ctrl
        if done:
            # 结束时如果接触(极近距离)给一个终止奖励(可正可负依据任务定义, 这里定义为负)
            if dist <= self.physics.separation_min:
                r -= 10.0
        return float(r)

    def _random_spawn(self) -> Tuple[np.ndarray, np.ndarray]:
        # 随机在水平面极坐标放置, 高度、速度正态采样
        rng = float(self._rng.uniform(self.spawn.min_range, self.spawn.max_range))
        ang = float(self._rng.uniform(-np.pi, np.pi))
        # 我机在原点附近, 敌机在极坐标(r, ang)处
        x_own, y_own = 0.0, 0.0
        x_bdt, y_bdt = rng * np.cos(ang), rng * np.sin(ang)
        # 航向取相对彼此的相反方向为初始化(更易形成交会)
        psi_own = self._wrap_angle(ang)
        psi_bdt = self._wrap_angle(ang + np.pi + self._rng.normal(0.0, 0.2))

        z_own = float(np.clip(self._rng.normal(self.spawn.alt_mean, self.spawn.alt_std), self.physics.min_alt + 500.0, self.physics.max_alt - 500.0))
        z_bdt = float(np.clip(self._rng.normal(self.spawn.alt_mean, self.spawn.alt_std), self.physics.min_alt + 500.0, self.physics.max_alt - 500.0))

        v_own = float(np.clip(self._rng.normal(self.spawn.v_mean, self.spawn.v_std), self.physics.min_speed + 10.0, self.physics.max_speed - 10.0))
        v_bdt = float(np.clip(self._rng.normal(self.spawn.v_mean, self.spawn.v_std), self.physics.min_speed + 10.0, self.physics.max_speed - 10.0))

        vz_own = float(self._rng.normal(0.0, 5.0))
        vz_bdt = float(self._rng.normal(0.0, 5.0))

        s_own = np.array([x_own, y_own, z_own, v_own, psi_own, vz_own], dtype=np.float32)
        s_bdt = np.array([x_bdt, y_bdt, z_bdt, v_bdt, psi_bdt, vz_bdt], dtype=np.float32)
        return s_own, s_bdt

    @staticmethod
    def _wrap_angle(a: float) -> float:
        # wrap to (-pi, pi]
        return float((a + np.pi) % (2 * np.pi) - np.pi)


# ===== 使用示例 =====
# from afruits.utils.RuleBasedTrajGenerator import RuleBasedTrajGenerator
# gen = RuleBasedTrajGenerator()
# data = gen.generate_trajectories(num_trajectories=10, horizon=60, with_reward=True)
# print(data["state_dim"], data["action_dim"], len(data["trajectories"]))