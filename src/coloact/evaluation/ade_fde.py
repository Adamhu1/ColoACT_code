import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 🟢 新增：用于导出 CSV 数据
import os

# --- 引入你的自定义模块 ---
from coloact.data.dataset import ColonDatasetDiffusion
from coloact.models.act import ACTPolicy 

# ================= ⚙️ 配置区域 (CONFIG) =================
CONFIG = {
    "data_dir": "data/h5",
    "model_path": "outputs/Checkpoints_ACT_RGBDE/act_best.pth",
    "input_mode": "rgbd_elev",
    "pred_horizon": 16,
    "dt": 0.1,
    
    # 你的归一化参数
    "ACTION_MAX": np.array([3.0, 3.0]), 
    
    "plot_samples": 4, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 🟢 新增：导出文件名
    "export_file": "eval_data_for_origin_RGBDE_try.csv" 
}
# =========================================================

def unnormalize_action(action_norm, max_val):
    return action_norm * max_val

def integrate_trajectory(actions, init_pose=(0,0,0), dt=0.1):
    x, y, theta = init_pose
    traj = [(x, y)]
    for v, w in actions:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        traj.append((x, y))
    return np.array(traj)

# 🟢 新增：专门用于导出数据给 Origin 的函数
def save_data_for_origin(plot_data, filename):
    print(f"💾 正在导出数据到 {filename} ...")
    data_rows = []
    
    for i, (gt, pred, ade) in enumerate(plot_data):
        # gt 和 pred 都是 (T+1, 2) 的数组
        # 假设 T=16，那么会有 17 个点 (包括起点 0,0)
        
        for t in range(len(gt)):
            row = {
                "Sample_ID": i + 1,       # 样本编号 1, 2, 3, 4
                "Time_Step": t,           # 时间步 0~16
                
                # 地面真值 (Ground Truth)
                # 注意：代码画图时用的是 plot(gt[:,1], gt[:,0])
                # 所以 1 是横轴(Lateral), 0 是纵轴(Forward)
                "GT_Lateral_X": gt[t, 1], 
                "GT_Forward_Y": gt[t, 0],
                
                # 预测值 (Prediction)
                "Pred_Lateral_X": pred[t, 1],
                "Pred_Forward_Y": pred[t, 0],
                
                # 该样本的误差
                "Sample_ADE": ade
            }
            data_rows.append(row)
            
    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"✅ 数据导出成功！请在 Origin 中打开: {filename}")
    print("   提示: Origin 中将 'GT_Lateral_X' 设为 X 轴, 'GT_Forward_Y' 设为 Y 轴即可复现轨迹。")

def evaluate():
    print(f"🚀 正在使用设备: {CONFIG['device']}")
    
    # 1. 准备数据
    full_dataset = ColonDatasetDiffusion(
        data_dir=CONFIG["data_dir"],
        pred_horizon=CONFIG["pred_horizon"],
        input_mode=CONFIG["input_mode"]
    )
    
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print(f"📊 验证集样本数: {len(val_dataset)}")

    # 2. 加载模型
    channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
    model = ACTPolicy(
        pred_horizon=CONFIG["pred_horizon"], 
        input_channels=channel_map[CONFIG["input_mode"]]
    ).to(CONFIG["device"])

    print(f"🔄 正在加载模型: {CONFIG['model_path']}")
    try:
        state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    model.eval()
    
    ade_losses = []
    fde_losses = []
    plot_data = [] 

    print("🏁 开始推理...")
    
    with torch.no_grad():
        for i, (obs, gt_actions) in enumerate(val_loader):
            obs = obs.to(CONFIG["device"])           
            gt_actions = gt_actions.to(CONFIG["device"]) 
            
            if isinstance(obs, (tuple, list)):
                pred_actions = model(obs[0], qpos=obs[1], actions=None)[0]
            else:
                output = model(obs, actions=None)
                pred_actions = output[0] if isinstance(output, tuple) else output

            pred_np_norm = pred_actions.cpu().numpy()[0] 
            gt_np_norm = gt_actions.cpu().numpy()[0]     
            
            pred_physical = unnormalize_action(pred_np_norm, CONFIG["ACTION_MAX"])
            gt_physical = unnormalize_action(gt_np_norm, CONFIG["ACTION_MAX"])
            
            traj_pred = integrate_trajectory(pred_physical, dt=CONFIG["dt"])
            traj_gt = integrate_trajectory(gt_physical, dt=CONFIG["dt"])
            
            errors = np.linalg.norm(traj_pred - traj_gt, axis=1)
            ade = np.mean(errors) 
            fde = errors[-1]      
            
            ade_losses.append(ade)
            fde_losses.append(fde)
            
            if len(plot_data) < CONFIG["plot_samples"]:
                plot_data.append((traj_gt, traj_pred, ade))

    print(f"\n📈 === 评估报告 ===")
    print(f"ADE: {np.mean(ade_losses):.4f} | FDE: {np.mean(fde_losses):.4f}")

    # 🟢 3. 导出数据给 Origin
    save_data_for_origin(plot_data, CONFIG["export_file"])

    # 4. 简单预览 (修复了 set_xlabel 缺失的问题)
    plot_results(plot_data)

def plot_results(plot_data):
    num_plots = len(plot_data)
    if num_plots == 0: return

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    if num_plots == 1: axes = [axes]
    
    for i, (gt, pred, ade) in enumerate(plot_data):
        ax = axes[i]
        
        # GT (X=Lateral, Y=Forward)
        ax.plot(gt[:, 1], gt[:, 0], c='black', label='Ground Truth', 
                linewidth=1.5, alpha=0.6, marker='.', markersize=6)
        
        # Pred
        ax.plot(pred[:, 1], pred[:, 0], c='red', label='ACT Pred', 
                linewidth=1.5, linestyle='--', marker='o', markersize=4)
        
        ax.scatter(gt[0, 1], gt[0, 0], c='green', s=80, marker='^', label='Start')
        
        ax.set_title(f"Sample {i+1} | ADE: {ade:.3f}")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # 🟢 修复：添加坐标轴标签
        # 既然没有绝对单位，使用 Relative Units 更严谨
        ax.set_xlabel("Lateral (rel. units)")
        ax.set_ylabel("Forward (rel. units)")
        
        if i == 0: ax.legend()
            
    plt.tight_layout()
    plt.savefig("eval_ade_chart.png")
    print(f"✅ 预览图已保存: eval_ade_chart.png")
    plt.show()

if __name__ == "__main__":
    evaluate()




# 3
# import torch
# from torch.utils.data import DataLoader, random_split
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # --- 引入你的自定义模块 ---
# from coloact.data.dataset import ColonDatasetDiffusion
# from coloact.models.act import ACTPolicy 

# # ================= ⚙️ 配置区域 (CONFIG) =================
# CONFIG = {
#     # 1. 路径设置 (确保指向 dataset2 和你刚训练好的 act_best.pth)
#     "data_dir": "data/h5",
#     "model_path": "outputs/Checkpoints_ACT_RGBD/act_best.pth",
    
#     # 2. 模型参数 (必须严格匹配 train.py)
#     "input_mode": "rgbd_elev",  # train.py 中使用的是这个
#     "pred_horizon": 16,         # train.py 中硬编码了 16
    
#     # 3. 物理参数 & 归一化 (这是解决“绝对距离”误差的核心！)
#     "dt": 0.1,                  # 采样时间间隔 (10Hz = 0.1s)
    
#     # 🔥🔥🔥 关键设置 🔥🔥🔥
#     # 你的 train.py 没有显示归一化逻辑，说明归一化发生在 dataset 类里。
#     # 请根据你 Dataset 的处理逻辑填入这里的最大值。
#     # 如果 Dataset 里做了 action / 3.0，这里就填 3.0。
#     # 如果 Dataset 里没做归一化，这里填 1.0。
#     "ACTION_MAX": np.array([3.0, 3.0]), # [线速度 v_max, 角速度 w_max]
    
#     "plot_samples": 4,          # 画图数量
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }
# # =========================================================

# def unnormalize_action(action_norm, max_val):
#     """
#     反归一化：将模型输出的 [-1, 1] 范围的值还原为物理单位 (m/s, rad/s)
#     """
#     return action_norm * max_val

# def integrate_trajectory(actions, init_pose=(0,0,0), dt=0.1):
#     """
#     积分：(v, w) -> (x, y)
#     """
#     x, y, theta = init_pose
#     traj = [(x, y)]
    
#     for v, w in actions:
#         x += v * np.cos(theta) * dt
#         y += v * np.sin(theta) * dt
#         theta += w * dt
#         traj.append((x, y))
        
#     return np.array(traj)

# def evaluate():
#     print(f"🚀 正在使用设备: {CONFIG['device']}")
    
#     # 1. 准备验证集
#     # 注意：因为 train.py 里的 random_split 没有固定种子，
#     # 这里我们固定种子 42，确保每次 eval 跑的数据是一样的。
#     full_dataset = ColonDatasetDiffusion(
#         data_dir=CONFIG["data_dir"],
#         pred_horizon=CONFIG["pred_horizon"],
#         input_mode=CONFIG["input_mode"]
#     )
    
#     # 保持和 train.py 一样的切分比例
#     val_ratio = 0.2
#     val_size = int(len(full_dataset) * val_ratio)
#     train_size = len(full_dataset) - val_size
    
#     # 固定种子切分
#     _, val_dataset = random_split(
#         full_dataset, [train_size, val_size], 
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
#     print(f"📊 验证集样本数: {len(val_dataset)}")

#     # 2. 初始化模型
#     channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
#     model = ACTPolicy(
#         pred_horizon=CONFIG["pred_horizon"], 
#         input_channels=channel_map[CONFIG["input_mode"]]
#     ).to(CONFIG["device"])

#     # 3. 加载权重 (匹配 train.py 的 torch.save(model.state_dict()))
#     print(f"🔄 正在加载模型权重: {CONFIG['model_path']}")
#     try:
#         # 直接加载 state_dict
#         state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
        
#         # 处理可能的 DataParallel 'module.' 前缀
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v
#             else:
#                 new_state_dict[k] = v
                
#         model.load_state_dict(new_state_dict)
#         print("✅ 模型加载成功！")
#     except Exception as e:
#         print(f"❌ 模型加载失败: {e}")
#         print("💡 提示: 请检查 model_act.py 是否和训练时完全一致。")
#         return

#     model.eval()
    
#     # 4. 评估循环
#     ade_losses = []
#     fde_losses = []
#     plot_data = [] 

#     print("🏁 开始推理...")
    
#     with torch.no_grad():
#         for i, (obs, gt_actions) in enumerate(val_loader):
#             obs = obs.to(CONFIG["device"])          
#             gt_actions = gt_actions.to(CONFIG["device"]) 
            
#             # --- ACT 推理 ---
#             # 你的 train.py 里调用是 model(obs, actions)，那是训练模式 (返回 loss)
#             # 推理模式调用: model(obs, actions=None) (返回 predicted_actions)
#             if isinstance(obs, (tuple, list)):
#                 pred_actions = model(obs[0], qpos=obs[1], actions=None)[0]
#             else:
#                 output = model(obs, actions=None)
#                 # 处理 ACT 可能返回 tuple 的情况
#                 pred_actions = output[0] if isinstance(output, tuple) else output

#             # 转 Numpy
#             pred_np_norm = pred_actions.cpu().numpy()[0] # (T, 2)
#             gt_np_norm = gt_actions.cpu().numpy()[0]     # (T, 2)
            
#             # --- 反归一化 (关键步骤) ---
#             # 将模型输出的 [-1, 1] 乘上最大速度，变回 m/s
#             pred_physical = unnormalize_action(pred_np_norm, CONFIG["ACTION_MAX"])
#             gt_physical = unnormalize_action(gt_np_norm, CONFIG["ACTION_MAX"])
            
#             # --- 积分计算轨迹 ---
#             traj_pred = integrate_trajectory(pred_physical, dt=CONFIG["dt"])
#             traj_gt = integrate_trajectory(gt_physical, dt=CONFIG["dt"])
            
#             # --- 计算误差 ---
#             errors = np.linalg.norm(traj_pred - traj_gt, axis=1)
#             ade = np.mean(errors) 
#             fde = errors[-1]      
            
#             ade_losses.append(ade)
#             fde_losses.append(fde)
            
#             if len(plot_data) < CONFIG["plot_samples"]:
#                 plot_data.append((traj_gt, traj_pred, ade))

#     # 5. 打印结果
#     print(f"\n📈 === 评估报告 ===")
#     print(f"Mode: {CONFIG['input_mode']} | Horizon: {CONFIG['pred_horizon']}")
#     print(f"ADE (平均轨迹误差): {np.mean(ade_losses):.4f} m")
#     print(f"FDE (终点位移误差): {np.mean(fde_losses):.4f} m")

#     # 6. 画图
#     plot_results(plot_data)

# def plot_results(plot_data):
#     num_plots = len(plot_data)
#     if num_plots == 0: return

#     fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
#     if num_plots == 1: axes = [axes]
    
#     for i, (gt, pred, ade) in enumerate(plot_data):
#         ax = axes[i]
        
#         # GT (黑色)
#         ax.plot(gt[:, 1], gt[:, 0], c='black', label='Ground Truth', 
#                 linewidth=1.5, alpha=0.6, marker='.', markersize=6)
        
#         # Pred (红色虚线)
#         ax.plot(pred[:, 1], pred[:, 0], c='red', label='ACT Pred', 
#                 linewidth=1.5, linestyle='--', marker='o', markersize=4)
        
#         # 起点/终点
#         ax.scatter(gt[0, 1], gt[0, 0], c='green', s=80, marker='^', label='Start')
#         ax.scatter(gt[-1, 1], gt[-1, 0], c='black', s=50, marker='x')
        
#         ax.set_title(f"Sample {i+1} | ADE: {ade:.3f}m")
#         ax.axis('equal')
#         ax.grid(True, alpha=0.3)
#         if i == 0: ax.legend()
            
#     plt.tight_layout()
#     plt.savefig("eval_ade_chart.png")
#     print(f"✅ 图表已保存: eval_ade_chart.png")
#     plt.show()

# if __name__ == "__main__":
#     evaluate()



# 2
# import torch
# from torch.utils.data import DataLoader, random_split
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # --- 你的自定义模块 ---
# # 请确保这两个文件在同一目录下，或者在 PYTHONPATH 中
# from coloact.data.dataset import ColonDatasetDiffusion
# from coloact.models.act import ACTPolicy 

# # ================= ⚙️ 配置区域 (CONFIG) =================
# CONFIG = {
#     # 路径设置
#     "data_dir": "data/h5",
#     "model_path": "outputs/Checkpoints_ACT_RGBD/act_best.pth",
    
#     # 模型参数 (必须与训练时严格一致!)
#     "input_mode": "rgbd_elev",  # rgb, rgbd, rgbd_elev
#     "pred_horizon": 16,         # ACT 预测的时间步长
    
#     # 物理参数
#     "dt": 0.1,                  # 采样时间间隔 (10Hz = 0.1s)
    
#     # 归一化统计量 (关键!)
#     # 如果训练时你的动作归一化是 v_norm = v / max_v
#     # 这里就需要定义 max_v 用于反归一化
#     "NORM_STATS": {
#         "action_max": np.array([3.0, 3.0]), # [v_max, w_max] 假设线速度角速度最大都是3
#         # 如果是用 Z-Score (mean/std)，请在这里修改
#     },
    
#     "plot_samples": 4,          # 画图数量
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }
# # =========================================================

# def unnormalize_action(action_norm, stats):
#     """
#     将模型输出的归一化动作还原为物理动作。
#     假设训练时采用的是 MinMax 归一化 (除以最大值)。
#     action_norm: (T, 2)
#     """
#     # 方式 A: 简单的最大值缩放 (对应 v_norm = v / v_max)
#     # 结果 = normalized_action * max_val
#     action_physical = action_norm * stats["action_max"]
    
#     # 方式 B: 如果你是 Z-Score 归一化 (v_norm = (v - mean) / std)
#     # action_physical = action_norm * stats["action_std"] + stats["action_mean"]
    
#     return action_physical

# def integrate_trajectory(actions, init_pose=(0,0,0), dt=0.1):
#     """
#     运动学积分: (v, w) -> (x, y)
#     actions: (T, 2) 真实的物理动作
#     """
#     x, y, theta = init_pose
#     traj = [(x, y)]
    
#     for v, w in actions:
#         # 差分驱动/独轮车模型积分
#         # x_next = x + v * cos(theta) * dt
#         x += v * np.cos(theta) * dt
#         y += v * np.sin(theta) * dt
#         theta += w * dt
        
#         traj.append((x, y))
        
#     return np.array(traj)

# def load_model_robust(model, path, device):
#     """健壮的模型加载函数，处理 checkpoint 字典包裹的情况"""
#     print(f"🔄 正在加载模型权重: {path}")
#     try:
#         loaded_content = torch.load(path, map_location=device)
        
#         # 1. 检查是否是 Checkpoint 字典 (包含 epoch, optimizer 等)
#         if isinstance(loaded_content, dict) and 'model_state_dict' in loaded_content:
#             print("   ℹ️ 检测到 Checkpoint 格式，提取 state_dict...")
#             state_dict = loaded_content['model_state_dict']
#         else:
#             state_dict = loaded_content

#         # 2. 处理 DataParallel 可能产生的 'module.' 前缀
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v
#             else:
#                 new_state_dict[k] = v
        
#         # 3. 加载权重
#         missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
#         if len(missing) > 0:
#             print(f"⚠️ 警告: 缺失键值 (可能不影响推理): {missing}")
#         if len(unexpected) > 0:
#             # ACT 有些版本在 eval 时不需要 loss head 的权重，这是正常的
#             pass 
            
#         print("✅ 模型加载成功！")
#         return True
#     except Exception as e:
#         print(f"❌ 模型加载严重失败: {e}")
#         return False

# def evaluate():
#     # 1. 准备验证集 (固定种子)
#     print("📦 正在加载数据集...")
#     full_dataset = ColonDatasetDiffusion(
#         data_dir=CONFIG["data_dir"],
#         pred_horizon=CONFIG["pred_horizon"],
#         input_mode=CONFIG["input_mode"]
#     )
    
#     # 切分验证集 (必须与训练逻辑一致)
#     val_size = int(0.2 * len(full_dataset))
#     train_size = len(full_dataset) - val_size
#     _, val_dataset = random_split(
#         full_dataset, [train_size, val_size], 
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     # Batch size = 1 便于单条轨迹分析
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
#     print(f"📊 验证集样本数: {len(val_dataset)}")

#     # 2. 初始化模型
#     channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
#     try:
#         model = ACTPolicy(
#             pred_horizon=CONFIG["pred_horizon"], 
#             input_channels=channel_map[CONFIG["input_mode"]]
#             # 如果你的类还需要 output_dim 或 hidden_dim，请在这里补充
#         ).to(CONFIG["device"])
#     except Exception as e:
#         print(f"❌ 模型初始化失败，请检查 CONFIG 参数是否匹配: {e}")
#         return

#     # 3. 加载权重
#     if not load_model_robust(model, CONFIG["model_path"], CONFIG["device"]):
#         return

#     model.eval()
    
#     # 4. 评估循环
#     ade_losses = []
#     fde_losses = []
#     plot_data = [] # 存储 (gt_traj, pred_traj, ade_val)

#     print("🚀 开始 Open-Loop ADE/FDE 评估...")
    
#     with torch.no_grad():
#         for i, (obs, gt_actions) in enumerate(val_loader):
#             obs = obs.to(CONFIG["device"])          # (1, C, H, W) 或类似
#             gt_actions = gt_actions.to(CONFIG["device"]) # (1, T, 2)
            
#             # --- ACT 推理 ---
#             # 标准 ACT 推理: 传入 obs, (可选 qpos), actions=None
#             # 返回: pred_actions (B, T, action_dim), is_pad, stats
#             if isinstance(obs, tuple) or isinstance(obs, list):
#                 # 如果你的 dataset 返回 (image, qpos)
#                  pred_actions = model(obs[0], qpos=obs[1], actions=None)[0]
#             else:
#                 # 如果只用视觉
#                 # 注意：有些 ACT 实现返回 (actions, latents)，这里取第一个
#                 output = model(obs, actions=None)
#                 pred_actions = output[0] if isinstance(output, tuple) else output

#             # 转 Numpy
#             pred_np_norm = pred_actions.cpu().numpy()[0] # (T, 2)
#             gt_np_norm = gt_actions.cpu().numpy()[0]     # (T, 2)
            
#             # --- 关键：反归一化 ---
#             # 我们需要在积分前把归一化的数值转回 m/s 和 rad/s
#             pred_physical = unnormalize_action(pred_np_norm, CONFIG["NORM_STATS"])
#             gt_physical = unnormalize_action(gt_np_norm, CONFIG["NORM_STATS"])
            
#             # --- 积分计算轨迹 ---
#             traj_pred = integrate_trajectory(pred_physical, dt=CONFIG["dt"])
#             traj_gt = integrate_trajectory(gt_physical, dt=CONFIG["dt"])
            
#             # --- 计算误差 ---
#             # 欧氏距离 (Euclidean Distance)
#             errors = np.linalg.norm(traj_pred - traj_gt, axis=1)
            
#             ade = np.mean(errors) # Average Displacement Error
#             fde = errors[-1]      # Final Displacement Error
            
#             ade_losses.append(ade)
#             fde_losses.append(fde)
            
#             # 保存用于绘图
#             if len(plot_data) < CONFIG["plot_samples"]:
#                 plot_data.append((traj_gt, traj_pred, ade))

#     # 5. 打印统计结果
#     print(f"\n📈 === 评估报告 ===")
#     print(f"Input Mode: {CONFIG['input_mode']}")
#     print(f"Horizon   : {CONFIG['pred_horizon']}")
#     print(f"--------------------")
#     print(f"ADE (平均轨迹误差): {np.mean(ade_losses):.4f} m")
#     print(f"FDE (终点位移误差): {np.mean(fde_losses):.4f} m")

#     # 6. 可视化绘图
#     plot_results(plot_data)

# def plot_results(plot_data):
#     """绘制对比图"""
#     num_plots = len(plot_data)
#     if num_plots == 0: return

#     fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
#     if num_plots == 1: axes = [axes]
    
#     for i, (gt, pred, ade) in enumerate(plot_data):
#         ax = axes[i]
        
#         # 绘制轨迹 (Y轴为 Forward, X轴为 Lateral)
#         # 注意: 机器人坐标系通常 X 前向, Y 左向。
#         # 这里为了直观，我们 plot(y, x) 让 X 轴朝上，或者直接 plot(x, y) 看平面图
        
#         # 假设 (x, y) = (Forward, Lateral)
#         ax.plot(gt[:, 1], gt[:, 0], c='black', label='Ground Truth', linewidth=2, alpha=0.7)
#         ax.plot(pred[:, 1], pred[:, 0], c='red', label='ACT Prediction', linewidth=2, linestyle='--')
        
#         # 标记起点和终点
#         ax.scatter(gt[0, 1], gt[0, 0], c='green', s=60, marker='^', label='Start') # 起点
#         ax.scatter(gt[-1, 1], gt[-1, 0], c='black', s=40, marker='x') # GT终点
#         ax.scatter(pred[-1, 1], pred[-1, 0], c='red', s=40, marker='x') # Pred终点
        
#         ax.set_title(f"Sample {i+1} | ADE: {ade:.3f}m")
#         ax.axis('equal') # 保持比例，这对于看轨迹非常重要
#         ax.grid(True, alpha=0.3)
        
#         if i == 0:
#             ax.legend()
            
#     plt.tight_layout()
#     save_path = "act_eval_result.png"
#     plt.savefig(save_path)
#     print(f"✅ 可视化结果已保存至: {save_path}")
#     plt.show()

# if __name__ == "__main__":
#     evaluate()

# import torch
# from torch.utils.data import DataLoader, random_split
# import numpy as np
# import matplotlib.pyplot as plt
# from coloact.data.dataset import ColonDatasetDiffusion
# from coloact.models.act import ACTPolicy # 确保这是你修改后带位置编码的正确模型类
# import os

# # ================= 配置区域 =================
# CONFIG = {
#     "data_dir": "data/h5",
#     "model_path": "outputs/Checkpoints_ACT_RGBD/act_best.pth", # 建议用 val loss 最低的模型
#     "input_mode": "rgbd_elev",
#     "pred_horizon": 16,
#     "dt": 0.1,  # 时间步长 (根据你的采集频率，10Hz=0.1s, 15Hz=0.067s)
#     "plot_samples": 4  # 画几张对比图
# }
# # ===========================================

# def integrate_trajectory(actions, init_pose=(0,0,0), dt=0.1):
#     """
#     将动作序列 (v, w) 积分为轨迹 (x, y)
#     actions: (T, 2)
#     """
#     x, y, theta = init_pose
#     traj = [(x, y)]
    
#     # 速度缩放 (根据你的训练数据归一化情况，这里假设直接输出真实物理量)
#     # 如果你的模型输出归一化了，这里记得反归一化，比如 action * 3.0
#     SPEED_SCALE = 3.0 
    
#     for v, w in actions:
#         v = v * SPEED_SCALE
#         w = w * SPEED_SCALE
        
#         # 简单差分驱动运动学
#         x += v * np.cos(theta) * dt
#         y += v * np.sin(theta) * dt
#         theta += w * dt
#         traj.append((x, y))
        
#     return np.array(traj)

# def evaluate():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. 准备验证集 (保持和训练时一样的切分逻辑，或者固定随机种子)
#     full_dataset = ColonDatasetDiffusion(
#         data_dir=CONFIG["data_dir"],
#         pred_horizon=CONFIG["pred_horizon"],
#         input_mode=CONFIG["input_mode"]
#     )
    
#     # 简单的切分策略：取最后 20% 作为测试
#     val_size = int(0.2 * len(full_dataset))
#     train_size = len(full_dataset) - val_size
#     _, val_dataset = random_split(
#         full_dataset, [train_size, val_size], 
#         generator=torch.Generator().manual_seed(42) # 🔥 固定种子，保证每次测的是同一批数据
#     )
    
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
#     print(f"📊 验证集大小: {len(val_dataset)}")

#     # 2. 加载模型
#     channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
#     model = ACTPolicy(
#         pred_horizon=CONFIG["pred_horizon"], 
#         input_channels=channel_map[CONFIG["input_mode"]]
#     ).to(device)
    
#     try:
#         model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
#         print("✅ 模型加载成功")
#     except Exception as e:
#         print(f"❌ 模型加载失败: {e}")
#         return

#     model.eval()
    
#     ade_losses = []
#     fde_losses = []
#     plot_data = []

#     print("🚀 开始评估 ADE/FDE...")
    
#     with torch.no_grad():
#         for idx, (obs, gt_actions) in enumerate(val_loader):
#             obs = obs.to(device)
#             gt_actions = gt_actions.to(device) # (1, 16, 2)

#             # 推理 (ACT 推理时 actions=None, z 默认为 0)
#             pred_actions, _, _ = model(obs, actions=None) 
            
#             # 转为 numpy
#             pred_np = pred_actions.cpu().numpy()[0] # (16, 2)
#             gt_np = gt_actions.cpu().numpy()[0]     # (16, 2)
            
#             # --- 核心：积分转为轨迹 (x, y) ---
#             traj_pred = integrate_trajectory(pred_np, dt=CONFIG["dt"])
#             traj_gt = integrate_trajectory(gt_np, dt=CONFIG["dt"])
            
#             # --- 计算误差 ---
#             # 欧氏距离 error = sqrt((x1-x2)^2 + (y1-y2)^2)
#             errors = np.linalg.norm(traj_pred - traj_gt, axis=1)
            
#             ade = np.mean(errors) # 平均位移误差
#             fde = errors[-1]      # 终点位移误差
            
#             ade_losses.append(ade)
#             fde_losses.append(fde)
            
#             # 收集几个样本画图用
#             if len(plot_data) < CONFIG["plot_samples"]:
#                 plot_data.append((traj_gt, traj_pred, ade))

#     print(f"\n📈 === 评估结果 ===")
#     print(f"平均轨迹误差 (ADE): {np.mean(ade_losses):.4f} m")
#     print(f"终点位移误差 (FDE): {np.mean(fde_losses):.4f} m")

#     # --- 画图 ---
#     fig, axes = plt.subplots(1, CONFIG["plot_samples"], figsize=(16, 4))
#     if CONFIG["plot_samples"] == 1: axes = [axes]
    
#     for i, (gt, pred, ade) in enumerate(plot_data):
#         ax = axes[i]
#         # 画 Ground Truth (人类)
#         ax.plot(gt[:, 1], gt[:, 0], 'k-', linewidth=2, label='Human (GT)') # 注意: 通常 Robot X是前向，画图时 Y轴作为前向更直观，这里交换了XY
#         # 画 Prediction (模型)
#         ax.plot(pred[:, 1], pred[:, 0], 'r--', linewidth=2, marker='o', markersize=3, label='ACT Pred')
        
#         # 标记起点
#         ax.plot(gt[0, 1], gt[0, 0], 'g^', markersize=8) # 起点
        
#         ax.set_title(f"Sample {i+1}\nADE: {ade:.3f}m")
#         ax.axis('equal')
#         if i == 0: ax.legend()
#         ax.grid(True, alpha=0.3)
#         ax.set_xlabel("Lateral (m)")
#         ax.set_ylabel("Forward (m)")

#     plt.suptitle(f"Open-Loop Evaluation (Validation Set) - ADE: {np.mean(ade_losses):.3f}m", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("eval_ade_chart.png")
#     print("✅ 图表已保存为 eval_ade_chart.png")
#     plt.show()

# if __name__ == "__main__":
#     evaluate()