"""
对比两个PatchTST模型的测试结果并可视化

对比：
1. 不加知识图谱特征的模型
2. 加知识图谱特征的模型
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

sys.stdout.reconfigure(encoding='utf-8')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
MODEL1_DIR = './models/patchtst'              # 不加KG模型
MODEL2_DIR = './models/patchtst_kg'           # 加KG模型（如果有）
OUTPUT_DIR = './results/comparison'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 加载结果 ====================
def load_model_results(model_dir, model_name):
    """加载模型测试结果"""
    results_path = os.path.join(model_dir, 'test_results.pkl')

    if not os.path.exists(results_path):
        print(f"❌ {model_name} 结果文件不存在: {results_path}")
        return None

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print(f"✓ 加载 {model_name}")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE: {results['mae']:.6f}")

    return results

def load_metadata(data_dir='./data/processed_features'):
    """加载metadata"""
    metadata_path = os.path.join(data_dir, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

# ==================== 可视化函数 ====================
def plot_overall_metrics_comparison(results1, results2, metadata):
    """绘制整体指标对比"""
    metrics = ['MSE', 'RMSE', 'MAE']

    model1_scores = [
        results1['mse_original'],
        results1['rmse_original'],
        results1['mae_original']
    ]

    if results2:
        model2_scores = [
            results2['mse_original'],
            results2['rmse_original'],
            results2['mae_original']
        ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, model1_scores, width, label='不加KG', color='steelblue')
    if results2:
        bars2 = ax.bar(x + width/2, model2_scores, width, label='加KG', color='coral')

    ax.set_xlabel('评估指标', fontsize=12)
    ax.set_ylabel('误差值', fontsize=12)
    ax.set_title('模型整体性能对比（原始尺度）', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1] + ([bars2] if results2 else []):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_overall_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 1_overall_metrics.png")
    plt.close()

def plot_feature_mae_comparison(results1, results2, metadata):
    """绘制各特征MAE对比"""
    pred1 = results1['predictions_original']
    true1 = results1['ground_truth_original']

    # 获取特征名
    if 'target_features' in metadata:
        features = metadata['target_features']
    else:
        features = metadata['feature_names']

    n_features = len(features)
    mae1_by_feature = []

    for i in range(n_features):
        mae = np.mean(np.abs(pred1[:, i] - true1[:, i]))
        mae1_by_feature.append(mae)

    if results2:
        pred2 = results2['predictions_original']
        true2 = results2['ground_truth_original']
        mae2_by_feature = []
        for i in range(n_features):
            mae = np.mean(np.abs(pred2[:, i] - true2[:, i]))
            mae2_by_feature.append(mae)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(features))
    width = 0.35

    bars1 = ax.bar(x - width/2, mae1_by_feature, width, label='不加KG', color='steelblue')
    if results2:
        bars2 = ax.bar(x + width/2, mae2_by_feature, width, label='加KG', color='coral')

    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('MAE（原始尺度）', fontsize=12)
    ax.set_title('各特征预测误差对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1] + ([bars2] if results2 else []):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_feature_mae.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 2_feature_mae.png")
    plt.close()

def plot_timestep_mae_comparison(results1, results2, metadata):
    """绘制各预测时间步MAE对比"""
    pred1 = results1['predictions_original']
    true1 = results1['ground_truth_original']

    if 'target_features' in metadata:
        n_features = metadata['n_target_features']
        horizon = metadata['forecast_horizon']
    else:
        n_features = metadata['n_features']
        horizon = metadata['forecast_horizon']

    pred1_reshaped = pred1.reshape(-1, horizon, n_features)
    true1_reshaped = true1.reshape(-1, horizon, n_features)

    mae1_by_timestep = []
    for t in range(horizon):
        mae = np.mean(np.abs(pred1_reshaped[:, t, :] - true1_reshaped[:, t, :]))
        mae1_by_timestep.append(mae)

    if results2:
        pred2 = results2['predictions_original']
        true2 = results2['ground_truth_original']
        pred2_reshaped = pred2.reshape(-1, horizon, n_features)
        true2_reshaped = true2.reshape(-1, horizon, n_features)

        mae2_by_timestep = []
        for t in range(horizon):
            mae = np.mean(np.abs(pred2_reshaped[:, t, :] - true2_reshaped[:, t, :]))
            mae2_by_timestep.append(mae)

    fig, ax = plt.subplots(figsize=(10, 6))

    timesteps = [f'+{i+1}h' for i in range(horizon)]
    x = np.arange(horizon)

    ax.plot(x, mae1_by_timestep, marker='o', linewidth=2, markersize=8,
            label='不加KG', color='steelblue')
    if results2:
        ax.plot(x, mae2_by_timestep, marker='s', linewidth=2, markersize=8,
                label='加KG', color='coral')

    ax.set_xlabel('预测时间步', fontsize=12)
    ax.set_ylabel('MAE（原始尺度）', fontsize=12)
    ax.set_title('不同预测时间步的误差对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(timesteps)
    ax.legend()
    ax.grid(alpha=0.3)

    # 添加数值标签
    for i, (x_pos, y_val) in enumerate(zip(x, mae1_by_timestep)):
        ax.text(x_pos, y_val, f'{y_val:.3f}', ha='center', va='bottom', fontsize=9)

    if results2:
        for i, (x_pos, y_val) in enumerate(zip(x, mae2_by_timestep)):
            ax.text(x_pos, y_val, f'{y_val:.3f}', ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_timestep_mae.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 3_timestep_mae.png")
    plt.close()

def plot_scatter_comparison(results1, results2, metadata):
    """绘制预测vs真实值散点图"""
    if 'target_features' in metadata:
        features = metadata['target_features']
    else:
        features = metadata['feature_names']

    n_features = len(features)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    pred1 = results1['predictions_original']
    true1 = results1['ground_truth_original']

    for i in range(min(n_features, 4)):
        ax = axes[i]

        # 模型1
        ax.scatter(true1[:, i], pred1[:, i], alpha=0.3, s=10,
                  color='steelblue', label='不加KG')

        # 模型2
        if results2:
            pred2 = results2['predictions_original']
            true2 = results2['ground_truth_original']
            ax.scatter(true2[:, i], pred2[:, i], alpha=0.3, s=10,
                      color='coral', label='加KG')

        # 对角线
        min_val = min(true1[:, i].min(), pred1[:, i].min())
        max_val = max(true1[:, i].max(), pred1[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel(f'真实值 - {features[i]}', fontsize=10)
        ax.set_ylabel(f'预测值 - {features[i]}', fontsize=10)
        ax.set_title(f'{features[i]} 预测对比', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_scatter_plots.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 4_scatter_plots.png")
    plt.close()

def plot_error_distribution(results1, results2, metadata):
    """绘制误差分布直方图"""
    if 'target_features' in metadata:
        features = metadata['target_features']
    else:
        features = metadata['feature_names']

    n_features = len(features)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    pred1 = results1['predictions_original']
    true1 = results1['ground_truth_original']

    for i in range(min(n_features, 4)):
        ax = axes[i]

        errors1 = pred1[:, i] - true1[:, i]

        ax.hist(errors1, bins=50, alpha=0.6, color='steelblue',
               label='不加KG', edgecolor='black')

        if results2:
            pred2 = results2['predictions_original']
            true2 = results2['ground_truth_original']
            errors2 = pred2[:, i] - true2[:, i]
            ax.hist(errors2, bins=50, alpha=0.6, color='coral',
                   label='加KG', edgecolor='black')

        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('预测误差', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_title(f'{features[i]} 误差分布', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_error_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 5_error_distribution.png")
    plt.close()

def plot_improvement_percentage(results1, results2, metadata):
    """绘制改进百分比"""
    if not results2:
        print("⚠ 只有一个模型，跳过改进百分比图")
        return

    if 'target_features' in metadata:
        features = metadata['target_features']
    else:
        features = metadata['feature_names']

    n_features = len(features)

    pred1 = results1['predictions_original']
    true1 = results1['ground_truth_original']
    pred2 = results2['predictions_original']
    true2 = results2['ground_truth_original']

    improvements = []
    for i in range(n_features):
        mae1 = np.mean(np.abs(pred1[:, i] - true1[:, i]))
        mae2 = np.mean(np.abs(pred2[:, i] - true2[:, i]))
        improvement = ((mae1 - mae2) / mae1) * 100
        improvements.append(improvement)

    # 整体改进
    overall_mae1 = results1['mae_original']
    overall_mae2 = results2['mae_original']
    overall_improvement = ((overall_mae1 - overall_mae2) / overall_mae1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.bar(features, improvements, color=colors, alpha=0.7, edgecolor='black')

    # 添加整体改进线
    ax.axhline(overall_improvement, color='blue', linestyle='--',
              linewidth=2, label=f'整体改进: {overall_improvement:.2f}%')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('改进百分比 (%)', fontsize=12)
    ax.set_title('加入知识图谱后各特征的改进程度', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center',
               va='bottom' if height > 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_improvement.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: 6_improvement.png")
    plt.close()

# ==================== 生成对比报告 ====================
def generate_comparison_report(results1, results2, metadata):
    """生成文本对比报告"""
    report_path = os.path.join(OUTPUT_DIR, 'comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PatchTST模型对比报告\n")
        f.write("=" * 80 + "\n\n")

        # 整体指标
        f.write("【1. 整体性能指标（原始尺度）】\n")
        f.write(f"{'指标':<15} {'不加KG':<20} {'加KG':<20} {'改进':<15}\n")
        f.write("-" * 80 + "\n")

        metrics = [
            ('MSE', 'mse_original'),
            ('RMSE', 'rmse_original'),
            ('MAE', 'mae_original')
        ]

        for name, key in metrics:
            val1 = results1[key]
            if results2:
                val2 = results2[key]
                improvement = ((val1 - val2) / val1) * 100
                f.write(f"{name:<15} {val1:<20.6f} {val2:<20.6f} {improvement:>+.2f}%\n")
            else:
                f.write(f"{name:<15} {val1:<20.6f} {'N/A':<20} {'N/A':<15}\n")

        # 各特征详细对比
        f.write("\n【2. 各特征MAE对比】\n")

        if 'target_features' in metadata:
            features = metadata['target_features']
        else:
            features = metadata['feature_names']

        pred1 = results1['predictions_original']
        true1 = results1['ground_truth_original']

        f.write(f"{'特征':<15} {'不加KG':<20} {'加KG':<20} {'改进':<15}\n")
        f.write("-" * 80 + "\n")

        for i, feat in enumerate(features):
            mae1 = np.mean(np.abs(pred1[:, i] - true1[:, i]))
            if results2:
                pred2 = results2['predictions_original']
                true2 = results2['ground_truth_original']
                mae2 = np.mean(np.abs(pred2[:, i] - true2[:, i]))
                improvement = ((mae1 - mae2) / mae1) * 100
                f.write(f"{feat:<15} {mae1:<20.6f} {mae2:<20.6f} {improvement:>+.2f}%\n")
            else:
                f.write(f"{feat:<15} {mae1:<20.6f} {'N/A':<20} {'N/A':<15}\n")

        # 各时间步对比
        f.write("\n【3. 各预测时间步MAE对比】\n")

        if 'forecast_horizon' in metadata:
            horizon = metadata['forecast_horizon']
            n_features = metadata.get('n_target_features', metadata.get('n_features'))

        pred1_reshaped = pred1.reshape(-1, horizon, n_features)
        true1_reshaped = true1.reshape(-1, horizon, n_features)

        f.write(f"{'时间步':<15} {'不加KG':<20} {'加KG':<20} {'改进':<15}\n")
        f.write("-" * 80 + "\n")

        for t in range(horizon):
            mae1 = np.mean(np.abs(pred1_reshaped[:, t, :] - true1_reshaped[:, t, :]))
            if results2:
                pred2_reshaped = results2['predictions_original'].reshape(-1, horizon, n_features)
                true2_reshaped = results2['ground_truth_original'].reshape(-1, horizon, n_features)
                mae2 = np.mean(np.abs(pred2_reshaped[:, t, :] - true2_reshaped[:, t, :]))
                improvement = ((mae1 - mae2) / mae1) * 100
                f.write(f"+{t+1}小时{'':<9} {mae1:<20.6f} {mae2:<20.6f} {improvement:>+.2f}%\n")
            else:
                f.write(f"+{t+1}小时{'':<9} {mae1:<20.6f} {'N/A':<20} {'N/A':<15}\n")

    print(f"✓ 保存: comparison_report.txt")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("PatchTST模型对比与可视化")
    print("=" * 80)

    # 加载metadata
    print("\n【加载数据】")
    metadata = load_metadata()

    # 加载模型1（不加KG）
    print("\n【加载模型结果】")
    results1 = load_model_results(MODEL1_DIR, "模型1: 不加KG")

    if results1 is None:
        print("❌ 模型1结果加载失败，退出")
        return

    # 加载模型2（加KG）
    results2 = load_model_results(MODEL2_DIR, "模型2: 加KG")

    # 生成可视化
    print("\n【生成对比图表】")
    plot_overall_metrics_comparison(results1, results2, metadata)
    plot_feature_mae_comparison(results1, results2, metadata)
    plot_timestep_mae_comparison(results1, results2, metadata)
    plot_scatter_comparison(results1, results2, metadata)
    plot_error_distribution(results1, results2, metadata)

    if results2:
        plot_improvement_percentage(results1, results2, metadata)

    # 生成报告
    print("\n【生成对比报告】")
    generate_comparison_report(results1, results2, metadata)

    print("\n" + "=" * 80)
    print(f"对比完成！结果保存在: {OUTPUT_DIR}")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - 1_overall_metrics.png       整体指标对比")
    print("  - 2_feature_mae.png           各特征MAE对比")
    print("  - 3_timestep_mae.png          各时间步MAE对比")
    print("  - 4_scatter_plots.png         预测vs真实值散点图")
    print("  - 5_error_distribution.png    误差分布直方图")
    if results2:
        print("  - 6_improvement.png           改进百分比图")
    print("  - comparison_report.txt       详细对比报告")

if __name__ == '__main__':
    main()
