"""
PatchTST模型对比分析
对比原始PatchTST与TSVec+PatchTST融合模型的性能

功能：
1. 自动加载两个模型的测试结果
2. 详细对比各项指标（MSE, RMSE, MAE）
3. 对比各特征的预测误差
4. 对比不同预测时间步的误差
5. 生成可视化对比图表
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# 设置matplotlib中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = '../data/processed_features'
PATCHTST_DIR = '../models/patchtst'
PATCHTST_TSVEC_DIR = '../models/patchtst_with_tsvec'
OUTPUT_DIR = '../results/comparison_patchtst_versions'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 加载结果 ====================
def load_results():
    """加载两个模型的测试结果"""
    print("=" * 80)
    print("加载测试结果")
    print("=" * 80)

    # 加载metadata
    with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # 加载原始PatchTST结果
    patchtst_results_path = os.path.join(PATCHTST_DIR, 'test_results.pkl')
    if not os.path.exists(patchtst_results_path):
        print(f"❌ 未找到原始PatchTST测试结果: {patchtst_results_path}")
        print(f"   请先运行: python test/test_patchtst.py")
        sys.exit(1)

    with open(patchtst_results_path, 'rb') as f:
        patchtst_results = pickle.load(f)

    print(f"✓ 原始PatchTST结果加载完成")
    print(f"  路径: {patchtst_results_path}")
    print(f"  样本数: {len(patchtst_results['predictions'])}")

    # 加载TSVec+PatchTST结果
    patchtst_tsvec_results_path = os.path.join(PATCHTST_TSVEC_DIR, 'test_results.pkl')
    if not os.path.exists(patchtst_tsvec_results_path):
        print(f"\n❌ 未找到TSVec+PatchTST测试结果: {patchtst_tsvec_results_path}")
        print(f"   请先运行: python test/test_patchtst_with_tsvec.py")
        sys.exit(1)

    with open(patchtst_tsvec_results_path, 'rb') as f:
        patchtst_tsvec_results = pickle.load(f)

    print(f"\n✓ TSVec+PatchTST结果加载完成")
    print(f"  路径: {patchtst_tsvec_results_path}")
    print(f"  样本数: {len(patchtst_tsvec_results['predictions'])}")

    return patchtst_results, patchtst_tsvec_results, metadata

# ==================== 指标对比 ====================
def compare_overall_metrics(patchtst_results, patchtst_tsvec_results):
    """对比整体指标"""
    print("\n" + "=" * 80)
    print("【1. 整体指标对比】")
    print("=" * 80)

    metrics = ['mse', 'rmse', 'mae', 'mse_original', 'rmse_original', 'mae_original']
    metric_names = {
        'mse': 'MSE（归一化）',
        'rmse': 'RMSE（归一化）',
        'mae': 'MAE（归一化）',
        'mse_original': 'MSE（原始尺度）',
        'rmse_original': 'RMSE（原始尺度）',
        'mae_original': 'MAE（原始尺度）'
    }

    comparison_data = {}

    print(f"\n{'指标':<25} | {'PatchTST':>15} | {'PatchTST+TSVec':>15} | {'改进':>12} | {'改进率':>10}")
    print("-" * 90)

    for metric in metrics:
        patchtst_val = patchtst_results[metric]
        tsvec_val = patchtst_tsvec_results[metric]

        improvement = patchtst_val - tsvec_val
        improvement_pct = (improvement / patchtst_val) * 100 if patchtst_val != 0 else 0

        comparison_data[metric] = {
            'patchtst': patchtst_val,
            'tsvec': tsvec_val,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

        # 格式化输出
        status = "↓" if improvement > 0 else "↑"
        print(f"{metric_names[metric]:<25} | {patchtst_val:>15.6f} | {tsvec_val:>15.6f} | "
              f"{improvement:>11.6f} {status} | {improvement_pct:>9.2f}%")

    return comparison_data

def compare_feature_errors(patchtst_results, patchtst_tsvec_results, metadata):
    """对比各特征的误差"""
    print("\n" + "=" * 80)
    print("【2. 各特征MAE对比（原始尺度）】")
    print("=" * 80)

    target_features = metadata['target_features']

    patchtst_pred = patchtst_results['predictions_original']
    patchtst_true = patchtst_results['ground_truth_original']

    tsvec_pred = patchtst_tsvec_results['predictions_original']
    tsvec_true = patchtst_tsvec_results['ground_truth_original']

    feature_comparison = {}

    print(f"\n{'特征':<20} | {'PatchTST MAE':>15} | {'TSVec MAE':>15} | {'改进':>12} | {'改进率':>10}")
    print("-" * 80)

    for i, feat in enumerate(target_features):
        patchtst_mae = np.mean(np.abs(patchtst_pred[:, i] - patchtst_true[:, i]))
        tsvec_mae = np.mean(np.abs(tsvec_pred[:, i] - tsvec_true[:, i]))

        improvement = patchtst_mae - tsvec_mae
        improvement_pct = (improvement / patchtst_mae) * 100 if patchtst_mae != 0 else 0

        feature_comparison[feat] = {
            'patchtst': patchtst_mae,
            'tsvec': tsvec_mae,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

        status = "↓" if improvement > 0 else "↑"
        print(f"{feat:<20} | {patchtst_mae:>15.4f} | {tsvec_mae:>15.4f} | "
              f"{improvement:>11.4f} {status} | {improvement_pct:>9.2f}%")

    return feature_comparison

def compare_timestep_errors(patchtst_results, patchtst_tsvec_results, metadata):
    """对比各预测时间步的误差"""
    print("\n" + "=" * 80)
    print("【3. 各预测时间步MAE对比（原始尺度）】")
    print("=" * 80)

    forecast_horizon = metadata['forecast_horizon']
    n_features = len(metadata['target_features'])

    patchtst_pred = patchtst_results['predictions_original'].reshape(-1, forecast_horizon, n_features)
    patchtst_true = patchtst_results['ground_truth_original'].reshape(-1, forecast_horizon, n_features)

    tsvec_pred = patchtst_tsvec_results['predictions_original'].reshape(-1, forecast_horizon, n_features)
    tsvec_true = patchtst_tsvec_results['ground_truth_original'].reshape(-1, forecast_horizon, n_features)

    timestep_comparison = {}

    print(f"\n{'时间步':<15} | {'PatchTST MAE':>15} | {'TSVec MAE':>15} | {'改进':>12} | {'改进率':>10}")
    print("-" * 75)

    for t in range(forecast_horizon):
        patchtst_mae_t = np.mean(np.abs(patchtst_pred[:, t, :] - patchtst_true[:, t, :]))
        tsvec_mae_t = np.mean(np.abs(tsvec_pred[:, t, :] - tsvec_true[:, t, :]))

        improvement = patchtst_mae_t - tsvec_mae_t
        improvement_pct = (improvement / patchtst_mae_t) * 100 if patchtst_mae_t != 0 else 0

        timestep_comparison[f"+{t+1}h"] = {
            'patchtst': patchtst_mae_t,
            'tsvec': tsvec_mae_t,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

        status = "↓" if improvement > 0 else "↑"
        print(f"+{t+1}小时{'':<9} | {patchtst_mae_t:>15.4f} | {tsvec_mae_t:>15.4f} | "
              f"{improvement:>11.4f} {status} | {improvement_pct:>9.2f}%")

    return timestep_comparison

# ==================== 可视化 ====================
def plot_overall_metrics(comparison_data):
    """绘制整体指标对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 归一化空间指标
    metrics_norm = ['mse', 'rmse', 'mae']
    labels_norm = ['MSE', 'RMSE', 'MAE']
    patchtst_vals_norm = [comparison_data[m]['patchtst'] for m in metrics_norm]
    tsvec_vals_norm = [comparison_data[m]['tsvec'] for m in metrics_norm]

    x = np.arange(len(labels_norm))
    width = 0.35

    axes[0].bar(x - width/2, patchtst_vals_norm, width, label='PatchTST', alpha=0.8)
    axes[0].bar(x + width/2, tsvec_vals_norm, width, label='PatchTST+TSVec', alpha=0.8)
    axes[0].set_ylabel('误差值')
    axes[0].set_title('整体指标对比（归一化空间）')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_norm)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 原始尺度指标
    metrics_orig = ['mse_original', 'rmse_original', 'mae_original']
    labels_orig = ['MSE', 'RMSE', 'MAE']
    patchtst_vals_orig = [comparison_data[m]['patchtst'] for m in metrics_orig]
    tsvec_vals_orig = [comparison_data[m]['tsvec'] for m in metrics_orig]

    axes[1].bar(x - width/2, patchtst_vals_orig, width, label='PatchTST', alpha=0.8)
    axes[1].bar(x + width/2, tsvec_vals_orig, width, label='PatchTST+TSVec', alpha=0.8)
    axes[1].set_ylabel('误差值')
    axes[1].set_title('整体指标对比（原始尺度）')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_orig)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '1_overall_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 整体指标对比图已保存: {output_path}")
    plt.close()

def plot_feature_comparison(feature_comparison):
    """绘制各特征误差对比"""
    features = list(feature_comparison.keys())
    patchtst_vals = [feature_comparison[f]['patchtst'] for f in features]
    tsvec_vals = [feature_comparison[f]['tsvec'] for f in features]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, patchtst_vals, width, label='PatchTST', alpha=0.8)
    ax.bar(x + width/2, tsvec_vals, width, label='PatchTST+TSVec', alpha=0.8)

    ax.set_ylabel('MAE（原始尺度）')
    ax.set_title('各特征MAE对比')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '2_feature_mae.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 各特征MAE对比图已保存: {output_path}")
    plt.close()

def plot_timestep_comparison(timestep_comparison):
    """绘制各时间步误差对比"""
    timesteps = list(timestep_comparison.keys())
    patchtst_vals = [timestep_comparison[t]['patchtst'] for t in timesteps]
    tsvec_vals = [timestep_comparison[t]['tsvec'] for t in timesteps]

    x = np.arange(len(timesteps))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, patchtst_vals, marker='o', label='PatchTST', linewidth=2, markersize=8)
    ax.plot(x, tsvec_vals, marker='s', label='PatchTST+TSVec', linewidth=2, markersize=8)

    ax.set_xlabel('预测时间步')
    ax.set_ylabel('MAE（原始尺度）')
    ax.set_title('各预测时间步MAE对比')
    ax.set_xticks(x)
    ax.set_xticklabels(timesteps)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '3_timestep_mae.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 各时间步MAE对比图已保存: {output_path}")
    plt.close()

def plot_improvement_rates(comparison_data, feature_comparison, timestep_comparison):
    """绘制改进率总览"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 整体指标改进率
    metrics = ['mae', 'mae_original']
    labels = ['MAE\n(归一化)', 'MAE\n(原始尺度)']
    improvement_rates = [comparison_data[m]['improvement_pct'] for m in metrics]

    axes[0].bar(labels, improvement_rates, alpha=0.8, color=['steelblue', 'darkorange'])
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_ylabel('改进率 (%)')
    axes[0].set_title('整体指标改进率')
    axes[0].grid(axis='y', alpha=0.3)

    # 各特征改进率
    features = list(feature_comparison.keys())
    feat_improvements = [feature_comparison[f]['improvement_pct'] for f in features]

    axes[1].bar(features, feat_improvements, alpha=0.8)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_ylabel('改进率 (%)')
    axes[1].set_title('各特征MAE改进率')
    axes[1].grid(axis='y', alpha=0.3)

    # 各时间步改进率
    timesteps = list(timestep_comparison.keys())
    timestep_improvements = [timestep_comparison[t]['improvement_pct'] for t in timesteps]

    axes[2].plot(range(len(timesteps)), timestep_improvements, marker='o', linewidth=2, markersize=8)
    axes[2].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[2].set_xlabel('预测时间步')
    axes[2].set_ylabel('改进率 (%)')
    axes[2].set_title('各时间步MAE改进率')
    axes[2].set_xticks(range(len(timesteps)))
    axes[2].set_xticklabels(timesteps)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '4_improvement_rates.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 改进率总览图已保存: {output_path}")
    plt.close()

def plot_prediction_samples(patchtst_results, patchtst_tsvec_results, metadata, n_samples=3):
    """绘制预测样本对比"""
    target_features = metadata['target_features']
    forecast_horizon = metadata['forecast_horizon']
    n_features = len(target_features)

    patchtst_pred = patchtst_results['predictions_original'].reshape(-1, forecast_horizon, n_features)
    patchtst_true = patchtst_results['ground_truth_original'].reshape(-1, forecast_horizon, n_features)

    tsvec_pred = patchtst_tsvec_results['predictions_original'].reshape(-1, forecast_horizon, n_features)

    fig, axes = plt.subplots(n_samples, n_features, figsize=(16, 3*n_samples))

    for sample_idx in range(n_samples):
        for feat_idx, feat_name in enumerate(target_features):
            ax = axes[sample_idx, feat_idx] if n_samples > 1 else axes[feat_idx]

            timesteps = list(range(1, forecast_horizon + 1))

            # 真实值
            ax.plot(timesteps, patchtst_true[sample_idx, :, feat_idx],
                   marker='o', label='真实值', linewidth=2, markersize=6)

            # PatchTST预测
            ax.plot(timesteps, patchtst_pred[sample_idx, :, feat_idx],
                   marker='s', label='PatchTST', linewidth=2, markersize=6, linestyle='--')

            # TSVec预测
            ax.plot(timesteps, tsvec_pred[sample_idx, :, feat_idx],
                   marker='^', label='PatchTST+TSVec', linewidth=2, markersize=6, linestyle='--')

            if sample_idx == 0:
                ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold')

            if feat_idx == 0:
                ax.set_ylabel(f'样本 #{sample_idx+1}\n数值', fontsize=10)

            if sample_idx == n_samples - 1:
                ax.set_xlabel('预测时间步（小时）', fontsize=10)

            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '5_prediction_samples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 预测样本对比图已保存: {output_path}")
    plt.close()

# ==================== 保存详细报告 ====================
def save_detailed_report(comparison_data, feature_comparison, timestep_comparison, metadata):
    """保存详细的对比报告"""
    report_path = os.path.join(OUTPUT_DIR, 'comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PatchTST模型对比分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"预测时间步: {metadata['forecast_horizon']}小时\n")
        f.write(f"目标特征: {', '.join(metadata['target_features'])}\n\n")

        # 1. 整体指标对比
        f.write("=" * 80 + "\n")
        f.write("【1. 整体指标对比】\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'指标':<25} | {'PatchTST':>15} | {'PatchTST+TSVec':>15} | {'改进':>12} | {'改进率':>10}\n")
        f.write("-" * 90 + "\n")

        metric_names = {
            'mse': 'MSE（归一化）',
            'rmse': 'RMSE（归一化）',
            'mae': 'MAE（归一化）',
            'mse_original': 'MSE（原始尺度）',
            'rmse_original': 'RMSE（原始尺度）',
            'mae_original': 'MAE（原始尺度）'
        }

        for metric, name in metric_names.items():
            data = comparison_data[metric]
            status = "↓" if data['improvement'] > 0 else "↑"
            f.write(f"{name:<25} | {data['patchtst']:>15.6f} | {data['tsvec']:>15.6f} | "
                   f"{data['improvement']:>11.6f} {status} | {data['improvement_pct']:>9.2f}%\n")

        # 2. 各特征对比
        f.write("\n" + "=" * 80 + "\n")
        f.write("【2. 各特征MAE对比（原始尺度）】\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'特征':<20} | {'PatchTST MAE':>15} | {'TSVec MAE':>15} | {'改进':>12} | {'改进率':>10}\n")
        f.write("-" * 80 + "\n")

        for feat, data in feature_comparison.items():
            status = "↓" if data['improvement'] > 0 else "↑"
            f.write(f"{feat:<20} | {data['patchtst']:>15.4f} | {data['tsvec']:>15.4f} | "
                   f"{data['improvement']:>11.4f} {status} | {data['improvement_pct']:>9.2f}%\n")

        # 3. 各时间步对比
        f.write("\n" + "=" * 80 + "\n")
        f.write("【3. 各预测时间步MAE对比（原始尺度）】\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'时间步':<15} | {'PatchTST MAE':>15} | {'TSVec MAE':>15} | {'改进':>12} | {'改进率':>10}\n")
        f.write("-" * 75 + "\n")

        for timestep, data in timestep_comparison.items():
            status = "↓" if data['improvement'] > 0 else "↑"
            f.write(f"{timestep:<15} | {data['patchtst']:>15.4f} | {data['tsvec']:>15.4f} | "
                   f"{data['improvement']:>11.4f} {status} | {data['improvement_pct']:>9.2f}%\n")

        # 4. 总结
        f.write("\n" + "=" * 80 + "\n")
        f.write("【4. 总结】\n")
        f.write("=" * 80 + "\n\n")

        mae_improvement = comparison_data['mae_original']['improvement_pct']
        if mae_improvement > 0:
            f.write(f"✓ TSVec+PatchTST模型相比原始PatchTST模型，在MAE指标上有 {mae_improvement:.2f}% 的改进\n")
        else:
            f.write(f"✗ TSVec+PatchTST模型相比原始PatchTST模型，在MAE指标上有 {abs(mae_improvement):.2f}% 的退化\n")

        f.write(f"\n各特征改进情况:\n")
        for feat, data in feature_comparison.items():
            if data['improvement_pct'] > 0:
                f.write(f"  ✓ {feat}: 改进 {data['improvement_pct']:.2f}%\n")
            else:
                f.write(f"  ✗ {feat}: 退化 {abs(data['improvement_pct']):.2f}%\n")

    print(f"✓ 详细报告已保存: {report_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("PatchTST模型对比分析")
    print("对比: 原始PatchTST vs TSVec+PatchTST")
    print("=" * 80)

    # 1. 加载结果
    patchtst_results, patchtst_tsvec_results, metadata = load_results()

    # 2. 对比分析
    comparison_data = compare_overall_metrics(patchtst_results, patchtst_tsvec_results)
    feature_comparison = compare_feature_errors(patchtst_results, patchtst_tsvec_results, metadata)
    timestep_comparison = compare_timestep_errors(patchtst_results, patchtst_tsvec_results, metadata)

    # 3. 生成可视化
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80 + "\n")

    plot_overall_metrics(comparison_data)
    plot_feature_comparison(feature_comparison)
    plot_timestep_comparison(timestep_comparison)
    plot_improvement_rates(comparison_data, feature_comparison, timestep_comparison)
    plot_prediction_samples(patchtst_results, patchtst_tsvec_results, metadata, n_samples=3)

    # 4. 保存报告
    print("\n" + "=" * 80)
    print("保存详细报告")
    print("=" * 80 + "\n")

    save_detailed_report(comparison_data, feature_comparison, timestep_comparison, metadata)

    # 5. 总结
    print("\n" + "=" * 80)
    print("对比分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {OUTPUT_DIR}")
    print(f"  - 可视化图表: 5张PNG图片")
    print(f"  - 详细报告: comparison_report.txt")

    mae_improvement = comparison_data['mae_original']['improvement_pct']
    print(f"\n核心结论:")
    if mae_improvement > 0:
        print(f"  ✓ TSVec+PatchTST模型的MAE相比原始PatchTST改进了 {mae_improvement:.2f}%")
    else:
        print(f"  ✗ TSVec+PatchTST模型的MAE相比原始PatchTST退化了 {abs(mae_improvement):.2f}%")

if __name__ == '__main__':
    main()
