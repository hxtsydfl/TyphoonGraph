"""
优化版台风特征工程 - 特征选择与提取
基于past_2000-2022typhoons历史数据，提取核心特征用于预测：
- 经度 (lng)
- 纬度 (lat)
- 风速 (speed)
- 气压 (pressure)

特征分类：
1. 基础特征 (6个): lng, lat, speed, pressure, move_speed, move_dir
2. 时序特征 (滞后、差分、二阶差分)
3. 空间特征 (位置变化、轨迹曲率)
4. 强度特征 (强度比率、强度等级)
5. 统计特征 (滚动均值、标准差、极值)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 配置参数 ====================
DATA_DIR = r'./data/input_data/past_2000-2022typhoons'
OUTPUT_DIR = r'./data/processed_features'

# 预测目标特征
TARGET_FEATURES = ['lng', 'lat', 'speed', 'pressure']

# 基础特征
BASE_FEATURES = ['lng', 'lat', 'speed', 'pressure', 'move_speed', 'move_dir']

# 时序窗口参数
LOOKBACK_WINDOW = 12    # 输入：过去12小时
FORECAST_HORIZON = 6    # 预测：未来6小时

# 数据集划分
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 滚动窗口大小
ROLLING_WINDOW = 3

# ==================== 方向转换 ====================
DIRECTION_MAP = {
    '北': 0, '偏北': 0,
    '东北偏北': 22.5, '北东北': 22.5, '东北北': 22.5, '北北东': 22.5,
    '东北': 45, '北东': 45,
    '东北偏东': 67.5, '东东北': 67.5, '东北东': 67.5,
    '东': 90, '偏东': 90,
    '东南偏东': 112.5, '东东南': 112.5, '东南东': 112.5,
    '东南': 135, '南东': 135,
    '东南偏南': 157.5, '东南南': 157.5, '南东南': 157.5, '南南东': 157.5,
    '南': 180, '偏南': 180,
    '西南偏南': 202.5, '西南南': 202.5, '南南西': 202.5,
    '西南': 225, '南西': 225,
    '西南偏西': 247.5, '西南西': 247.5, '西西南': 247.5, '南西南': 247.5,
    '西': 270, '偏西': 270,
    '西北偏西': 292.5, '西北西': 292.5, '西西北': 292.5,
    '西北': 315, '北西': 315,
    '西北偏北': 337.5, '西北北': 337.5, '北西北': 337.5, '北北西': 337.5,
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
    'ENE 2': 67.5, '北偏西': 337.5,
}

def direction_to_angle(direction):
    """将方向描述转换为角度"""
    if pd.isna(direction) or direction == '':
        return np.nan
    direction_str = str(direction).strip()
    angle = DIRECTION_MAP.get(direction_str, np.nan)
    if pd.isna(angle):
        try:
            angle = float(direction_str)
        except:
            return np.nan
    return angle

# ==================== 数据加载 ====================
def load_raw_data():
    """加载past_2000-2022typhoons原始数据"""
    print("=" * 80)
    print("步骤1: 数据加载")
    print("=" * 80)

    all_typhoons = []
    typhoon_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    print(f"发现 {len(typhoon_files)} 个CSV文件")

    for filename in typhoon_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath, encoding='utf-8')

            # 检查必需列
            if not all(col in df.columns for col in BASE_FEATURES):
                continue

            df = df[BASE_FEATURES].copy()

            # 转换方向
            df['move_dir'] = df['move_dir'].apply(direction_to_angle)

            # 删除缺失值
            df = df.dropna()

            # 需要足够的数据
            min_length = LOOKBACK_WINDOW + FORECAST_HORIZON + ROLLING_WINDOW + 2
            if len(df) >= min_length:
                all_typhoons.append({
                    'name': filename.replace('.csv', ''),
                    'data': df.reset_index(drop=True)
                })
        except Exception as e:
            print(f"  跳过 {filename}: {str(e)}")
            continue

    print(f"\n✓ 成功加载 {len(all_typhoons)} 个台风")

    # 统计信息
    total_samples = sum(len(t['data']) for t in all_typhoons)
    print(f"  总观测点数: {total_samples}")
    print(f"  平均每个台风: {total_samples / len(all_typhoons):.1f} 个观测点")

    return all_typhoons

# ==================== 特征工程 ====================
class OptimizedFeatureEngineer:
    """优化版特征工程类"""

    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}

    def engineer_all_features(self, typhoons_data):
        """提取完整特征集"""
        print("\n" + "=" * 80)
        print("步骤2: 特征工程")
        print("=" * 80)

        enhanced_typhoons = []

        for typhoon in typhoons_data:
            df = typhoon['data'].copy()
            typhoon_name = typhoon['name']

            # 1. 滞后特征
            df = self._add_lag_features(df)

            # 2. 差分特征（变化率）
            df = self._add_diff_features(df)

            # 3. 二阶差分（加速度）
            df = self._add_second_diff(df)

            # 4. 周期性编码
            df = self._add_cyclic_encoding(df)

            # 5. 空间特征
            df = self._add_spatial_features(df)

            # 6. 强度特征
            df = self._add_intensity_features(df)

            # 7. 滚动统计特征
            df = self._add_rolling_stats(df)

            # 8. 交互特征
            df = self._add_interaction_features(df)

            # 删除NaN
            df = df.dropna().reset_index(drop=True)

            if len(df) >= LOOKBACK_WINDOW + FORECAST_HORIZON:
                enhanced_typhoons.append({
                    'name': typhoon_name,
                    'data': df
                })

        # 记录特征名
        if len(enhanced_typhoons) > 0:
            self.feature_names = list(enhanced_typhoons[0]['data'].columns)
            print(f"\n✓ 特征工程完成")
            print(f"  处理台风数: {len(enhanced_typhoons)}")
            print(f"  总特征数: {len(self.feature_names)}")

        return enhanced_typhoons

    def _add_lag_features(self, df):
        """滞后特征 - 捕获历史信息"""
        for feature in ['speed', 'pressure', 'move_speed']:
            if feature in df.columns:
                df[f'{feature}_lag1'] = df[feature].shift(1)
                df[f'{feature}_lag2'] = df[feature].shift(2)
        return df

    def _add_diff_features(self, df):
        """一阶差分 - 变化率"""
        for feature in ['speed', 'pressure', 'move_speed', 'move_dir']:
            if feature in df.columns:
                df[f'{feature}_change'] = df[feature].diff()
        return df

    def _add_second_diff(self, df):
        """二阶差分 - 加速度"""
        for feature in ['speed', 'pressure']:
            if feature in df.columns:
                df[f'{feature}_accel'] = df[feature].diff().diff()
        return df

    def _add_cyclic_encoding(self, df):
        """周期性编码 - 处理方向的周期性"""
        if 'move_dir' in df.columns:
            df['move_dir_sin'] = np.sin(np.radians(df['move_dir']))
            df['move_dir_cos'] = np.cos(np.radians(df['move_dir']))
        return df

    def _add_spatial_features(self, df):
        """空间特征 - 位置和轨迹"""
        if 'lng' in df.columns and 'lat' in df.columns:
            # 位置变化
            df['position_change'] = np.sqrt(df['lng'].diff()**2 + df['lat'].diff()**2)

            # 轨迹曲率
            lng_diff2 = df['lng'].diff().diff()
            lat_diff2 = df['lat'].diff().diff()
            df['trajectory_curvature'] = np.sqrt(lng_diff2**2 + lat_diff2**2)

            # 距离参考点（海口: 110.3°E, 20.0°N）
            df['dist_to_coast'] = np.sqrt((df['lng'] - 110.3)**2 + (df['lat'] - 20.0)**2)

            # 移动方向（从东的角度）
            df['dir_from_east'] = np.arctan2(df['lat'].diff(), df['lng'].diff()) * 180 / np.pi
        return df

    def _add_intensity_features(self, df):
        """强度特征"""
        if 'speed' in df.columns and 'pressure' in df.columns:
            # 强度比率
            df['intensity_ratio'] = df['speed'] / (1010 - df['pressure'] + 1)

            # 强度等级 (根据中国台风分级)
            df['intensity_category'] = pd.cut(df['speed'],
                                              bins=[0, 10.8, 17.2, 24.5, 32.7, 41.5, 51, 100],
                                              labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)

            # 是否增强中
            df['is_intensifying'] = (df['speed'].diff() > 0).astype(int)
        return df

    def _add_rolling_stats(self, df):
        """滚动窗口统计特征"""
        window = ROLLING_WINDOW

        if 'speed' in df.columns:
            df['speed_roll_mean'] = df['speed'].rolling(window).mean()
            df['speed_roll_std'] = df['speed'].rolling(window).std()
            df['speed_roll_max'] = df['speed'].rolling(window).max()

        if 'pressure' in df.columns:
            df['pressure_roll_mean'] = df['pressure'].rolling(window).mean()
            df['pressure_roll_min'] = df['pressure'].rolling(window).min()

        if 'move_speed' in df.columns:
            df['move_speed_roll_mean'] = df['move_speed'].rolling(window).mean()
        return df

    def _add_interaction_features(self, df):
        """交互特征"""
        if 'speed' in df.columns and 'move_speed' in df.columns:
            df['kinetic_indicator'] = df['speed'] * df['move_speed']

        if 'speed' in df.columns and 'pressure' in df.columns:
            df['strength_composite'] = df['speed'] / (df['pressure'] / 1000)

        if 'move_speed' in df.columns:
            df['movement_consistency'] = df['move_speed'].rolling(ROLLING_WINDOW).std()
        return df


    def select_top_features(self):
        """选择12个核心特征"""
        # 直接指定12个重要特征
        selected_features = [
            'speed',                    # 1. 风速
            'pressure',                 # 2. 气压
            'lng',                      # 3. 经度
            'lat',                      # 4. 纬度
            'intensity_ratio',          # 5. 强度比率
            'speed_change',             # 6. 风速变化率
            'move_speed',               # 7. 移动速度
            'move_dir',                 # 8. 移动方向
            'trajectory_curvature',     # 9. 轨迹曲率
            'speed_roll_mean',          # 10. 6小时滚动平均风速
            'pressure_change',          # 11. 气压变化率
            'lat',                      # 12. 纬度（用于计算变化率）
        ]

        # 验证特征是否存在
        available_features = [f for f in selected_features if f in self.feature_names]

        print(f"\n✓ 选择了 {len(available_features)} 个核心特征")
        print(f"  特征列表: {', '.join(available_features)}")

        return available_features

# ==================== 序列构建 ====================
def create_sequences(typhoons_data, input_features, target_features):
    """
    构建训练序列，同时记录每个样本的台风ID和时间索引

    Returns:
        X: (n_samples, lookback, n_input_features)
        y: (n_samples, forecast, n_target_features)
        sample_info: list of dict 包含 {'typhoon_id', 'start_index'}
    """

    X_list, y_list = [], []
    sample_info_list = []  # 新增：记录样本元信息

    for typhoon_idx, typhoon in enumerate(typhoons_data):
        typhoon_id = typhoon['name']  # 获取台风ID
        X_data = typhoon['data'][input_features].values
        y_data = typhoon['data'][target_features].values

        for i in range(len(X_data) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1):
            X = X_data[i:i + LOOKBACK_WINDOW]
            y = y_data[i + LOOKBACK_WINDOW:i + LOOKBACK_WINDOW + FORECAST_HORIZON]

            X_list.append(X)
            y_list.append(y)

            # 记录样本元信息
            sample_info_list.append({
                'typhoon_id': typhoon_id,
                'start_index': i  # 样本在台风序列中的起始索引
            })


            print(f"\n【第1个台风示例】: {typhoon['name']}")
            print(f"  原始数据长度: {len(X_data)}")
            if(len(X_data)<18):
                print("数据错误")
            # print(f"  生成序列数: {len(X_data) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1}")
            #
            # if len(X_list) > 0:
            #     print(f"\n  第1个样本的X (输入序列):")
            #     print(f"    形状: {X_list[0].shape} (lookback={LOOKBACK_WINDOW}, features={len(input_features)})")
            #     print(f"    前3个时间步数据:")
            #     for t in range(min(3, LOOKBACK_WINDOW)):
            #         print(f"      时间步{t}: {X_list[0][t]}")
            #
            #     print(f"\n  第1个样本的y (目标序列):")
            #     print(f"    形状: {y_list[0].shape} (horizon={FORECAST_HORIZON}, targets={len(target_features)})")
            #     print(f"    未来6小时预测目标:")
            #     for t in range(FORECAST_HORIZON):
            #         print(f"      +{t+1}小时: {y_list[0][t]} (lng, lat, speed, pressure)")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n✓ 序列构建完成")
    print(f"  总样本数: {len(X)}")
    print(f"  输入形状: {X.shape} (samples={len(X)}, lookback={LOOKBACK_WINDOW}, input_features={len(input_features)})")
    print(f"  输出形状: {y.shape} (samples={len(y)}, horizon={FORECAST_HORIZON}, target_features={len(target_features)})")
    print(f"  样本元信息: {len(sample_info_list)} 条记录")

    print(f"\n数据验证:")
    print(f"  X的数据范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y的数据范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  X包含NaN: {np.isnan(X).any()}")
    print(f"  y包含NaN: {np.isnan(y).any()}")

    # 显示样本元信息示例
    if len(sample_info_list) > 0:
        print(f"\n样本元信息示例 (前3个):")
        for i, info in enumerate(sample_info_list[:3]):
            print(f"  样本{i}: 台风={info['typhoon_id']}, 起始索引={info['start_index']}")

    return X, y, sample_info_list

# ==================== 数据集划分与归一化 ====================
def split_and_normalize(X, y, sample_info):
    """划分数据集并归一化，同时划分样本元信息"""
    print("\n" + "=" * 80)
    print("步骤5: 数据集划分与归一化")
    print("=" * 80)

    # 划分（使用索引保持一致性）
    indices = np.arange(len(X))
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, indices, test_size=TEST_RATIO, random_state=42
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=42
    )

    # 划分样本元信息
    sample_info_train = [sample_info[i] for i in idx_train]
    sample_info_val = [sample_info[i] for i in idx_val]
    sample_info_test = [sample_info[i] for i in idx_test]

    print(f"✓ 数据集划分:")
    print(f"  训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  验证集: {len(X_val)} 样本 ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.1f}%)")

    # 归一化 - 分别为 X 和 y 创建 scaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    n_samples, n_timesteps, n_input_features = X_train.shape
    _, _, n_target_features = y_train.shape

    # 归一化 X
    X_train_2d = X_train.reshape(-1, n_input_features)
    scaler_X.fit(X_train_2d)

    X_train = scaler_X.transform(X_train_2d).reshape(n_samples, n_timesteps, n_input_features)
    X_val = scaler_X.transform(X_val.reshape(-1, n_input_features)).reshape(len(X_val), n_timesteps, n_input_features)
    X_test = scaler_X.transform(X_test.reshape(-1, n_input_features)).reshape(len(X_test), n_timesteps, n_input_features)

    # 归一化 y
    y_train_2d = y_train.reshape(-1, n_target_features)
    scaler_y.fit(y_train_2d)

    y_train = scaler_y.transform(y_train_2d).reshape(len(y_train), FORECAST_HORIZON, n_target_features)
    y_val = scaler_y.transform(y_val.reshape(-1, n_target_features)).reshape(len(y_val), FORECAST_HORIZON, n_target_features)
    y_test = scaler_y.transform(y_test.reshape(-1, n_target_features)).reshape(len(y_test), FORECAST_HORIZON, n_target_features)

    print(f"✓ 归一化完成")
    print(f"  输入特征scaler: {n_input_features}个特征")
    print(f"  目标特征scaler: {n_target_features}个特征")

    return (X_train, y_train, sample_info_train), (X_val, y_val, sample_info_val), (X_test, y_test, sample_info_test), (scaler_X, scaler_y)

# ==================== 保存数据 ====================
def save_processed_data(train_data, val_data, test_data, scalers, input_features, target_features):
    """保存处理后的数据（包含样本元信息）"""
    print("\n" + "=" * 80)
    print("步骤6: 保存处理后的数据")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scaler_X, scaler_y = scalers

    # 解包数据（包含sample_info）
    X_train, y_train, sample_info_train = train_data
    X_val, y_val, sample_info_val = val_data
    X_test, y_test, sample_info_test = test_data

    # 将sample_info转换为结构化数组以便保存到npz
    def convert_sample_info_to_arrays(sample_info_list):
        typhoon_ids = np.array([s['typhoon_id'] for s in sample_info_list], dtype='U50')
        start_indices = np.array([s['start_index'] for s in sample_info_list], dtype=np.int32)
        return typhoon_ids, start_indices

    train_typhoon_ids, train_start_indices = convert_sample_info_to_arrays(sample_info_train)
    val_typhoon_ids, val_start_indices = convert_sample_info_to_arrays(sample_info_val)
    test_typhoon_ids, test_start_indices = convert_sample_info_to_arrays(sample_info_test)

    # 保存数据集（包含样本元信息）
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'train_data.npz'),
        X=X_train, y=y_train,
        typhoon_ids=train_typhoon_ids, start_indices=train_start_indices
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'val_data.npz'),
        X=X_val, y=y_val,
        typhoon_ids=val_typhoon_ids, start_indices=val_start_indices
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'test_data.npz'),
        X=X_test, y=y_test,
        typhoon_ids=test_typhoon_ids, start_indices=test_start_indices
    )

    # 保存元数据
    metadata = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'lookback_window': LOOKBACK_WINDOW,
        'forecast_horizon': FORECAST_HORIZON,
        'input_features': input_features,
        'target_features': target_features,
        'n_input_features': len(input_features),
        'n_target_features': len(target_features)
    }

    with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # 保存特征列表
    df_features = pd.DataFrame({
        'feature_type': ['input']*len(input_features) + ['target']*len(target_features),
        'feature_name': input_features + target_features
    })
    df_features.to_csv(
        os.path.join(OUTPUT_DIR, 'feature_list.csv'), index=False, encoding='utf-8-sig'
    )

    print(f"✓ 数据已保存到: {OUTPUT_DIR}")
    print(f"\n文件列表:")
    print(f"  - train_data.npz: 训练集")
    print(f"  - val_data.npz: 验证集")
    print(f"  - test_data.npz: 测试集")
    print(f"  - metadata.pkl: 元数据（scaler等）")
    print(f"  - feature_list.csv: 特征列表")

# ==================== 主函数 ====================
def main():

    print("=" * 80)
    print(f"\n数据来源: {DATA_DIR}")
    print(f"预测目标: {', '.join(TARGET_FEATURES)}")
    print(f"时序窗口: lookback={LOOKBACK_WINDOW}, forecast={FORECAST_HORIZON}")

    # 1. 加载数据
    all_typhoons = load_raw_data()

    # 2. 特征工程
    engineer = OptimizedFeatureEngineer()
    enhanced_typhoons = engineer.engineer_all_features(all_typhoons)

    # 3. 选择12个核心输入特征
    input_features = engineer.select_top_features()

    # 4. 构建序列（输入使用12个特征，输出使用4个目标特征，同时记录样本元信息）
    X, y, sample_info = create_sequences(enhanced_typhoons, input_features, TARGET_FEATURES)

    # 5. 划分与归一化
    train_data, val_data, test_data, scalers = split_and_normalize(X, y, sample_info)

    # 6. 保存
    save_processed_data(train_data, val_data, test_data, scalers,
                       input_features, TARGET_FEATURES)

    print("\n" + "=" * 80)
    print("特征工程完成！")
    print("=" * 80)
    print(f"\n输入特征数: {len(input_features)}")
    print(f"目标特征数: {len(TARGET_FEATURES)}")
    print(f"训练样本数: {len(train_data[0])}")
    print(f"\n输入特征: {', '.join(input_features)}")
    print(f"目标特征: {', '.join(TARGET_FEATURES)}")

if __name__ == '__main__':
    main()
