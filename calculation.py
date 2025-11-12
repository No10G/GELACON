import pandas as pd
import numpy as np
import json
from collections import defaultdict
from datetime import datetime, date, timedelta
import os

# --- 1. 定数とファイル名 ---
# 傾斜率（気温補正に使用）
GRADIENT_RATE = 0.6  
# キャッシュファイル名
PAST_CACHE_FILE = 'past_data.json'   
FUTURE_CACHE_FILE = 'CF_data.json' 
OUTPUT_CACHE_FILE = 'XGBoost_Features_Cache.json'

# アメダス観測所の標高（概算）
AMEDAS_ELEVATIONS = {'Kandatsu': 340, 'Marunuma': 370}  
# コース標高（予測ターゲット）
COURSE_TARGETS = {
    'Kandatsu': [900, 700, 500],
    'Marunuma': [1950, 1700, 1500, 1300]
}

# 過去データ観測所と未来データリゾートの対応
PAST_FUTURE_MAPPING = {
    'yuzawa': 'Kandatsu',
    'minakami': 'Marunuma'
}

# XGBoostモデルが期待する特徴量の順序 (8列)
MODEL_FEATURE_ORDER = [
    'MaxSnowDepth', 'Snowfall', 'AvgWindSpeed', 'Adj_Temp_Min', 
    'Night_Chill_Factor', 'Cumulative_Heat_History', 'Surface_Hardening_Risk', 'Course_Elev'
]

# --- 2. メインの特徴量計算関数 ---
def generate_xgboost_features():
    
    print("定数とファイル名の設定が完了しました。")
    
    # 1. フルパスの計算とJSONファイルのロード
    
    # スクリプトの絶対パスを取得し、ベースディレクトリとする
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd() 

    past_full_path = os.path.join(base_dir, PAST_CACHE_FILE)
    future_full_path = os.path.join(base_dir, FUTURE_CACHE_FILE)

    try:
        # 過去データの読み込み
        print(f"過去データを探しています: {past_full_path}")
        with open(past_full_path, 'r', encoding='utf-8') as f:
            past_cache = json.load(f)
            
        # 未来データの読み込み
        print(f"未来データを探しています: {future_full_path}")
        with open(future_full_path, 'r', encoding='utf-8') as f:
            future_cache = json.load(f)
            
        print("✅ 過去データと未来データのキャッシュファイルを正常に読み込みました。")
    
    except FileNotFoundError as e:
        print("\n" + "="*50)
        print("🚨 致命的なファイル読み込みエラーが発生しました (FileNotFoundError) 🚨")
        print(f"アクセスを試みたファイル: {e.filename}")
        print("ファイル名またはパスを確認してください。")
        print("="*50)
        return
    except json.JSONDecodeError:
        print("\n" + "="*50)
        print("🚨 致命的なJSON解析エラーが発生しました (JSONDecodeError) 🚨")
        print(f"ファイル {PAST_CACHE_FILE} または {FUTURE_CACHE_FILE} の内容が不正です。")
        print("JSONファイルの構文を確認してください。")
        print("="*50)
        return

    # 過去データで使用されたリゾートキーを抽出
    past_resort_keys = [k for k in past_cache.keys() if k != 'metadata']
    
    # 2. 過去データから初期値（ベースライン）を計算 (Night Chill Factorと熱履歴の基点)
    initial_history = {}
    for past_key in past_resort_keys:
        # 過去データ (N日分) をPandasに変換
        df_past = pd.DataFrame(past_cache[past_key])
        
        # 数値変換 (エラーを考慮し、文字列データを数値に変換)
        for col in ['temp_max_c', 'temp_min_c', 'snow_depth_max_cm', 'wind_max_ms']:
            if col in df_past.columns:
                 df_past[col] = pd.to_numeric(df_past[col], errors='coerce')
        
        # 過去データの不足チェック
        if len(df_past) < 2:
            print(f"エラー: {past_key} の過去データが不足しています（2日未満）。スキップします。")
            continue
            
        # ⚠️ 動的な日付基点: 最終日とその前日を動的に特定 ⚠️
        prev_day_max_temp = df_past['temp_max_c'].iloc[-2] # 最新日の前日の最高気温
        df_latest_day = df_past.iloc[-1] # 最新日のデータ
        
        # 補正値の計算 (過去データ取得時の補正値を仮定)
        future_key = PAST_FUTURE_MAPPING[past_key]
        amedas_elev = AMEDAS_ELEVATIONS[future_key]
        course_elev_top = COURSE_TARGETS[future_key][0] # トップコースの標高を基準に
        elev_diff = course_elev_top - amedas_elev
        adj_val = (elev_diff / 100) * GRADIENT_RATE 
        
        # 累積熱履歴の再計算 (過去データ全体で実行)
        df_past['Adj_Max'] = df_past['temp_max_c'] - adj_val
        df_past['Heat_Penalty_Daily'] = np.maximum(0, df_past['Adj_Max'] - 0)
        df_past['CumHeatHistory'] = df_past['Heat_Penalty_Daily'].cumsum()
        
        # 最新日 (Futureの予測開始日の前日) の値を取得
        cum_heat_base = df_past['CumHeatHistory'].iloc[-1]
        max_snow_depth = df_latest_day['snow_depth_max_cm'] 

        initial_history[past_key] = {
            # 補正後の最高気温をベースとする
            'PrevDayMaxTemp': prev_day_max_temp - adj_val, 
            'CumulativeHeatHistoryBase': cum_heat_base,
            'MaxSnowDepth': max_snow_depth,
            'BaseAdjVal': adj_val
        }

    # 3. 未来予報データ (CF_data.json) を準備
    all_features_for_model = {}
    
    for base_resort in ['Kandatsu', 'Marunuma']:
        
        # 予報データのDataFrameを作成
        forecast_data = future_cache.get(base_resort) 
        if not forecast_data:
             print(f"注意: {base_resort} の予報データが見つかりません。スキップします。")
             continue
             
        forecast_df = pd.DataFrame(forecast_data)
        
        # 🚨 キー統一後の数値変換 🚨
        # CF_data.jsonのキーを使用
        numeric_cols = ['temp_max_c', 'temp_min_c', 'wind_avg_ms', 'snowfall_cm']
        for col in numeric_cols:
            forecast_df[col] = pd.to_numeric(forecast_df[col], errors='coerce') 

        # 4. コースごとの特徴量計算ループ
        for course_elev in COURSE_TARGETS[base_resort]:
            
            # 状態変数の初期化 (過去データから引き継ぐ)
            past_key = 'yuzawa' if base_resort == 'Kandatsu' else 'minakami'
            if past_key not in initial_history:
                continue 
                
            past_base = initial_history[past_key]
            
            prev_day_max_adj = past_base['PrevDayMaxTemp'] # 補正済みの最高気温
            cum_heat = past_base['CumulativeHeatHistoryBase']
            max_snow_depth = past_base['MaxSnowDepth']

            # 補正値の計算 (コースごとに異なる補正を適用)
            amedas_elev = AMEDAS_ELEVATIONS[base_resort]
            elev_diff = course_elev - amedas_elev
            adjustment_value = (elev_diff / 100) * GRADIENT_RATE
            
            course_features = []
            
            for index, day_data in forecast_df.iterrows():
                
                # A. 標高補正 (統一キーを使用)
                adj_min = day_data['temp_min_c'] - adjustment_value
                adj_max = day_data['temp_max_c'] - adjustment_value
                
                # B. Night Chill Factor (急冷度)
                night_chill = prev_day_max_adj - adj_min
                
                # C. 累積熱履歴の更新
                heat_daily = np.maximum(0, adj_max - 0)
                cum_heat = cum_heat + heat_daily 
                
                # D. 雪面硬化リスク (統一キーを使用)
                hardening_risk = day_data['wind_avg_ms']**2 * (1.5 if adj_min < 0 else 1.0)
                
                # E. XGBoostモデル用のレコード作成 (順序厳守)
                record = {
                    'Date': day_data['date'], 
                    'MaxSnowDepth': max_snow_depth, 
                    'Snowfall': day_data['snowfall_cm'], # 統一キーを使用
                    'AvgWindSpeed': day_data['wind_avg_ms'], # 統一キーを使用
                    'Adj_Temp_Min': adj_min, 
                    'Night_Chill_Factor': night_chill, 
                    'Cumulative_Heat_History': cum_heat,
                    'Surface_Hardening_Risk': hardening_risk, 
                    'Course_Elev': course_elev
}
                
                # モデルの期待する特徴量のみを、期待する順序で格納
                feature_values = [
                    float(record[feat]) if isinstance(record[feat], (int, float, np.number)) else record[feat] 
                    for feat in MODEL_FEATURE_ORDER
                ]
                # 日付情報と特徴量を格納
                final_record = {'Date': record['Date'], 'Course': course_elev, 'Features': feature_values}
                course_features.append(final_record)
                
                # F. 翌日のために状態を更新
                prev_day_max_adj = adj_max # 補正後の最高気温を翌日の基点にする

            all_features_for_model[f"{base_resort}_{course_elev}m"] = course_features
            
    # 5. 最終JSONファイルへの出力
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": all_features_for_model
    }
    output_filename_full_path = os.path.join(base_dir, OUTPUT_CACHE_FILE)
    
    with open(output_filename_full_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 特徴量計算が完了し、XGBoost予測用キャッシュ '{output_filename_full_path}' が生成されました。")


# --- 実行 ---
if __name__ == '__main__':
    generate_xgboost_features()