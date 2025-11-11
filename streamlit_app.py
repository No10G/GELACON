import pandas as pd
import numpy as np
import requests
import datetime
from collections import defaultdict

# --- 予測に必要な定数 (学習データと一致) ---
GRADIENT_RATE = 0.6  
MODEL_FEATURE_ORDER = [
    'MaxSnowDepth', 'Snowfall', 'AvgWindSpeed', 'Adj_Temp_Min', 
    'Night_Chill_Factor', 'Cumulative_Heat_History', 'Surface_Hardening_Risk', 'Course_Elev'
]

# リゾートごとの設定（API座標、補正値、標高）
ADJUSTMENT_MAP = {
    '神立スノーリゾート (1000m)': {'adj': 3.96, 'elev': 1000, 'lat': 36.942, 'lon': 138.810},
    '丸沼高原スキー場 (2000m)': {'adj': 9.78, 'elev': 2000, 'lat': 36.815, 'lon': 139.331}
}
API_KEY = "712944967f82ebaa54544d29577bd6c6" # ★APIキーをここに設定★
def prepare_and_predict_forecast(course_name, api_key, cached_history):
    """
    指定されたコース名に基づき、未来5日間のAPIデータを取得・整形し、モデル入力用のDataFrameを生成する。
    
    cached_history: ファイルキャッシュから読み込んだ前日MaxTempなどの情報。
    """
    
    resort_info = ADJUSTMENT_MAP.get(course_name)
    if not resort_info:
        return pd.DataFrame(), "エラー: コース設定が見つかりません。"
    
    # 補正値と座標の設定
    adjustment_value = resort_info['adj']
    course_elev = resort_info['elev']
    lat, lon = resort_info['lat'], resort_info['lon']
    
    # --- 1. APIデータの取得 ---
    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast" 
    params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': api_key}
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        api_data = response.json()
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"APIエラー: {e}"

    # --- 2. 日別集計と特徴量計算の初期値設定 ---
    daily_data = defaultdict(lambda: {'temp_max': -float('inf'), 'temp_min': float('inf'), 'winds': [], 'snows': [], 'date_str': ''})
    today = datetime.date.today()
    
    # キャッシュから初期値を取得
    prev_day_max_temp = cached_history.get('PrevDayMaxTemp', 5.0)  # 前日MaxTempの初期値 (5.0℃と仮定)
    cumulative_heat_history = cached_history.get('CumulativeHeatHistory', 0.0) # 累積熱履歴の初期値
    max_snow_depth = cached_history.get('MaxSnowDepth', 150) # 積雪深 (変化しないと仮定)

    # --- 3. 3時間ごとのデータ処理と集計 ---
    for item in api_data.get('list', []):
        dt_object = datetime.datetime.fromtimestamp(item['dt'])
        date_key = dt_object.date()
        
        # 今日から5日間のデータに限定
        if date_key < today or (date_key - today).days >= 5:
            continue

        # 日別集計
        daily_data[date_key]['temp_max'] = max(daily_data[date_key]['temp_max'], item['main']['temp_max'])
        daily_data[date_key]['temp_min'] = min(daily_data[date_key]['temp_min'], item['main']['temp_min'])
        daily_data[date_key]['winds'].append(item['wind']['speed'])
        daily_data[date_key]['snows'].append(item.get('snow', {}).get('3h', 0))
        daily_data[date_key]['date_str'] = date_key.strftime('%Y-%m-%d')
    
    # --- 4. 最終DataFrameの構築と特徴量計算 ---
    final_records = []
    
    for date_key in sorted(daily_data.keys()):
        day_data = daily_data[date_key]
        
        # A. 標高補正
        adj_min = day_data['temp_min'] - adjustment_value
        adj_max = day_data['temp_max'] - adjustment_value
        
        # B. Night Chill Factor (急冷度)
        night_chill = prev_day_max_temp - adj_min
        
        # C. 累積熱履歴の更新と計算
        heat_daily = np.maximum(0, adj_max - 0)
        cumulative_heat_history += heat_daily # 前日の累積に当日の熱を足す
        
        # D. 雪面硬化リスク
        avg_wind = np.mean(day_data['winds'])
        hardening_risk = avg_wind**2 * (1.5 if adj_min < 0 else 1.0)
        
        # E. 降雪量合計 (mmをcmに変換)
        snowfall_cm = sum(day_data['snows']) / 10 
        
        # F. XGBoostモデルに渡すレコードを作成 (順序厳守)
        record = {
            'MaxSnowDepth': max_snow_depth,
            'Snowfall': snowfall_cm,
            'AvgWindSpeed': avg_wind,
            'Adj_Temp_Min': adj_min,
            'Night_Chill_Factor': night_chill,
            'Cumulative_Heat_History': cumulative_heat_history,
            'Surface_Hardening_Risk': hardening_risk,
            'Course_Elev': course_elev,
            # (表示用に日付を含める)
            'Date': day_data['date_str']
        }
        
        final_records.append(record)
        
        # G. 翌日の Night Chill Factor のために PrevDay_MaxTemp を更新
        prev_day_max_temp = day_data['temp_max'] 

    # 最終的な DataFrame を構築し、XGBoostの入力順に並べ替える
    df_predict = pd.DataFrame(final_records)
    
    # 最終的な出力カラムの順序を決定 (Dateは表示用なので除く)
    prediction_cols = [col for col in MODEL_FEATURE_ORDER]
    
    return df_predict, df_predict[prediction_cols]