import pandas as pd
import numpy as np
import requests
import datetime
import json
from collections import defaultdict
import time
import os

# ---  定数とリゾート設定 ---
API_KEY = "APIkey" 
TODAY = datetime.date.today()
TARGET_FORECAST_DAYS = 5 

RESORT_SETTINGS = {
    'Kandatsu': {
        'name': '神立スノーリゾート', 'elev': 1000, 'adj_val': 3.96,
        'lat': 36.565, 'lon': 138.486
    },
    'Marunuma': {
        'name': '丸沼高原スキー場', 'elev': 2000, 'adj_val': 9.78,
        'lat': 36.464, 'lon': 138.579 
    }
}

# --- 未来の予報データ取得 (OpenWeatherMap API) ---
def get_future_weather_forecast_owm(api_key, lat, lon):
    """今日から未来5日間のOpenWeatherMap予報データを取得する"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast" 
    params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': api_key}
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status() 
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"APIアクセスエラーが発生しました: {e}")
        return None

# --- メイン処理 ---

def generate_full_cache_file():
    master_cache = {}
    master_cache['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    for resort_key, settings in RESORT_SETTINGS.items():
        
        # 4.2 未来の予報データ (API) を取得
        api_json = get_future_weather_forecast_owm(API_KEY, settings['lat'], settings['lon'])
        
        if api_json:
            # 標高補正値 (adj_val) を取得
            correction_value = settings['adj_val'] 
            
            #JSON形式で格納
            daily_data = defaultdict(lambda: {
                'temp_max': -float('inf'), 
                'temp_min': float('inf'), 
                'temp_sum': 0.0, # 平均気温算出用
                'temp_count': 0, # 平均気温算出用
                'winds': [], 
                'snowfall_cm': 0.0, 
                'precipitation_total_mm': 0.0, # 降水量
                'wind_max_ms': -float('inf'), # 最大風速
                'date_str': ''
            })
            
            for item in api_json.get('list', []):
                dt_object = datetime.datetime.fromtimestamp(item['dt'])
                date_key = dt_object.date()
                
                # 今日から5日間のデータに限定
                if date_key < TODAY or (date_key - TODAY).days >= TARGET_FORECAST_DAYS:
                    continue

                # === 1. 標高補正の適用 (気温) ===
                temp_corrected = item['main']['temp'] - correction_value
                temp_max_corrected = item['main']['temp_max'] - correction_value
                temp_min_corrected = item['main']['temp_min'] - correction_value
                
                # === 2. 日別集計 ===
                daily_data[date_key]['temp_max'] = max(daily_data[date_key]['temp_max'], temp_max_corrected)
                daily_data[date_key]['temp_min'] = min(daily_data[date_key]['temp_min'], temp_min_corrected)
                
                # 平均気温算出のため、全要素を積算
                daily_data[date_key]['temp_sum'] += temp_corrected
                daily_data[date_key]['temp_count'] += 1
                
                daily_data[date_key]['winds'].append(item['wind']['speed'])
                
                # 最大風速を更新
                daily_data[date_key]['wind_max_ms'] = max(daily_data[date_key]['wind_max_ms'], item['wind']['speed'])

                # 降雪量 を積算し、cmに変換
                daily_data[date_key]['snowfall_cm'] += item.get('snow', {}).get('3h', 0) / 10 
                
                # 降水量 を積算し、mmで保持
                daily_data[date_key]['precipitation_total_mm'] += item.get('rain', {}).get('3h', 0)
                
                daily_data[date_key]['date_str'] = date_key.strftime('%m月%d日')

            # 4.4 最終的なリストを作成 (過去データキーに統一)
            forecast_list = []
            for date_key, values in sorted(daily_data.items()):
                d = values
                # データがない日 (スキップされた日) を除外
                if d['temp_count'] == 0:
                    continue
                    
                # 平均気温の計算
                temp_avg = d['temp_sum'] / d['temp_count']
                
                forecast_list.append({
                    "date": d['date_str'],
                    "precipitation_total_mm": str(round(d['precipitation_total_mm'], 1)), # 文字列, 小数点1桁
                    "temp_avg_c": str(round(temp_avg, 1)),                              # 文字列, 小数点1桁
                    "temp_max_c": str(round(d['temp_max'], 1)),                           # 文字列, 小数点1桁
                    "temp_min_c": str(round(d['temp_min'], 1)),                           # 文字列, 小数点1桁
                    "wind_avg_ms": str(round(np.mean(d['winds']), 1)),                 # 文字列, 小数点1桁
                    "wind_max_ms": str(round(d['wind_max_ms'], 1)),                     # 文字列, 小数点1桁
                    "sunshine_h": "NaN", # OpenWeatherMapには日照時間がないため、NaN
                    "snowfall_cm": str(round(d['snowfall_cm'], 1)),                     # 文字列, 小数点1桁
                    "snow_depth_max_cm": "NaN" # OpenWeatherMapには最深積雪がないため、NaN
                })

            # 同じ構造に格納
            master_cache[resort_key] = forecast_list 

        else:
            print(f"Skipping {resort_key} due to API error.")

    # --- JSONファイルへの出力 ---
    output_json_filename = 'GELACON/CF_data.json'
    
    with open(output_json_filename, 'w', encoding='utf-8') as f:
        master_cache_final = {
            "metadata": {
                "date_run": datetime.datetime.now().isoformat(),
                "target_period_days": TARGET_FORECAST_DAYS,
                "data_source": "OpenWeatherMap API Forecast"
            },
        }
        master_cache_final.update({k: v for k, v in master_cache.items() if k != 'timestamp'})
        
        json.dump(master_cache_final, f, ensure_ascii=False, indent=4)

    print("\n" + "="*60)
    print(f"気象のデータ取得とキャッシュファイル '{output_json_filename}' の生成が完了しました。")
    print("="*60)
# --- 実行 ---
if __name__ == '__main__':
    generate_full_cache_file()