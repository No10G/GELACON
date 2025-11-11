import pandas as pd
import numpy as np
import requests
import datetime
import json
from collections import defaultdict
import time
import os

# --- 1. 定数とリゾート設定 (学習データと一致させる) ---
# ★APIキーをここに設定してください (セキュリティのため、本番環境では環境変数を使用)
API_KEY = "712944967f82ebaa54544d29577bd6c6" 
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

# --- 3. 未来の予報データ取得 (OpenWeatherMap API) ---
def get_future_weather_forecast_owm(api_key, lat, lon):
    """今日から未来5日間のOpenWeatherMap予報データを取得する"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast" 
    params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': api_key}
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status() # 応答コードが200以外ならエラーを発生させる
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"APIアクセスエラーが発生しました: {e}")
        return None

# --- 4. メイン処理: データの取得、計算、JSON保存 ---

def generate_full_cache_file():
    master_cache = {}
    master_cache['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    master_cache['resorts'] = {}

    for resort_key, settings in RESORT_SETTINGS.items():
        
        
        # 4.2 未来の予報データ (API) を取得
        api_json = get_future_weather_forecast_owm(API_KEY, settings['lat'], settings['lon'])
        
        if api_json:
            # 4.3 3時間ごとのデータを日別に集計し、JSONフレンドリーな形式で格納
            daily_data = defaultdict(lambda: {'temp_max': -float('inf'), 'temp_min': float('inf'), 'winds': [], 'snowfall_cm': 0.0, 'date_str': ''})
            
            for item in api_json.get('list', []):
                dt_object = datetime.datetime.fromtimestamp(item['dt'])
                date_key = dt_object.date()
                
                # 今日から5日間のデータに限定
                if date_key < TODAY or (date_key - TODAY).days >= TARGET_FORECAST_DAYS:
                    continue

                daily_data[date_key]['temp_max'] = max(daily_data[date_key]['temp_max'], item['main']['temp_max'])
                daily_data[date_key]['temp_min'] = min(daily_data[date_key]['temp_min'], item['main']['temp_min'])
                daily_data[date_key]['winds'].append(item['wind']['speed'])
                daily_data[date_key]['snowfall_cm'] += item.get('snow', {}).get('3h', 0) / 10 # mmをcmに変換
                daily_data[date_key]['date_str'] = date_key.strftime('%m月%d日')

            # 4.4 最終的なリストを作成 (DataFrame変換前にリスト形式で準備)
            forecast_list = []
            for date_key, values in sorted(daily_data.items()):
                d = values
                forecast_list.append({
                    'Date': d['date_str'],
                    'MaxTemp': round(d['temp_max'], 2),
                    'MinTemp': round(d['temp_min'], 2),
                    'AvgWindSpeed': round(np.mean(d['winds']), 2),
                    'Snowfall': round(d['snowfall_cm'], 2)
                })

            # 4.5 キャッシュに格納
            master_cache['resorts'][resort_key] = {
                'forecast_data': forecast_list
            }
        else:
            print(f"Skipping {resort_key} due to API error.")

    # --- 5. JSONファイルへの出力 ---
    output_json_filename = 'latest_weather_cache.json'
    
    with open(output_json_filename, 'w', encoding='utf-8') as f:
        json.dump(master_cache, f, ensure_ascii=False, indent=4)

    print("\n" + "="*60)
    print(f"✅ 全てのデータ取得とキャッシュファイル '{output_json_filename}' の生成が完了しました。")
    print("このファイルを Webアプリが読み込みます。")
    print("="*60)

# --- 実行 ---
if __name__ == '__main__':
    generate_full_cache_file()