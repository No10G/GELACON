import pandas as pd
import numpy as np
import requests
import datetime
import json
from collections import defaultdict
import time
import os
from bs4 import BeautifulSoup # Webスクレイピングに必要
import urllib.parse # Webスクレイピングに必要

# --- 共通設定 (過去データ) ---
# 観測所のパラメータ
OBSERVATORIES = {
    "yuzawa": {'name': "湯沢", 'prec_no': 54, 'block_no': '0544'},
    "minakami": {'name': "水上", 'prec_no': 42, 'block_no': '1019'}
}
TARGET_DAYS = 5
PAST_OUTPUT_FILENAME = "past_data.json" 
DATA_KEYS = ['date', 'precipitation_total_mm', 'temp_avg_c', 'temp_max_c', 'temp_min_c', 
             'wind_avg_ms', 'wind_max_ms', 'sunshine_h', 'snowfall_cm', 'snow_depth_max_cm']

def run_past_data_acquisition(base_dir):
    """過去データ（past_data.json）のWebスクレイピングと保存を実行"""
    
    TODAY = datetime.date.today()
    all_resort_data = {}
    BASE_URL = "https://www.data.jma.go.jp/stats/etrn/view/daily_a1.php"

    try:
        for obs_code, obs_settings in OBSERVATORIES.items():
            
            END_DATE = TODAY
            START_DATE = END_DATE - datetime.timedelta(days=TARGET_DAYS - 1)
            weather_data_list = []

            # 月またぎ対応
            target_months = sorted(list(set([
                (START_DATE.year, START_DATE.month),
                (END_DATE.year, END_DATE.month)
            ])))
            
            for year, month in target_months:
                params = {
                    'prec_no': obs_settings['prec_no'],
                    'block_no': obs_settings['block_no'],
                    'year': year, 'month': month, 'day': 1, 'view': 'p1'
                }
                full_url = BASE_URL + '?' + urllib.parse.urlencode(params)

                response = requests.get(full_url, timeout=10)
                response.encoding = 'EUC-JP'
                soup = BeautifulSoup(response.text, 'html.parser')

                data_table = soup.find('table', id='tablefix1')
                if not data_table: continue

                rows = data_table.find_all('tr')
                
                for i in range(2, len(rows)):
                    cols = [ele.text.strip() for ele in rows[i].find_all(['td', 'th'])]
                    if cols and cols[0].isdigit():
                        day = int(cols[0])
                        current_date = datetime.date(year, month, day)

                        if START_DATE <= current_date <= END_DATE:
                            # [0:日付, 1:降水計, 4:気温平均, 5:最高, 6:最低, 9:風速平均, 10:最大風速, 15:日照, 16:降雪計, 17:最深積雪]
                            raw_data = [f"{month}月{day}日"] + [cols[i] for i in [1, 4, 5, 6, 9, 10, 15, 16, 17]]
                            day_dict = dict(zip(DATA_KEYS, raw_data))
                            weather_data_list.append(day_dict)
            
            # 日付順に並び替え (年を補完)
            weather_data_list.sort(key=lambda x: datetime.datetime.strptime(str(TODAY.year) + x['date'], '%Y%m月%d日'))
            all_resort_data[obs_code] = weather_data_list

        # 最終的なJSON構造にまとめる
        final_json_output = {
            "metadata": {"date_run": datetime.datetime.now().isoformat(), "target_period_days": TARGET_DAYS, "data_source": "JMA Past Weather Data (Web Scraping)"},
            **all_resort_data
        }

        # ファイルに保存
        output_path = os.path.join(base_dir, PAST_OUTPUT_FILENAME)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 過去データ ({PAST_OUTPUT_FILENAME}) の更新完了。")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"過去データ取得中のリクエストエラー: {e}")
        return False
    except Exception as e:
        print(f"過去データ取得中の予期せぬエラー: {e}")
        return False

# --- (以下に run_future_data_acquisition と run_feature_calculation が続くと仮定) ---