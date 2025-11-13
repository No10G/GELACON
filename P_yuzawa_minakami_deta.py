import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse
import sys
import json
from collections import defaultdict

# --- 1. 共通設定 ---
# 取得したいデータの基準日
TODAY = datetime.date(2025, 11, 11) # 動作確認のため固定。実際は datetime.date.today() を使用してください。
TARGET_DAYS = 5
OUTPUT_FILENAME = "GELACON/past_data.json" 

# 観測所のパラメータ（JSON出力のために地点名も追加）
OBSERVATORIES = {
    "yuzawa": { # 湯沢
        "name": "湯沢",
        "prec_no": 54,
        "block_no": '0544'
    },
    "minakami": { # 水上（みなかみ）
        "name": "水上",
        "prec_no": 42,
        "block_no": '1019'
    }
}

# ヘッダー項目 (JSONのキーとして使用)
DATA_KEYS = [
    'date', 'precipitation_total_mm', 'temp_avg_c', 'temp_max_c', 'temp_min_c', 
    'wind_avg_ms', 'wind_max_ms', 'sunshine_h', 'snowfall_cm', 'snow_depth_max_cm'
]
# ---------------------


# --- 2. 過去の実績データ取得 (Webスクレイピング) ---
def get_past_weather_data(today, target_days, obs_code):
    """指定された観測所の過去N日間の気象庁実績データを取得する"""
    
    print(f"\n### 過去データ取得: {OBSERVATORIES[obs_code]['name']} ({target_days}日間)")
    
    # 取得期間の定義
    END_DATE = today
    START_DATE = END_DATE - datetime.timedelta(days=target_days - 1)
    BASE_URL = "https://www.data.jma.go.jp/stats/etrn/view/daily_a1.php"
    weather_data_list = [] # データを辞書で格納するリスト
    
    # 処理する月を特定 (月またぎ対応)
    target_months = set([
        (START_DATE.year, START_DATE.month),
        (END_DATE.year, END_DATE.month)
    ])
    
    # 月ごとにループ処理を実行
    for year, month in sorted(list(target_months)):
        
        # URLパラメータの設定
        params = {
            'prec_no': OBSERVATORIES[obs_code]['prec_no'],
            'block_no': OBSERVATORIES[obs_code]['block_no'],
            'year': year,
            'month': month,
            'day': 1, 
            'view': 'p1'
        }
        
        full_url = BASE_URL + '?' + urllib.parse.urlencode(params)
        print(f"-> アクセス: {year}年{month}月")

        try:
            response = requests.get(full_url, timeout=10)
            response.encoding = 'EUC-JP'
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"エラー: {year}年{month}月のURLへのアクセス中にエラーが発生しました: {e}")
            continue

        data_table = soup.find('table', id='tablefix1')
        if not data_table:
            continue

        rows = data_table.find_all('tr')
        
        # データ行の処理 (3行目から)
        for i in range(2, len(rows)):
            cols = rows[i].find_all(['td', 'th'])
            cols = [ele.text.strip() for ele in cols]
            
            if cols and cols[0].isdigit():
                day = int(cols[0])
                current_date = datetime.date(year, month, day)

                # 範囲内のデータのみを抽出
                if START_DATE <= current_date <= END_DATE:
                    
                    # 必要な要素を抽出 (リスト形式)
                    # [0:日付, 1:降水計, 4:気温平均, 5:最高, 6:最低, 9:風速平均, 10:最大風速, 15:日照, 16:降雪計, 17:最深積雪]
                    raw_data = [f"{month}月{day}日"] + [cols[i] for i in [1, 4, 5, 6, 9, 10, 15, 16, 17]]
                    
                    # データを辞書形式に変換
                    day_dict = dict(zip(DATA_KEYS, raw_data))
                    weather_data_list.append(day_dict)

    # 日付順に並び替え
    weather_data_list.sort(key=lambda x: datetime.datetime.strptime(str(today.year) + x['date'], '%Y%m月%d日'))
    return weather_data_list


# --- 3. メイン処理とJSONファイル出力 ---

# 湯沢と水上のデータを取得
yuzawa_data = get_past_weather_data(TODAY, TARGET_DAYS, "yuzawa")
minakami_data = get_past_weather_data(TODAY, TARGET_DAYS, "minakami")

# 最終的なJSON構造にまとめる
final_json_output = {
    "metadata": {
        "date_run": datetime.datetime.now().isoformat(),
        "target_period_days": TARGET_DAYS,
        "data_source": "JMA Past Weather Data (Web Scraping)"
    },
    "yuzawa": yuzawa_data,
    "minakami": minakami_data
}

# JSON形式でファイルに保存 
try:
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        # indent=4 で整形し、ensure_ascii=False で日本語をそのまま保存
        json.dump(final_json_output, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*50)
    print(f"データ保存完了！")
    print(f"データはファイル '{OUTPUT_FILENAME}' にJSON形式で保存されました。")
    print("="*50)

except Exception as e:
    print(f"\n--- エラー: ファイル保存に失敗しました ---")
    print(f"詳細: {e}")