import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse
import sys
from collections import defaultdict
import numpy as np

# --- 1. 共通設定 ---
# 取得したいデータの基準日（今日）
TODAY = datetime.date(2025, 11, 8) # 動作確認のため固定。実際は datetime.date.today() を使用してください。
TARGET_DAYS = 5
API_KEY = "1a56b1626e30118ca94615f08b7005c5" 
# みなかみ町の座標（気象庁のデータ地点に近い）
LATITUDE = 36.815 
LONGITUDE = 139.331
# ---------------------


# --- 2. 過去の実績データ取得 (気象庁Webスクレイピングのロジックを統合) ---
def get_past_weather_data(today, target_days):
    """今日から過去N日間の気象庁実績データを取得する"""
    
    print("\n### 過去の実績データ取得開始 (気象庁)...")
    
    # 取得期間の定義
    END_DATE = today
    START_DATE = END_DATE - datetime.timedelta(days=target_days - 1)
    BASE_URL = "https://www.data.jma.go.jp/stats/etrn/view/daily_a1.php"
    weather_data = []

    # 処理する月を特定 (開始月から終了月まで)
    target_months = set([
        (START_DATE.year, START_DATE.month),
        (END_DATE.year, END_DATE.month)
    ])
    
    print(f"取得期間: {START_DATE} から {END_DATE} まで")

    # 月ごとにループ処理を実行
    for year, month in sorted(list(target_months)):
        
        # URLパラメータの設定 (月の最初の日にアクセス) - みなかみ（群馬県）
        params = {
            'prec_no': 42,
            'block_no': 1019,
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
                    
                    # 必要な要素を抽出
                    # 抽出インデックス: 0(日付), 1(降水計), 4(平均気温), 5(最高), 6(最低), 9(平均風速), 10(最大風速), 15(日照), 16(降雪計), 17(最深積雪)
                    data_row_fixed = [f"{month}月{day}日"] + cols[1:18]
                    simple_row_indices_fixed = [0, 1, 4, 5, 6, 9, 10, 15, 16, 17]
                    
                    simple_row = [data_row_fixed[i] for i in simple_row_indices_fixed]
                    weather_data.append(simple_row)

    # 日付順に並び替え
    weather_data.sort(key=lambda x: datetime.datetime.strptime(str(today.year) + x[0], '%Y%m月%d日'))
    return weather_data


# --- 3. 未来の予報データ取得 (OpenWeatherMap APIのロジックを統合) ---
def get_future_weather_forecast(today, target_days, api_key, lat, lon):
    """今日から未来N日間のOpenWeatherMap予報データを取得する"""
    
    print("\n### 🤖 未来の予報データ取得開始 (OpenWeatherMap)...")
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast" 
    
    params = {
        'lat': lat,
        'lon': lon,
        'units': 'metric', # 単位をメートル法に設定
        'appid': api_key,
        'lang': 'ja'
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"\n--- 致命的なエラー ---")
        print(f"APIアクセスエラーが発生しました: {e}")
        return []

    # 3時間ごとのデータを日別に集計する
    daily_data = defaultdict(lambda: {
        'temp_max': -float('inf'), 'temp_min': float('inf'), 
        'winds': [], 'rains': [], 'snows': []
    })

    for item in data.get('list', []):
        
        dt_object = datetime.datetime.fromtimestamp(item['dt'])
        date_key = dt_object.strftime('%Y-%m-%d')
        date_str = dt_object.strftime('%m月%d日')
        
        # 予報が今日以降のデータであることを確認
        if dt_object.date() < today:
            continue
        # N日間の予報のみを処理
        if len(daily_data) >= target_days and date_key not in daily_data:
            continue

        # 日別の最高/最低気温を更新
        daily_data[date_key]['date_str'] = date_str
        daily_data[date_key]['temp_max'] = max(daily_data[date_key]['temp_max'], item['main']['temp_max'])
        daily_data[date_key]['temp_min'] = min(daily_data[date_key]['temp_min'], item['main']['temp_min'])
        
        # 風速と降水量・降雪量をリストに追加
        daily_data[date_key]['winds'].append(item['wind']['speed'])
        daily_data[date_key]['rains'].append(item.get('rain', {}).get('3h', 0))
        daily_data[date_key]['snows'].append(item.get('snow', {}).get('3h', 0))

    final_forecast = []
    for date_key, values in sorted(daily_data.items()):
        
        # 3時間ごとの降水量・降雪量を合計する
        total_rain_snow = sum(values['rains']) + sum(values['snows'])
        
        # 降雪量はmm
        final_forecast.append([
            values['date_str'],
            f"{total_rain_snow:.1f}", 
            f"{(values['temp_max'] + values['temp_min']) / 2:.1f}", 
            f"{values['temp_max']:.1f}",
            f"{values['temp_min']:.1f}",
            f"{np.mean(values['winds']):.1f}",
            f"{max(values['winds']):.1f}", 
            # 予報では欠損しているデータは「-」とする
            '-', # 日照時間
            f"{sum(values['snows']):.1f}", # 降雪量(mm)
            '-' # 最深積雪
        ])
    
    return final_forecast


# --- 4. メイン処理と結果表示 ---

# 過去データと未来データを取得
past_data = get_past_weather_data(TODAY, TARGET_DAYS)
future_data = get_future_weather_forecast(TODAY, TARGET_DAYS, API_KEY, LATITUDE, LONGITUDE)

# 最終的なヘッダーを定義
final_header = [
    '日付', '降水量 合計(mm)', 
    '気温 平均(℃)', '最高(℃)', '最低(℃)', 
    '平均風速(m/s)', '最大風速(m/s)', 
    '日照時間(h)', '降雪・降雪深さ/合計(cm/mm)', '最深積雪(cm)'
]

print("\n" + "="*50)
print(f"### 📊 気象データ統合結果 (みなかみ: {TARGET_DAYS}日間の実績と{TARGET_DAYS}日間の予報)")
print(f"**基準日:** {TODAY}")
print("="*50)

# 過去データ表示
print("\n#### 過去の実績データ (気象庁: Webスクレイピング)")
print("| " + " | ".join(final_header) + " |")
print("|" + " :--- |" * len(final_header))
for row in past_data:
    # 過去データでは「降雪の深さの合計(cm)」が9列目、「最深積雪(cm)」が10列目
    # ヘッダーに合わせるため、9列目と10列目をマージして表示し、ヘッダーとデータの列数を合わせる
    # 実際のデータは10要素
    display_row = row[0:8] + [f"{row[8]} / {row[9]}", row[9]] # 降雪/積雪をマージ
    print("| " + " | ".join(display_row) + " |")

# 未来データ表示
print("\n#### 未来の予報データ (OpenWeatherMap API)")
print("| " + " | ".join(final_header) + " |")
print("|" + " :--- |" * len(final_header))
for row in future_data:
    # 未来データでは「降雪量(mm)」が9列目、「最深積雪」は10列目（ハイフン）
    # ヘッダーに合わせるため、9列目と10列目をマージして表示し、ヘッダーとデータの列数を合わせる
    # 実際のデータは10要素
    display_row = row[0:8] + [row[8], row[9]] # 降雪量(mm)と最深積雪(-)
    print("| " + " | ".join(display_row) + " |")

# 注意事項
print("\n> **注記:** 過去データと未来データで利用しているデータソースが異なるため、特に降雪量や日照時間の単位・有無が異なります。")
print("> **過去データ:** 降雪・降雪深さ/合計は **降雪の深さの合計(cm)** です。最深積雪も(cm)です。")
print("> **未来データ:** 降雪・降雪深さ/合計は **降雪量(mm)** です。日照時間/最深積雪はAPIで提供されないため「-」です。")