import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse
import sys
# collectionsとnumpyは未来予報の集計でしか使わないため削除
# from collections import defaultdict
# import numpy as np 

# --- 1. 共通設定 ---
# 取得したいデータの基準日（今日）
TODAY = datetime.date(2025, 11, 11) # 動作確認のため固定。実際は datetime.date.today() を使用してください。
TARGET_DAYS = 8
# ---------------------


# --- 2. 過去の実績データ取得 (気象庁Webスクレイピングのロジック) ---
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
    
    print(f"取得期間: {START_DATE} から {END_DATE} まで (みなかみ)")

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
        print(f"-> アクセス: {year}年{month}月 ({full_url})")

        try:
            # タイムアウトを設定し、エラー処理を強化
            response = requests.get(full_url, timeout=10)
            response.encoding = 'EUC-JP'
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"エラー: {year}年{month}月のURLへのアクセス中にエラーが発生しました: {e}")
            continue

        # 安定したテーブル特定方法（ID検索）
        data_table = soup.find('table', id='tablefix1')

        if not data_table:
            print(f"警告: {year}年{month}月分のデータテーブルが見つかりませんでした。")
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


# --- 3. メイン処理と結果表示 ---

# 過去データを取得
past_data = get_past_weather_data(TODAY, TARGET_DAYS)

# 最終的なヘッダーを定義 (過去データに特化)
final_header = [
    '日付', '降水量 合計(mm)', 
    '気温 平均(℃)', '最高(℃)', '最低(℃)', 
    '平均風速(m/s)', '最大風速(m/s)', 
    '日照時間(h)', '降雪の深さの合計(cm)', '最深積雪(cm)'
]

print("\n" + "="*50)
print(f"### 📊 過去の実績データ取得結果 (みなかみ: {TARGET_DAYS}日間)")
print(f"**基準日:** {TODAY}")
print("="*50)

# 過去データ表示
print("| " + " | ".join(final_header) + " |")
print("|" + " :--- |" * len(final_header))
for row in past_data:
    print("| " + " | ".join(row) + " |")

print("\n--- 処理完了 ---")