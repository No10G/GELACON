import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import json
import os
import time
from datetime import datetime, date

# --- 0. ファイルと定数の設定 ---
MODEL_FILE = 'gelecon_predictor_model.pkl'
CACHE_FILE = 'latest_weather_cache.json'
GRADIENT_RATE = 0.6

# ★ 補正値とコース定義（学習データと一致させる）★
# 湯沢(神立) - 340m基準 / 水上(丸沼) - 370m基準
COURSE_SETTINGS = {
    # 神立スノーリゾート (湯沢基準: 3.96℃補正)
    'Kandatsu_900m': {'resort': '神立', 'adj': 3.36, 'elev': 900},
    'Kandatsu_700m': {'resort': '神立', 'adj': 2.16, 'elev': 700},
    'Kandatsu_500m': {'resort': '神立', 'adj': 0.96, 'elev': 500},
    # 丸沼高原 (水上基準: 9.78℃補正)
    'Marunuma_1950m': {'resort': '丸沼', 'adj': 9.48, 'elev': 1950},
    'Marunuma_1700m': {'resort': '丸沼', 'adj': 7.98, 'elev': 1700},
    'Marunuma_1500m': {'resort': '丸沼', 'adj': 6.78, 'elev': 1500},
    'Marunuma_1300m': {'resort': '丸沼', 'adj': 5.58, 'elev': 1300},
}
CONDITIONS = {
    0: 'パウダー ✨', 1: '神バーン 💎', 2: 'アイスバーン ⚠️', 3: 'ゴロゴロ/シャバ雪 ☀️'
}
MODEL_FEATURE_ORDER = [
    'MaxSnowDepth', 'Snowfall', 'AvgWindSpeed', 'Adj_Temp_Min', 
    'Night_Chill_Factor', 'Cumulative_Heat_History', 'Surface_Hardening_Risk', 'Course_Elev'
]

# --- 1. モデルとキャッシュのロード ---
try:
    model = joblib.load(MODEL_FILE)
    # キャッシュファイルの読み込み
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    model_loaded = True
except Exception as e:
    st.error(f"エラー: モデルまたはキャッシュファイルが見つかりません。学習とデータ収集が完了しているか確認してください。({e})")
    model_loaded = False


# --- 2. 特徴量計算と予測実行関数 ---
def calculate_and_predict(course_data, course_key, past_history):
    """日別のデータフレームを受け取り、特徴量計算、モデル予測、結果を返す"""
    
    settings = COURSE_SETTINGS[course_key]
    adj_val = settings['adj']
    elev_val = settings['elev']
    
    # 履歴データの抽出
    prev_day_max_temp = past_history['PrevDayMaxTemp']
    cumulative_heat_history = past_history['CumulativeHeatHistoryBase']
    max_snow_depth = past_history['MaxSnowDepth']

    predictions_list = []

    for index, day_data in course_data.iterrows():
        
        # A. 標高補正
        adj_min = day_data['MinTemp'] - adj_val
        adj_max = day_data['MaxTemp'] - adj_val
        
        # B. Night Chill Factor: (急冷度 = 前日Max - 当日補正Min)
        night_chill = prev_day_max_temp - adj_min
        
        # C. 累積熱履歴の更新と計算
        heat_daily = np.maximum(0, adj_max - 0)
        cumulative_heat_history += heat_daily 
        
        # D. 雪面硬化リスク
        hardening_risk = day_data['AvgWindSpeed']**2 * (1.5 if adj_min < 0 else 1.0)
        
        # E. モデル入力DataFrameの作成 (★順序厳守★)
        X_predict = pd.DataFrame({
            'MaxSnowDepth': [max_snow_depth], 'Snowfall': [day_data['Snowfall']], 
            'AvgWindSpeed': [day_data['AvgWindSpeed']], 'Adj_Temp_Min': [adj_min], 
            'Night_Chill_Factor': [night_chill], 'Cumulative_Heat_History': [cumulative_heat_history],
            'Surface_Hardening_Risk': [hardening_risk], 'Course_Elev': [elev_val] 
        }, columns=MODEL_FEATURE_ORDER)
        
        # F. XGBoost予測実行
        probabilities = model.predict_proba(X_predict)[0]
        prediction_code = np.argmax(probabilities)
        
        # G. 翌日のために状態を更新
        prev_day_max_temp = day_data['MaxTemp'] # 当日のMaxTempを翌日のPrevDayMaxTempとして使用

        predictions_list.append({
            'Date': day_data['Date'],
            'Condition': CONDITIONS.get(prediction_code),
            'Probabilities': probabilities,
            'Adj_Min_Temp': adj_min
        })

    return predictions_list


# --- 3. Streamlit UI (メインルーチン) ---

st.title("❄️ GELECON AIバーン予測システム")
st.markdown("##### 複数リゾート・コース対応 (標高補正済み)")

if model_loaded:
    
    # リゾートの選択 (サイドバー)
    resort_options = ['神立スノーリゾート', '丸沼高原スキー場']
    selected_resort = st.sidebar.selectbox("🏔️ リゾートを選択", resort_options)
    st.sidebar.markdown("---")

    # A. 選択リゾートに属するコースをフィルタリング
    if selected_resort == '神立スノーリゾート':
        target_keys = [k for k in COURSE_SETTINGS.keys() if 'Kandatsu' in k]
        api_data = cache_data['resorts']['Kandatsu']
    else:
        target_keys = [k for k in COURSE_SETTINGS.keys() if 'Marunuma' in k]
        api_data = cache_data['resorts']['Marunuma']
        
    st.header(f"予測対象: {selected_resort}")
    st.markdown(f"###### データ取得時刻: {cache_data['timestamp']}")
    
    # B. コースごとの予測結果表示ループ (メイン画面)
    
    # APIから取得した未来予報データをDataFrameに変換
    forecast_df = pd.DataFrame(api_data['forecast_data'])
    past_history = api_data['history']

    for course_key in target_keys:
        course_elev = COURSE_SETTINGS[course_key]['elev']
        
        # 3. 予測の実行 (日別データを計算)
        predictions = calculate_and_predict(forecast_df.copy(), course_key, past_history)
        
        st.markdown(f"### 🏂 {course_key} ({course_elev}m)")
        st.markdown("---")

        # 結果を日別で表示
        for result in predictions:
            
            prob_df = pd.DataFrame({'Condition': list(CONDITIONS.values()), 'Probability': result['Probabilities']})
            prob_df['Probability'] = (prob_df['Probability'] * 100).round(1)

            with st.expander(f"🗓️ **{result['Date']}** - 予測: **{result['Condition']}** ({result['Adj_Min_Temp']:.1f}℃)"):
                
                # 予測の根拠 (確率)
                st.subheader("予測の確信度と内訳")
                
                col_chart, col_data = st.columns([2, 1])
                
                # グラフ表示 (例: Plotly/Altairを使用するとStreamlitで綺麗に表示されるが、ここではPandasで代用)
                # グラフの代わりに、確率の高い順に表示
                top_prob = prob_df.sort_values('Probability', ascending=False).iloc[0]
                col_chart.metric(f"最も確信度が高い予測", f"{top_prob['Condition']}", f"{top_prob['Probability']}%")

                # パーセンテージ内訳
                col_data.markdown("###### 確率の内訳")
                for _, row in prob_df.head(4).iterrows():
                    col_data.write(f"- {row['Condition']}: **{row['Probability']:.1f}%**")