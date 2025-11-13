import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import json
import os
from collections import defaultdict
from datetime import datetime, date
import plotly.express as px 
import sys # エラー処理に利用


# --- 0. ファイルと定数の設定 ---
MODEL_FILE = 'gelecon_predictor_model.pkl'
# 過去データファイル
PAST_CACHE_FILE = 'past_data.json'
FUTURE_CACHE_FILE = 'CF_data.json' 
GRADIENT_RATE = 0.6

# 補正値とコース定義
COURSE_TARGETS = {
    'Kandatsu': [900, 700, 500],
    'Marunuma': [1950, 1700, 1500, 1300]
}
AMEDAS_ELEVATIONS = {'Kandatsu': 340, 'Marunuma': 370} 
CONDITIONS = {0: 'パウダー', 1: '神バーン', 2: 'アイスバーン', 3: 'シャバ雪'}
CONDITION_EMOJIS = {'パウダー': '✨', '神バーン': '💎', 'アイスバーン': '⚠️', 'シャバ雪': '🧊'}
MODEL_FEATURE_ORDER = [
    'MaxSnowDepth', 'Snowfall', 'AvgWindSpeed', 'Adj_Temp_Min', 
    'Night_Chill_Factor', 'Cumulative_Heat_History', 'Surface_Hardening_Risk', 'Course_Elev'
]

# 🚨 修正: 変数を必ず最初に初期化する 🚨
model_loaded = False 

# --- 1. モデルとキャッシュのロード ---
# ファイルが実行ファイルと同じディレクトリにあることを前提とします
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd() 

try:
    # 予測モデルをロード
    model = joblib.load(os.path.join(base_dir, MODEL_FILE))
    
    # 過去データと未来データをJSONからロード
    with open(os.path.join(base_dir, PAST_CACHE_FILE), 'r', encoding='utf-8') as f:
        past_cache_data = json.load(f)
    
    with open(os.path.join(base_dir, FUTURE_CACHE_FILE), 'r', encoding='utf-8') as f:
        future_cache_data = json.load(f)
        
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"エラー: 必要なファイルが見つかりません。パスを確認してください: {e.filename}")
except Exception as e:
    st.error(f"エラー: モデルまたはキャッシュファイル ({e.__class__.__name__}) の読み込みに失敗しました。詳細: {e}")

# --- 2. 特徴量計算と予測実行関数 (ダミー/暫定) ---
# この関数は、本番では 'XGBoost_Features_Cache.json' の読み込みに置き換えることを推奨します。
def calculate_and_predict_dummy(forecast_df, course_elev):
    """グラフ表示のため、ダミーの予測結果を生成する (本番ではモデル予測に置き換え)"""
    
    predictions = []
    
    for i in range(len(forecast_df)):
        # ダミー確率を生成 (4つの条件の合計が1になるように正規化)
        probs = np.random.rand(len(CONDITIONS))
        probs /= probs.sum()
        
        predictions.append({
            'Date': forecast_df['date'].iloc[i],
            'Condition': CONDITIONS[np.argmax(probs)],
            'Probabilities': probs.tolist(), 
            'Course_Elev': course_elev
        })
        
    return predictions


# --- 3. Streamlit UI (メインルーチン) ---

st.set_page_config(layout="wide")
st.title("❄️ GELECON ゲレンデコンディション予測システム")
st.markdown("##### AIによる未来5日間のバーン予測")

if model_loaded:
    
    # リゾートの選択 (サイドバー)
    st.sidebar.header("🏔️ リゾート選択")
    resort_options = ['神立スノーリゾート', '丸沼高原スキー場']
    selected_resort = st.sidebar.selectbox("予測リゾートを選択", resort_options)
    st.sidebar.markdown("---")

    # A. 選択リゾートの設定をフィルタリング
    base_key = 'Kandatsu' if selected_resort == '神立スノーリゾート' else 'Marunuma'
    past_key_map = 'yuzawa' if base_key == 'Kandatsu' else 'minakami'
    
    try:
        # past_cache_dataから初期値を取得 (今回は使わないが構造チェック)
        past_history_check = past_cache_data[past_key_map] 
    except KeyError:
        st.error(f"エラー: 過去データファイル内にリゾートキー '{past_key_map}' が見つかりません。")
        st.stop()
        
    # 未来予報データ (JSONからDataFrameへ変換)
    forecast_data = future_cache_data.get(base_key, [])
    if not forecast_data:
        st.error(f"エラー: 未来予報ファイル '{FUTURE_CACHE_FILE}' 内に {base_key} のデータがありません。")
        st.stop()
        
    forecast_df = pd.DataFrame(forecast_data)
    
    # 予測結果を格納するリストとDataFrame
    all_predictions_df = []
    
    st.header(f"予測対象: {selected_resort}")
    st.markdown("---")
    
    # ターゲット標高リストを取得
    target_elevations = COURSE_TARGETS[base_key]
    
    # B. コースごとの予測実行ループ
    for course_elev in target_elevations:
        
        # 1. 予測の実行 (日別データを計算)
        # ⚠️ 暫定: ダミー予測を実行 ⚠️
        predictions = calculate_and_predict_dummy(forecast_df.copy(), course_elev)
        
        # 予測結果をDataFrameに変換して統合
        df_course = pd.DataFrame(predictions)
        df_course['Course_Elev'] = df_course['Course_Elev'].astype(str) + 'm'
        all_predictions_df.append(df_course)

    df_combined = pd.concat(all_predictions_df)
    
    # --- UI表示のメイン部分 ---
    
    # 1. 標高ごとのコンディションサマリ（左上）
    st.subheader("1. 📉 標高ごとの予測コンディションマップ")
    
    # 各日付で最も確率の高いコンディションを取得
    df_combined['Top_Condition'] = df_combined.apply(lambda row: CONDITIONS[np.argmax(row['Probabilities'])], axis=1)
    df_combined['Top_Condition_Emoji'] = df_combined['Top_Condition'].map(CONDITION_EMOJIS)
    
    # Plotly Heatmap (imshow) の作成
    # Course_Elevを逆順にして、高い標高が上に来るようにする
    pivot_table = df_combined.pivot_table(
        index='Course_Elev', 
        columns='Date', 
        values='Top_Condition_Emoji', 
        aggfunc='first'
    ).reindex([str(e) + 'm' for e in target_elevations[::-1]]) # 標高を降順でreindex
    
    fig = px.imshow(
        pivot_table,
        text_auto=True,
        aspect="auto",
        labels=dict(x="日付", y="コース標高", color=""), # カラーバーを非表示にするためラベルを空に
        title=f"{selected_resort} - 5日間の予測コンディションマップ",
        color_continuous_scale=px.colors.qualitative.Plotly # 定性的な色スケールを使用
    )
    # 軸の調整
    fig.update_xaxes(side="top")
    fig.update_layout(height=450, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.sidebar.markdown("デバッグ情報:") # ← ここまで実行されたか確認
    st.sidebar.json(future_cache_data.get(base_key, [])) # 読み込んだ未来データを表示
    st.sidebar.dataframe(forecast_df.head()) # DataFrameの先頭を表示
    # 2. ドロップダウン選択による詳細確率グラフ
    st.subheader("2. 📊 詳細予測確率 (バーンコンディションの割合)")
    
    # 日付とコースの選択
    col1, col2 = st.columns(2)
    
    # 一意の選択肢を確保
    unique_dates = df_combined['Date'].unique()
    unique_elevs = df_combined['Course_Elev'].unique()

    with col1:
        selected_date = st.selectbox("予測日を選択", unique_dates)
        
    with col2:
        selected_elev = st.selectbox("コース標高を選択 (m)", unique_elevs)
        
    # フィルタリング
    df_filtered = df_combined[
        (df_combined['Date'] == selected_date) & 
        (df_combined['Course_Elev'] == selected_elev)
    ].iloc[0] # 該当する1行を取得
    
    
    # 円グラフ用データフレームの作成
    prob_data = pd.DataFrame({
        'Condition': list(CONDITIONS.values()),
        'Probability': df_filtered['Probabilities']
    })
    
    # 確率をパーセンテージに変換し、降順にソート
    prob_data['Probability'] = (prob_data['Probability'] * 100).round(1)
    prob_data = prob_data.sort_values(by='Probability', ascending=False)


    # 円グラフの描画
    prob_fig = px.pie(
        prob_data, 
        values='Probability', 
        names='Condition', 
        title=f"{selected_elev} / {selected_date} のバーン確率",
        color='Condition',
        color_discrete_map={
            'パウダー': 'lightblue', 
            '神バーン': 'green', 
            'アイスバーン': 'red', 
            'シャバ雪': 'orange'
        }
    )
    prob_fig.update_traces(textinfo='percent+label') # パーセンテージとラベルを表示
    st.plotly_chart(prob_fig, use_container_width=True)

# --- 実行 ---
if __name__ == '__main__':
    pass