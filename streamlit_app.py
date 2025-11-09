import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import requests
import os

# --- A. 定数とモデルのロード ---

# 補正値とコース定義（学習データと一致）
ADJUSTMENT_MAP = {
    '神立スキー場 (1000m)': 3.96,
    '丸沼高原 (2000m)': 9.78
}
ELEVATION_MAP = {
    '神立スキー場 (1000m)': 1000,
    '丸沼高原 (2000m)': 2000
}
# TARGET_CODEを人間に分かりやすい名前に変換する辞書
CONDITIONS = {
    0: 'パウダー ✨', 1: '神バーン 💎', 2: 'アイスバーン ⚠️', 3: 'ゴロゴロ/シャバ雪 ☀️'
}

try:
    model = joblib.load('gelecon_predictor_model.pkl')
    model_loaded = True
except FileNotFoundError:
    st.error("エラー: 'gelecon_predictor_model.pkl'が見つかりません。XGBoostの学習が完了しているか確認してください。")
    model_loaded = False


# --- B. 予測に必要なカスタム特徴量の計算関数 ---
def calculate_features_for_prediction(user_data, adjustment_value, course_elev):
    """
    ユーザー入力をモデルが学習した8つのカスタム特徴量に変換する
    """
    
    # 1. 標高補正
    adj_temp_min = user_data['MinTemp'] - adjustment_value
    adj_temp_max = user_data['MaxTemp'] - adjustment_value
    
    # 2. Night Chill Factor: (ユーザー入力値で計算)
    # PrevDayMaxTemp (前日の熱) - Adj_Temp_Min (当日の真の冷え込み)
    night_chill = user_data['PrevDayMaxTemp'] - adj_temp_min
    
    # 3. Cumulative Heat History: (過去7日間の0度超え日数で推定)
    # 簡易計算: 5 * 0度超え日数 (熱履歴のペナルティ)
    heat_history = 5 * user_data['HeatDays']
    
    # 4. Surface Hardening Risk: (風速^2 * 低温時の重み)
    # Adj_Temp_Minが0度以下なら1.5倍の重みをかける
    hardening_risk = user_data['AvgWindSpeed']**2 * (1.5 if adj_temp_min < 0 else 1.0)
    
    # 5. ★モデルが期待する厳密な順序のDataFrameを作成★
    X_predict = pd.DataFrame({
        'MaxSnowDepth': [user_data['MaxSnowDepth']],
        'Snowfall': [user_data['Snowfall']],
        'AvgWindSpeed': [user_data['AvgWindSpeed']],
        'Adj_Temp_Min': [adj_temp_min],
        'Night_Chill_Factor': [night_chill],
        'Cumulative_Heat_History': [heat_history],
        'Surface_Hardening_Risk': [hardening_risk],
        'Course_Elev': [course_elev] 
    })
    return X_predict


# --- C. Streamlit UI (ユーザーインターフェース) ---

st.title("❄️ GELECON AIバーン予測システム")
st.markdown("##### ZOZO面接デモ：カスタム特徴量に基づく予測")

if model_loaded:
    
    st.header("1. コースと基本条件の入力")
    
    col1, col2 = st.columns(2)
    course_name = col1.selectbox("予測コースを選択", list(ADJUSTMENT_MAP.keys()))
    adjustment_val = ADJUSTMENT_MAP[course_name]
    elev_val = ELEVATION_MAP[course_name]

    col2.markdown(f"**推定標高**: {elev_val}m")
    col2.markdown(f"**気温補正**: -{adjustment_val:.2f}℃")
    
    col3, col4, col5 = st.columns(3)
    
    # 基本情報
    max_snow = col3.number_input("最深積雪 (cm)", min_value=10, max_value=300, value=150)
    snowfall = col4.number_input("新雪量 (cm)", min_value=0.0, max_value=50.0, value=5.0)
    avg_wind = col5.number_input("平均風速 (m/s)", min_value=0.0, max_value=15.0, value=3.0)

    # カスタム特徴量のための入力
    st.subheader("2. 凍結・熱履歴の推定入力")
    
    col6, col7, col8 = st.columns(3)
    
    min_temp = col6.number_input("当日の最低気温 (℃) - 山頂推定", min_value=-30.0, max_value=5.0, value=-8.0)
    prev_day_max_temp = col7.number_input("前日の最高気温 (℃) - 観測地", min_value=-5.0, max_value=15.0, value=5.0)
    heat_days = col8.number_input("過去7日の0℃超え日数", min_value=0, max_value=7, value=1)
    
    
    # --- 予測の実行 ---
    
    if st.button("🏔️ 雪質を予測する"):
        
        # ユーザー入力をディクショナリに格納
        user_input_data = {
            'MaxSnowDepth': max_snow, 'Snowfall': snowfall, 'AvgWindSpeed': avg_wind,
            'MinTemp': min_temp, 'PrevDayMaxTemp': prev_day_max_temp, 'HeatDays': heat_days,
        }
        
        # モデル入力形式に変換 (8つの特徴量を計算)
        X_predict = calculate_features_for_prediction(user_input_data, adjustment_val, elev_val)
        
        # 予測実行
        prediction_code = model.predict(X_predict)[0]
        prediction_proba = model.predict_proba(X_predict)[0]
        
        final_condition = CONDITIONS.get(prediction_code, "不明")
        confidence = prediction_proba[prediction_code] * 100

        
        st.markdown("---")
        st.header("4. GELECON AI予測結果")
        
        # 最終結果の表示
        if prediction_code == 3 or prediction_code == 2:
             st.error(f"予測結果: **{final_condition}** ({confidence:.1f}% 信頼度) 🚫")
        else:
             st.success(f"予測結果: **{final_condition}** ({confidence:.1f}% 信頼度) ✅")
             
        # 予測理由 (カスタムアナリシス)
        st.subheader("予測の根拠 (AIアナリシス)")
        
        st.markdown(f"""
        - **夜間急冷度**: {X_predict['Night_Chill_Factor'].iloc[0]:.2f} pt (前日の熱と当日の冷え込みの差)
        - **熱履歴**: {X_predict['Cumulative_Heat_History'].iloc[0]:.2f}pt (過去の雪質劣化の蓄積)
        - **硬化リスク**: {X_predict['Surface_Hardening_Risk'].iloc[0]:.2f}pt (風と低温による硬化度合い)
        """)

# --- 実行方法の案内 ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ アプリの実行方法")
st.sidebar.code("streamlit run streamlit_app.py")