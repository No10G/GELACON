import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import json
import os
import plotly.express as px 
import sys 

# --- 0. ファイルと定数の設定 ---
MODEL_FILE = 'gelacon_predictor_model.pkl'
PAST_CACHE_FILE = 'past_data.json'
FUTURE_CACHE_FILE = 'CF_data.json' 
FEATURE_CACHE_FILE = 'XGBoost_Features_Cache.json'

# 補正値とコース定義
COURSE_TARGETS = {
	'Kandatsu': [900, 700, 500],
	'Marunuma': [1950, 1700, 1500, 1300]
}
AMEDAS_ELEVATIONS = {'Kandatsu': 340, 'Marunuma': 370} 
CONDITIONS = {0: 'パウダー', 1: '神バーン', 2: 'アイスバーン', 3: 'シャバ雪/ゴロゴロ雪'}
CONDITION_EMOJIS = {'パウダー': '✨', '神バーン': '💎', 'アイスバーン': '⚠️', 'シャバ雪/ゴロゴロ雪': '💧'} 
MODEL_FEATURE_ORDER = [
	'MaxSnowDepth', 'Snowfall', 'AvgWindSpeed', 'Adj_Temp_Min', 
	'Night_Chill_Factor', 'Cumulative_Heat_History', 'Surface_Hardening_Risk', 'Course_Elev'
]

# 変数の初期化 
model_loaded = False 
feature_cache_data = None
past_cache_data = None
future_cache_data = None

# --- コメント定義関数 ---
def get_snow_condition_comment(condition):
	if condition == 'パウダー':
		return "予測される雪面状態は、低密度の新雪（パウダー）です。高い浮力が得られるため、サーフボード等による滑走が推奨されます。"
	elif condition == '神バーン':
		return "予測される雪面状態は、締まった圧雪バーンです。雪面硬度が高く、エッジの食い込みが安定するため、攻めたカービングに最適なコンディションです。"
	elif condition == 'アイスバーン':
		return "予測される雪面状態は、雪面が氷結しているリスクが高いアイスバーンです。エッジ角が不十分な場合、制御不能に陥る可能性があります。低速での慎重なアプローチが必要です。"
	elif condition == 'シャバ雪/ゴロゴロ雪':
		return "予測される雪面状態は、水分含有率が高い融解雪または再凍結で粒が粗くなった状態です。滑走抵抗が大きいため、ワックスの選択（低温用・湿雪用）と、雪崩等のリスク管理に注意してください。"
	else:
		return "現在の雪質は不明です。現地の情報をご確認ください。"
# --------------------

# --- 1. モデルとキャッシュのロード ---
try:
	# スクリプトの絶対パスを取得し、ベースディレクトリとする
	base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
	base_dir = os.getcwd() 

try:
	# 予測モデルをロード
	model = joblib.load(os.path.join(base_dir, MODEL_FILE))
	
	# 過去データと未来データをJSONからロード (存在しない場合はエラーにしない)
	try:
		with open(os.path.join(base_dir, PAST_CACHE_FILE), 'r', encoding='utf-8') as f:
			past_cache_data = json.load(f)
		with open(os.path.join(base_dir, FUTURE_CACHE_FILE), 'r', encoding='utf-8') as f:
			future_cache_data = json.load(f)
	except FileNotFoundError as e:
		st.warning(f"注意: 依存ファイル ({e.filename}) が見つかりません。予測コアロジックには影響しませんが、サイドバーのデバッグ情報等は不完全になります。")

	# 必須: 特徴量キャッシュをロード 
	with open(os.path.join(base_dir, FEATURE_CACHE_FILE), 'r', encoding='utf-8') as f:
		feature_cache_data = json.load(f)
		
	model_loaded = True

except FileNotFoundError as e:
	st.error(f"エラー: 必要なファイルが見つかりません。特に '{MODEL_FILE}' または '{FEATURE_CACHE_FILE}' を確認してください。パス: {e.filename}")
except Exception as e:
	st.error(f"エラー: モデルまたはキャッシュファイル ({e.__class__.__name__}) の読み込みに失敗しました。詳細: {e}")
	
# --- 2. 予測実行関数 (特徴量キャッシュを使用) ---
def run_model_prediction(feature_data_list, course_elev):
	
	if not feature_data_list:
		return []
	
	predictions = []
	
	# 特徴量リストをNumpy配列に変換 (XGBoostモデルへの入力)
	features_array = np.array([item['Features'] for item in feature_data_list])

	try:
		# モデルによる予測を実行 (確率を出力) 
		# 出力は [サンプル数, クラス数(4)] の確率配列
		probabilities = model.predict_proba(features_array)
	except Exception as e:
		st.error(f"モデル予測エラー: {e}")
		return []

	for item, probs in zip(feature_data_list, probabilities):
		
		# 最も確率の高い条件を決定
		predicted_class = np.argmax(probs)
		top_condition = CONDITIONS.get(predicted_class, '不明')
		
		predictions.append({
			'Date': item['Date'],
			'Condition': top_condition,
			'Probabilities': probs.tolist(), 
			'Course_Elev': course_elev
		})
			
	return predictions

# --- 3. Streamlit UI (メインルーチン) ---

st.set_page_config(layout="wide")

# アプリケーションの概要とタイトル
st.markdown("<h1 style='text-align: center;'>GELACON ゲレンデコンディション予測システム</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: #777; font-size: 1.1em;'>
    気象情報をもとに、ゲレンデの標高ごとのバーン状態を予測するシステムとなっております。<br>
    これはあくまで予測なので、実際のバーン状況とは異なる可能性がございます。
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(" AIによる5日間先のバーン予測")


if model_loaded and feature_cache_data:
	
	# リゾートの選択 (サイドバー)
	st.sidebar.header("🏔️ リゾート選択")
	resort_options = ['神立スノーリゾート', '丸沼高原スキー場']
	selected_resort = st.sidebar.selectbox("予測リゾートを選択", resort_options)
	st.sidebar.markdown("---")

	# A. 選択リゾートの設定をフィルタリング
	base_key = 'Kandatsu' if selected_resort == '神立スノーリゾート' else 'Marunuma'
	
	# 予測結果を格納するリストとDataFrame
	all_predictions_df = []
	
	st.header(f"予測対象: {selected_resort}")
	st.markdown("---")
	
	# ターゲット標高リストを取得
	target_elevations = COURSE_TARGETS[base_key]
	
	# B. コースごとの予測実行ループ
	for course_elev in target_elevations:
		
		# 特徴量キャッシュから該当するデータセットを取得するためのキーを作成
		feature_key = f"{base_key}_{course_elev}m"
		
		# 該当する特徴量データリストを取得
		feature_data_list = feature_cache_data['features'].get(feature_key, [])
		
		if not feature_data_list:
			st.warning(f"注意: {feature_key} の特徴量データがキャッシュに見つかりません。スキップします。")
			continue # データがない場合はスキップ
		
		# 1. 予測の実行 (日別データを計算)
		predictions = run_model_prediction(feature_data_list, course_elev)
		
		# 予測結果をDataFrameに変換して統合
		df_course = pd.DataFrame(predictions)
		df_course['Course_Elev'] = df_course['Course_Elev'].astype(str) + 'm'
		all_predictions_df.append(df_course)

	# 予測データが存在する場合のみUIを表示
	if all_predictions_df:
		df_combined = pd.concat(all_predictions_df)
		
		# --- UI表示のメイン部分 ---
		
		# 1. 標高ごとのコンディションサマリ（左上）
		st.subheader("1. 🗺️ コンディションマップ")
		
		# 各日付で最も確率の高いコンディションを取得
		df_combined['Top_Condition'] = df_combined.apply(lambda row: CONDITIONS[np.argmax(row['Probabilities'])], axis=1)
		
		# 修正: 絵文字と日本語名を結合した新しいカラムを作成
		def format_condition(row):
			emoji = CONDITION_EMOJIS[row['Top_Condition']]
			# 略称を使用せず、完全な名称を使用
			name = row['Top_Condition'] 
			return f"{emoji} {name}"
			
		df_combined['Formatted_Condition'] = df_combined.apply(format_condition, axis=1)
		
		# Plotly Heatmap (imshow) の代わりにPandas Stylerを使用
		# 標高(index)と日付(columns)でピボットテーブルを作成
		pivot_table_formatted = df_combined.pivot_table(
			index='Course_Elev', 
			columns='Date', 
			values='Formatted_Condition', # 結合した文字列を使用
			aggfunc='first'
		# 修正後のポイント: target_elevations をそのまま使用 (降順であるため)
		).reindex([str(e) + 'm' for e in target_elevations])
		
		# 標高の数値だけを取り出し、ソートしてグラデーションの基準を作成
		elev_floats = [float(e.replace('m', '')) for e in pivot_table_formatted.index]
		min_elev = min(elev_floats)
		max_elev = max(elev_floats)
		
		# --- グラデーション関数 ---
		def elevation_gradient(s):
			# 各標高行に対応するグラデーションを適用する
			styles = []
			elev_str = s.name
			elev_val = float(elev_str.replace('m', ''))
			
			# 0.1から0.7の範囲で青の濃淡を計算
			normalized_elev = (elev_val - min_elev) / (max_elev - min_elev) if max_elev > min_elev else 0.5
			# Hue=240(青), Saturation=70%, Lightness=70% - (normalized)*30% (標高が高いほど色が濃い青)
			lightness = 70 - (normalized_elev * 30) 
			
			bg_color = f"hsl(240, 70%, {lightness}%)"
			
			
			for _ in s.index:
				styles.append(f'background-color: {bg_color}; color: white; text-align: center; font-size: 0.75em;')
			return styles
		# ------------------------------------

		# Stylerを適用してHTMLテーブルとしてStreamlitに表示
		st.dataframe(
			pivot_table_formatted.style.apply(elevation_gradient, axis=1), 
			use_container_width=True,
			height=len(target_elevations) * 70
		)
		
		st.markdown("---")
		
		# 2. ドロップダウン選択による詳細確率グラフとコメント
		st.subheader("2. 📊 詳細予測確率とアドバイス")
		
		col1, col2 = st.columns(2)
		
		unique_dates = df_combined['Date'].unique()
		unique_elevs = df_combined['Course_Elev'].unique()

		with col1:
			selected_date = st.selectbox("予測日を選択", unique_dates)
			
		with col2:
			selected_elev = st.selectbox("コース標高を選択 (m)", unique_elevs)
			
		df_filtered = df_combined[
			(df_combined['Date'] == selected_date) & 
			(df_combined['Course_Elev'] == selected_elev)
		].iloc[0]
		

		st.markdown("<br><br>", unsafe_allow_html=True) 
		
		st.markdown("#### 💬 今日のアドバイス")
		top_condition_for_comment = df_filtered['Top_Condition']
		st.info(get_snow_condition_comment(top_condition_for_comment))
		
		st.markdown("<br><br>", unsafe_allow_html=True) 
		
		prob_data = pd.DataFrame({
			'Condition': list(CONDITIONS.values()),
			'Probability': df_filtered['Probabilities']
		})
		
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
				'シャバ雪/ゴロゴロ雪': 'orange'
			}
		)
		prob_fig.update_traces(textinfo='percent+label')
		st.plotly_chart(prob_fig, use_container_width=True)

	else:
		st.warning("選択したリゾート、またはコースの予測データが見つかりませんでした。")

else:
	st.error("予測システムを起動できません。必要なファイルが揃っているか確認してください。")

# --- 実行 ---
if __name__ == '__main__':
	pass