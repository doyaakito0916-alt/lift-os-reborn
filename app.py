import streamlit as st
import pandas as pd
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit.components.v1 as components
import openai
from sklearn.linear_model import LinearRegression
import numpy as np

# --- 設定 ---
# スプレッドシート名（作成したものと一字一句同じにする）
SPREADSHEET_NAME = 'muscle_db'

# 種目定義
EXERCISES = {
    "胸": ["ベンチプレス", "インクラインベンチプレス", "インクラインダンベルプレス", "ディップス", "ペックフライ", "マシンプレス"],
    "背中": ["デッドリフト", "フロントプル", "ラットプル", "ローロー", "チンニング"],
    "脚": ["スクワット", "レッグエクステンション", "レッグカール", "レッグプレス", "ブルガリアンスクワット"],
    "肩": ["サイドレイズ", "ダンベルショルダープレス", "バーベルショルダープレス"],
    "腕": ["スカルクラッシャー", "インクラインカール", "バーベルカール", "ケーブルプレスダウン"]
}

def get_body_part(exercise_name):
    """種目名から部位を逆引き"""
    for part, exercises in EXERCISES.items():
        if exercise_name in exercises:
            return part
    return "その他"

# --- JSタイマー機能 ---
def render_js_timer():
    timer_html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap');
        .timer-container {
            text-align: center;
            margin-top: 20px;
        }
        .time-display {
            font-family: 'Montserrat', sans-serif;
            font-size: 3.5rem;
            font-weight: 600;
            color: #31333F;
            line-height: 1;
            margin-bottom: 15px;
        }
        .btn-group {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        button {
            background: transparent;
            border: 1px solid #FF4B4B;
            color: #FF4B4B;
            padding: 8px 16px;
            border-radius: 30px;
            font-family: 'Montserrat', sans-serif;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:hover {
            background: #FF4B4B;
            color: white;
        }
        .reset-btn {
            border-color: #999;
            color: #999;
        }
        .reset-btn:hover {
            background: #999;
            color: white;
        }
    </style>
    
    <div class="timer-container">
        <div id="timer" class="time-display">00:00</div>
        <div class="btn-group">
            <button onclick="startTimer(90)">90s</button>
            <button onclick="startTimer(120)">120s</button>
            <button onclick="resetTimer()" class="reset-btn">STOP</button>
        </div>
    </div>

    <script>
        let interval;
        function startTimer(duration) {
            clearInterval(interval);
            let timer = duration, minutes, seconds;
            const display = document.querySelector('#timer');
            
            // 即時反映
            updateDisplay(timer, display);

            interval = setInterval(function () {
                if (--timer < 0) {
                    clearInterval(interval);
                    display.textContent = "00:00";
                    // 必要なら音を鳴らす処理
                } else {
                    updateDisplay(timer, display);
                }
            }, 1000);
        }
        
        function updateDisplay(timer, display) {
            let minutes = parseInt(timer / 60, 10);
            let seconds = parseInt(timer % 60, 10);
            minutes = minutes < 10 ? "0" + minutes : minutes;
            seconds = seconds < 10 ? "0" + seconds : seconds;
            display.textContent = minutes + ":" + seconds;
        }

        function resetTimer() {
            clearInterval(interval);
            document.querySelector('#timer').textContent = "00:00";
        }
    </script>
    """
    components.html(timer_html, height=200)


def display_progressive_overload_stats(df, selected_event):
    """前回の記録と自己ベストを表示"""
    # データが空、または該当種目がない場合
    if df.empty:
        return

    # 種目でフィルタリング
    event_df = df[df['種目名'] == selected_event].copy()
    
    if event_df.empty:
        st.sidebar.info("初めての種目です！まずは記録を作りましょう。")
        return

    # 日付順にソート (すでにdatetimeになっている前提だが念のため)
    if '日付' in event_df.columns:
        event_df['日付'] = pd.to_datetime(event_df['日付'])
    event_df = event_df.sort_values('日付')

    # 前回記録（直近の行）
    last_record = event_df.iloc[-1]
    last_weight = last_record['重量(kg)']
    last_reps = last_record['回数(レップ)']
    
    # 自己ベスト（重量の最大値）
    event_df['重量(kg)'] = pd.to_numeric(event_df['重量(kg)'], errors='coerce').fillna(0)
    
    max_weight_idx = event_df['重量(kg)'].idxmax()
    pr_record = event_df.loc[max_weight_idx]
    pr_weight = pr_record['重量(kg)']
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(
            label="前回の記録",
            value=f"{last_weight}kg x {last_reps}回",
            delta="この記録を超えよう！",
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="🏆 自己ベスト",
            value=f"{pr_weight} kg",
            help=f"達成日: {pr_record['日付'].strftime('%Y-%m-%d')}"
        )






def predict_next_weight(df, target_event):
    """
    特定種目の過去データから、次回扱うべき重量を予測する関数
    """
    # その種目のデータだけ抜き出す
    df_event = df[df['種目名'] == target_event].copy()
    
    # データが少なすぎる場合（3回未満）は予測しない
    if len(df_event) < 3:
        return None

    # --- データ加工 (Feature Engineering) ---
    # 日付を「基準日からの経過日数」という数値に変換する（AIは日付型を読めないため）
    df_event['日付'] = pd.to_datetime(df_event['日付'])
    start_date = df_event['日付'].min()
    df_event['days_passed'] = (df_event['日付'] - start_date).dt.days
    
    # 重量(kg)を数値に変換（念のため）
    df_event['重量(kg)'] = pd.to_numeric(df_event['重量(kg)'], errors='coerce')
    df_event = df_event.dropna(subset=['重量(kg)'])

    # 説明変数 X (経過日数) と 目的変数 y (重量) を用意
    X = df_event[['days_passed']]
    y = df_event['重量(kg)']

    # --- 機械学習 (Model Training) ---
    # 線形回帰モデルを作成して学習（.fit）
    model = LinearRegression()
    model.fit(X, y)

    # --- 未来予測 (Prediction) ---
    # 「今日」が開始日から何日目かを計算
    today_days = (datetime.datetime.now() - start_date).days
    # 次回（例えば今日）の重量を予測
    predicted_weight = model.predict([[today_days]])[0]

    # --- 安全策 & 補正 ---
    # 予測値があまりに突飛な数字にならないよう丸める（2.5kg刻みなど）
    # ここでは単純に小数を丸める
    return round(predicted_weight, 1)


# --- AIエージェント機能 ---
def get_ai_agent_advice(df):
    """
    データフレームを受け取り、ユーザーレベルに応じたアドバイスを生成するエージェント
    """
    if df.empty:
        return "データがありません。まずは初回のトレーニングを記録しましょう！"

    # --- 1. 現状分析（Context） ---
    # 直近のデータを取得
    last_record = df.iloc[0] # dfは日付降順ソート済みと想定されるため、0番目が最新
    # ただし、関数呼び出し元のdfがソートされているか確認が必要。
    # display_progressive_overload_stats内ではソートしているが、ここでも安全のためソートする
    df_sorted = df.sort_values('日付', ascending=False)
    last_record = df_sorted.iloc[0]
    
    last_date = pd.to_datetime(last_record['日付'])
    today = datetime.datetime.now()
    days_since_last = (today - last_date).days
    
    # ターゲット種目 (前回の種目)
    target_event = last_record['種目名']

    # --- 予測機能の統合 ---
    predicted_kg = predict_next_weight(df, target_event)
    
    # 継続期間や総負荷量から「レベル」を判定（仮ロジック）
    # 例: データ行数が30行未満ならビギナー
    is_beginner = len(df) < 30

    # --- 2. 人格の切り替え（Persona） ---
    if is_beginner:
        # Phase 1: ビギナーモード（優しい・思考停止させる）
        system_prompt = """
        あなたはユーザーを溺愛する「過保護なトレーニングマネージャー」です。
        以下の制約を守ってください：
        1. 難しい専門用語は一切使わないでください。
        2. 「とにかくジムに来たこと」や「記録したこと」を大げさに褒めてください。
        3. ユーザーが迷わないよう、今日のメニューを断定的に指示してください。
        4. 口調は明るく、絵文字を多用してください。
        """
    else:
        # Phase 2: プロモード（厳しい・データ重視）
        system_prompt = """
        あなたはデータ重視の「冷徹なAI分析官」です。
        以下の制約を守ってください：
        1. 「漸進的過負荷」や「ボリューム」などの観点から論理的に話してください。
        2. 褒める必要はありません。データの事実と改善点だけを伝えてください。
        3. 前回の記録を超えられるような、具体的な重量設定を提案してください。
        4. 口調は敬語ですが、事務的でクールにしてください。
        """

    # --- 予測結果のテキスト化 ---
    if predicted_kg:
        ai_prediction_text = f"過去の成長トレンドに基づくと、今日の適正重量は【{predicted_kg}kg】です。"
    else:
        ai_prediction_text = "データ不足のため予測できません。まずはデータを溜めましょう。"

    # --- 3. 指示書（User Prompt） ---
    # AIに渡す「今の状況」
    user_prompt = f"""
    【ユーザーデータ】
    - 前回のトレーニング日: {last_date.strftime('%Y-%m-%d')} ({days_since_last}日前)
    - 前回の種目: {last_record['種目名']}
    - 前回の重量: {last_record['重量(kg)']}kg
    - 前回の回数: {last_record['回数(レップ)']}回
    
    【AI予測モデルの推奨】
    {ai_prediction_text}
    
    この推奨値を参考に、ユーザーに今日の目標を伝えてください。
    無理そうなら少し下げてもいいと伝えてください。
    アドバイスは150文字以内で出力してください。
    """

    # --- 4. 生成（Generation） ---
    try:
        # SecretsからAPIキーを取得 (st.secrets["OPENAI_API_KEY"])
        # ローカルテスト等でキーがない場合のハンドリングも考慮
        if "OPENAI_API_KEY" in st.secrets:
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        else:
            return "OpenAI APIキーが設定されていません。"
    except Exception as e:
        return f"AIエージェント接続エラー: {e}"


# --- Googleスプレッドシート接続機能 ---
def get_worksheet():
    """スプレッドシートに接続してワークシートを返す"""
    # Secretsから認証情報を取得
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    client = gspread.authorize(creds)
    
    # スプレッドシートを開く
    try:
        sheet = client.open(SPREADSHEET_NAME)
        return sheet.sheet1 # 1枚目のシートを使う
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"スプレッドシート '{SPREADSHEET_NAME}' が見つかりません。共有設定か名前を確認してください。")
        st.stop()

def load_data():
    """スプレッドシートからデータを読み込む"""
    worksheet = get_worksheet()
    data = worksheet.get_all_values()
    
    # データが空、またはヘッダーしかない場合
    if not data:
        return pd.DataFrame(columns=['日付', '部位', '種目名', '重量(kg)', '回数(レップ)'])
    
    # 1行目をヘッダーとして読み込む
    header = data[0]
    rows = data[1:]
    
    if not rows:
        return pd.DataFrame(columns=header)

    df = pd.DataFrame(rows, columns=header)
    return df

def save_new_data(date, body_part, exercise, weight, reps):
    """新規データをスプレッドシートの末尾に追加"""
    worksheet = get_worksheet()
    # データをリスト形式で用意（全て文字列にしておくと安全）
    row = [str(date), body_part, exercise, str(weight), str(reps)]
    
    # ヘッダーが無い場合は追加する処理（初回のみ）
    if len(worksheet.get_all_values()) == 0:
        worksheet.append_row(['日付', '部位', '種目名', '重量(kg)', '回数(レップ)'])
        
    worksheet.append_row(row)

def update_all_data(df):
    """編集後のデータを丸ごと上書き保存（編集機能用）"""
    worksheet = get_worksheet()
    worksheet.clear() # 一旦全消去
    # ヘッダーとデータを書き込む
    data_to_write = [df.columns.tolist()] + df.astype(str).values.tolist()
    worksheet.update(data_to_write)


# ... (Existing imports and setup code remains the same until main)

# --- ページ遷移管理 ---
def init_session_state():
    if 'current_view' not in st.session_state:
        st.session_state['current_view'] = 'dashboard'
    if 'selected_exercise' not in st.session_state:
        st.session_state['selected_exercise'] = None
    if 'selected_body_part' not in st.session_state:
        st.session_state['selected_body_part'] = 'All'

def navigate_to(view, exercise=None):
    st.session_state['current_view'] = view
    if exercise:
        st.session_state['selected_exercise'] = exercise
    st.rerun()

# --- ダッシュボード (メイン画面) ---
def render_dashboard(df):
    # CSSでスタイル調整 (Merged New & Existing AI Styles)
    st.markdown("""
    <style>
        /* --- 1. 全体の背景と文字色 (New) --- */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* --- 2. カッコいいタイトルの定義 (New) --- */
        .custom-title {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            font-size: 3rem;
            background: -webkit-linear-gradient(45deg, #00FF00, #00FFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            line-height: 1.2;
        }

        /* --- 3. ボタンをダサくなくする (New) --- */
        .stButton button {
            background-color: transparent;
            border: 1px solid #4CAF50;
            color: #4CAF50;
            border-radius: 20px; /* 丸くする */
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #4CAF50;
            color: white;
            box-shadow: 0 0 10px #4CAF50;
            border-color: #4CAF50;
        }

        /* --- 4. カードデザイン (New & Merged) --- */
        .exercise-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
        }

        /* --- 5. AI Agent & Parts (Existing Preserved) --- */
        .dashboard-header {
            padding: 20px;
            background-color: #1E1E1E;
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .ai-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .ai-message {
            font-size: 0.95rem;
            color: #E0E0E0;
            line-height: 1.5;
        }
        .part-badge {
            background-color: #444;
            color: #ddd;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            margin-right: 8px;
        }
        .last-record {
            color: #888;
            font-size: 0.85rem;
            margin-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ★修正1: タイトルを表示
    st.markdown('<div class="custom-title">LIFT OS</div>', unsafe_allow_html=True)

    # 1. AIエージェントエリア
    with st.container():
        st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
        col_ai_icon, col_ai_text = st.columns([1, 6])
        with col_ai_icon:
            st.image("https://api.dicebear.com/7.x/bottts/svg?seed=WorkoutAI", width=60) # 仮のアイコン
        with col_ai_text:
            st.markdown('<div class="ai-title">AI Coach Agent</div>', unsafe_allow_html=True)
            if not df.empty:
                # 毎回APIを叩くと重いので、キャッシュするか、ボタンで発火させるか要検討。今回は要望通りシンプルに表示
                # 本番では直近のアドバイスをsession_state保存推奨
                if 'ai_advice' not in st.session_state:
                     st.session_state['ai_advice'] = "今日も頑張りましょう！トレーニングを開始してください。"
                
                st.markdown(f'<div class="ai-message">{st.session_state["ai_advice"]}</div>', unsafe_allow_html=True)
                if st.button("今日のアドバイスを更新", key="refresh_ai"):
                    with st.spinner("思考中..."):
                        advice = get_ai_agent_advice(df)
                        st.session_state['ai_advice'] = advice
                        st.rerun()
            else:
                st.markdown('<div class="ai-message">データがありません。初回のトレーニングを記録しましょう！</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. ナビゲーション & フィルタ
    # 部位フィルタ
    parts = ["All"] + list(EXERCISES.keys())
    
    # st.pills (Streamlit 1.40+) があれば使う、なければradioを横並び風に
    # ここでは既存環境に合わせて st.columns でボタン風に実装するか、selectboxで妥協するか。
    # ユーザー要望のUIに近づけるため、columnsで並べる
    
    st.write("##### 部位フィルター")
    cols = st.columns(len(parts))
    for i, part in enumerate(parts):
        if cols[i].button(part, key=f"filter_{part}", use_container_width=True, type="primary" if st.session_state['selected_body_part'] == part else "secondary"):
            st.session_state['selected_body_part'] = part
            st.rerun()

    # 3. 種目リスト
    st.markdown("### 種目一覧")
    
    target_part = st.session_state['selected_body_part']
    if target_part == "All":
        target_exercises = []
        for p in EXERCISES:
            target_exercises.extend(EXERCISES[p])
    else:
        target_exercises = EXERCISES[target_part]

    for exercise in target_exercises:
        # 直近記録の取得
        last_rec_text = "記録なし"
        if not df.empty:
            ex_df = df[df['種目名'] == exercise].sort_values('日付', ascending=False)
            if not ex_df.empty:
                last = ex_df.iloc[0]
                last_rec_text = f"{last['重量(kg)']}kg x {last['回数(レップ)']} ({last['日付'].strftime('%m/%d')})"

        # 行で表示
        with st.container(border=True): # カード風枠線
            c1, c2 = st.columns([4, 1.2]) # ボタン幅を少し確保
            with c1:
                st.markdown(f"**{exercise}**")
                st.caption(f"{get_body_part(exercise)} • {last_rec_text}")
            with c2:
                # ★修正2: ボタン名を「記録」に変更
                if st.button("記録", key=f"nav_{exercise}"):
                    navigate_to('detail', exercise)

# --- 詳細画面 (入力 & グラフ) ---
def render_detail_view(df, exercise_name):
    # ヘッダー (戻るボタン & タイトル)
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("< Back"):
            navigate_to('dashboard')
    with c2:
        # ★修正3: 詳細画面のタイトルもカッコよく
        st.markdown(f'<div class="custom-title" style="font-size: 2rem;">{exercise_name}</div>', unsafe_allow_html=True)

    # 既存データの抽出
    if not df.empty:
        ex_df = df[df['種目名'] == exercise_name].sort_values('日付')
        # 数値変換
        ex_df['重量(kg)'] = pd.to_numeric(ex_df['重量(kg)'], errors='coerce').fillna(0)
        ex_df['回数(レップ)'] = pd.to_numeric(ex_df['回数(レップ)'], errors='coerce').fillna(0)
    else:
        ex_df = pd.DataFrame()

    # Stats Header
    # 推定1RMの計算 (Epley formula: Weight * (1 + Reps/30))
    # ★修正4: 「記録数」をやめて「最高記録(PR)」にする
    if not ex_df.empty:
        ex_df['1RM'] = ex_df['重量(kg)'] * (1 + ex_df['回数(レップ)'] / 30)
        last_item = ex_df.iloc[-1]
        last_date = last_item['日付'].strftime('%m/%d')
        count = len(ex_df)
        
        # 最高記録 (Personal Record) の計算
        max_weight = ex_df['重量(kg)'].max()
        pr_text = f"{int(max_weight)} kg"
    else:
        last_date = "-"
        count = 0
        pr_text = "-- kg"

    h1, h2, h3 = st.columns(3)
    h1.metric("部位", get_body_part(exercise_name))
    h2.metric("前回", last_date)
    # ここを変更
    h3.metric("👑 最高記録", pr_text)

    st.markdown("---")

    # グラフ (推移)
    st.subheader("推移 (推定1RM)")
    if not ex_df.empty and count > 1:
        # グラフ作成
        chart_data = ex_df[['日付', '1RM']].set_index('日付')
        st.line_chart(chart_data, color="#4CAF50") # 緑系
    else:
        st.info("データが2件以上あるとグラフが表示されます。")

    st.markdown("---")

    # タイマー (詳細画面にはあった方が便利なので追加)
    with st.expander("⏱ インターバルタイマー"):
        render_js_timer()

    # 入力フォーム (画面下部固定風にしたいがStreamlitでは難しいので、一番下に配置)
    st.subheader("新規記録")
    
    with st.form("record_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            input_date = st.date_input("日付", datetime.date.today())
        with f2:
            input_weight = st.number_input("重量 (kg)", min_value=0.0, step=1.0)
        with f3:
            input_reps = st.number_input("回数", min_value=0, step=1)
        
        submitted = st.form_submit_button("記録を保存", type="primary", use_container_width=True)
        
        if submitted:
            if input_weight > 0 and input_reps > 0:
                body_part = get_body_part(exercise_name)
                save_new_data(input_date, body_part, exercise_name, input_weight, input_reps)
                st.success("保存しました！")
                st.rerun() # リロードしてグラフ更新
            else:
                st.error("重量と回数を入力してください。")

    # 履歴リスト
    st.subheader("履歴")
    if not ex_df.empty:
        # 新しい順に表示
        display_df = ex_df.sort_values('日付', ascending=False)[['日付', '重量(kg)', '回数(レップ)', '1RM']]
        # フォーマット調整
        display_df['日付'] = display_df['日付'].dt.strftime('%Y/%m/%d')
        display_df['1RM'] = display_df['1RM'].apply(lambda x: f"{x:.1f}kg")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="LIFT OS", layout="centered") 
    
    # CSS注入 (共通デザイン)
    st.markdown("""
    <style>
        /* ダークモード前提の配色強化 */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* ボタンのスタイル上書き */
        .stButton button {
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    
    # データ読み込み
    try:
        df = load_data()
        if not df.empty:
            df['日付'] = pd.to_datetime(df['日付'])
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return

    # ルーティング
    if st.session_state['current_view'] == 'dashboard':
        render_dashboard(df)
    elif st.session_state['current_view'] == 'detail':
        render_detail_view(df, st.session_state['selected_exercise'])

if __name__ == '__main__':
    main()