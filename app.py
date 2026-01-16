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
SPREADSHEET_NAME = 'muscle_db'

EXERCISES = {
    "胸": ["ベンチプレス", "インクラインベンチプレス", "インクラインダンベルプレス", "ディップス", "ペックフライ", "マシンプレス"],
    "背中": ["デッドリフト", "フロントプル", "ラットプル", "ローロー", "チンニング"],
    "脚": ["スクワット", "レッグエクステンション", "レッグカール", "レッグプレス", "ブルガリアンスクワット"],
    "肩": ["サイドレイズ", "ダンベルショルダープレス", "バーベルショルダープレス"],
    "腕": ["スカルクラッシャー", "インクラインカール", "バーベルカール", "ケーブルプレスダウン"]
}

def get_body_part(exercise_name):
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
            updateDisplay(timer, display);
            interval = setInterval(function () {
                if (--timer < 0) {
                    clearInterval(interval);
                    display.textContent = "00:00";
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

def predict_next_weight(df, target_event):
    df_event = df[df['種目名'] == target_event].copy()
    if len(df_event) < 3:
        return None
    df_event['日付'] = pd.to_datetime(df_event['日付'])
    start_date = df_event['日付'].min()
    df_event['days_passed'] = (df_event['日付'] - start_date).dt.days
    df_event['重量(kg)'] = pd.to_numeric(df_event['重量(kg)'], errors='coerce')
    df_event = df_event.dropna(subset=['重量(kg)'])
    X = df_event[['days_passed']]
    y = df_event['重量(kg)']
    model = LinearRegression()
    model.fit(X, y)
    today_days = (datetime.datetime.now() - start_date).days
    predicted_weight = model.predict([[today_days]])[0]
    return round(predicted_weight, 1)

def get_ai_agent_advice(df):
    if df.empty:
        return "データがありません。まずは初回のトレーニングを記録しましょう！"
    df_sorted = df.sort_values('日付', ascending=False)
    last_record = df_sorted.iloc[0]
    last_date = pd.to_datetime(last_record['日付'])
    today = datetime.datetime.now()
    days_since_last = (today - last_date).days
    target_event = last_record['種目名']
    predicted_kg = predict_next_weight(df, target_event)
    is_beginner = len(df) < 30
    if is_beginner:
        system_prompt = """
        あなたはユーザーを溺愛する「過保護なトレーニングマネージャー」です。
        以下の制約を守ってください：
        1. 難しい専門用語は一切使わないでください。
        2. 「とにかくジムに来たこと」や「記録したこと」を大げさに褒めてください。
        3. ユーザーが迷わないよう、今日のメニューを断定的に指示してください。
        4. 口調は明るく、絵文字を多用してください。
        """
    else:
        system_prompt = """
        あなたはデータ重視の「冷徹なAI分析官」です。
        以下の制約を守ってください：
        1. 「漸進的過負荷」や「ボリューム」などの観点から論理的に話してください。
        2. 褒める必要はありません。データの事実と改善点だけを伝えてください。
        3. 前回の記録を超えられるような、具体的な重量設定を提案してください。
        4. 口調は敬語ですが、事務的でクールにしてください。
        """
    if predicted_kg:
        ai_prediction_text = f"過去の成長トレンドに基づくと、今日の適正重量は【{predicted_kg}kg】です。"
    else:
        ai_prediction_text = "データ不足のため予測できません。まずはデータを溜めましょう。"
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
    try:
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

def get_worksheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    client = gspread.authorize(creds)
    try:
        sheet = client.open(SPREADSHEET_NAME)
        return sheet.sheet1
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"スプレッドシート '{SPREADSHEET_NAME}' が見つかりません。")
        st.stop()

def load_data():
    worksheet = get_worksheet()
    data = worksheet.get_all_values()
    
    if not data:
        return pd.DataFrame(columns=['日付', '部位', '種目名', '重量(kg)', '回数(レップ)', 'ユーザー名'])
    
    header = data[0]
    
    # スキーママイグレーション: ユーザー名カラムがない場合に追加
    if 'ユーザー名' not in header:
        try:
            # 6列目(F列)にヘッダーを追加
            worksheet.update_cell(1, 6, 'ユーザー名')
            header.append('ユーザー名')
        except Exception as e:
            st.warning(f"スキーマ更新中にエラーが発生しましたが続行します: {e}")

    rows = data[1:]
    if not rows:
        return pd.DataFrame(columns=header)
    
    # 行の長さがヘッダーと異なる場合の補完処理
    aligned_rows = []
    for row in rows:
        if len(row) < len(header):
            row += [''] * (len(header) - len(row))
        aligned_rows.append(row)

    df = pd.DataFrame(aligned_rows, columns=header)
    
    # 現在のユーザーでフィルタリング
    current_user = st.session_state.get('username')
    if current_user:
         df = df[df['ユーザー名'] == current_user]
    
    return df

def save_new_data(date, body_part, exercise, weight, reps):
    worksheet = get_worksheet()
    current_user = st.session_state.get('username', 'Unknown')
    
    row = [str(date), body_part, exercise, str(weight), str(reps), str(current_user)]
    
    # シートが空の場合のヘッダー作成
    if len(worksheet.get_all_values()) == 0:
        worksheet.append_row(['日付', '部位', '種目名', '重量(kg)', '回数(レップ)', 'ユーザー名'])
    
    worksheet.append_row(row)

def init_session_state():
    if 'current_view' not in st.session_state:
        st.session_state['current_view'] = 'dashboard'
    if 'selected_exercise' not in st.session_state:
        st.session_state['selected_exercise'] = None
    if 'selected_body_part' not in st.session_state:
        st.session_state['selected_body_part'] = 'All'
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'is_logged_in' not in st.session_state:
        st.session_state['is_logged_in'] = False

def render_login():
    st.markdown("""
    <style>
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            flex-direction: column;
        }
        .login-title {
            font-size: 3rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00FF00, #00FFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-title" style="text-align: center;">LIFT OS</div>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("### ユーザーログイン")
        username = st.text_input("ユーザー名")
        submitted = st.form_submit_button("Start", type="primary", use_container_width=True)
        
        if submitted:
            if username:
                st.session_state['username'] = username
                st.session_state['is_logged_in'] = True
                st.rerun()
            else:
                st.error("ユーザー名を入力してください")

def navigate_to(view, exercise=None):
    st.session_state['current_view'] = view
    if exercise:
        st.session_state['selected_exercise'] = exercise
    st.rerun()

def logout():
    st.session_state['username'] = None
    st.session_state['is_logged_in'] = False
    st.session_state['current_view'] = 'dashboard'
    st.rerun()

# --- ダッシュボード (メイン画面) ---
def render_dashboard(df):
    # CSS注入 (統合版)
    st.markdown("""
    <style>
        /* ベーススタイル */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* 1. タイトル */
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

        /* 2. ボタンデザイン */
        .stButton button {
            background-color: transparent;
            border: 1px solid #4CAF50;
            color: #4CAF50;
            border-radius: 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #4CAF50;
            color: white;
            box-shadow: 0 0 10px #4CAF50;
            border-color: #4CAF50;
        }
        
        /* 認証済みヘッダー調整 */
        .user-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* 3. AIエリアの文字スタイル */
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
    </style>
    """, unsafe_allow_html=True)

    # タイトル & ユーザー情報
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f'<div class="custom-title">LIFT OS</div>', unsafe_allow_html=True)
    with c2:
        st.write(f"User: **{st.session_state['username']}**")
        if st.button("Logout", key="logout_btn", use_container_width=True):
            logout()

    # 1. AIエージェントエリア (修正: 枠線コンテナにして謎の四角を消去)
    with st.container(border=True):
        col_ai_icon, col_ai_text = st.columns([1, 6])
        with col_ai_icon:
            st.image("https://api.dicebear.com/7.x/bottts/svg?seed=WorkoutAI", width=60)
        with col_ai_text:
            st.markdown('<div class="ai-title">AI Coach Agent</div>', unsafe_allow_html=True)
            if not df.empty:
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

    # 2. ナビゲーション & フィルタ
    st.write("##### 部位フィルター")
    parts = ["All"] + list(EXERCISES.keys())
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
        last_rec_text = "記録なし"
        if not df.empty:
            ex_df = df[df['種目名'] == exercise].sort_values('日付', ascending=False)
            if not ex_df.empty:
                last = ex_df.iloc[0]
                last_rec_text = f"{last['重量(kg)']}kg x {last['回数(レップ)']} ({last['日付'].strftime('%m/%d')})"

        with st.container(border=True):
            c1, c2 = st.columns([4, 1.5])
            with c1:
                st.markdown(f"**{exercise}**")
                st.caption(f"{get_body_part(exercise)} • {last_rec_text}")
            with c2:
                if st.button("記録", key=f"nav_{exercise}", use_container_width=True):
                    navigate_to('detail', exercise)

# --- 詳細画面 (入力 & グラフ) ---
def render_detail_view(df, exercise_name):
    # ヘッダー
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("< Back"):
            navigate_to('dashboard')
    with c2:
        st.markdown(f'<div class="custom-title" style="font-size: 2rem;">{exercise_name}</div>', unsafe_allow_html=True)

    if not df.empty:
        ex_df = df[df['種目名'] == exercise_name].sort_values('日付')
        ex_df['重量(kg)'] = pd.to_numeric(ex_df['重量(kg)'], errors='coerce').fillna(0)
        ex_df['回数(レップ)'] = pd.to_numeric(ex_df['回数(レップ)'], errors='coerce').fillna(0)
    else:
        ex_df = pd.DataFrame()

    # Stats Header
    if not ex_df.empty:
        ex_df['1RM'] = ex_df['重量(kg)'] * (1 + ex_df['回数(レップ)'] / 30)
        last_item = ex_df.iloc[-1]
        last_date = last_item['日付'].strftime('%m/%d')
        
        max_weight = ex_df['重量(kg)'].max()
        pr_text = f"{int(max_weight)} kg"
        
        count = len(ex_df)
    else:
        last_date = "-"
        pr_text = "-- kg"
        count = 0

    h1, h2, h3 = st.columns(3)
    h1.metric("部位", get_body_part(exercise_name))
    h2.metric("前回", last_date)
    h3.metric("👑 最高記録", pr_text)

    st.markdown("---")

    # グラフ
    st.subheader("推移 (推定1RM)")
    if not ex_df.empty and count > 1:
        chart_data = ex_df[['日付', '1RM']].set_index('日付')
        st.line_chart(chart_data, color="#4CAF50")
    else:
        st.info("データが2件以上あるとグラフが表示されます。")

    st.markdown("---")

    with st.expander("⏱ インターバルタイマー"):
        render_js_timer()

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
                st.rerun()
            else:
                st.error("重量と回数を入力してください。")

    st.subheader("履歴")
    if not ex_df.empty:
        display_df = ex_df.sort_values('日付', ascending=False)[['日付', '重量(kg)', '回数(レップ)', '1RM']]
        display_df['日付'] = display_df['日付'].dt.strftime('%Y/%m/%d')
        display_df['1RM'] = display_df['1RM'].apply(lambda x: f"{x:.1f}kg")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title="LIFT OS", layout="centered") 
    init_session_state()

    if not st.session_state['is_logged_in']:
        render_login()
        return

    try:
        df = load_data()
        if not df.empty:
            df['日付'] = pd.to_datetime(df['日付'])
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return

    if st.session_state['current_view'] == 'dashboard':
        render_dashboard(df)
    elif st.session_state['current_view'] == 'detail':
        render_detail_view(df, st.session_state['selected_exercise'])

if __name__ == '__main__':
    main()