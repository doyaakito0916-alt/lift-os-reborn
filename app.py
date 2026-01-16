import streamlit as st
import pandas as pd
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit.components.v1 as components
import openai
from sklearn.linear_model import LinearRegression
import numpy as np

import json
import os

# --- 設定 ---
SPREADSHEET_NAME = 'muscle_db'
EXERCISES_FILE = 'exercises.json'

DEFAULT_EXERCISES = {
    "胸": ["ベンチプレス", "インクラインベンチプレス", "インクラインダンベルプレス", "ディップス", "ペックフライ", "マシンプレス"],
    "背中": ["デッドリフト", "フロントプル", "ラットプル", "ローロー", "チンニング"],
    "脚": ["スクワット", "レッグエクステンション", "レッグカール", "レッグプレス", "ブルガリアンスクワット"],
    "肩": ["サイドレイズ", "ダンベルショルダープレス", "バーベルショルダープレス"],
    "腕": ["スカルクラッシャー", "インクラインカール", "バーベルカール", "ケーブルプレスダウン"]
}

def load_exercises():
    if os.path.exists(EXERCISES_FILE):
        try:
            with open(EXERCISES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return DEFAULT_EXERCISES
    return DEFAULT_EXERCISES

def save_exercises(exercises):
    with open(EXERCISES_FILE, 'w', encoding='utf-8') as f:
        json.dump(exercises, f, ensure_ascii=False, indent=4)

def get_body_part(exercise_name):
    # セッションステートから取得、なければデフォルト
    exercises = st.session_state.get('exercises', DEFAULT_EXERCISES)
    for part, ex_list in exercises.items():
        if exercise_name in ex_list:
            return part
    return "その他"

def get_recovery_status(df):
    status = {}
    if df.empty:
        return status
    
            # ユーザー固有のデータでフィルタリング済みであることを前提とする
    # 各部位の最終トレーニング日を取得
    exercises_dict = st.session_state.get('exercises', DEFAULT_EXERCISES)
    for part in exercises_dict.keys():
        # その部位に関連する種目を抽出
        exercises = exercises_dict[part]
        part_df = df[df['種目名'].isin(exercises)]
        
        if not part_df.empty:
            last_date = pd.to_datetime(part_df['日付']).max()
            days_since = (datetime.datetime.now() - last_date).days
            status[part] = days_since
        else:
            status[part] = 999  # 未実施
            
    return status

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

def get_ai_agent_advice(df, mode):
    if df.empty:
        return "データがありません。まずは初回のトレーニングを記録しましょう！"

    # --- 1. 現状分析 (Context) ---
    df_sorted = df.sort_values('日付', ascending=False)
    last_record = df_sorted.iloc[0]
    last_date = pd.to_datetime(last_record['日付'])
    last_part = last_record['部位'] # 部位カラムを使う
    days_since = (datetime.datetime.now() - last_date).days
    
    # 回復状況計算
    recovery_status = get_recovery_status(df)
    # 値が999(未実施)を除外してソートするか、そのまま使うか。
    # ここでは未実施(999)は除外せずに、単純に日数が多い順(回復している順)に提案する
    sorted_recovery = sorted(recovery_status.items(), key=lambda x: x[1], reverse=True)
    recommended_part = sorted_recovery[0][0]

    # --- 2. モード別プロンプト分岐 ---
    # --- 2. モード別プロンプト分岐 ---
    if mode == "🔥 鬼軍曹":
        system_prompt = """
        あなたは地獄の鬼軍曹です。ユーザーは新兵です。
        甘えは一切許しません。以下の口調で、次に鍛えるべき部位を命令してください。
        
        【口調のルール】
        - 「貴様」「～だ！」「甘えるな！」などの強い言葉を使う。
        - 褒めない。煽ってやる気を引き出す。
        - 絵文字は🔥や💢のみ使用可。
        - 回復している部位（サボっている部位）を徹底的に攻めるよう命令する。
        - 100文字以内で短く怒鳴るように。
        """
        user_prompt = f"""
        新兵の状況: 前回 {days_since}日前に {last_part} を実施。
        最もサボっている部位: {recommended_part} ({recovery_status[recommended_part]}日経過)
        
        新兵を罵倒し、ジムへ叩き出してください。
        """

    elif mode == "✨ 励ましエンジェル":
        system_prompt = """
        あなたはユーザーを推している「アイドルのような天使」です。
        とにかくハイテンションで、ユーザーの努力を全肯定してください。
        
        【口調のルール】
        - 「すごい！」「えらい！」「優勝！」など、ポジティブな言葉を連発する。
        - 絵文字（✨💖🥺🎉）を多用する。
        - 回復している部位を「次はここを育てようね♡」と優しく提案する。
        - 120文字以内で、読むだけで元気がでるメッセージを。
        """
        status_text = "\n".join([f"- {k}: {v}日お休み中" for k, v in recovery_status.items() if v != 999])
        user_prompt = f"""
        推しの状況: 前回 {days_since}日前に {last_part} を頑張った！
        今の回復状況:\n{status_text}
        おすすめ: {recommended_part}
        
        最高の笑顔で応援してください。
        """

    elif mode == "🤖 システムOS":
        system_prompt = """
        あなたは近未来のトレーニング支援OS「LIFT OS」のシステムボイスです。
        感情を持たず、機械的かつクールに状況を報告してください。
        
        【口調のルール】
        - 「スキャン完了」「推奨」「プロトコル開始」などのSF用語を使う。
        - ユーザーを「パイロット」と呼ぶ。
        - 感情的な言葉は排除し、事実と推奨事項のみを伝える。
        - 100文字以内。
        """
        user_prompt = f"""
        Pilot Status: Last Workout {days_since} days ago ({last_part}).
        Target Recommendation: {recommended_part}.
        
        Generate mission briefing.
        """
    else:
         return "モードエラー: 不明なモードです"

    # --- 3. 生成 ---
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
            return "APIキー設定なし"
    except Exception as e:
        return f"エラー: {e}"

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
    if 'exercises' not in st.session_state:
        st.session_state['exercises'] = load_exercises()
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
    
    st.markdown('<div class="login-title" style="text-align: center;">PLUS ULTRA</div>', unsafe_allow_html=True)
    
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
        st.markdown(f'<div class="custom-title">PLUS ULTRA</div>', unsafe_allow_html=True)
    with c2:
        st.write(f"User: **{st.session_state['username']}**")
        if st.button("Logout", key="logout_btn", use_container_width=True):
            logout()
    # ★追加: サイドバーでAIモード設定
    with st.sidebar:
        st.markdown("### ⚙️ 設定")
        ai_mode = st.radio(
            "AIコーチングモード",
            ["✨ 励ましエンジェル", "🔥 鬼軍曹", "🤖 システムOS", "🤐 OFF"],
            index=0
        )
        st.divider()
        
        # --- 種目管理 ---
        with st.expander("🛠 種目管理"):
            st.caption("新しい種目の追加")
            new_ex_name = st.text_input("種目名", key="new_ex_name")
            new_ex_part = st.selectbox("部位", list(st.session_state['exercises'].keys()), key="new_ex_part")
            if st.button("追加", key="add_ex_btn"):
                if new_ex_name and new_ex_part:
                    if new_ex_name not in st.session_state['exercises'][new_ex_part]:
                        st.session_state['exercises'][new_ex_part].append(new_ex_name)
                        save_exercises(st.session_state['exercises'])
                        st.success(f"{new_ex_name} を追加しました")
                        st.rerun()
                    else:
                        st.warning("その種目は既に存在します")
            
            st.divider()
            st.caption("種目の削除")
            del_part = st.selectbox("部位選択", list(st.session_state['exercises'].keys()), key="del_part_select")
            del_ex = st.selectbox("削除する種目", st.session_state['exercises'][del_part], key="del_ex_select")
            if st.button("削除", key="del_ex_btn"):
                if del_ex in st.session_state['exercises'][del_part]:
                    st.session_state['exercises'][del_part].remove(del_ex)
                    save_exercises(st.session_state['exercises'])
                    st.success(f"{del_ex} を削除しました")
                    st.rerun()

    # 1. AIエージェントエリア (OFFなら表示しない)
    if ai_mode != "🤐 OFF":
        with st.container(border=True):
            col_ai_icon, col_ai_text = st.columns([1, 6])
            with col_ai_icon:
                st.image("https://api.dicebear.com/7.x/bottts/svg?seed=WorkoutAI", width=60)
            with col_ai_text:
                st.markdown('<div class="ai-title">AI Coach Agent</div>', unsafe_allow_html=True)
                if not df.empty:
                    # モードが変わったらアドバイスも再生成したいので、キーにモードを含める
                    advice_key = f'ai_advice_{ai_mode}'
                    
                    if advice_key not in st.session_state:
                         # 初回ロード時はとりあえずデフォルトメッセージ（API節約）
                         st.session_state[advice_key] = "今日も限界を超えていきましょう。" if ai_mode == "🔥 鬼軍曹" else "今日も頑張りましょう！"
                    
                    st.markdown(f'<div class="ai-message">{st.session_state[advice_key]}</div>', unsafe_allow_html=True)
                    
                    if st.button("アドバイスを更新", key="refresh_ai"):
                        with st.spinner("思考中..."):
                            # ★変更: モードを渡す
                            advice = get_ai_agent_advice(df, ai_mode)
                            st.session_state[advice_key] = advice
                            st.rerun()
                else:
                    st.markdown('<div class="ai-message">データがありません。</div>', unsafe_allow_html=True)

    # 2. ナビゲーション & フィルタ
    st.write("##### 部位フィルター")
    exercises_dict = st.session_state['exercises']
    parts = ["All"] + list(exercises_dict.keys())
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
        for p in exercises_dict:
            target_exercises.extend(exercises_dict[p])
    else:
        target_exercises = exercises_dict[target_part]

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
        if st.button("戻る"):
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