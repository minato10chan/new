import streamlit as st
import subprocess
import threading
import sys
import os

# srcディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.text_processing import process_text_file
from src.services.pinecone_service import PineconeService
from src.components.file_upload import render_file_upload
from src.components.chat import render_chat
from src.components.settings import render_settings
from src.components.agent import render_agent
from src.components.property_upload import render_property_upload
from src.config.settings import DEFAULT_SYSTEM_PROMPT, DEFAULT_RESPONSE_TEMPLATE
import os
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

# LangSmithの設定
client = Client()
tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "response_template" not in st.session_state:
    st.session_state.response_template = DEFAULT_RESPONSE_TEMPLATE

# Pineconeサービスの初期化 - session stateでキャッシュ
if "pinecone_service" not in st.session_state:
    try:
        st.session_state.pinecone_service = PineconeService()
    except Exception as e:
        st.error(f"Pineconeサービスの初期化に失敗しました: {str(e)}")
        st.stop()

pinecone_service = st.session_state.pinecone_service

# インデックスの状態を確認（キャッシュされた結果を使用）
if "index_stats" not in st.session_state:
    st.session_state.index_stats = pinecone_service.get_index_stats()
stats = st.session_state.index_stats

if stats['total_vector_count'] == 0:
    st.info("データベースは空です。物件情報を登録してください。")
else:
    st.write(f"データベースの状態: {stats['total_vector_count']}件のドキュメント")

def read_file_content(file) -> str:
    """ファイルの内容を適切なエンコーディングで読み込む"""
    encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp']
    content = file.getvalue()
    
    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    raise ValueError("ファイルのエンコーディングを特定できませんでした。UTF-8、Shift-JIS、CP932、EUC-JPのいずれかで保存されているファイルをアップロードしてください。")

def start_flask_server():
    """Flaskサーバーを起動する"""
    subprocess.run(["python", "reacttest.py"])

# Flaskサーバーを別スレッドで起動
flask_thread = threading.Thread(target=start_flask_server)
flask_thread.daemon = True
flask_thread.start()

def main():
    # サイドバーにメニューを配置
    with st.sidebar:
        st.title("管理者メニュー")
        page = st.radio(
            "機能を選択",
            ["チャット", "物件情報登録", "ファイルアップロード", "設定", "Agent"],
            index={
                "chat": 0,
                "property": 1,
                "upload": 2,
                "settings": 3,
                "agent": 4
            }[st.session_state.current_page]
        )
        st.session_state.current_page = {
            "チャット": "chat",
            "物件情報登録": "property",
            "ファイルアップロード": "upload",
            "設定": "settings",
            "Agent": "agent"
        }[page]

    # メインコンテンツの表示
    if st.session_state.current_page == "chat":
        render_chat(pinecone_service)
    elif st.session_state.current_page == "property":
        render_property_upload(pinecone_service)
    elif st.session_state.current_page == "upload":
        render_file_upload(pinecone_service)
    elif st.session_state.current_page == "agent":
        render_agent(pinecone_service)
    else:
        render_settings(pinecone_service)

if __name__ == "__main__":
    main()
