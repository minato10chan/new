import streamlit as st
import json
import csv
import io
from datetime import datetime
from src.services.pinecone_service import PineconeService
from src.services.langchain_service import LangChainService
from src.config.settings import (
    load_prompt_templates
)
import streamlit.components.v1 as components

def save_chat_history(messages, filename=None):
    """チャット履歴をCSVファイルとして保存"""
    if filename is None:
        filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # CSVデータを作成
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)  # すべてのフィールドをクォート
    writer.writerow(["timestamp", "role", "content", "details"])
    
    for message in messages:
        # 行を書き込み
        writer.writerow([
            message.get("timestamp", datetime.now().isoformat()),  # 既存のタイムスタンプがあれば使用
            message["role"],
            message["content"],
            json.dumps(message.get("details", {}), ensure_ascii=False) if "details" in message else ""
        ])
    
    return output.getvalue(), filename

def load_chat_history(file):
    """チャット履歴をCSVファイルから読み込み"""
    messages = []
    # ファイルをテキストモードで読み込む
    content = file.getvalue().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    
    for row in reader:
        message = {
            "timestamp": row["timestamp"],
            "role": row["role"],
            "content": row["content"]
        }
        
        # detailsが存在する場合はJSONとしてパース
        if row["details"] and row["details"].strip():
            try:
                message["details"] = json.loads(row["details"])
            except json.JSONDecodeError:
                message["details"] = {}
        
        messages.append(message)
    
    return messages

@st.cache_data(ttl=300)
def get_property_list(pinecone_service: PineconeService) -> list:
    """物件情報の一覧を取得"""
    try:
        # Pineconeから物件情報の一覧を取得
        results = pinecone_service.list_vectors(namespace="property")
        
        if not results:
            return []
            
        properties = []
        for match in results:
            # テキストから物件情報を抽出
            text = match.metadata["text"]
            lines = text.split('\n')
            
            # 物件名と場所を抽出（最初の2行を想定）
            name = lines[0].strip() if len(lines) > 0 else "不明"
            location = lines[1].strip() if len(lines) > 1 else "不明"
            
            properties.append({
                "id": match.id,
                "name": name,
                "location": location,
                "text": text
            })
            
        return properties
    except Exception as e:
        st.error(f"物件情報の取得中にエラーが発生しました: {str(e)}")
        return []

def get_property_info(property_id: str, pinecone_service: PineconeService) -> str:
    """選択された物件の詳細情報を取得"""
    try:
        # Pineconeから物件情報を取得
        result = pinecone_service.get_by_id(property_id, namespace="property")
        
        if not result:
            return "物件情報が見つかりませんでした。"
            
        # テキストを取得
        return result.get("text", "物件情報が見つかりませんでした。")
    except Exception as e:
        return f"物件情報の取得中にエラーが発生しました: {str(e)}"

def render_chat(pinecone_service: PineconeService):
    """チャット機能のUIを表示"""
    st.title("チャット")
    st.write("アップロードしたドキュメントについて質問できます。")

    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # LangChainサービスの初期化
    if "langchain_service" not in st.session_state:
        from streamlit_app import callback_manager
        st.session_state.langchain_service = LangChainService(callback_manager=callback_manager)
        # API使用状況の確認
        st.session_state.langchain_service.check_api_usage()
    
    # プロンプトテンプレートの読み込み（毎回最新の状態を取得）
    prompt_templates, _, _ = load_prompt_templates()
    st.session_state.prompt_templates = prompt_templates
    
    # サイドバーに履歴管理機能を配置
    with st.sidebar:
        st.header("チャット履歴管理")
        
        # 会話履歴の最適化状態を表示
        if "langchain_service" in st.session_state:
            history_tokens = sum(st.session_state.langchain_service.count_tokens(msg.content) 
                               for msg in st.session_state.langchain_service.message_history.messages)
            max_tokens = 12000  # LangChainServiceのoptimize_chat_historyと同じ値
            optimization_status = "最適化済み" if history_tokens <= max_tokens else "最適化が必要"
            
            st.write(f"会話履歴の状態:")
            st.write(f"- トークン数: {history_tokens:,} / {max_tokens:,}")
            st.write(f"- 最適化状態: {optimization_status}")
            st.write(f"- メッセージ数: {len(st.session_state.messages)}")
        
        # プロンプトテンプレートの選択
        st.header("プロンプトテンプレート")
        template_names = [template["name"] for template in st.session_state.prompt_templates]
        selected_template = st.selectbox(
            "使用するテンプレートを選択",
            template_names,
            index=0
        )
        
        # 選択されたテンプレートの内容を表示
        selected_template_data = next(
            template for template in st.session_state.prompt_templates 
            if template["name"] == selected_template
        )
        st.subheader("選択中のテンプレート")
        template_tab1, template_tab2 = st.tabs(["システムプロンプト", "応答テンプレート"])
        with template_tab1:
            st.text_area("システムプロンプト", value=selected_template_data["system_prompt"], disabled=True)
        with template_tab2:
            st.text_area("応答テンプレート", value=selected_template_data["response_template"], disabled=True)
            
        # 物件情報の選択
        st.header("物件情報")
        properties = get_property_list(pinecone_service)
        
        if properties:
            # 物件の選択肢を作成（物件名と場所を表示）
            property_options = [f"{p['name']} - {p['location']}" for p in properties]
            selected_property = st.selectbox(
                "物件を選択",
                options=property_options,
                index=0
            )
            
            # 選択された物件のIDを取得
            selected_property_id = properties[property_options.index(selected_property)]["id"]
            
            # 選択された物件の詳細情報を取得
            st.session_state.property_info = get_property_info(selected_property_id, pinecone_service)
            
            # 物件の詳細情報を表示
            st.subheader("選択中の物件情報")
            st.markdown(st.session_state.property_info)
        else:
            st.warning("物件情報が登録されていません。")
            st.session_state.property_info = "物件情報が登録されていません。"
        
        # 履歴の保存 (ローカルダウンロード)
        st.write(f"現在のメッセージ数: {len(st.session_state.messages)}")
        if len(st.session_state.messages) > 0:
            csv_data, filename = save_chat_history(st.session_state.messages)
            st.download_button(
                label="履歴をダウンロード",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_history"
            )
        else:
            st.button("履歴をダウンロード", disabled=True, key="download_history_disabled")
        
        # 履歴の読み込み
        uploaded_file = st.file_uploader("保存した履歴を読み込む", type=['csv'])
        if uploaded_file is not None and "load_history" not in st.session_state:
            try:
                # 新しい履歴を読み込む
                loaded_messages = load_chat_history(uploaded_file)
                
                # セッション状態を更新
                st.session_state.messages = loaded_messages.copy()
                
                # LangChainの会話履歴を更新
                st.session_state.langchain_service.clear_memory()
                for message in loaded_messages:
                    if message["role"] == "user":
                        st.session_state.langchain_service.message_history.add_user_message(message["content"])
                    elif message["role"] == "assistant":
                        st.session_state.langchain_service.message_history.add_ai_message(message["content"])
                
                st.session_state.load_history = True
                st.success("履歴を読み込みました")
                st.rerun()
            except Exception as e:
                st.error(f"履歴の読み込みに失敗しました: {str(e)}")
        
        # 履歴のクリア
        if st.button("履歴をクリア"):
            st.session_state.messages = []
            st.session_state.langchain_service.clear_memory()
            if "load_history" in st.session_state:
                del st.session_state.load_history
            st.success("履歴をクリアしました")
            st.rerun()
    
    # メインコンテンツ
    # メインのチャット表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "details" in message and message["details"]:
                # 詳細情報を表示するボタン
                if st.button("詳細情報を表示", key=f"details_{message['timestamp']}"):
                    details = message["details"]
                    
                    # タブを使用して詳細情報を表示
                    tabs = st.tabs(["トークン数", "送信テキスト", "その他の情報"])
                    
                    # トークン数タブ
                    with tabs[0]:
                        if "トークン数" in details:
                            st.json(details["トークン数"])
                    
                    # 送信テキストタブ
                    with tabs[1]:
                        if "送信テキスト" in details:
                            sent_text = details["送信テキスト"]
                            st.text_area("システムプロンプト", sent_text["システムプロンプト"], height=100)
                            st.text_area("チャット履歴", "\n".join([f"[{msg['type']}]: {msg['content']}" for msg in sent_text["チャット履歴"]]), height=200)
                            st.text_area("参照文脈", sent_text["参照文脈"], height=100)
                            
                            # 参照文脈の詳細情報を表示
                            if "参照文脈の詳細" in sent_text:
                                st.markdown("**参照文脈の詳細**")
                                for i, detail in enumerate(sent_text["参照文脈の詳細"], 1):
                                    st.markdown(f"**参照 {i}**")
                                    st.write(f"ファイル名: {detail['ファイル名']}")
                                    st.write(f"ページ番号: {detail['ページ番号']}")
                                    st.write(f"セクション: {detail['セクション']}")
                                    st.write(f"スコア: {detail['スコア']}")
                                    st.text_area(f"テキスト {i}", detail['テキスト'], height=100)
                            
                            st.text_area("物件情報", sent_text["物件情報"], height=100)
                            st.text_area("ユーザー入力", sent_text["ユーザー入力"], height=100)
                    
                    # その他の情報タブ
                    with tabs[2]:
                        other_details = {k: v for k, v in details.items() 
                                      if k not in ["トークン数", "送信テキスト"]}
                        if other_details:
                            st.json(other_details)

    # ユーザー入力
    if prompt := st.chat_input("メッセージを入力してください"):
        # ユーザーメッセージを追加
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # 選択されたテンプレートを取得
        selected_template_data = next(
            template for template in st.session_state.prompt_templates 
            if template["name"] == selected_template
        )
        
        # LangChainを使用して応答を生成
        with st.spinner("応答を生成中..."):
            # 会話履歴をLangChainのメッセージ形式に変換
            chat_history = []
            for msg in st.session_state.messages:  # すべてのメッセージを含める
                if msg["role"] == "user":
                    chat_history.append(("human", msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(("ai", msg["content"]))
            
            # 会話履歴を逆順にして、最新の会話から処理
            chat_history.reverse()
            
            response, details = st.session_state.langchain_service.get_response(
                prompt,
                system_prompt=selected_template_data["system_prompt"],
                response_template=selected_template_data["response_template"],
                property_info=st.session_state.get("property_info", "物件情報はありません。"),
                chat_history=chat_history  # 会話履歴を渡す
            )
            
            # アシスタントの応答を追加
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "details": details,
                "timestamp": datetime.now().isoformat()
            })
        
        st.rerun()  