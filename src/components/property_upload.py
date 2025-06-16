import streamlit as st
from src.services.pinecone_service import PineconeService
import pandas as pd
import json
import traceback
from datetime import datetime
import tiktoken
from src.config.settings import PROPERTY_MAX_TOKENS

# 都道府県と市区町村のデータ
PREFECTURES = [
    "埼玉県", "千葉県", "東京都", "神奈川県"
]

# 主要な市区町村のデータ（例として東京都の区を記載）
CITIES = {
    "埼玉県": [
        "川越市", "さいたま市"
    ],
    # 他の都道府県の市区町村も同様に追加可能
}

def split_property_data(property_data: dict, max_tokens: int = PROPERTY_MAX_TOKENS) -> list:
    """物件データを複数のチャンクに分割する（簡素化版）"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    
    # 基本情報（常に含める）
    base_info = {
        "property_name": property_data["property_name"],
        "property_type": property_data["property_type"],
        "prefecture": property_data["prefecture"],
        "city": property_data["city"],
        "detailed_address": property_data["detailed_address"],
        "latitude": property_data.get("latitude", "0.0"),
        "longitude": property_data.get("longitude", "0.0")
    }
    
    # 詳細情報を分割
    details = property_data.get("property_details", "")
    if not details:
        return [{"text": json.dumps(base_info, ensure_ascii=False), "metadata": base_info}]
    
    base_info_text = json.dumps(base_info, ensure_ascii=False)
    base_tokens = len(encoding.encode(base_info_text))
    
    available_tokens = max_tokens - base_tokens - 100  # 100トークンはマージンとメタデータ用
    
    print(f"基本情報トークン数: {base_tokens}")
    print(f"詳細情報用利用可能トークン数: {available_tokens}")
    
    # 詳細情報を段落で分割
    paragraphs = [p.strip() for p in details.split('\n') if p.strip()]
    print(f"段落数: {len(paragraphs)}")
    
    chunks = []
    current_chunk_paragraphs = []
    current_tokens = 0
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_tokens = len(encoding.encode(paragraph))
        print(f"段落 {i+1}/{len(paragraphs)} のトークン数: {paragraph_tokens}")
        
        if paragraph_tokens > available_tokens:
            print(f"段落が大きすぎるため文字数で分割: {paragraph_tokens} > {available_tokens}")
            
            # 現在のチャンクを保存（空でない場合）
            if current_chunk_paragraphs:
                chunk_info = base_info.copy()
                chunk_info["property_details"] = "\n".join(current_chunk_paragraphs)
                chunk_info["chunk_number"] = len(chunks) + 1
                
                chunk = {
                    "text": json.dumps(chunk_info, ensure_ascii=False),
                    "metadata": chunk_info
                }
                chunks.append(chunk)
                current_chunk_paragraphs = []
                current_tokens = 0
            
            chars_per_token = len(paragraph) / paragraph_tokens if paragraph_tokens > 0 else 1
            max_chars = int(available_tokens * chars_per_token * 0.9)  # 90%マージン
            
            for start in range(0, len(paragraph), max_chars):
                sub_paragraph = paragraph[start:start + max_chars]
                
                chunk_info = base_info.copy()
                chunk_info["property_details"] = sub_paragraph
                chunk_info["chunk_number"] = len(chunks) + 1
                
                chunk = {
                    "text": json.dumps(chunk_info, ensure_ascii=False),
                    "metadata": chunk_info
                }
                chunks.append(chunk)
            
        elif current_tokens + paragraph_tokens > available_tokens:
            # 現在のチャンクを保存
            if current_chunk_paragraphs:
                chunk_info = base_info.copy()
                chunk_info["property_details"] = "\n".join(current_chunk_paragraphs)
                chunk_info["chunk_number"] = len(chunks) + 1
                
                chunk = {
                    "text": json.dumps(chunk_info, ensure_ascii=False),
                    "metadata": chunk_info
                }
                chunks.append(chunk)
            
            # 新しいチャンクを開始
            current_chunk_paragraphs = [paragraph]
            current_tokens = paragraph_tokens
        else:
            # 現在のチャンクに追加
            current_chunk_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens
    
    # 最後のチャンクを処理
    if current_chunk_paragraphs:
        chunk_info = base_info.copy()
        chunk_info["property_details"] = "\n".join(current_chunk_paragraphs)
        chunk_info["chunk_number"] = len(chunks) + 1
        
        chunk = {
            "text": json.dumps(chunk_info, ensure_ascii=False),
            "metadata": chunk_info
        }
        chunks.append(chunk)
    
    # 総チャンク数を更新
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    # 各チャンクのトークン数を確認
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(encoding.encode(chunk["text"]))
        if chunk_tokens > max_tokens:
            print(f"警告: チャンク {i+1} がmax_tokens({max_tokens})を超えています: {chunk_tokens} tokens")
    
    return chunks

def render_property_upload(pinecone_service: PineconeService):
    """物件情報のアップロードUIを表示"""
    st.title("🏠 物件情報のアップロード")
    
    with st.form("property_upload_form"):
        st.markdown("### 物件情報の入力")
        
        # 物件名
        property_name = st.text_input("物件名", help="物件の名称を入力してください")
        
        # 物件種別
        property_type = st.selectbox(
            "物件種別",
            ["一戸建て", "土地", "マンション"],
            help="物件の種別を選択してください"
        )
        
        # 都道府県と市区町村の選択
        col1, col2 = st.columns(2)
        with col1:
            prefecture = st.selectbox(
                "都道府県",
                PREFECTURES,
                help="物件の所在地の都道府県を選択してください"
            )
        
        with col2:
            # 選択された都道府県に基づいて市区町村のリストを表示
            cities = CITIES.get(prefecture, [])
            city = st.selectbox(
                "市区町村",
                cities,
                help="物件の所在地の市区町村を選択してください"
            )
        
        # 詳細住所
        detailed_address = st.text_input("詳細住所", help="物件の詳細な住所を入力してください")
        
        # 物件の詳細情報
        property_details = st.text_area(
            "物件の詳細情報",
            help="物件の詳細な情報を入力してください（長い文章は自動的に分割されます）"
        )
        
        # 緯度・経度
        col3, col4 = st.columns(2)
        with col3:
            latitude = st.text_input(
                "緯度",
                value="0.0",
                help="物件の緯度を入力してください"
            )
        with col4:
            longitude = st.text_input(
                "経度",
                value="0.0",
                help="物件の経度を入力してください"
            )
        
        # アップロードボタン
        submit_button = st.form_submit_button("アップロード")
        
        if submit_button:
            try:
                # 必須項目のチェック
                if not all([property_name, property_type, prefecture, city]):
                    st.error("❌ 必須項目（物件名、物件種別、都道府県、市区町村）を入力してください")
                    return
                
                # 物件情報の構造化
                property_data = {
                    "property_name": property_name,
                    "property_type": property_type,
                    "prefecture": prefecture,
                    "city": city,
                    "detailed_address": detailed_address,
                    "property_details": property_details,
                    "latitude": latitude,
                    "longitude": longitude
                }
                
                # 物件データをチャンクに分割
                chunks = split_property_data(property_data)
                
                # チャンクごとにIDを付与
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                for i, chunk in enumerate(chunks):
                    chunk["id"] = f"property_{timestamp}_{i}"
                
                # Pineconeへのアップロード
                pinecone_service.upload_chunks(chunks, namespace="property")
                
                st.success(f"✅ 物件情報を{len(chunks)}件のチャンクに分割してアップロードしました")
                
            except Exception as e:
                st.error(f"❌ アップロードに失敗しました: {str(e)}")
                st.error(f"🔍 エラーの詳細: {type(e).__name__}")
                st.error(f"📜 スタックトレース:\n{traceback.format_exc()}")    