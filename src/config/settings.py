"""
アプリケーションの設定を管理するモジュール
"""

import streamlit as st
import os
import sys
import json
from dotenv import load_dotenv
from datetime import datetime

# 環境変数の読み込み
load_dotenv()

# API Keys
try:
    if 'streamlit' in sys.modules:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("pinecone_key", None)
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai_api_key", None)
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or st.secrets.get("index_name", "default-index")
        PINECONE_ASSISTANT_NAME = os.getenv("PINECONE_ASSISTANT_NAME") or st.secrets.get("assistant_name", "default-assistant")
    else:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "default-index")
        PINECONE_ASSISTANT_NAME = os.getenv("PINECONE_ASSISTANT_NAME", "default-assistant")
except Exception:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "default-index")
    PINECONE_ASSISTANT_NAME = os.getenv("PINECONE_ASSISTANT_NAME", "default-assistant")

# Text Processing Settings
CHUNK_SIZE = 500  # テキストを分割する際の1チャンクあたりの文字数
BATCH_SIZE = 100  # Pineconeへのアップロード時のバッチサイズ
PROPERTY_MAX_TOKENS = 8000  # 物件情報チャンクの最大トークン数（Pineconeの40KB制限に基づく）

# OpenAI Settings
EMBEDDING_MODEL = "text-embedding-3-large"  # 使用する埋め込みモデル
EMBEDDING_DIMENSION = 3072  # 埋め込みベクトルの次元数

# Search Settings
DEFAULT_TOP_K = 10  # デフォルトの検索結果数
SIMILARITY_THRESHOLD = 0.7  # 類似度のしきい値（0-1の範囲）

# Metadata Settings
DEFAULT_CREATION_DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # メタデータの作成日が空の場合のデフォルト値

def get_metadata_creation_date(metadata):
    """
    メタデータの作成日を取得する関数
    作成日が空の場合は現在の日時を返す
    """
    if not metadata or "creation_date" not in metadata or not metadata["creation_date"]:
        return DEFAULT_CREATION_DATE
    return metadata["creation_date"]

# Prompt Settings
# プロンプトテンプレートの保存と読み込み
PROMPT_TEMPLATES_FILE = "prompt_templates.json"

def save_prompt_templates(templates):
    """プロンプトテンプレートを保存"""
    with open(PROMPT_TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

def load_prompt_templates():
    """プロンプトテンプレートを読み込み"""
    if os.path.exists(PROMPT_TEMPLATES_FILE):
        with open(PROMPT_TEMPLATES_FILE, "r", encoding="utf-8") as f:
            templates = json.load(f)
            # デフォルトテンプレートを取得
            default_template = next((t for t in templates if t["name"] == "デフォルト"), None)
            if default_template:
                return templates, default_template["system_prompt"], default_template["response_template"]
    return [], "", ""

# プロンプトテンプレートの読み込み
PROMPT_TEMPLATES, DEFAULT_SYSTEM_PROMPT, DEFAULT_RESPONSE_TEMPLATE = load_prompt_templates()

# メタデータ設定
METADATA_CATEGORIES = {
    "大カテゴリ": [
        "物件概要",
        "地域特性・街のプロフィール",
        "教育・子育て",
        "交通・アクセス",
        "安全・防災",
        "行政施策・政策",
        "生活利便性",
        "不動産市場",
        "地域コミュニティ",
        "その他（個別の懸念・特殊事情）"
    ],
    "中カテゴリ": {
        "物件概要": [
            "完成時期",
            "販売開始",
            "建築確認",
            "間取り・仕様",
            "設備・オプション",
            "デザイン・外観",
            "建築会社・デベロッパー",
            "価格・費用",
            "資産価値・売却",
            "契約・手続き"
        ],
        "地域特性・街のプロフィール": [
            "概要・エリア区分",
            "人口・居住特性",
            "街の歴史・地域史",
            "地理的特性",
            "自然環境",
            "地域イベント・伝統行事",
            "都市連携・姉妹都市情報",
            "治安・騒音・環境整備",
            "風景・景観・街並み",
            "観光・地元特産品・名産・グルメ"
        ],
        "教育・子育て": [
            "保育園・幼稚園",
            "小学校・中学校",
            "学童・放課後支援",
            "習い事・塾",
            "子育て支援制度",
            "公園・遊び場",
            "病院・小児科",
            "学区・教育水準",
            "待機児童・入園状況",
            "学校イベント・行事"
        ],
        "交通・アクセス": [
            "最寄り駅・路線",
            "電車の混雑状況",
            "バス路線・本数",
            "駐輪場・駐車場",
            "道路交通量・渋滞",
            "車移動のしやすさ",
            "通勤・通学時間",
            "高速道路・インター",
            "タクシー・ライドシェア",
            "空港・新幹線アクセス"
        ],
        "安全・防災": [
            "防犯カメラ・交番の有無",
            "避難場所・防災拠点",
            "ハザードマップ（洪水・地震）",
            "土砂災害リスク",
            "耐震性・建物強度",
            "火災リスク・消防体制",
            "夜道の安全性",
            "台風・風害・雪害対策",
            "地震・液状化リスク",
            "交通事故・子ども見守り"
        ],
        "行政施策・政策": [
            "市政・行政組織",
            "再開発・都市計画",
            "交通インフラ整備",
            "公共施設運営・市民サービス",
            "ゴミ収集・清掃環境",
            "商業・産業振興策",
            "住宅政策・住環境整備",
            "福祉・医療支援",
            "補助金・助成制度",
            "行政評価・市民参加"
        ],
        "生活利便性": [
            "スーパー・買い物環境",
            "コンビニ・ドラッグストア",
            "銀行・金融機関・郵便局",
            "公共施設",
            "病院・クリニック・夜間救急",
            "文化施設・美術館・劇場",
            "スポーツ施設・ジム",
            "娯楽施設・カラオケ・映画館",
            "飲食店・グルメスポット",
            "宅配サービス・ネットスーパー"
        ],
        "不動産市場": [
            "地価の変動・推移",
            "将来の売却しやすさ",
            "賃貸需要・投資価値",
            "住宅ローン・金利動向",
            "人気エリアの傾向",
            "空き家・中古市場動向",
            "固定資産税・税制優遇",
            "マンション vs 戸建て",
            "住み替え・転勤時の影響",
            "不動産会社の評判・実績"
        ],
        "地域コミュニティ": [
            "自治会・町内会の活動",
            "地域の祭り",
            "ボランティア活動",
            "住民の意識・口コミ",
            "高齢者支援・福祉施設",
            "外国人コミュニティ",
            "風評・悪評の実態",
            "市民幸福度・満足度",
            "地域振興・コミュニティ活性化"
        ],
        "その他（個別の懸念・特殊事情）": [
            "スマートシティ・DX施策",
            "エコ対策・再生可能エネルギー",
            "観光客・短期滞在者の影響",
            "景観条例・建築規制",
            "駐車場問題・車両ルール",
            "宅配便・物流インフラ"
        ]
    },
    "市区町村": [
        "川越市",
        "さいたま市",
        "千葉市",
        "横浜市",
        "川崎市",
        "相模原市",
    ]
}            