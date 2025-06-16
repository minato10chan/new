#!/usr/bin/env python3
"""
Standalone test script for property text chunking functionality
"""
import sys
import os
import json
import tiktoken

def split_property_data_test(property_data: dict, max_tokens: int = 8000) -> list:
    """物件データを複数のチャンクに分割する（テスト用簡素化版）"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    
    base_info = {
        "property_name": property_data["property_name"],
        "property_type": property_data["property_type"],
        "prefecture": property_data["prefecture"],
        "city": property_data["city"],
        "detailed_address": property_data["detailed_address"],
        "latitude": property_data.get("latitude", "0.0"),
        "longitude": property_data.get("longitude", "0.0")
    }
    
    details = property_data.get("property_details", "")
    if not details:
        return [{"text": json.dumps(base_info, ensure_ascii=False), "metadata": base_info}]
    
    base_info_text = json.dumps(base_info, ensure_ascii=False)
    base_tokens = len(encoding.encode(base_info_text))
    
    available_tokens = max_tokens - base_tokens - 100  # 100トークンはマージンとメタデータ用
    
    print(f"基本情報トークン数: {base_tokens}")
    print(f"詳細情報用利用可能トークン数: {available_tokens}")
    
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
            if current_chunk_paragraphs:
                chunk_info = base_info.copy()
                chunk_info["property_details"] = "\n".join(current_chunk_paragraphs)
                chunk_info["chunk_number"] = len(chunks) + 1
                
                chunk = {
                    "text": json.dumps(chunk_info, ensure_ascii=False),
                    "metadata": chunk_info
                }
                chunks.append(chunk)
            
            current_chunk_paragraphs = [paragraph]
            current_tokens = paragraph_tokens
        else:
            current_chunk_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens
    
    if current_chunk_paragraphs:
        chunk_info = base_info.copy()
        chunk_info["property_details"] = "\n".join(current_chunk_paragraphs)
        chunk_info["chunk_number"] = len(chunks) + 1
        
        chunk = {
            "text": json.dumps(chunk_info, ensure_ascii=False),
            "metadata": chunk_info
        }
        chunks.append(chunk)
    
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    print(f"最終的なチャンク数: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(encoding.encode(chunk["text"]))
        print(f"チャンク {i+1} のトークン数: {chunk_tokens}")
        if chunk_tokens > max_tokens:
            print(f"警告: チャンク {i+1} がmax_tokens({max_tokens})を超えています")
    
    return chunks

def test_chunking():
    """Test the chunking logic with various text lengths"""
    encoding = tiktoken.encoding_for_model('text-embedding-3-large')
    max_tokens = 8000
    
    long_text = """
この物件は東京都心部に位置する高級マンションです。最寄り駅から徒歩3分という抜群の立地条件を誇り、都心へのアクセスが非常に便利です。建物は地上20階建ての鉄筋コンクリート造で、2022年に竣工した新築物件です。

外観は洗練されたモダンデザインを採用し、エントランスには24時間有人管理のコンシェルジュサービスが配置されています。セキュリティ面では、オートロック、防犯カメラ、宅配ボックスなど最新の設備が完備されています。

各住戸は南向きの角部屋を中心とした設計で、大きな窓から豊富な自然光が差し込みます。室内は高級感のある内装仕上げで、システムキッチン、浴室乾燥機、床暖房、エアコンなどの設備が標準装備されています。

共用施設として、屋上庭園、フィットネスジム、ゲストルーム、キッズルーム、パーティールームなどが用意されており、住民の快適な生活をサポートします。また、敷地内には来客用駐車場も完備されています。

周辺環境は非常に充実しており、徒歩圏内にスーパーマーケット、コンビニエンスストア、銀行、郵便局、病院、薬局などの生活に必要な施設が揃っています。また、近隣には有名な公園があり、緑豊かな環境でリラックスできます。

教育環境も優れており、評判の良い小学校、中学校が学区内にあります。また、有名私立学校へのアクセスも良好で、子育て世代にとって理想的な環境です。保育園や幼稚園も複数あり、待機児童の心配も少ない地域です。

交通アクセスは複数路線が利用可能で、主要ターミナル駅まで乗り換えなしでアクセスできます。朝の通勤ラッシュ時でも比較的混雑が少なく、快適な通勤が可能です。また、羽田空港や成田空港へのアクセスも良好で、出張や旅行の際にも便利です。

商業施設も充実しており、大型ショッピングモール、デパート、専門店街などが近隣にあります。レストラン、カフェ、居酒屋なども多数あり、外食やエンターテイメントにも困りません。

将来性についても、この地域は再開発計画が進行中で、さらなる発展が期待されています。新しい商業施設や公共施設の建設も予定されており、資産価値の向上も見込まれます。

管理体制は信頼できる大手管理会社が担当し、建物の維持管理、清掃、設備点検などが適切に行われています。管理費や修繕積立金も適正な水準に設定されており、長期的な資産価値の維持が期待できます。

この物件は投資用としても魅力的で、賃貸需要が高い立地条件により、安定した賃料収入が見込めます。また、将来的な売却時にも高い流動性が期待できる優良物件です。

価格については、同エリアの類似物件と比較して競争力のある設定となっており、コストパフォーマンスに優れています。住宅ローンの金利優遇制度も利用可能で、購入しやすい条件が整っています。

内見は随時受け付けており、実際の住環境を確認していただけます。また、詳細な資料や図面、周辺環境の情報なども提供可能です。ご興味をお持ちの方は、お気軽にお問い合わせください。

この物件は限定販売のため、早期の検討をお勧めします。人気の高い立地条件と充実した設備により、完売が予想される注目の物件です。
""" * 10  # Make it 10x longer to test chunking

    property_data = {
        'property_name': 'テスト高級マンション',
        'property_type': 'マンション',
        'prefecture': '東京都',
        'city': '渋谷区',
        'detailed_address': '渋谷1-1-1',
        'property_details': long_text,
        'latitude': '35.6580',
        'longitude': '139.7016'
    }

    print(f"=== Property Text Chunking Test ===")
    print(f"Original property_details length: {len(long_text)} characters")
    print(f"Max tokens setting: {max_tokens}")
    
    json_str = json.dumps(property_data, ensure_ascii=False)
    json_bytes = json_str.encode('utf-8')
    
    print(f"Full JSON string length: {len(json_str)} characters")
    print(f"Full JSON bytes size: {len(json_bytes)} bytes")
    print(f"Pinecone limit: 40KB = {40 * 1024} bytes")
    
    if len(json_bytes) > 40 * 1024:
        print("❌ PROBLEM: Full JSON exceeds Pinecone's 40KB metadata limit!")
        print(f"Excess: {len(json_bytes) - 40 * 1024} bytes")
        print("✅ This is why chunking is needed")
    else:
        print("✅ Full JSON fits within Pinecone's 40KB metadata limit")
    
    tokens = len(encoding.encode(json_str))
    print(f"Full JSON token count: {tokens} tokens")
    
    print(f"\n=== Testing Chunking Function ===")
    
    try:
        chunks = split_property_data_test(property_data, max_tokens)
        print(f"✅ Successfully created {len(chunks)} chunks")
        
        all_chunks_valid = True
        total_size = 0
        
        for i, chunk in enumerate(chunks):
            chunk_json = chunk["text"]
            chunk_bytes = chunk_json.encode('utf-8')
            chunk_tokens = len(encoding.encode(chunk_json))
            
            print(f"Chunk {i+1}: {len(chunk_json)} chars, {len(chunk_bytes)} bytes, {chunk_tokens} tokens")
            total_size += len(chunk_bytes)
            
            if len(chunk_bytes) > 40 * 1024:
                print(f"❌ WARNING: Chunk {i+1} exceeds 40KB limit!")
                all_chunks_valid = False
            
            if chunk_tokens > max_tokens:
                print(f"❌ WARNING: Chunk {i+1} exceeds token limit!")
                all_chunks_valid = False
        
        print(f"\nTotal size of all chunks: {total_size} bytes")
        
        if all_chunks_valid:
            print("✅ All chunks are within limits!")
        else:
            print("❌ Some chunks exceed limits!")
            
        return chunks
        
    except Exception as e:
        print(f"❌ ERROR in chunking: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_short_text():
    """Test with short text that shouldn't need chunking"""
    print(f"\n=== Testing Short Text ===")
    
    short_property_data = {
        'property_name': 'シンプル物件',
        'property_type': 'マンション',
        'prefecture': '東京都',
        'city': '渋谷区',
        'detailed_address': '渋谷2-2-2',
        'property_details': '駅近の便利な物件です。',
        'latitude': '35.6580',
        'longitude': '139.7016'
    }
    
    try:
        chunks = split_property_data_test(short_property_data, 8000)
        print(f"✅ Short text created {len(chunks)} chunk(s)")
        
        if len(chunks) == 1:
            print("✅ Short text correctly created single chunk")
        else:
            print(f"❌ Short text unexpectedly created {len(chunks)} chunks")
            
        return chunks
        
    except Exception as e:
        print(f"❌ ERROR with short text: {e}")
        return None

if __name__ == "__main__":
    print("Starting property text chunking tests...")
    
    long_chunks = test_chunking()
    
    short_chunks = test_short_text()
    
    print(f"\n=== Test Summary ===")
    if long_chunks and short_chunks:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
