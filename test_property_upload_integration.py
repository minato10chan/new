#!/usr/bin/env python3
"""
Integration test for property upload with long text
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_property_upload_integration():
    """Test the complete property upload workflow"""
    print("=== Property Upload Integration Test ===")
    
    try:
        from src.components.property_upload import split_property_data
        from src.config.settings import PROPERTY_MAX_TOKENS
        
        print(f"✅ Successfully imported modules")
        print(f"PROPERTY_MAX_TOKENS: {PROPERTY_MAX_TOKENS}")
        
        long_description = """
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
""" * 15  # Make it 15x longer to really test chunking
        
        test_property = {
            'property_name': 'テスト統合物件',
            'property_type': 'マンション',
            'prefecture': '東京都',
            'city': '渋谷区',
            'detailed_address': '渋谷1-1-1',
            'property_details': long_description,
            'latitude': '35.6580',
            'longitude': '139.7016'
        }
        
        print(f"Property details length: {len(long_description)} characters")
        
        chunks = split_property_data(test_property)
        print(f"✅ Created {len(chunks)} chunks")
        
        all_valid = True
        total_size = 0
        
        for i, chunk in enumerate(chunks):
            chunk_json = chunk["text"]
            chunk_bytes = chunk_json.encode('utf-8')
            total_size += len(chunk_bytes)
            
            print(f"Chunk {i+1}: {len(chunk_bytes)} bytes")
            
            if len(chunk_bytes) > 40 * 1024:
                print(f"❌ Chunk {i+1} exceeds 40KB limit!")
                all_valid = False
            
            try:
                metadata = chunk["metadata"]
                assert "property_name" in metadata
                assert "chunk_number" in metadata
                assert "total_chunks" in metadata
                print(f"  ✅ Chunk {i+1} has valid metadata structure")
            except Exception as e:
                print(f"❌ Chunk {i+1} has invalid metadata: {e}")
                all_valid = False
        
        print(f"Total size of all chunks: {total_size} bytes")
        
        if all_valid:
            print("✅ All chunks are valid and within limits!")
            return True
        else:
            print("❌ Some chunks are invalid!")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_property_upload_integration()
    sys.exit(0 if success else 1)
