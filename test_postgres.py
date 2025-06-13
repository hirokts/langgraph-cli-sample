import asyncio
import os
from dotenv import load_dotenv
from agent import create_checkpointer

load_dotenv()

async def test_postgres_connection():
    """PostgreSQL接続テスト"""
    try:
        print("PostgreSQL接続テストを開始...")
        
        # PostgreSQL設定を強制的に設定
        os.environ["CHECKPOINT_TYPE"] = "postgres"
        
        async with create_checkpointer() as checkpointer:
            print("チェックポインター作成成功")
            
            # 何らかの基本操作を実行してテーブル作成を促す
            
            try:
                # 最初にsetupを実行してテーブルを作成
                await checkpointer.setup()
                print("PostgreSQLテーブル初期化成功")
                
            except Exception as e:
                print(f"初期化エラー: {e}")
                
    except Exception as e:
        print(f"PostgreSQL接続エラー: {e}")

if __name__ == "__main__":
    asyncio.run(test_postgres_connection())
