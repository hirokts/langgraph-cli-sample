import asyncio
import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

async def test_postgres_setup():
    """PostgreSQLのセットアップとテーブル作成をテスト"""
    try:
        print("PostgreSQL接続テストを開始...")
        
        # 接続文字列を構築
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "langgraph")
        user = os.getenv("POSTGRES_USER", "langgraph_user")
        password = os.getenv("POSTGRES_PASSWORD", "langgraph_password")
        
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        print(f"接続文字列: {connection_string}")
        
        # AsyncPostgresSaverを作成して使用
        async with AsyncPostgresSaver.from_conn_string(connection_string) as checkpointer:
            print("チェックポインター接続成功")
            
            # setupメソッドを実行してテーブル作成
            print("テーブル作成を実行中...")
            await checkpointer.setup()
            print("テーブル作成完了")
            
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_postgres_setup())
