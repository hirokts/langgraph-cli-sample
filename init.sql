-- PostgreSQL初期化スクリプト
-- LangGraphのチェックポイント用データベースの設定

-- チェックポイント用のテーブルが作成されるまで待機
-- LangGraphが自動的にテーブルを作成するため、ここでは特別な設定は不要

-- 必要に応じて追加の設定をここに記載
GRANT ALL PRIVILEGES ON DATABASE langgraph TO langgraph_user;
