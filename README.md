# LangGraph CLI with Azure OpenAI

Azure OpenAIとLangGraphを使用したインテリジェントなCLIツールです。
エージェントによる推論とツール実行機能を備え、リアルタイムストリーミングレスポンスを提供します。

## 🚀 特徴

- **マルチプロバイダー対応**: Azure OpenAIまたはOpenAI APIを環境変数で切り替え可能
- **エージェント**: LangGraphによる論理的推論と行動パターン
- **リアルタイムストリーミング**: `astream_events`による文字レベルのライブ応答
- **ツール実行**: 時刻取得や計算などの外部ツールとの連携
- **Rich UI**: カラフルで見やすいコンソール出力

## 📋 前提条件

- Python 3.8+
- Azure OpenAI アカウントとAPIキー
- uv

## ⚙️ セットアップ

1. **リポジトリのクローン**
```bash
git clone <repository-url>
cd langgraph-cli-sample
```

2. **依存関係のインストール**
```bash
# uvを使用する場合
uv sync
```

3. **環境変数の設定**
`.env`ファイルを作成し、使用するAIプロバイダーに応じて変数を設定してください：

**Azure OpenAIを使用する場合:**
```bash
# プロバイダー選択
USE_AZURE=true

# Azure OpenAI 設定
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_DEPLOYMENT_ID=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21

# モデル設定
LLM_MODEL=gpt-4o
```

**OpenAIを使用する場合:**
```bash
# プロバイダー選択
USE_AZURE=false

# OpenAI 設定
OPENAI_API_KEY=your_openai_api_key

# モデル設定
LLM_MODEL=gpt-4o
```

## 🛠️ 利用可能なツール

### 1. 時刻取得ツール (`get_current_time`)
現在の日付と時刻を取得します。

### 2. 計算ツール (`calculator`)
数式を評価して計算結果を返します。

## 💻 使用方法

### 基本的な使用法
```bash
python main.py send "メッセージをここに入力"
```

### ツールを使用しないシンプルな質問
```bash
python main.py send "こんにちは、調子はどうですか？"
```

### 時刻取得ツールを使用する例
```bash
# 現在時刻を聞く
python main.py send "今何時ですか？"
python main.py send "現在の日付と時刻を教えてください"
python main.py send "今日は何月何日ですか？"
```

### 計算ツールを使用する例
```bash
# 基本的な計算
python main.py send "2 + 3を計算してください"
python main.py send "10 × 5の結果を教えて"
python main.py send "100 ÷ 4はいくつですか？"

# 複雑な計算
python main.py send "15 * 8 + 25 - 7を計算して"
python main.py send "(50 + 30) * 2の値は？"
```

### 複数のツールを組み合わせた例
```bash
# 時刻確認＋計算
python main.py send "現在の時刻を確認してから、10 + 5を計算してください"

# 連続した計算タスク
python main.py send "以下を順番に計算してください：1. 25 × 4、2. 100 ÷ 5、3. 結果を合計"

# 複雑なリクエスト
python main.py send "今日の日付を確認して、その月の数字を2倍してください。そして結果を説明してください。"
```

### より実用的な例
```bash
# データ分析的なタスク
python main.py send "現在の時刻を記録して、作業開始から3時間30分後の時刻を計算してください"

# 問題解決
python main.py send "会議が14:30から始まって90分続く予定です。終了時刻を計算してください"

# 複数ステップのタスク
python main.py send "今の時刻を確認して、8時間後の時刻を計算し、さらにその時刻から15 * 60分を引いてください"
```

## 📊 出力例

### シンプルな質問の場合
```
Azure OpenAI設定:
  エンドポイント: https://your-resource.openai.azure.com/
  デプロイメント: your-deployment
  APIバージョン: 2024-10-21

🤖 エージェントが思考中...

🤖 回答:
こんにちは！調子は良好です。何かお手伝いできることはありますか？
✅ 完了
```

### ツールを使用する場合
```
Azure OpenAI設定:
  エンドポイント: https://your-resource.openai.azure.com/
  デプロイメント: your-deployment
  APIバージョン: 2024-10-21

🤖 エージェントが思考中...

🔧 ツール実行中: get_current_time({})
✅ ツール結果: content='2025-06-12 15:54:18' name='get_current_time' tool_call_id='call_xyz'

🔧 ツール実行中: calculator({'expression': '10 * 5'})
✅ ツール結果: content='50' name='calculator' tool_call_id='call_abc'

🤖 回答:
現在の時刻は2025年6月12日15時54分18秒です。また、10 × 5の計算結果は50です。
✅ 完了
```

## 🎨 UI説明

- 🤖 **青色**: エージェントの状態
- 🔧 **青色**: ツール実行開始
- ✅ **黄色**: ツール実行結果
- 🤖 **緑色**: 最終回答
- ✅ **緑色**: 処理完了
- ❌ **赤色**: エラーメッセージ

## 🔧 技術詳細

### アーキテクチャ
- **LangGraph**: エージェントパターンの実装
- **Azure OpenAI**: GPT-4oモデルによる自然言語処理
- **Typer**: モダンなCLIインターフェース
- **Rich**: 高品質なコンソール出力
- **async/await**: 非同期ストリーミング処理

### ストリーミング実装
`astream_events`を使用したリアルタイムストリーミング：
- `on_chat_model_stream`: 文字レベルでの逐次表示
- `on_tool_start`: ツール実行開始の即座通知
- `on_tool_end`: ツール実行結果の詳細表示

## 🐛 トラブルシューティング

### よくある問題

1. **環境変数エラー**
```
エラー: 以下の環境変数が設定されていません: AZURE_OPENAI_API_KEY
```
→ `.env`ファイルに必要な環境変数が設定されているか確認してください

2. **API接続エラー**
```
エラーが発生しました: 404 Not Found
```
→ Azure OpenAIのエンドポイントとデプロイメント名を確認してください

3. **計算エラー**
```
計算エラー: invalid syntax
```
→ 数式の構文が正しいか確認してください（例：`2+3`、`10*5`など）

## 📚 参考リンク

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Azure OpenAI Service](https://azure.microsoft.com/products/ai-services/openai-service)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)

## 📄 ライセンス

MIT License
