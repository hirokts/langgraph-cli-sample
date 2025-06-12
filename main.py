import os
import typer
import asyncio
from typing import Annotated
from rich.console import Console
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent import create_agent_graph

# .envファイルから環境変数を読み込み
load_dotenv()

console = Console()
app = typer.Typer()


async def send_message(message: str) -> None:
    """
    LangGraphのStateGraphを使用してストリーミングレスポンスを返す
    """
    # USE_AZURE環境変数でプロバイダーを決定
    use_azure = os.getenv("USE_AZURE", "false").lower() == "true"
    
    if use_azure:
        # Azure OpenAI APIの環境変数確認
        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_ENDPOINT", "AZURE_OPENAI_API_DEPLOYMENT_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            console.print(f"[red]エラー: 以下のAzure OpenAI環境変数が設定されていません: {', '.join(missing_vars)}[/red]")
            return
    else:
        # OpenAI APIの環境変数確認
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            console.print(f"[red]エラー: 以下のOpenAI環境変数が設定されていません: {', '.join(missing_vars)}[/red]")
            return
    
    # デバッグ情報の表示
    if use_azure:
        console.print("[dim]Azure OpenAI設定:[/dim]")
        console.print(f"[dim]  エンドポイント: {os.getenv('AZURE_ENDPOINT')}[/dim]")
        console.print(f"[dim]  デプロイメント: {os.getenv('AZURE_OPENAI_API_DEPLOYMENT_ID')}[/dim]")
        console.print(f"[dim]  APIバージョン: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21')}[/dim]")
    else:
        console.print("[dim]OpenAI設定:[/dim]")
        console.print(f"[dim]  モデル: {os.getenv('LLM_MODEL', 'gpt-4o')}[/dim]")
    console.print()
    
    # StateGraphエージェントの作成
    agent = create_agent_graph()

    # create_react_agentを使う場合
    # agent = create_react_agent_graph()
    
    # 初期状態の設定
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "next_action": "call_model"
    }
    
    # ストリーミングレスポンスの処理
    console.print("[cyan]🤖 エージェントが思考中...[/cyan]\n")
    
    response_started = False
    
    try:
        async for event in agent.astream_events(initial_state, version="v1"):
            # チャットモデルのストリーミングイベントを処理
            if event.get("event") == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if not response_started:
                        console.print("[green]🤖 回答:[/green]")
                        response_started = True
                    console.print(chunk.content, end="")
            
            # ツール実行開始イベント
            elif event.get("event") == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event["data"].get("input", {})
                console.print(f"\n[blue]🔧 ツール実行中: {tool_name}({tool_input})[/blue]")
            
            # ツール実行終了イベント
            elif event.get("event") == "on_tool_end":
                tool_name = event.get("name", "unknown")
                tool_output = event["data"].get("output", "")
                console.print(f"[yellow]✅ ツール結果: {tool_output}[/yellow]")
        
        if response_started:
            console.print()  # 最後に改行
        console.print(f"[green]✅ 完了[/green]")
    
    except Exception as e:
        console.print(f"[red]エラーが発生しました: {e}[/red]")


@app.command()
def send(
    message: Annotated[str, typer.Argument(help="送信するメッセージ")]
):
    """
    メッセージを送信するCLIエンドポイント
    """
    asyncio.run(send_message(message))

@app.command()
def mock(message: str):
    """
    GUIからメッセージを送信する関数
    """
    console.print(f"[green]メッセージを送信しました: {message}[/green]")


if __name__ == "__main__":
    # CLIモードとして実行
    print("CLIモードで実行中...")
    app()
