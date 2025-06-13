import os
import typer
import asyncio
import uuid
from typing import Annotated, Optional
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from agent import (
    create_agent_graph,
    create_checkpointer,
    get_session_history,
    list_sessions,
)

# .envファイルから環境変数を読み込み
load_dotenv()

console = Console()
app = typer.Typer()


async def send_message(message: str, session_id: Optional[str] = None, show_checkpoint: bool = False) -> None:
    """
    LangGraphのStateGraphを使用してストリーミングレスポンスを返す
    """
    # USE_AZURE環境変数でプロバイダーを決定
    use_azure = os.getenv("USE_AZURE", "false").lower() == "true"

    if use_azure:
        # Azure OpenAI APIの環境変数確認
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_ENDPOINT",
            "AZURE_OPENAI_API_DEPLOYMENT_ID",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            console.print(
                f"[red]エラー: 以下のAzure OpenAI環境変数が設定されていません: {', '.join(missing_vars)}[/red]"
            )
            return
    else:
        # OpenAI APIの環境変数確認
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            console.print(
                f"[red]エラー: 以下のOpenAI環境変数が設定されていません: {', '.join(missing_vars)}[/red]"
            )
            return

    # デバッグ情報の表示
    if use_azure:
        console.print("[dim]Azure OpenAI設定:[/dim]")
        console.print(f"[dim]  エンドポイント: {os.getenv('AZURE_ENDPOINT')}[/dim]")
        console.print(
            f"[dim]  デプロイメント: {os.getenv('AZURE_OPENAI_API_DEPLOYMENT_ID')}[/dim]"
        )
        console.print(
            f"[dim]  APIバージョン: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21')}[/dim]"
        )
    else:
        console.print("[dim]OpenAI設定:[/dim]")
        console.print(f"[dim]  モデル: {os.getenv('LLM_MODEL', 'gpt-4o')}[/dim]")

    # セッションIDが指定されていない場合は自動生成
    if not session_id:
        session_id = f"auto_{uuid.uuid4().hex[:8]}"
        console.print(f"[cyan]セッションIDが自動生成されました: {session_id}[/cyan]")

    # セッション情報の表示
    console.print(f"[dim]  セッションID: {session_id}[/dim]")
    console.print()

    # チェックポインターとエージェントの作成（常にセッションIDありで実行）
    async with create_checkpointer() as checkpointer:
        agent = create_agent_graph(checkpointer=checkpointer)
        config = RunnableConfig(configurable={"thread_id": session_id})
        console.print(
            f"[yellow]セッションIDが設定されました: {session_id}[/yellow]"
        )

        # 既存のセッション履歴を取得
        existing_messages = await get_session_history(checkpointer, session_id)
        if existing_messages:
            console.print(f"[blue]📚 既存セッション履歴を読み込みました（{len(existing_messages)}件のメッセージ）[/blue]\n")

        # 初期状態の設定（既存履歴を含む）
        all_messages = existing_messages + [HumanMessage(content=message)]
        initial_state = {
            "messages": all_messages,
            "next_action": "call_model",
        }

        # ストリーミングレスポンスの処理
        console.print("[cyan]🤖 エージェントが思考中...[/cyan]\n")

        try:
            result = await agent.ainvoke(initial_state, config=config)

            # 最後のAIメッセージを取得して表示
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    console.print("[green]🤖 回答:[/green]")
                    console.print(message.content)
                    break
            console.print("[green]✅ 完了[/green]")

            # チェックポイント情報の表示（フラグが有効な場合のみ）
            if show_checkpoint:
                config_for_checkpoint = RunnableConfig(
                    configurable={"thread_id": session_id}
                )

                # 最新のチェックポイントを取得
                try:
                    count = 0
                    latest_id = None
                    async for latest_checkpoint in checkpointer.alist(
                        config_for_checkpoint
                    ):
                        if latest_checkpoint:
                            # 現在のcheckpoint_id (checkpointフィールドから)
                            current_id = (
                                latest_checkpoint.checkpoint.get("id")
                                if latest_checkpoint.checkpoint
                                else None
                            )
                            messages = latest_checkpoint.checkpoint.get(
                                "channel_values", {}
                            ).get("messages", [])
                            content, response_metadata = (messages[-1].content, messages[-1].response_metadata) if messages else ("", None)
                            meta_summary = f"({response_metadata.get('model_name')} {response_metadata.get('finish_reason', '')})" if response_metadata else ""
                            message_role = "(AI)" if response_metadata else "(user)"
                            if count == 0:
                                console.print(f"[green]📍 現在のチェックポイントID: {current_id} [/green] ")
                                latest_id = current_id
                            else:
                                console.print(
                                    f"[yellow]📍 [{count}]個前のチェックポイントID: {current_id}[/yellow]"
                                )
                            console.print(f"{message_role} {content} {meta_summary}")
                            count += 1
                            console.print()

                    console.print("[dim]リプレイコマンド例:[/dim]")
                    console.print(f"[dim]  python main.py replay -s {session_id} -c {latest_id}[/dim]")

                except Exception as checkpoint_error:
                    console.print(
                        f"[yellow]チェックポイント情報の取得に失敗しました: {checkpoint_error}[/yellow]"
                    )

        except Exception as e:
            console.print(f"[red]エラーが発生しました: {e}[/red]")


async def send_replay(session_id: str, checkpoint_id: str) -> None:
    """
    LangGraphのStateGraphを使用してストリーミングレスポンスをリプレイ
    """
    # USE_AZURE環境変数でプロバイダーを決定
    use_azure = os.getenv("USE_AZURE", "false").lower() == "true"

    if use_azure:
        # Azure OpenAI APIの環境変数確認
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_ENDPOINT",
            "AZURE_OPENAI_API_DEPLOYMENT_ID",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            console.print(
                f"[red]エラー: 以下のAzure OpenAI環境変数が設定されていません: {', '.join(missing_vars)}[/red]"
            )
            return
    else:
        # OpenAI APIの環境変数確認
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            console.print(
                f"[red]エラー: 以下のOpenAI環境変数が設定されていません: {', '.join(missing_vars)}[/red]"
            )
            return

    # デバッグ情報の表示
    if use_azure:
        console.print("[dim]Azure OpenAI設定:[/dim]")
        console.print(f"[dim]  エンドポイント: {os.getenv('AZURE_ENDPOINT')}[/dim]")
        console.print(
            f"[dim]  デプロイメント: {os.getenv('AZURE_OPENAI_API_DEPLOYMENT_ID')}[/dim]"
        )
        console.print(
            f"[dim]  APIバージョン: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21')}[/dim]"
        )
    else:
        console.print("[dim]OpenAI設定:[/dim]")
        console.print(f"[dim]  モデル: {os.getenv('LLM_MODEL', 'gpt-4o')}[/dim]")

    # チェックポインターとエージェントの作成
    async with create_checkpointer() as checkpointer:
        agent = create_agent_graph(checkpointer=checkpointer)

        config = RunnableConfig(
            configurable={"thread_id": session_id, "checkpoint_id": checkpoint_id}
        )
        console.print(
            f"[yellow]セッションID、チェックポイントIDが指定されました: {session_id} - {checkpoint_id}[/yellow]"
        )

        # ストリーミングレスポンスの処理
        console.print("[cyan]🤖 エージェントが思考中...[/cyan]\n")

        try:
            # 結果を取得して表示
            result = await agent.ainvoke(None, config=config)

            # 最後のAIメッセージを取得して表示
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    console.print("[green]🤖 回答:[/green]")
                    console.print(message.content)
                    break
            console.print("[green]✅ 完了[/green]")

        except Exception as e:
            console.print(f"[red]エラーが発生しました: {e}[/red]")


@app.command()
def send(
    message: Annotated[str, typer.Argument(help="送信するメッセージ")],
    session_id: Annotated[
        Optional[str], typer.Option("--session-id", "-s", help="セッションID")
    ] = None,
    show_checkpoint: Annotated[
        bool, typer.Option("--show-checkpoint", "-sc", help="チェックポイント情報を表示する")
    ] = False,
):
    """
    メッセージを送信するCLIエンドポイント
    """
    asyncio.run(send_message(message, session_id, show_checkpoint))


@app.command()
def replay(
    session_id: Annotated[str, typer.Option("--session-id", "-s", help="セッションID")],
    checkpoint_id: Annotated[
        str, typer.Option("--checkpoint-id", "-c", help="チェックポイントID")
    ],
):
    """
    メッセージをリプレイするCLIエンドポイント
    """
    asyncio.run(send_replay(session_id, checkpoint_id))


@app.command()
def sessions():
    """
    すべてのセッションを一覧表示
    """

    async def list_sessions_async():
        try:
            async with create_checkpointer() as checkpointer:
                session_list = await list_sessions(checkpointer)

                if not session_list:
                    console.print("[yellow]セッションが見つかりません[/yellow]")
                    return

                table = Table(title="セッション一覧")
                table.add_column("セッションID", style="cyan")
                table.add_column("作成日", style="magenta")

                for session_id in session_list:
                    # 簡易版では作成日を表示しない
                    table.add_row(session_id, "不明")

                console.print(table)
        except Exception as e:
            console.print(f"[red]エラー: {e}[/red]")

    asyncio.run(list_sessions_async())


@app.command()
def history(session_id: Annotated[str, typer.Argument(help="セッションID")]):
    """
    指定されたセッションの履歴を表示
    """

    async def show_history_async():
        try:
            async with create_checkpointer() as checkpointer:
                messages = await get_session_history(checkpointer, session_id)

                if not messages:
                    console.print(
                        f"[yellow]セッション '{session_id}' の履歴が見つかりません[/yellow]"
                    )
                    return

                console.print(f"[cyan]セッション '{session_id}' の履歴:[/cyan]\n")

                for i, message in enumerate(messages, 1):
                    if isinstance(message, HumanMessage):
                        console.print(
                            f"[green]🙋‍♂️ ユーザー ({i}):[/green] {message.content}"
                        )
                    elif isinstance(message, AIMessage):
                        console.print(f"[blue]🤖 AI ({i}):[/blue] {message.content}")
                    console.print()
        except Exception as e:
            console.print(f"[red]エラー: {e}[/red]")

    asyncio.run(show_history_async())


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
