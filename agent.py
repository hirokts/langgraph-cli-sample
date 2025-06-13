import os
from typing import TypedDict, List, Any, Dict
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr
from tools import tools, tool_map


# USE_AZURE環境変数でプロバイダーを決定
UZUE_AZURE = os.getenv("USE_AZURE", "false").lower() == "true"

# StateGraph用のState定義
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | ToolMessage]
    next_action: str


def call_model(state: AgentState) -> AgentState:
    """LLMを呼び出してレスポンスを生成"""
    messages = state["messages"]
    
    # プロンプトテンプレートの作成
    system_prompt = """あなたは親切で知識豊富なAIアシスタントです。
ユーザーの質問に答えるために、必要に応じて利用可能なツールを使用してください。"""
    
    
    # LLMの設定
    if UZUE_AZURE:
        llm = AzureChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")) or None,
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_ID"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=0,
            streaming=True
        ).bind_tools(tools)
    else:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")) or None,
            temperature=0,
            streaming=True
        ).bind_tools(tools)
    
    # システムメッセージを追加
    full_messages = [AIMessage(content=system_prompt)] + messages
    
    # LLMを呼び出し
    response = llm.invoke(full_messages)
    
    # レスポンスをstateに追加
    updated_messages = messages + [response]

    # 型安全のため、HumanMessage, AIMessage, ToolMessageのみを残す
    filtered_messages: List[HumanMessage | AIMessage | ToolMessage] = [
        m for m in updated_messages if isinstance(m, (HumanMessage, AIMessage, ToolMessage))
    ]
    
    # 次のアクションを決定（AIMessageかつtool_callsがある場合のみツール実行）
    if isinstance(response, AIMessage) and hasattr(response, 'tool_calls') and response.tool_calls:
        next_action = "call_tools"
    else:
        next_action = "end"
    
    return {
        "messages": filtered_messages,
        "next_action": next_action
    }

def call_tools(state: AgentState) -> AgentState:
    """ツールを実行して結果を取得"""
    messages = state["messages"]
    last_message = messages[-1]
    
    new_messages = []
    
    # AIMessageでtool_callsがある場合のみツール呼び出しを実行
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    new_messages.append(tool_message)
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"エラー: {e}",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    new_messages.append(error_message)
    
    updated_messages = messages + new_messages
    
    return {
        "messages": updated_messages,
        "next_action": "call_model"
    }


def should_continue(state: AgentState) -> str:
    """次のステップを決定"""
    return state["next_action"]


def create_agent_graph(checkpointer=None):
    """エージェントグラフを作成"""
    workflow = StateGraph(AgentState)
    
    # ノードを追加
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tools", call_tools)
    
    # エッジを追加
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tools": "call_tools",
            "end": END
        }
    )
    workflow.add_edge("call_tools", "call_model")
    
    return workflow.compile(checkpointer=checkpointer)


def create_react_agent_graph():
    # LLMの設定
    if UZUE_AZURE:
        llm = AzureChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")) or None,
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_ID"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=0,
            streaming=True
        )
    else:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")) or None,
            temperature=0,
            streaming=True
        )
    return create_react_agent(llm, tools)


def create_checkpointer(db_path: str = "checkpoints.db"):
    """チェックポインターを作成（SQLite または PostgreSQL）"""
    from contextlib import asynccontextmanager
    
    checkpoint_type = os.getenv("CHECKPOINT_TYPE", "sqlite").lower()
    
    if checkpoint_type == "postgres":
        # PostgreSQL接続文字列を構築
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "langgraph")
        user = os.getenv("POSTGRES_USER", "langgraph_user")
        password = os.getenv("POSTGRES_PASSWORD", "langgraph_password")
        
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
        @asynccontextmanager
        async def postgres_checkpointer():
            async with AsyncPostgresSaver.from_conn_string(connection_string) as checkpointer:
                # 初回セットアップを実行
                try:
                    await checkpointer.setup()
                except Exception:
                    # セットアップが既に完了している場合はエラーを無視
                    pass
                yield checkpointer
        
        return postgres_checkpointer()
    else:
        # SQLite (デフォルト)
        abs_db_path = os.path.abspath(db_path)
        return AsyncSqliteSaver.from_conn_string(abs_db_path)


async def get_session_history(checkpointer, session_id: str) -> List[HumanMessage | AIMessage | ToolMessage]:
    """セッション履歴を取得（SQLite/PostgreSQL対応）"""
    try:
        # RunnableConfigを作成して最新のチェックポイントを取得
        config = RunnableConfig(configurable={"thread_id": session_id})
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            # チェックポイントから状態を取得
            checkpoint_dict = dict(checkpoint_tuple.checkpoint)
            if "channel_values" in checkpoint_dict:
                channel_values = checkpoint_dict["channel_values"]
                if isinstance(channel_values, dict) and "messages" in channel_values:
                    messages = channel_values["messages"]
                    return messages
        
        return []
    except Exception as e:
        print(f"セッション履歴取得エラー: {e}")
        return []


async def list_sessions(checkpointer) -> List[str]:
    """すべてのセッションIDを取得（SQLite/PostgreSQL対応）"""
    try:
        # チェックポイントから直接セッション一覧を取得
        sessions = set()
        async for checkpoint_tuple in checkpointer.alist(RunnableConfig(configurable={})):
            if checkpoint_tuple.config and "configurable" in checkpoint_tuple.config:
                thread_id = checkpoint_tuple.config["configurable"].get("thread_id")
                if thread_id:
                    sessions.add(thread_id)
        return sorted(list(sessions))
    except Exception as e:
        print(f"セッション一覧取得エラー: {e}")
        return []
