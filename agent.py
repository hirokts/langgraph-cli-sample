import os
from typing import TypedDict, List
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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


def create_agent_graph():
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
    
    return workflow.compile()


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
