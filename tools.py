from langchain_core.tools import tool


@tool("get_current_time_tool", parse_docstring=True)
def get_current_time():
    """現在の時刻を取得します。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool("calculator_tool", parse_docstring=True)
def calculator(expression: str) -> str:
    """簡単な数式を計算します。
    
    Args:
        expression: 計算する数式（例: '2 + 3', '10 * 5'）
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"

# ツールマッピング
tools = [get_current_time, calculator]
tool_map = {tool.name: tool for tool in tools}
