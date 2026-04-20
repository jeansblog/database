import httpx
import json

# =========================
# 設定
# =========================
MCP_URL = "http://localhost:8900/mcp"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "gemma4:e2b"

# =========================
# system prompt（ツール辞書生成）
# =========================
TOOL_DICT_SYSTEM_PROMPT = """
あなたはツール定義を整理するアシスタントです。

以下は MCP サーバで提供されている tools の一覧です。
各 tool について、どんな intent で使われるかを考えてください。

制約：
- JSON のキーは必ず tool の name をそのまま使う
- 新しい tool 名を作らない
- 出力は JSON のみ

形式：
{
  "<tool_name>": ["intent1", "intent2"]
}
"""

# =========================
# system prompt（引数抽出）
# =========================
ARG_EXTRACT_SYSTEM_PROMPT = """
あなたはツール呼び出し用の引数抽出アシスタントです。

inputSchema を元に、ユーザー入力から引数を抽出してください。

制約：
- 出力は JSON のみ
- inputSchema に存在しないキーは禁止
- 値が不明な場合は null を使う
"""

# =========================
# JSON-RPC helper
# =========================
def rpc(method, params=None, id=1):
    payload = {
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return payload

# =========================
# Ollama 呼び出し
# =========================
def ask_ollama(messages):
    r = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "stream": False,
        },
        timeout=120,
    )
    return r.json()["choices"][0]["message"]["content"]

# =========================
# MCP tools/list 取得
# =========================
def get_tools_list(client, headers):
    res = client.post(
        MCP_URL,
        headers=headers,
        json=rpc("tools/list"),
        timeout=30,
    ).json()

    # FastApiMCP 形式
    if "tools" in res:
        return res["tools"]

    # JSON-RPC 標準形式
    if "result" in res and "tools" in res["result"]:
        return res["result"]["tools"]

    raise RuntimeError(f"Unexpected tools/list response: {res}")

# =========================
# ツール辞書生成
# =========================
def build_tool_intent_dict(tools):
    tools_info = [
        {"name": t["name"], "description": t.get("description", "")}
        for t in tools
    ]

    raw = ask_ollama(
        [
            {"role": "system", "content": TOOL_DICT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(tools_info, ensure_ascii=False)},
        ]
    ).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    result = json.loads(raw)

    tool_names = {t["name"] for t in tools}
    for k in result:
        if k not in tool_names:
            raise RuntimeError(f"未知の tool 名: {k}")

    return result

# =========================
# intent vocab 生成
# =========================
def build_valid_intents(tool_intent_dict):
    s = set()
    for v in tool_intent_dict.values():
        s.update(v)
    s.add("other")
    return s

# =========================
# intent 分類
# =========================
def classify_intent(user_input, valid_intents):
    intent_list = "\n".join(f"- {i}" for i in sorted(valid_intents))
    return ask_ollama(
        [
            {
                "role": "system",
                "content": (
                    "次のユーザ入力を intent に分類してください。\n\n"
                    "使える intent:\n"
                    f"{intent_list}\n\n"
                    "出力は intent のみ。"
                ),
            },
            {"role": "user", "content": user_input},
        ]
    ).strip().lower()

# =========================
# tool 選択
# =========================
def select_tool(intent, tool_intent_dict):
    for tool, intents in tool_intent_dict.items():
        if intent in intents:
            return tool
    return None

# =========================
# 引数抽出
# =========================
def extract_tool_arguments(user_input, tool):
    schema = tool.get("inputSchema")
    if not schema:
        return {}

    raw = ask_ollama(
        [
            {"role": "system", "content": ARG_EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tool_name": tool["name"],
                        "inputSchema": schema,
                        "user_input": user_input,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    ).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)

# =========================
# 必須引数チェック
# =========================
def missing_required_args(tool, args):
    required = tool.get("inputSchema", {}).get("required", [])
    return [k for k in required if k not in args or args[k] is None]

# =========================
# メイン
# =========================
with httpx.Client() as client:
    # initialize（Accept ヘッダ必須）
    init = client.post(
        MCP_URL,
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json=rpc(
            "initialize",
            {
                "protocolVersion": "1.0",
                "capabilities": {},
                "clientInfo": {"name": "react-agent-chat", "version": "1.0"},
            },
        ),
    )

    session_id = init.headers["mcp-session-id"]

    # ★ ここで headers を 1 回だけ作る（最重要）
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "mcp-session-id": session_id,
    }

    # initialized 通知
    client.post(
        MCP_URL,
        headers=headers,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )

    # 初期化処理
    tools = get_tools_list(client, headers)
    tool_intent_dict = build_tool_intent_dict(tools)
    valid_intents = build_valid_intents(tool_intent_dict)

    print("[INFO] tool_intent_dict =", tool_intent_dict)
    print("MCP ReAct Agent (type 'exit' to quit)")

    # CLI ループ
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        intent = classify_intent(user_input, valid_intents)
        tool_name = select_tool(intent, tool_intent_dict)

        if tool_name:
            tool = next(t for t in tools if t["name"] == tool_name)
            args = extract_tool_arguments(user_input, tool)

            missing = missing_required_args(tool, args)
            if missing:
                print(f"{missing[0]}を教えてください。")
                continue

            result = client.post(
                MCP_URL,
                headers=headers,
                json=rpc(
                    "tools/call",
                    {"name": tool_name, "arguments": args},
                ),
            ).json()

            observation = result["result"]["content"][0]["text"]
        else:
            observation = user_input

        response = ask_ollama(
            [
                {
                    "role": "system",
                    "content": "以下を自然な日本語で答えてください。",
                },
                {"role": "user", "content": observation},
            ]
        )
        print(response)
