import httpx
import json
import argparse
from pathlib import Path

import yaml

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
# 設定ロード
# =========================
DEFAULT_CONFIG = {
    "mcp": {"url": "http://localhost:8900/mcp"},
    "llm": {
        "provider": "ollama",
        "timeout": 120,
        "temperature": 0,
        "stream": False,
        "providers": {
            "ollama": {
                "base_url": "http://localhost:11434/v1/chat/completions",
                "model": "gemma4:e2b",
                "api_key": None,
            },
            "vllm": {
                "base_url": "http://localhost:8000/v1/chat/completions",
                "model": "your-model-name",
                "api_key": None,
            },
        },
    },
}


def deep_merge(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    p = Path(path)
    if p.exists():
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        deep_merge(cfg, loaded)
    return cfg

# =========================
# JSON-RPC helper
# =========================
def rpc(method, params=None, id=1):
    payload = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        payload["params"] = params
    return payload

# =========================
# LLM 呼び出し（Ollama / vLLM 共通）
# =========================
def ask_llm(messages, *, provider_cfg, timeout, temperature=0, stream=False):
    url = provider_cfg["base_url"]
    model = provider_cfg["model"]
    api_key = provider_cfg.get("api_key")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = httpx.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# =========================
# MCP tools/list 取得
# =========================
def get_tools_list(client, headers, mcp_url):
    res = client.post(
        mcp_url,
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
def build_tool_intent_dict(tools, ask_fn):
    tools_info = [{"name": t["name"], "description": t.get("description", "") } for t in tools]

    raw = ask_fn(
        [
            {"role": "system", "content": TOOL_DICT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(tools_info, ensure_ascii=False)},
        ]
    ).strip()

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].strip()
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
def classify_intent(user_input, valid_intents, ask_fn):
    intent_list = "\n".join(f"- {i}" for i in sorted(valid_intents))
    return (
        ask_fn(
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
        )
        .strip()
        .lower()
    )

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
def extract_tool_arguments(user_input, tool, ask_fn):
    schema = tool.get("inputSchema")
    if not schema:
        return {}

    raw = ask_fn(
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
        raw = raw.split("```", 2)[1].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)

# =========================
# 必須引数チェック
# =========================
def missing_required_args(tool, args):
    required = tool.get("inputSchema", {}).get("required", [])
    return [k for k in required if k not in args or args[k] is None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--provider", default=None, help="Override provider (ollama|vllm)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    mcp_url = cfg["mcp"]["url"]
    llm_cfg = cfg["llm"]
    provider_name = args.provider or llm_cfg["provider"]
    provider_cfg = llm_cfg["providers"][provider_name]

    timeout = llm_cfg.get("timeout", 120)
    temperature = llm_cfg.get("temperature", 0)
    stream = llm_cfg.get("stream", False)

    def ask_fn(messages):
        return ask_llm(
            messages,
            provider_cfg=provider_cfg,
            timeout=timeout,
            temperature=temperature,
            stream=stream,
        )

    with httpx.Client() as client:
        init = client.post(
            mcp_url,
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

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id,
        }

        client.post(
            mcp_url,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

        tools = get_tools_list(client, headers, mcp_url)
        tool_intent_dict = build_tool_intent_dict(tools, ask_fn)
        valid_intents = build_valid_intents(tool_intent_dict)

        print("[INFO] provider =", provider_name)
        print("[INFO] model =", provider_cfg["model"])
        print("[INFO] tool_intent_dict =", tool_intent_dict)
        print("MCP ReAct Agent (type 'exit' to quit)")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ("exit", "quit"):
                break

            intent = classify_intent(user_input, valid_intents, ask_fn)
            tool_name = select_tool(intent, tool_intent_dict)

            if tool_name:
                tool = next(t for t in tools if t["name"] == tool_name)
                args2 = extract_tool_arguments(user_input, tool, ask_fn)

                missing = missing_required_args(tool, args2)
                if missing:
                    print(f"{missing[0]}を教えてください。")
                    continue

                result = client.post(
                    mcp_url,
                    headers=headers,
                    json=rpc(
                        "tools/call",
                        {"name": tool_name, "arguments": args2},
                    ),
                ).json()

                observation = result["result"]["content"][0]["text"]
            else:
                observation = user_input

            response = ask_fn(
                [
                    {"role": "system", "content": "以下を自然な日本語で答えてください。"},
                    {"role": "user", "content": observation},
                ]
            )
            print(response)


if __name__ == "__main__":
    main()