
import os
import re
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

# LangChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# ---- SQLite fallback for Chroma (Cloudでの古いsqlite対策) ----
try:
    import sys, pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except Exception:
    pass

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# BM25 は “あれば使う”
try:
    from langchain_community.retrievers import BM25Retriever
    HAS_BM25 = True
except Exception:
    BM25Retriever = None  # 型ヒント用
    HAS_BM25 = False

# -----------------------------
# ページ設定・定数
# -----------------------------
st.set_page_config(page_title="Streamlit RAG Starter — LangChain + OpenAI + Chroma", layout="centered")

BASE_DIR = Path(__file__).parent.resolve()
RES_DIR = BASE_DIR / "resources"
HERO_IMG = BASE_DIR / "assets" / "myakujii.png"

# 既存DBが1536次元想定なので small を採用（変えるなら保存先も変える）
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
VECTOR_DIR = RES_DIR / f"note_{EMBED_MODEL.replace('-', '_')}"
PERSIST_DIR = str(VECTOR_DIR)  # ← Chromaには必ず str を渡す

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
K_VECT = 20
K_BM25 = 10
REL_THRESHOLD = 0.28

ACL_CHOICES = ["public", "finance", "engineering", "sales"]

# -----------------------------
# .env / モデル
# -----------------------------
api_key = None

# 1) Cloud/ローカルで secrets.toml がある場合（例外安全）
try:
    api_key = st.secrets["OPENAI_API_KEY"]  # Cloud でもローカルでも OK
except Exception:
    api_key = None  # secrets 未設定ならスルー

# 2) 見つからなければ .env を読む（ローカル運用）
if not api_key:
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")

# 3) 無ければエラー表示
if not api_key:
    st.error(
        "OPENAI_API_KEY が設定されていません。\n"
        "ローカル: .env に  OPENAI_API_KEY=sk-...\n"
        "Cloud: 『Edit secrets』に下記を保存\n\n"
        "[default]\nOPENAI_API_KEY = \"sk-...\""
    )
    st.stop()

# ---（任意）デバッグ表示：ファイル有無だけ確認（st.secrets には触れない）---
st.sidebar.caption(f"DEBUG: .env exists? {Path('.env').exists()}  |  .streamlit/secrets.toml exists? {Path('.streamlit/secrets.toml').exists()}")

# LangChain / OpenAI にキーを渡す
LLM_MODEL = "gpt-4.1-mini"
llm = ChatOpenAI(api_key=api_key, model=LLM_MODEL)
rewriter = ChatOpenAI(api_key=api_key, model=LLM_MODEL)

# -----------------------------
# UIヘッダ
# -----------------------------
# --- 2段タイトル ---
st.markdown("""
<style>
h1.app-title{
  font-weight: 800;
  line-height: 1.15;
  margin: .2rem 0 .8rem;
  /* 画面幅に応じてサイズ調整 */
  font-size: clamp(2.2rem, 4vw + 1rem, 3.6rem);
}
.nowrap{ white-space: nowrap; }  /* LangChain / OpenAI / Chroma などを途中で折り返さない */
</style>

<h1 class="app-title">
  Streamlit RAG Starter <br/>
  <span class="nowrap">— LangChain + OpenAI + Chroma</span>
</h1>
""", unsafe_allow_html=True)
mid = st.columns([1, 2, 1])[1]
with mid:
    if HERO_IMG.exists():
        st.image(str(HERO_IMG), width=280)
st.markdown(
    """
<div style="text-align:center; font-weight:700; margin: 8px 0 18px; font-size:1.05rem;">
私はミャクじぃ！世の中、だれも知らない架空のキャラクターじゃ。ChatGPTに聞いても、絶対にワシのことは知らないはず。<br>
このアプリにのみ、ミャクじぃに関する情報がRAGとして登録されている。もしRAGの凄さを体験したければ、ミャクじぃについて、質問するのじゃ！Tech0の旅費規程（架空）も登録しているぞ！
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# サイドバー（運用っぽい制御）
# -----------------------------
st.sidebar.subheader("ボット設定")
user_acl = st.sidebar.multiselect("あなたの権限（ACL）", ACL_CHOICES, default=["public"])
mode = st.sidebar.selectbox("フォールバック方針", ["安全設計 (a→c→b)", "使い勝手重視 (a→b→c)"])
allow_general = st.sidebar.toggle("RAGに無いとき一般回答も許可する（b）", value=False)
st.sidebar.caption("a: 明確化質問 / b: 一般回答 / c: エスカレーション")

# 再構築ボタン
if st.sidebar.button("🔁 インデックス再構築", help="resources配下を再読み込みしてベクトルDBを作り直します"):
    st.cache_resource.clear()
    shutil.rmtree(VECTOR_DIR, ignore_errors=True)
    st.session_state["_reindexed"] = True
    st.rerun()
if st.session_state.get("_reindexed"):
    st.sidebar.success("インデックスを再構築しました。")
    st.session_state["_reindexed"] = False

# --- 履歴クリア（サイドバー） ---
st.sidebar.divider()
if st.sidebar.button("🧹 チャット履歴をクリア", use_container_width=True,
                     help="会話ログを消して最初からやり直します"):
    st.session_state.pop("messages", None)  # ← 履歴を消す
    st.sidebar.success("履歴を消去しました。")
    st.rerun()  # 画面を再描画

# -----------------------------
# 読み込みユーティリティ
# -----------------------------
def _list_source_paths() -> List[Path]:
    # 必要になれば *.md, *.pdf を追加
    return sorted((RES_DIR).glob("*.txt"))

def _sanitize_meta(meta: dict) -> dict:
    # Chromaのメタは str/int/float/bool/None のみ
    clean = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def _load_and_chunk(paths: List[Path]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    out: List[Document] = []
    for p in paths:
        tdocs = TextLoader(str(p), encoding="utf-8").load()
        splits = splitter.split_documents(tdocs)
        for i, d in enumerate(splits):
            d.metadata.update(
                _sanitize_meta(
                    {
                        "source": p.name,     # 例: tech0_travel_policy.txt
                        "title": p.stem,      # 例: tech0_travel_policy
                        "chunk": i,
                        "acl": "public",      # 文字列で保存（list禁止）
                        "domain": "internal-policy",
                    }
                )
            )
        out.extend(splits)
    return out

# -----------------------------
# ストア構築（キャッシュ）
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_indices() -> Tuple[Any, Optional[Any], List[Document]]:
    embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)

    def rebuild() -> Tuple[Any, List[Document]]:
        paths = _list_source_paths()
        if not paths:
            raise FileNotFoundError(f"{RES_DIR} に *.txt がありません。")
        docs_ = _load_and_chunk(paths)
        if not docs_:
            raise ValueError("読み込んだドキュメントが0件です。ファイル内容/文字コードを確認してください。")
        vs_ = Chroma.from_documents(documents=docs_, embedding=embeddings, persist_directory=PERSIST_DIR)
        return vs_, docs_

    if VECTOR_DIR.exists():
        vs = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIR)
        try:
            n = vs._collection.count()
        except Exception:
            n = 0
        if n == 0:
            shutil.rmtree(VECTOR_DIR, ignore_errors=True)
            vs, docs = rebuild()
        else:
            raw = vs._collection.get(include=["metadatas", "documents"])
            docs = [
                Document(page_content=c, metadata=(m or {}))
                for c, m in zip(raw.get("documents", []), raw.get("metadatas", []))
            ]
            if not docs:
                shutil.rmtree(VECTOR_DIR, ignore_errors=True)
                vs, docs = rebuild()
    else:
        vs, docs = rebuild()

    bm25: Optional[Any] = None
    if HAS_BM25 and docs:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = K_BM25

    return vs, bm25, docs

VS, BM25, ALL_DOCS = build_indices()

# 見える化
st.sidebar.caption(f"Persist dir: {PERSIST_DIR}")
st.sidebar.caption(f"Indexed files: {', '.join(sorted({d.metadata.get('source') for d in ALL_DOCS}))}")
st.sidebar.caption(f"Chunks: {len(ALL_DOCS)}")

# -----------------------------
# 検索系
# -----------------------------
def domain_router(q: str) -> str:
    internal_kw = ["ミャクじぃ", "ひろじぃ", "RAG", "ベクタ", "Chroma", "講義", "デモ",
                   "Tech0", "旅費", "旅費規程", "出張", "宿泊", "日当"]
    if any(k.lower() in q.lower() for k in internal_kw):
        return "internal"
    both_kw = ["設計", "要件", "仕様", "社内", "手順", "規程"]
    if any(k in q for k in both_kw):
        return "both"
    return "general"

def normalize_query(q: str) -> str:
    q = q.replace("　", " ").strip()
    return re.sub(r"\s+", " ", q)

def condense_query(history: List[Dict], q: str) -> str:
    last = "\n".join([f"{'ユーザー' if m['role']=='user' else 'アシスタント'}: {m['content']}" for m in history[-6:]])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "あなたは検索クエリ整形の専門家です。"
         "【絶対条件】固有名詞・社内用語（例: ミャクじぃ/Tech0 等）は必ずそのまま残す。"
         "省略・言い換え・削除はしない。日本語1文、短く。出力はクエリのみ。"),
        ("human", "履歴:\n{h}\n\n直近質問:\n{q}")
    ])
    out = rewriter.invoke(prompt.format_messages(h=last, q=q)).content.strip()
    return out or q


def expand_queries(q: str, original: str) -> List[str]:
    seeds = [original]  # ←原文を最優先で保持
    # 固有名詞の手作り追加（落ちない保険）
    if "ミャク" in original:
        seeds += ["ミャクじぃ とは", "ミャクじぃ キャラクター", "ミャクじぃ 説明"]
    p = ChatPromptTemplate.from_messages([
        ("system", "与えたクエリに対し意味の異なる日本語言い換えを3つ、改行で。説明不要。"),
        ("human", "{q}")
    ])
    out = rewriter.invoke(p.format_messages(q=q)).content.strip()
    seeds += [s.strip(" ・-•\n") for s in out.splitlines() if s.strip()]
    # 重複除去
    seen, ret = set(), []
    for s in seeds:
        if s not in seen:
            seen.add(s); ret.append(s)
    return ret[:4]



def uniq_id(doc: Document) -> str:
    key = f"{doc.metadata.get('source','')}-{doc.metadata.get('chunk','')}"
    return key + "-" + hashlib.md5(doc.page_content[:200].encode("utf-8")).hexdigest()

def _allow(meta_acl: Optional[str], user_acl_list: List[str]) -> bool:
    if meta_acl is None:
        return True
    return str(meta_acl) in set(user_acl_list)

def hybrid_retrieve(queries: List[str], acl: List[str]) -> List[Tuple[Document, float, str]]:
    pool: Dict[str, Tuple[Document, float, str]] = {}

    # ---- ベクトル（Chromaのfilterは使わず、Python側でACLチェック）----
    for q in queries:
        docs_scores = VS.similarity_search_with_relevance_scores(q, k=K_VECT)
        for d, s in docs_scores:
            if not _allow(d.metadata.get("acl"), acl):
                continue
            uid = uniq_id(d)
            if uid not in pool and s >= REL_THRESHOLD:
                pool[uid] = (d, float(s), "vect")

    # ---- BM25（ある場合のみ）----
    if BM25:
        for q in queries:
            for d in BM25.get_relevant_documents(q):
                if not _allow(d.metadata.get("acl"), acl):
                    continue
                uid = uniq_id(d)
                if uid not in pool:
                    pool[uid] = (d, 0.30, "bm25")
                else:
                    old = pool[uid]
                    pool[uid] = (old[0], max(old[1], 0.40), old[2])

    return sorted(pool.values(), key=lambda x: x[1], reverse=True)[:8]

SYSTEM_RAG = """あなたは企業内RAGアシスタントです。与えられた context だけを根拠に日本語で簡潔に答えてください。
- contextに十分な根拠がない場合は、推測せず「このRAGでは該当情報が見つかりませんでした。」と答えてください。
- 社外情報や一般知識を混ぜないでください（フォールバックは別で処理します）。
- 箇条書き歓迎。各主張の後に [1], [2] のように参照番号を付け、最後に「参考: タイトル#チャンク…」を列挙してください。
"""
PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RAG),
        ("human", "context:\n{context}\n\n質問: {q}"),
    ]
)

SYSTEM_GENERAL = """あなたは有能な日本語アシスタントです。事実に基づき、簡潔かつ丁寧に回答してください。
ミャクじぃ（社内固有）の情報は使わず、一般的な説明のみで回答してください。必要なら注意点も添えてください。"""
PROMPT_GENERAL = ChatPromptTemplate.from_messages([("system", SYSTEM_GENERAL), ("human", "{q}")])

def compose_with_citations(q: str, docs: List[Tuple[Document, float, str]]) -> Tuple[str, float, str]:
    if not docs:
        return "", 0.0, ""
    refs, parts = [], []
    for i, (d, s, src) in enumerate(docs, start=1):
        parts.append(f"[{i}] {d.page_content}")
        title = d.metadata.get("title", d.metadata.get("source", "doc"))
        refs.append(f"[{i}] {title}#{d.metadata.get('chunk','')}")
    context = "\n\n".join(parts)
    msgs = PROMPT_RAG.format_messages(context=context, q=q)
    out = llm.invoke(msgs).content
    max_rel = max(score for _, score, _ in docs)
    return out, float(max_rel), "参考: " + ", ".join(refs)

def confidence_check(text: str) -> float:
    p = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは回答の自己評価器です。以下の回答が、与えられた社内コンテキストだけで十分に裏付けられているかを0〜1で出力。数値のみ。"),
            ("human", "{ans}"),
        ]
    )
    try:
        s = llm.invoke(p.format_messages(ans=text)).content.strip()
        m = re.findall(r"1(?:\.0+)?|0\.\d+|0", s)
        return max(0.0, min(1.0, float(m[0]))) if m else 0.5
    except Exception:
        return 0.5

def fallback_strategy(q: str, policy: str, allow_general_flag: bool) -> str:
    clarify = "もう少し具体的に教えてください。（例：対象範囲／時点／キーワード）"
    escalate = "社内窓口にエスカレーションしてください（例：#help-desk / 担当: support@example.com）。"
    general = (
        llm.invoke(PROMPT_GENERAL.format_messages(q=q)).content
        if allow_general_flag
        else "一般回答は現在オフになっています（サイドバーで有効化できます）。"
    )
    if policy.startswith("安全設計"):
        return f"{clarify}\n\n{escalate}\n\n{general if allow_general_flag else ''}".strip()
    else:
        return f"{clarify}\n\n{general}\n\n{escalate}".strip()

# -----------------------------
# セッション
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {'role': 'user'|'assistant', 'content': str}

# 既存ログ表示（右=ユーザー／左=アシスタント）
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        message(m["content"], is_user=True, avatar_style="personas", seed="hiroji", key=f"msg_{i}_user")
    else:
        message(m["content"], is_user=False, avatar_style="bottts", seed="myakuji", key=f"msg_{i}_bot")

# -----------------------------
# 入力 → 典型フロー実行
# -----------------------------
q_raw = st.chat_input("聞きたいことを入力してください:")
if q_raw:
    st.session_state.messages.append({"role": "user", "content": q_raw})
    with st.spinner("検索＆生成中…"):
        domain = domain_router(q_raw)
        q_norm = normalize_query(q_raw)
        q_condensed = condense_query(st.session_state.messages, q_norm)
        queries = expand_queries(q_condensed, original=q_norm)  # ←原文を渡す

        answer_text = ""
        citations = ""
        conf = 0.0

        if domain in ("internal", "both"):
            docs = hybrid_retrieve(queries, acl=user_acl)
            if docs:
                answer_text, relmax, citations = compose_with_citations(q_norm, docs)
                conf_llm = confidence_check(answer_text)
                conf = (relmax + conf_llm) / 2.0

        if not answer_text or conf < 0.35:
            if domain in ("general", "both"):
                if allow_general:
                    answer_text = llm.invoke(PROMPT_GENERAL.format_messages(q=q_norm)).content
                else:
                    answer_text = fallback_strategy(q_norm, policy=mode, allow_general_flag=allow_general)
            else:
                answer_text = fallback_strategy(q_norm, policy=mode, allow_general_flag=allow_general)

        if citations:
            answer_text = f"{answer_text}\n\n{citations}"

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.rerun()
