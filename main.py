import os
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
# ── ENV ───────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PORT         = int(os.environ.get("PORT", 8000))

# ── DB POOL ───────────────────────────────────────────────────────────────────
db_pool: asyncpg.Pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    print("✓ DB pool ready")
    yield
    await db_pool.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
groq = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

# ── SNAPSHOT CACHE ────────────────────────────────────────────────────────────
_snapshot_cache: dict[str, dict] = {}
CACHE_TTL = 600  # 10 minutes


# ── REQUEST MODELS ────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    business_id: str
    history: Optional[list[ChatMessage]] = []


# ── DETECT ACTIVE ORDER STATUSES ─────────────────────────────────────────────
# We never assume a fixed status string. Instead we fetch all distinct statuses
# for the business and exclude obviously-dead ones (cancelled / refunded).
EXCLUDED_STATUSES = {"cancelled", "canceled", "refunded", "failed", "void"}

async def get_active_statuses(conn: asyncpg.Connection, business_id: str) -> list[str]:
    rows = await conn.fetch(
        'SELECT DISTINCT status FROM "order" WHERE business_id = $1',
        business_id,
    )
    all_statuses = [r["status"] for r in rows if r["status"]]
    active = [s for s in all_statuses if s.lower() not in EXCLUDED_STATUSES]
    # If somehow everything is excluded, fall back to all statuses
    return active if active else all_statuses

def status_filter(statuses: list[str]) -> str:
    """Return a SQL fragment like: AND o.status IN ('completed','paid',...)"""
    quoted = ", ".join(f"'{s}'" for s in statuses)
    return f"AND o.status IN ({quoted})"


# ── QUERY PLANNER ─────────────────────────────────────────────────────────────
QUERY_PLANNER_PROMPT = """You are a PostgreSQL query planner for a business analytics system.

Database schema (relevant tables):
- "order"(id, business_id, customer_id, total, status, order_voucher, order_discount, created_at)
- order_item(id, order_id, product_id, quantity, unit_price, item_discount, attributes)
- product(id, business_id, name, price, cost, stock, created_at)
- customer(id, business_id, full_name, email, phone_number, segment)
- product_segment(business_id, product_id, cluster, cluster_name, job_id, updated_at)
- product_cluster_summary(id, business_id, cluster, cluster_name, num_products, avg_profit,
    total_profit, avg_revenue, total_revenue, avg_price, avg_cost, avg_margin, avg_stock,
    avg_quantity, revenue_share_pct, profit_share_pct)
- customer_cluster_summary(id, business_id, cluster, segment_name, num_customers,
    recency_median, frequency_median, monetary_median, monetary_sum, aov_median,
    churn_risk, priority, channel, offer)

Rules:
1. ALWAYS filter by business_id = $1 (server-injected — never hardcode it).
2. Revenue = SUM(oi.unit_price * oi.quantity - COALESCE(oi.item_discount, 0))
3. Profit  = SUM((oi.unit_price - COALESCE(p.cost, 0)) * oi.quantity - COALESCE(oi.item_discount, 0))
4. NEVER filter by order status — the server injects the correct status filter automatically.
   Do NOT write AND o.status = anything. Leave status filtering out entirely.
5. "order" is a reserved word — always quote it as "order".

Return ONLY a JSON object — no markdown, no explanation:
{
  "intent": "one-line description of what the user is asking",
  "queries": [
    { "label": "descriptive label", "sql": "SELECT ..." }
  ]
}

Generate 1–4 targeted queries that together fully answer the question."""


async def plan_queries(user_message: str, business_id: str) -> dict:
    resp = await groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": QUERY_PLANNER_PROMPT},
            {"role": "user",   "content": f"Business ID: {business_id}\nUser question: {user_message}"},
        ],
        temperature=0,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


async def execute_queries(plan: dict, business_id: str, status_clause: str) -> str:
    """Execute planner queries, injecting the status filter before WHERE business_id."""
    parts = []
    async with db_pool.acquire() as conn:
        for q in plan.get("queries", []):
            sql = q["sql"]
            # Inject status filter: append after business_id = $1 condition
            sql = sql.replace(
                "business_id = $1",
                f"business_id = $1 {status_clause}"
            )
            try:
                rows = await conn.fetch(sql, business_id)
                data = [dict(r) for r in rows]
                serialized = json.dumps(data, default=str, indent=2)
                parts.append(f"[{q['label']}]:\n{serialized}")
            except Exception as e:
                parts.append(f"[{q['label']}]: Query error — {e}")
    return "\n\n".join(parts)


# ── BUSINESS SNAPSHOT ─────────────────────────────────────────────────────────
async def get_business_snapshot(business_id: str) -> str:
    cached = _snapshot_cache.get(business_id)
    if cached and (datetime.now(timezone.utc).timestamp() - cached["ts"]) < CACHE_TTL:
        return cached["data"]

    async with db_pool.acquire() as conn:
        # 1. Detect active statuses for this business
        active_statuses = await get_active_statuses(conn, business_id)
        sc = status_filter(active_statuses)
        print(f"[snapshot] business={business_id} active_statuses={active_statuses}")

        snap = await conn.fetchrow(
            f"""SELECT
                 COUNT(DISTINCT o.id) AS total_orders,
                 COALESCE(SUM(oi.unit_price * oi.quantity - COALESCE(oi.item_discount,0)), 0) AS total_revenue,
                 COALESCE(SUM((oi.unit_price - COALESCE(p.cost,0)) * oi.quantity - COALESCE(oi.item_discount,0)), 0) AS total_profit,
                 AVG(p.price) AS avg_price,
                 MIN(o.created_at) AS first_order,
                 MAX(o.created_at) AS last_order
               FROM "order" o
               JOIN order_item oi ON oi.order_id = o.id
               JOIN product    p  ON p.id = oi.product_id
               WHERE o.business_id = $1 {sc}""",
            business_id,
        )

        yearly = await conn.fetch(
            f"""SELECT EXTRACT(YEAR FROM o.created_at)::int AS year,
                      SUM(oi.unit_price * oi.quantity - COALESCE(oi.item_discount,0)) AS revenue,
                      SUM((oi.unit_price - COALESCE(p.cost,0)) * oi.quantity - COALESCE(oi.item_discount,0)) AS profit
               FROM "order" o
               JOIN order_item oi ON oi.order_id = o.id
               JOIN product    p  ON p.id = oi.product_id
               WHERE o.business_id = $1 {sc}
               GROUP BY 1 ORDER BY 1""",
            business_id,
        )

        monthly = await conn.fetch(
            f"""SELECT TO_CHAR(DATE_TRUNC('month', o.created_at), 'YYYY-MM') AS month,
                      SUM(oi.unit_price * oi.quantity - COALESCE(oi.item_discount,0)) AS revenue
               FROM "order" o
               JOIN order_item oi ON oi.order_id = o.id
               WHERE o.business_id = $1 {sc}
                 AND o.created_at >= NOW() - INTERVAL '12 months'
               GROUP BY 1 ORDER BY 1""",
            business_id,
        )

        top_products = await conn.fetch(
            f"""SELECT p.name,
                      SUM(oi.unit_price * oi.quantity - COALESCE(oi.item_discount,0)) AS revenue
               FROM order_item oi
               JOIN "order"  o ON o.id = oi.order_id
               JOIN product  p ON p.id = oi.product_id
               WHERE o.business_id = $1 {sc}
               GROUP BY p.id, p.name ORDER BY revenue DESC LIMIT 5""",
            business_id,
        )

        top_customers = await conn.fetch(
            f"""SELECT c.full_name,
                      SUM(oi.unit_price * oi.quantity - COALESCE(oi.item_discount,0)) AS revenue
               FROM "order" o
               JOIN customer   c  ON c.id = o.customer_id
               JOIN order_item oi ON oi.order_id = o.id
               WHERE o.business_id = $1 {sc}
               GROUP BY c.id, c.full_name ORDER BY revenue DESC LIMIT 5""",
            business_id,
        )

    total_revenue = float(snap["total_revenue"] or 0)
    total_profit  = float(snap["total_profit"]  or 0)
    margin    = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    avg_price = float(snap["avg_price"] or 0)

    def fmt(n): return f"{float(n):,.0f}"

    yearly_rows  = [dict(r) for r in yearly]
    yearly_lines = []
    for i, r in enumerate(yearly_rows):
        rev  = float(r["revenue"])
        prof = float(r["profit"])
        if i > 0:
            prev    = float(yearly_rows[i - 1]["revenue"])
            yoy     = ((rev - prev) / prev * 100) if prev else 0
            yoy_str = f" | YoY: {yoy:+.1f}%"
        else:
            yoy_str = " (base year)"
        yearly_lines.append(f"  {r['year']}: Revenue={fmt(rev)} EGP | Profit={fmt(prof)} EGP{yoy_str}")

    cagr_str = ""
    if len(yearly_rows) >= 2:
        first = float(yearly_rows[0]["revenue"])
        last  = float(yearly_rows[-1]["revenue"])
        n     = yearly_rows[-1]["year"] - yearly_rows[0]["year"]
        if first > 0 and n > 0:
            cagr     = ((last / first) ** (1 / n) - 1) * 100
            cagr_str = f"\n  Revenue CAGR: {cagr:.1f}%"

    first_order = snap["first_order"].strftime("%b %Y") if snap["first_order"] else "N/A"
    last_order  = snap["last_order"].strftime("%b %Y")  if snap["last_order"]  else "N/A"
    statuses_label = ", ".join(active_statuses)

    snapshot = f"""
════════════════════════════════════════════════
FUSE BUSINESS INTELLIGENCE REPORT
════════════════════════════════════════════════

▸ PORTFOLIO SNAPSHOT
  Total Orders  : {int(snap['total_orders'] or 0):,}
  Total Revenue : {fmt(total_revenue)} EGP
  Total Profit  : {fmt(total_profit)} EGP
  Avg Margin    : {margin:.1f}%
  Avg Price     : {fmt(avg_price)} EGP
  Data Range    : {first_order} → {last_order}
  Order Statuses: {statuses_label}

▸ YEAR-ON-YEAR PERFORMANCE
{chr(10).join(yearly_lines)}{cagr_str}

▸ LAST 12 MONTHS — MONTHLY REVENUE
{chr(10).join(f"  {r['month']}: {fmt(r['revenue'])} EGP" for r in monthly)}

▸ TOP 5 PRODUCTS — REVENUE
{chr(10).join(f"  {r['name']}: {fmt(r['revenue'])} EGP" for r in top_products)}

▸ TOP 5 CUSTOMERS — REVENUE
{chr(10).join(f"  {r['full_name']}: {fmt(r['revenue'])} EGP" for r in top_customers)}
════════════════════════════════════════════════"""

    _snapshot_cache[business_id] = {"data": snapshot, "ts": datetime.now(timezone.utc).timestamp()}
    return snapshot


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
def build_system_prompt(snapshot: str) -> str:
    return f"""You are FUSE AI — a senior business advisor embedded inside this company.
You have an MBA-level grasp of strategy, finance, pricing, operations, and growth,
AND you have complete visibility into this business's live data.

You are NOT a chatbot that summarizes data. You are a thinking advisor who reads
the data, spots what it means, and tells the owner what to DO about it.

════════════════════════════════════════════════
BUSINESS INTELLIGENCE (your foundation — know this cold)
════════════════════════════════════════════════
{snapshot}
════════════════════════════════════════════════

━━━ LANGUAGE PROTOCOL ━━━
- Mirror the user's EXACT language and script. No exceptions. Ever.
- English → English only. No Arabic, no other scripts.
- Egyptian colloquial Arabic (عامية) → write like a smart Egyptian friend texting — natural, warm, zero formality. Arabic script only.
- Modern Standard Arabic (فصحى) → formal, structured, confident. Arabic script only.
- NEVER mix languages or scripts unless the user does first.
- CRITICAL: Zero tolerance for stray characters from other languages.

━━━ HOW TO ANSWER — THE CONSULTANT STANDARD ━━━
1. ANCHOR IN DATA FIRST. Open with the most relevant hard number. No fluff opener.
2. DIAGNOSE WHAT THE DATA IS TELLING YOU. Read patterns, not just figures.
3. APPLY BUSINESS EXPERTISE. Layer in the "so what" — pricing, product mix, seasonality, etc.
4. FOR FUTURE QUESTIONS: Extrapolate from the trend. Give the number. Say what they need to do.
   NEVER say "I don't have future data." You're a forward-looking advisor.
5. CLOSE WITH ONE SHARP ACTION. Concrete, specific next step.

━━━ ANTI-PATTERNS — NEVER DO THESE ━━━
- NEVER repeat the same point twice in different words.
- NEVER open by echoing the user's question back.
- NEVER use filler openers or "I believe" as a crutch — state directly with confidence.
- No walls of bullets. Flowing, punchy paragraphs.

━━━ TONE ━━━
Direct. Sharp. Confident. Like a senior partner, not an intern.
Monetary values always in EGP.

━━━ OUT OF SCOPE ━━━
Only deflect if the question has ZERO business connection (weather, recipes, personal life).
- EN:  "That's outside my scope — I'm here to help with your business."
- EGY: "ده برا نطاقي — أنا هنا عشان أساعدك في شغلك."
Everything else — engage."""


# ── GREETING DETECTION ────────────────────────────────────────────────────────
GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|howdy|sup|yo|good\s*(morning|afternoon|evening)|"
    r"مرحبا|هاي|هلو|السلام عليكم|صباح الخير|مساء الخير|اهلا|أهلا|ازيك)\s*[!?.\u061f]*\s*$",
    re.IGNORECASE,
)

def detect_lang(text: str) -> str:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return "ar" if arabic > len(text) * 0.2 else "en"


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/snapshot/{business_id}")
async def snapshot_route(business_id: str):
    try:
        data = await get_business_snapshot(business_id)
        return {"snapshot": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if GREETING_RE.match(req.message.strip()):
        lang  = detect_lang(req.message)
        reply = (
            "هلا! أنا FUSE AI، مستشارك التجاري. إيه اللي تحب تعرفه عن شغلك؟"
            if lang == "ar"
            else "Hey! I'm FUSE AI, your business advisor. What would you like to know about your business?"
        )
        return {"reply": reply}

    try:
        # 1. Detect active statuses once (also warms the snapshot cache)
        async with db_pool.acquire() as conn:
            active_statuses = await get_active_statuses(conn, req.business_id)
        sc = status_filter(active_statuses)

        # 2. Business snapshot (cached)
        snapshot = await get_business_snapshot(req.business_id)

        # 3. Plan & execute dynamic queries
        query_context = ""
        try:
            plan          = await plan_queries(req.message, req.business_id)
            query_context = await execute_queries(plan, req.business_id, sc)
        except Exception as e:
            query_context = f"[Query planning/execution failed: {e}]"

        # 4. Build messages for advisor LLM
        recent_history = [{"role": m.role, "content": m.content} for m in (req.history or [])[-10:]]
        messages = [
            {"role": "system", "content": build_system_prompt(snapshot)},
            *recent_history,
            {
                "role": "user",
                "content": (
                    "[CRITICAL LANGUAGE RULE: Detect the script of the question below and reply ONLY "
                    "in that script. Arabic → Arabic script only. English → English only. Dialect stays dialect.]\n\n"
                    f"Question: {req.message}\n\n"
                    f"Live Query Results for this question:\n{query_context}"
                ),
            },
        ]

        # 5. Advisor LLM
        response = await groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return {"reply": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)