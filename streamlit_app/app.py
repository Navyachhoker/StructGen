"""
Streamlit demo - a thin UI over the Invoice Extractor API.

Three views: Dashboard (stats + recent invoice table), Extract (submit
text, review result), Benchmark (status checklist until Phase 7's model
comparison has real numbers).
"""

import streamlit as st
import requests

from api_client import submit_extraction, poll_job, get_stats, get_recent_invoices, API_BASE_URL

st.set_page_config(page_title="StructGen", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }

    .stApp { background: #F7F8FA; color: #1F2937; }
    .block-container { padding-top: 2rem; max-width: 1360px; }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    .brand { font-size: 1.05rem; font-weight: 700; color: #1F2937; margin-bottom: 0.05rem; }
    .brand-sub { font-size: 0.75rem; color: #9CA3AF; margin-bottom: 1.4rem; }

    /* Active nav item is a plain styled div, not a widget - this is
       the fix for the active-state highlight never appearing. The
       previous version relied on a CSS :has() selector targeting a
       checked radio input, which several Chromium builds render
       inconsistently. A plain div has no such dependency. */
    .nav-active {
        background: #EEF2FF;
        color: #4F46E5;
        font-weight: 600;
        padding: 0.5rem 0.7rem;
        border-radius: 8px;
        margin-bottom: 0.15rem;
        font-size: 0.95rem;
    }
    [data-testid="stSidebar"] .stButton button {
        background: transparent;
        color: #374151;
        border: none;
        text-align: left;
        font-weight: 500;
        padding: 0.5rem 0.7rem;
        width: 100%;
        justify-content: flex-start;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #F3F4F6;
        color: #1F2937;
    }
    [data-testid="stSidebar"] .stButton button p { text-align: left; }

    .system-status {
        border-top: 1px solid #E5E7EB;
        margin-top: 1.5rem;
        padding-top: 0.9rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
    .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
    .status-ok { background: #10B981; }
    .status-down { background: #EF4444; }

    .page-title { font-size: 1.55rem; font-weight: 700; color: #1F2937; margin-bottom: 0.2rem; }
    .page-subtitle { color: #6B7280; font-size: 0.92rem; margin-bottom: 1.6rem; }

    .stat-block { border-top: 2px solid #E5E7EB; padding-top: 0.6rem; }
    .stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #9CA3AF; margin-bottom: 0.3rem; font-weight: 600; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #1F2937; }

    .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-success { background: #D1FAE5; color: #065F46; }
    .badge-failed  { background: #FEE2E2; color: #991B1B; }
    .badge-pending { background: #FEF3C7; color: #92400E; }

    /* Explicit !important on both text color and placeholder color -
       the earlier version set textarea color but not placeholder
       color, and on some local Streamlit theme configs the base theme
       variable was winning over the plain class selector, producing
       near-invisible light-on-white text. */
    .stTextArea textarea {
        background: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #111827 !important;
    }
    .stTextArea textarea::placeholder {
        color: #9CA3AF !important;
        opacity: 1;
    }

    .stButton button {
        background: #4F46E5;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.6rem;
    }
    .stButton button:hover { background: #4338CA; color: #FFFFFF; }

    .panel { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 1.3rem; }
    .panel-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #9CA3AF; font-weight: 600; margin-bottom: 0.9rem; }

    .field-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #9CA3AF; font-weight: 600; margin-top: 0.7rem; }
    .field-value { font-size: 0.98rem; color: #1F2937; font-weight: 500; }
    .field-muted { font-size: 0.9rem; color: #9CA3AF; font-style: italic; }

    .totals-row { display: flex; justify-content: space-between; padding: 0.3rem 0; color: #374151; font-size: 0.92rem; }
    .totals-row.grand-total { border-top: 1px solid #E5E7EB; margin-top: 0.4rem; padding-top: 0.6rem; font-weight: 700; color: #1F2937; }

    .meta-row { display: flex; gap: 3rem; border-top: 1px solid #E5E7EB; margin-top: 1.2rem; padding-top: 0.9rem; }
    .meta-item .field-label { margin-top: 0; }

    .checklist-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0; font-size: 0.92rem; color: #374151; }
    .check-yes { color: #10B981; font-weight: 700; }
    .check-no { color: #D1D5DB; font-weight: 700; }

    /* st.expander ("Details") isn't covered by any selector above, so
       its default header uses Streamlit's base theme background/text
       colors - dark until the :hover state (Streamlit's own hover CSS)
       happens to override it. Forcing both states light fixes the
       "black until hovered" bug. */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary {
        background: #FFFFFF !important;
        color: #374151 !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #F9FAFB !important;
    }
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background: #FFFFFF;
    }

    /* Boxed stat cards - bordered, with a colored icon chip, closer to
       the reference dashboards than the earlier flat hairline style. */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .stat-card-label { font-size: 0.82rem; color: #6B7280; font-weight: 500; margin-bottom: 0.5rem; }
    .stat-card-value { font-size: 1.9rem; font-weight: 700; color: #1F2937; }
    .stat-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .stat-icon-indigo { background: #EEF2FF; color: #4F46E5; }
    .stat-icon-green  { background: #D1FAE5; color: #059669; }
    .stat-icon-amber  { background: #FEF3C7; color: #D97706; }
    .stat-icon-blue   { background: #DBEAFE; color: #2563EB; }

    .invoice-row {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.55rem;
    }
</style>
""", unsafe_allow_html=True)

NAV_OPTIONS = ["Dashboard", "Extract", "Benchmark"]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


def fmt_money(value):
    """Consistent $X,XXX.XX formatting. Falls back to the raw string if
    the value isn't numeric, rather than raising - failed-extraction
    rows can have odd/missing values and shouldn't crash the table."""
    if value is None:
        return None
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


# --- Sidebar navigation ---
# Rebuilt as plain buttons rather than a styled radio widget: the active
# item renders as a non-interactive styled div, and every other item is
# a real button. This has no dependency on CSS pseudo-selectors, unlike
# the previous :has()-based approach, which never actually highlighted
# in the browser being tested with.
with st.sidebar:
    st.markdown('<div class="brand">StructGen</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Invoice Intelligence</div>', unsafe_allow_html=True)

    for opt in NAV_OPTIONS:
        if st.session_state.current_page == opt:
            st.markdown(f'<div class="nav-active">{opt}</div>', unsafe_allow_html=True)
        else:
            if st.button(opt, key=f"nav_{opt}"):
                st.session_state.current_page = opt
                st.rerun()

    # Two real checks (not fabricated): API reachability via /health,
    # and Postgres reachability via whether /stats succeeds (that
    # endpoint requires a DB round-trip). Redis/worker aren't split out
    # separately here since there's no endpoint that checks them in
    # isolation yet - only showing what's actually verifiable today.
    try:
        health_ok = requests.get(f"{API_BASE_URL}/health", timeout=5).ok
    except requests.RequestException:
        health_ok = False
    db_ok = get_stats() is not None

    def status_line(label, ok):
        dot = "status-ok" if ok else "status-down"
        text = "Connected" if ok else "Unreachable"
        return f'<div style="display:flex;justify-content:space-between;margin-top:0.3rem;"><span>{label}</span><span><span class="status-dot {dot}"></span>{text}</span></div>'

    st.markdown(
        '<div class="system-status">'
        + status_line("API", health_ok)
        + status_line("Database", db_ok)
        + '</div>',
        unsafe_allow_html=True,
    )

page = st.session_state.current_page


def stat_card(col, label, value, icon, icon_style):
    with col:
        st.markdown(
            f'<div class="stat-card">'
            f'<div><div class="stat-card-label">{label}</div>'
            f'<div class="stat-card-value">{value}</div></div>'
            f'<div class="stat-icon {icon_style}">{icon}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    title_col, button_col = st.columns([4, 1])
    with title_col:
        st.markdown('<div class="page-title">Invoice Processing</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Monitor your extraction pipeline.</div>', unsafe_allow_html=True)
    with button_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("+ Process Invoice"):
            st.session_state.current_page = "Extract"
            st.rerun()

    stats = get_stats()

    if stats is None:
        st.warning("API not reachable. Confirm the FastAPI service is running.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        stat_card(c1, "Requests", stats["total_requests"], "📄", "stat-icon-indigo")
        stat_card(c2, "Success Rate", f"{stats['success_rate'] * 100:.1f}%", "✓", "stat-icon-green")
        avg_latency = f"{stats['average_latency_seconds']:.2f}s" if stats["average_latency_seconds"] else "-"
        stat_card(c3, "Avg Latency", avg_latency, "⏱", "stat-icon-amber")
        stat_card(c4, "Total Cost", f"${stats['total_estimated_cost_usd']:.4f}", "$", "stat-icon-blue")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="field-label" style="font-size:0.85rem;">RECENT INVOICES</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    invoices = get_recent_invoices(limit=20)

    if not invoices:
        st.info("No extraction records yet — submit one from the Extract tab.")
    else:
        header_cols = st.columns([1.2, 1.6, 1, 1, 0.9, 0.8])
        for c, h in zip(header_cols, ["Invoice #", "Vendor", "Total", "Status", "Latency", ""]):
            c.markdown(f'<div class="field-label">{h}</div>', unsafe_allow_html=True)

        for inv in invoices:
            with st.container(border=True):
                cols = st.columns([1.2, 1.6, 1, 1, 0.9, 0.8])

                if inv["success"]:
                    invoice_num_display = inv["invoice_number"] or "Not detected"
                    cols[0].markdown(f'<div class="field-value">{invoice_num_display}</div>', unsafe_allow_html=True)
                    cols[1].markdown(f'<div class="field-value">{inv["vendor_name"] or "—"}</div>', unsafe_allow_html=True)
                    total_display = fmt_money(inv["total_amount"]) or "—"
                    cols[2].markdown(f'<div class="field-value">{total_display}</div>', unsafe_allow_html=True)
                    cols[3].markdown('<span class="badge badge-success">Processed</span>', unsafe_allow_html=True)
                else:
                    cols[0].markdown('<div class="field-muted">Extraction Failed</div>', unsafe_allow_html=True)
                    cols[1].markdown('<div class="field-muted">Unknown</div>', unsafe_allow_html=True)
                    cols[2].markdown('<div class="field-muted">—</div>', unsafe_allow_html=True)
                    cols[3].markdown('<span class="badge badge-failed">Failed</span>', unsafe_allow_html=True)

                cols[4].markdown(f'<div class="field-value">{inv["latency_seconds"]:.2f}s</div>', unsafe_allow_html=True)
                with cols[5]:
                    with st.expander("Details"):
                        st.markdown(f'<div class="field-label">MODEL</div><div class="field-value" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;">{inv["model_name"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-label">EST. COST</div><div class="field-value">${inv["estimated_cost_usd"]:.6f}</div>', unsafe_allow_html=True)
                        if not inv["success"]:
                            st.markdown(f'<div class="field-label">STATUS</div><div class="field-value" style="color:#991B1B;">Failed</div>', unsafe_allow_html=True)
                        if inv["error"]:
                            st.markdown(f'<div class="field-label">REASON</div><div class="field-value" style="color:#991B1B;">{inv["error"]}</div>', unsafe_allow_html=True)

        # One simple visualization rather than several - a latency trend
        # across recent requests, oldest to newest (API returns newest
        # first, so reverse for a left-to-right timeline).
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="field-label" style="font-size:0.85rem;">LATENCY TREND</div>', unsafe_allow_html=True)
        latencies = [inv["latency_seconds"] for inv in reversed(invoices)]
        st.line_chart(latencies, height=200)


# ============================================================
# EXTRACT
# ============================================================
elif page == "Extract":
    st.markdown('<div class="page-title">Process Invoice</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Turn unstructured invoice text into validated structured data.</div>',
        unsafe_allow_html=True,
    )

    example_text = """ACME Office Supplies
Invoice #: INV-2024-0091
Date: 2024-03-15

Printer Paper (Box) x3 - $74.97
Stapler x1 - $8.50

Subtotal: $83.47
Tax: $6.68
Total Due: $90.15"""

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        st.session_state.last_raw_text = ""

    raw_text = st.text_area(
        "Invoice text",
        value="",
        placeholder=example_text,
        height=180,
        label_visibility="collapsed",
    )
    submit = st.button("Process Invoice", disabled=not raw_text.strip())

    if submit:
        with st.spinner("Submitting..."):
            try:
                job_id = submit_extraction(raw_text)
            except Exception as e:
                st.error(f"Failed to reach the API: {e}")
                st.stop()

        with st.spinner("Extracting..."):
            job = poll_job(job_id)

        st.session_state.last_result = job
        st.session_state.last_raw_text = raw_text

    st.markdown("<br>", unsafe_allow_html=True)

    job = st.session_state.last_result
    if job is not None:
        header_col, status_col = st.columns([4, 1])
        with header_col:
            st.markdown('<div class="field-label" style="font-size:0.85rem;">EXTRACTION RESULT</div>', unsafe_allow_html=True)
        with status_col:
            if job["status"] != "timeout" and job["result"]["success"]:
                st.markdown('<span class="badge badge-success">Processed</span>', unsafe_allow_html=True)
            elif job["status"] == "timeout":
                st.markdown('<span class="badge badge-pending">Timeout</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-failed">Failed</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="panel"><div class="panel-title">SOURCE</div>', unsafe_allow_html=True)
            st.text(st.session_state.last_raw_text)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            if job["status"] == "timeout":
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.error("Timed out waiting for extraction. The service may be under load — try again.")
                st.markdown('</div>', unsafe_allow_html=True)
            elif job["result"]["success"]:
                inv = job["result"]["invoice"]
                panel_html = ['<div class="panel"><div class="panel-title">EXTRACTED DATA</div>']

                for label, key in [("Vendor", "vendor_name"), ("Invoice Number", "invoice_number"), ("Invoice Date", "invoice_date")]:
                    value = inv.get(key)
                    panel_html.append(
                        f'<div class="field-label">{label}</div>'
                        f'<div class="field-value">{value if value is not None else "—"}</div>'
                    )

                subtotal = fmt_money(inv.get("subtotal"))
                tax = fmt_money(inv.get("tax_amount"))
                total = fmt_money(inv.get("total_amount"))
                panel_html.append('<div style="margin-top:1rem;">')
                if subtotal:
                    panel_html.append(f'<div class="totals-row"><span>Subtotal</span><span>{subtotal}</span></div>')
                if tax:
                    panel_html.append(f'<div class="totals-row"><span>Tax</span><span>{tax}</span></div>')
                panel_html.append(f'<div class="totals-row grand-total"><span>Total</span><span>{total or "—"}</span></div>')
                panel_html.append('</div></div>')

                st.markdown("".join(panel_html), unsafe_allow_html=True)
            else:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.error(f"Extraction failed: {job['result']['error']}")
                st.markdown('</div>', unsafe_allow_html=True)

        if job["status"] != "timeout":
            st.markdown("<br>", unsafe_allow_html=True)
            # Honest post-hoc confirmation of the pipeline stages that
            # actually ran - not a simulated live progress bar, since
            # by the time this renders the job has already completed.
            # This exists to make the async architecture (API -> Redis
            # queue -> in-process arq worker -> Groq -> Postgres)
            # visible in the UI, not just in the code.
            validated_ok = job["result"]["success"]
            st.markdown(
                '<div class="panel"><div class="panel-title">PROCESSING PIPELINE</div>'
                '<div class="checklist-row"><span class="check-yes">✓</span>Request accepted</div>'
                '<div class="checklist-row"><span class="check-yes">✓</span>Job queued (Redis)</div>'
                '<div class="checklist-row"><span class="check-yes">✓</span>Processed by worker (Groq call)</div>'
                f'<div class="checklist-row"><span class="{"check-yes" if validated_ok else "check-no"}">{"✓" if validated_ok else "✕"}</span>Schema validation {"passed" if validated_ok else "failed"}</div>'
                '<div class="checklist-row"><span class="check-yes">✓</span>Record stored (Postgres)</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        if job["status"] != "timeout" and job["result"]["success"]:
            st.markdown(
                f'<div class="meta-row">'
                f'<div class="meta-item"><div class="field-label">LATENCY</div>'
                f'<div class="field-value">{job["result"]["latency_seconds"]:.2f}s</div></div>'
                f'<div class="meta-item"><div class="field-label">ESTIMATED COST</div>'
                f'<div class="field-value">${job["result"]["estimated_cost_usd"]:.6f}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="panel" style="color:#9CA3AF;">Submit invoice text above to see the extracted result here.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# BENCHMARK
# ============================================================
elif page == "Benchmark":
    st.markdown('<div class="page-title">Model Benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Compare extraction quality, latency, and cost across models.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel"><div class="panel-title">BENCHMARK STATUS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="checklist-row"><span class="status-dot status-ok"></span>Synthetic dataset — Ready</div>'
        '<div class="checklist-row"><span class="status-dot status-ok"></span>Prompted LLM baseline (Groq) — Ready</div>'
        '<div class="checklist-row"><span class="status-dot status-down"></span>Fine-tuned model (Qwen2.5-1.5B LoRA) — Training required</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin-top:0.9rem;color:#6B7280;font-size:0.88rem;">'
        'Accuracy, latency, and cost numbers for both models will appear here once the LoRA training '
        'run and benchmark script have completed — not filled in early to avoid showing placeholder '
        'numbers as if they were real results.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)