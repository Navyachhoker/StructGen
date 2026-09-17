"""
StructGen Streamlit Dashboard

UI for the Invoice Extractor system.

Views:
    - Dashboard
    - Extract
    - Benchmark

The Streamlit app is intentionally kept as a presentation layer.
All extraction logic remains inside the FastAPI + arq backend.
"""

import os
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from api_client import (
    get_recent_invoices,
    get_stats,
    poll_job,
    submit_extraction,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StructGen",
    page_icon="SG",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "StructGen"
APP_SUBTITLE = "Invoice Intelligence"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

PRIMARY_MODEL = "qwen-invoice-lora-v3"
FALLBACK_MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------

# These are the final corrected benchmark results from the project.
# Keep dataset labels explicit because Synthetic and Real are different
# evaluation sets.

BENCHMARK_RESULTS = [
    {
        "Model": "Groq",
        "Dataset": "Synthetic",
        "Valid JSON": 100.00,
        "Field Accuracy": 89.80,
        "Item Recall": 94.92,
        "Item Precision": 100.00,
        "Complete Item": 83.05,
    },
    {
        "Model": "Groq",
        "Dataset": "Real",
        "Valid JSON": 100.00,
        "Field Accuracy": 79.12,
        "Item Recall": 97.56,
        "Item Precision": 95.24,
        "Complete Item": 60.98,
    },
    {
        "Model": "LoRA Qwen v3",
        "Dataset": "Real",
        "Valid JSON": 92.31,
        "Field Accuracy": 54.95,
        "Item Recall": 58.54,
        "Item Precision": 70.59,
        "Complete Item": 29.27,
    },
    {
        "Model": "Base Qwen",
        "Dataset": "Real",
        "Valid JSON": 92.31,
        "Field Accuracy": 46.15,
        "Item Recall": 60.98,
        "Item Precision": 92.59,
        "Complete Item": 24.39,
    },
]

BENCHMARK_DF = pd.DataFrame(BENCHMARK_RESULTS)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* ------------------------------------------------------------------ */
    /* Global                                                              */
    /* ------------------------------------------------------------------ */

    .stApp {
        background: #f6f7fb;
        color: #172033;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #172033;
        letter-spacing: -0.02em;
    }

    p {
        color: #667085;
    }

    /* ------------------------------------------------------------------ */
    /* Sidebar                                                             */
    /* ------------------------------------------------------------------ */

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        padding: 0.25rem 0.75rem 1.5rem 0.75rem;
    }

    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.15rem;
    }

    .sidebar-brand-subtitle {
        font-size: 0.72rem;
        color: #98a2b3;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-section {
        font-size: 0.68rem;
        color: #98a2b3;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.5rem 0.75rem 0.4rem 0.75rem;
    }

    /* ------------------------------------------------------------------ */
    /* Header                                                              */
    /* ------------------------------------------------------------------ */

    .page-header {
        margin-bottom: 1.5rem;
    }

    .page-kicker {
        color: #667085;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.15;
        color: #172033;
        margin-bottom: 0.35rem;
    }

    .page-description {
        color: #667085;
        font-size: 0.95rem;
        max-width: 800px;
    }

    /* ------------------------------------------------------------------ */
    /* Cards                                                               */
    /* ------------------------------------------------------------------ */

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .card-title {
        color: #172033;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .card-subtitle {
        color: #98a2b3;
        font-size: 0.78rem;
        margin-bottom: 1rem;
    }

    /* ------------------------------------------------------------------ */
    /* KPI cards                                                           */
    /* ------------------------------------------------------------------ */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.15rem 1.2rem;
        min-height: 118px;
    }

    .metric-label {
        color: #667085;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: #172033;
        font-size: 1.65rem;
        font-weight: 750;
        line-height: 1.1;
    }

    .metric-note {
        color: #98a2b3;
        font-size: 0.72rem;
        margin-top: 0.45rem;
    }

    /* ------------------------------------------------------------------ */
    /* Section headings                                                    */
    /* ------------------------------------------------------------------ */

    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #172033;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* ------------------------------------------------------------------ */
    /* Status badges                                                       */
    /* ------------------------------------------------------------------ */

    .status-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.6rem;
        font-size: 0.68rem;
        font-weight: 700;
        line-height: 1;
    }

    .status-success {
        background: #ecfdf3;
        color: #027a48;
    }

    .status-failed {
        background: #fef3f2;
        color: #b42318;
    }

    .status-warning {
        background: #fffaeb;
        color: #b54708;
    }

    .status-neutral {
        background: #f2f4f7;
        color: #475467;
    }

    /* ------------------------------------------------------------------ */
    /* Extraction result                                                   */
    /* ------------------------------------------------------------------ */

    .result-field {
        background: #f8fafc;
        border: 1px solid #eaecf0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.65rem;
    }

    .field-label {
        color: #98a2b3;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .field-value {
        color: #172033;
        font-size: 0.88rem;
        font-weight: 600;
        word-break: break-word;
    }

    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-top: 0.8rem;
    }

    .meta-item {
        flex: 1;
        min-width: 150px;
        background: #f8fafc;
        border: 1px solid #eaecf0;
        border-radius: 8px;
        padding: 0.7rem;
    }

    /* ------------------------------------------------------------------ */
    /* Pipeline                                                            */
    /* ------------------------------------------------------------------ */

    .pipeline {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }

    .checklist-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #475467;
        font-size: 0.82rem;
    }

    .check-yes {
        color: #039855;
        font-weight: 700;
    }

    .check-neutral {
        color: #98a2b3;
        font-weight: 700;
    }

    /* ------------------------------------------------------------------ */
    /* Benchmark cards                                                     */
    /* ------------------------------------------------------------------ */

    .benchmark-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
    }

    .benchmark-model {
        font-size: 0.95rem;
        font-weight: 700;
        color: #172033;
    }

    .benchmark-dataset {
        font-size: 0.7rem;
        color: #667085;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .benchmark-score {
        font-size: 1.45rem;
        font-weight: 750;
        color: #172033;
        margin-top: 0.75rem;
    }

    .benchmark-label {
        color: #98a2b3;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .finding-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #4f46e5;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0;
    }

    .improvement-row {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }

    .improvement-before {
        font-size: 1.5rem;
        font-weight: 700;
        color: #98a2b3;
    }

    .improvement-arrow {
        font-size: 1.3rem;
        color: #98a2b3;
    }

    .improvement-after {
        font-size: 2rem;
        font-weight: 750;
        color: #172033;
    }

    .improvement-delta {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        background: #ecfdf3;
        color: #027a48;
    }

    .finding-title {
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.35rem;
    }

    .finding-text {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ------------------------------------------------------------------ */
    /* Buttons                                                             */
    /* ------------------------------------------------------------------ */

    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
        min-height: 2.5rem;
    }

    /* ------------------------------------------------------------------ */
    /* Text area                                                           */
    /* ------------------------------------------------------------------ */

    textarea {
        background: #ffffff !important;
        color: #172033 !important;
        border-color: #d0d5dd !important;
        font-family: "IBM Plex Mono", monospace !important;
        font-size: 0.82rem !important;
        line-height: 1.55 !important;
    }

    /* ------------------------------------------------------------------ */
    /* Tables                                                              */
    /* ------------------------------------------------------------------ */

    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
    }

    .recent-table-wrap {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .recent-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }

    .recent-table th {
        text-align: left;
        color: #667085;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.7rem 0.9rem;
        background: #f8fafc;
        border-bottom: 1px solid #e5e7eb;
    }

    .recent-table td {
        color: #172033;
        padding: 0.65rem 0.9rem;
        border-bottom: 1px solid #f2f4f7;
    }

    .recent-table tr:last-child td {
        border-bottom: none;
    }

    /* ------------------------------------------------------------------ */
    /* Alerts                                                              */
    /* ------------------------------------------------------------------ */

    .stAlert {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def safe_value(value: Any, default: str = "Not detected") -> str:
    """Return a display-safe value for missing or empty fields."""
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text


def format_currency(value: Any, currency: Any = None) -> str:
    """Format monetary values consistently."""
    if value is None or str(value).strip() == "":
        return "Not detected"

    currency_code = safe_value(currency, "")

    try:
        amount = float(value)

        if currency_code:
            return f"{currency_code} {amount:,.2f}"

        return f"{amount:,.2f}"

    except (TypeError, ValueError):
        if currency_code:
            return f"{currency_code} {value}"

        return str(value)


def format_latency(value: Any) -> str:
    """Format latency values."""
    if value is None:
        return "Not available"

    try:
        return f"{float(value):.2f}s"

    except (TypeError, ValueError):
        return str(value)


def format_cost(value: Any) -> str:
    """Format estimated cost."""
    if value is None:
        return "$0.000000"

    try:
        return f"${float(value):.6f}"

    except (TypeError, ValueError):
        return str(value)


def render_page_header(
    kicker: str,
    title: str,
    description: str,
) -> None:
    """Render the shared page header."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-kicker">{kicker}</div>
            <div class="page-title">{title}</div>
            <div class="page-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(
    label: str,
    value: str,
    note: str = "",
) -> None:
    """Render a KPI metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_improvement(
    label: str,
    before_value: float,
    after_value: float,
    before_label: str = "Base Qwen",
    after_label: str = "LoRA Qwen v3",
) -> None:
    """Render a prominent before → after comparison."""
    delta = after_value - before_value

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{label}</div>
            <div class="card-subtitle">
                {before_label} → {after_label}
            </div>
            <div class="improvement-row">
                <div class="improvement-before">{before_value:.2f}%</div>
                <div class="improvement-arrow">→</div>
                <div class="improvement-after">{after_value:.2f}%</div>
                <span class="improvement-delta">
                    +{delta:.2f} percentage points
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_field(label: str, value: Any) -> None:
    """Render a single extracted field."""
    st.markdown(
        f"""
        <div class="result-field">
            <div class="field-label">{label}</div>
            <div class="field-value">{safe_value(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(status: str) -> str:
    """Return HTML for a status badge."""
    normalized = str(status).lower()

    if normalized in {"success", "complete", "completed"}:
        return (
            '<span class="status-badge status-success">'
            "SUCCESS"
            "</span>"
        )

    if normalized in {"failed", "error"}:
        return (
            '<span class="status-badge status-failed">'
            "FAILED"
            "</span>"
        )

    if normalized in {"queued", "in_progress", "pending"}:
        return (
            '<span class="status-badge status-warning">'
            "PROCESSING"
            "</span>"
        )

    return (
        '<span class="status-badge status-neutral">'
        f"{str(status).upper()}"
        "</span>"
    )


def get_invoice_from_result(result: dict) -> dict:
    """Safely retrieve the invoice payload."""
    invoice = result.get("invoice")

    if isinstance(invoice, dict):
        return invoice

    return {}


def derive_record_status(item: dict) -> str:
    """Derive display status from the persisted success boolean."""
    if item.get("success") is True:
        return "success"

    if item.get("success") is False:
        return "failed"

    return "unknown"


def render_recent_table(rows: list[dict]) -> None:
    """Render the Recent Extractions table as hand-built HTML."""
    body_rows = "".join(
        f"""
        <tr>
            <td>{row['Invoice']}</td>
            <td>{row['Vendor']}</td>
            <td>{row['Total']}</td>
            <td>{render_status(row['StatusKey'])}</td>
            <td>{row['Model']}</td>
            <td>{row['Latency']}</td>
        </tr>
        """
        for row in rows
    )

    st.markdown(
        f"""
        <div class="recent-table-wrap">
            <table class="recent-table">
                <thead>
                    <tr>
                        <th>Invoice</th>
                        <th>Vendor</th>
                        <th>Total</th>
                        <th>Status</th>
                        <th>Model</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {body_rows}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">StructGen</div>
            <div class="sidebar-brand-subtitle">
                Invoice Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Workspace</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Extract",
            "Benchmark",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section">System</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"API: {API_BASE_URL}")

    # Simple API health indicator.
    try:
        import requests

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=10,
        )

        if response.ok:
            st.success("API connected")
        else:
            st.warning("API unavailable")

    except requests.RequestException:
        st.error("API offline")

    st.markdown("---")

    st.caption(
        "Primary: Qwen2.5-1.5B + LoRA v3"
    )

    st.caption(
        "Fallback: Groq GPT-OSS-120B"
    )


# ===========================================================================
# DASHBOARD
# ===========================================================================

if page == "Dashboard":

    render_page_header(
        "Overview",
        "Invoice extraction dashboard",
        "Monitor extraction activity, model usage, latency, and recent results.",
    )

    # -----------------------------------------------------------------------
    # KPI cards
    # -----------------------------------------------------------------------

    stats = get_stats()

    if stats is None:
        st.warning(
            "Dashboard statistics are currently unavailable. "
            "Make sure the FastAPI service and database are running."
        )

        stats = {}

    total_requests = stats.get(
        "total_requests",
        stats.get("requests", 0),
    )

    successful_requests = stats.get(
        "successful_requests",
        stats.get("successful", 0),
    )

    success_rate = stats.get(
        "success_rate",
        0,
    )

    avg_latency = stats.get(
        "avg_latency_seconds",
        stats.get("average_latency_seconds", 0),
    )

    total_cost = stats.get(
        "total_estimated_cost_usd",
        stats.get("total_cost_usd", 0),
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric(
            "Requests",
            str(total_requests),
            "Total extraction jobs",
        )

    with col2:
        if success_rate:
            success_display = f"{float(success_rate):.1f}%"

        elif total_requests:
            success_display = (
                f"{(successful_requests / total_requests) * 100:.1f}%"
            )

        else:
            success_display = "0.0%"

        render_metric(
            "Success Rate",
            success_display,
            "Completed successfully",
        )

    with col3:
        render_metric(
            "Avg Latency",
            format_latency(avg_latency),
            "Across recorded jobs",
        )

    with col4:
        render_metric(
            "Estimated Cost",
            format_cost(total_cost),
            "Recorded model cost",
        )

    # -----------------------------------------------------------------------
    # Quick action
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Quick action</div>',
        unsafe_allow_html=True,
    )

    action_col1, action_col2 = st.columns([1, 3])

    with action_col1:

        if st.button(
            "Process Invoice",
            type="primary",
            width="stretch",
        ):
            st.session_state["page_override"] = "Extract"
            st.rerun()

    with action_col2:

        st.markdown(
            """
            <div style="
                padding:0.65rem 0.9rem;
                color:#667085;
                font-size:0.82rem;
            ">
                Submit invoice or receipt text and StructGen will process it
                asynchronously through the FastAPI + arq pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Recent invoices
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Recent extractions</div>',
        unsafe_allow_html=True,
    )

    recent = get_recent_invoices(limit=20)

    if recent is None:
        st.info("No recent extraction records are available.")

    elif not recent:
        st.info(
            "No extraction records yet. Process an invoice to populate this table."
        )

    else:

        table_rows = []

        for item in recent:

            invoice_number = safe_value(
                item.get("invoice_number")
            )

            vendor_name = safe_value(
                item.get("vendor_name")
            )

            total_amount = item.get(
                "total_amount"
            )

            currency = item.get(
                "currency"
            )

            status_key = derive_record_status(item)

            latency = item.get(
                "latency_seconds"
            )

            model_name = safe_value(
                item.get("model_name"),
                "Unknown",
            )

            table_rows.append(
                {
                    "Invoice": invoice_number,
                    "Vendor": vendor_name,
                    "Total": format_currency(
                        total_amount,
                        currency,
                    ),
                    "StatusKey": status_key,
                    "Model": model_name,
                    "Latency": format_latency(latency),
                }
            )

        render_recent_table(table_rows)

    # -----------------------------------------------------------------------
    # Latency chart
    # -----------------------------------------------------------------------

    if recent:

        # Built in chronological order (oldest first) so "Job 1" reads
        # left-to-right the way a trend should.
        latency_rows = []

        for item in reversed(recent):

            latency = item.get("latency_seconds")

            if latency is None:
                continue

            latency_rows.append(
                {
                    "Invoice": safe_value(
                        item.get("invoice_number"),
                        "Unknown",
                    ),
                    "Latency (s)": float(latency),
                }
            )

        if latency_rows:

            st.markdown(
                '<div class="section-heading">Latency trend</div>',
                unsafe_allow_html=True,
            )

            latency_df = pd.DataFrame(latency_rows)

            latency_df["Job"] = [
                f"Job {i + 1}"
                for i in range(len(latency_df))
            ]

            latency_chart = (
                alt.Chart(latency_df)
                .mark_line(point=True, color="#4f46e5")
                .encode(
                    x=alt.X(
                        "Job:N",
                        sort=None,
                        title="Extraction order",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y(
                        "Latency (s):Q",
                        title="Latency (seconds)",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Invoice:N",
                            title="Invoice",
                        ),
                        alt.Tooltip(
                            "Latency (s):Q",
                            title="Latency",
                            format=".1f",
                        ),
                    ],
                )
                .properties(height=280)
            )

            st.altair_chart(
                latency_chart,
                width="stretch",
            )


# ===========================================================================
# EXTRACT
# ===========================================================================

elif page == "Extract":

    render_page_header(
        "Extraction",
        "Extract structured invoice data",
        "Submit raw invoice or receipt text and inspect the structured JSON returned by the model.",
    )

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------

    left_col, right_col = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with left_col:

        st.markdown(
            """
            <div class="card-title">Invoice / receipt text</div>
            <div class="card-subtitle">
                Paste the raw document text below.
            </div>
            """,
            unsafe_allow_html=True,
        )

        default_text = ""

        raw_text = st.text_area(
            "Invoice text",
            value=default_text,
            height=430,
            placeholder=(
                "Paste invoice or receipt text here...\n\n"
                "Example:\n"
                "TechMart\n"
                "Invoice No: INV-2001\n"
                "Date: 2026-09-16\n\n"
                "Laptop Stand  1  45.02  45.02\n"
                "Mousepad       2  26.32  52.64\n"
                "USB Hub        1  12.55  12.55\n\n"
                "Subtotal: 110.21\n"
                "Tax: 8.82\n"
                "Total: 119.03\n"
                "Currency: USD"
            ),
            label_visibility="collapsed",
        )

        process_button = st.button(
            "Process Invoice",
            type="primary",
            width="stretch",
        )

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------

    if process_button:

        if not raw_text.strip():

            st.error(
                "Please enter invoice or receipt text before processing."
            )

        else:

            try:

                with st.spinner(
                    "Submitting invoice to the extraction pipeline..."
                ):

                    job_id = submit_extraction(
                        raw_text
                    )

                st.session_state[
                    "last_job_id"
                ] = job_id

                st.session_state[
                    "last_source_text"
                ] = raw_text

                st.success(
                    f"Job submitted successfully: {job_id}"
                )

                status_placeholder = st.empty()

                def _show_live_status(body: dict) -> None:
                    live_status = body.get(
                        "status",
                        "unknown",
                    )

                    status_placeholder.markdown(
                        f"""
                        <div class="card" style="margin-bottom:0;">
                            <div class="card-title">
                                Processing invoice...
                            </div>

                            <div class="card-subtitle"
                                 style="margin-bottom:0;">
                                Job ID: {job_id}
                            </div>

                            <div style="margin-top:0.5rem;">
                                {render_status(live_status)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                job = poll_job(
                    job_id,
                    on_update=_show_live_status,
                )

                status_placeholder.empty()

                if job.get("status") == "complete":
                    st.success("✓ Extraction completed")

                st.session_state[
                    "last_job"
                ] = job

            except Exception as exc:

                st.error(
                    f"Extraction request failed: {exc}"
                )

    # -----------------------------------------------------------------------
    # Existing result
    # -----------------------------------------------------------------------

    job = st.session_state.get(
        "last_job"
    )

    source_text = st.session_state.get(
        "last_source_text",
        raw_text,
    )

    if job:

        status = job.get(
            "status",
            "unknown",
        )

        if status == "complete":

            result = job.get(
                "result"
            ) or {}

            invoice = get_invoice_from_result(
                result
            )

            success = result.get(
                "success",
                False,
            )

            if not success:

                st.error(
                    safe_value(
                        result.get("error"),
                        "Extraction failed.",
                    )
                )

            # ---------------------------------------------------------------
            # Result heading
            # ---------------------------------------------------------------

            st.markdown(
                '<div class="section-heading">Extraction result</div>',
                unsafe_allow_html=True,
            )

            source_col, result_col = st.columns(
                [0.9, 1.1],
                gap="large",
            )

            # ---------------------------------------------------------------
            # Source panel
            # ---------------------------------------------------------------

            with source_col:

                st.markdown(
                    """
                    <div class="card">
                        <div class="card-title">
                            Source document
                        </div>

                        <div class="card-subtitle">
                            Raw text submitted to the API.
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.text_area(
                    "Source",
                    value=source_text,
                    height=470,
                    disabled=True,
                    label_visibility="collapsed",
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ---------------------------------------------------------------
            # Structured result
            # ---------------------------------------------------------------

            with result_col:

                st.markdown(
                    """
                    <div class="card">
                        <div class="card-title">
                            Structured data
                        </div>

                        <div class="card-subtitle">
                            Validated invoice fields returned by the pipeline.
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                invoice_col1, invoice_col2 = st.columns(2)

                with invoice_col1:

                    render_field(
                        "Vendor",
                        invoice.get(
                            "vendor_name"
                        ),
                    )

                    render_field(
                        "Invoice Number",
                        invoice.get(
                            "invoice_number"
                        ),
                    )

                with invoice_col2:

                    render_field(
                        "Invoice Date",
                        invoice.get(
                            "invoice_date"
                        ),
                    )

                    render_field(
                        "Currency",
                        invoice.get(
                            "currency"
                        ),
                    )

                # -----------------------------------------------------------
                # Line items
                # -----------------------------------------------------------

                st.markdown(
                    '<div class="field-label" style="margin-top:0.9rem;">'
                    "LINE ITEMS"
                    "</div>",
                    unsafe_allow_html=True,
                )

                line_items = invoice.get(
                    "line_items",
                    [],
                )

                if line_items:

                    line_rows = []

                    currency = invoice.get(
                        "currency"
                    )

                    for item in line_items:

                        line_rows.append(
                            {
                                "Description": safe_value(
                                    item.get("description")
                                ),
                                "Qty": item.get(
                                    "quantity",
                                    "Not detected",
                                ),
                                "Unit Price": format_currency(
                                    item.get("unit_price"),
                                    currency,
                                ),
                                "Total": format_currency(
                                    item.get("total"),
                                    currency,
                                ),
                            }
                        )

                    line_df = pd.DataFrame(
                        line_rows
                    )

                    st.dataframe(
                        line_df,
                        width="stretch",
                        hide_index=True,
                    )

                else:

                    st.info(
                        "No line items detected."
                    )

                # -----------------------------------------------------------
                # Totals
                # -----------------------------------------------------------

                currency = invoice.get(
                    "currency"
                )

                subtotal = invoice.get(
                    "subtotal"
                )

                tax_amount = invoice.get(
                    "tax_amount"
                )

                total_amount = invoice.get(
                    "total_amount"
                )

                total_col1, total_col2, total_col3 = st.columns(3)

                with total_col1:

                    render_field(
                        "Subtotal",
                        format_currency(
                            subtotal,
                            currency,
                        ),
                    )

                with total_col2:

                    render_field(
                        "Tax",
                        format_currency(
                            tax_amount,
                            currency,
                        ),
                    )

                with total_col3:

                    render_field(
                        "Total",
                        format_currency(
                            total_amount,
                            currency,
                        ),
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------------------
            # Pipeline metadata
            # ----------------------------------------------------------------

            st.markdown(
                '<div class="section-heading">Processing details</div>',
                unsafe_allow_html=True,
            )

            model_name = result.get(
                "model_name",
                "unknown",
            )

            fallback_used = result.get(
                "fallback_used",
                False,
            )

            latency = result.get(
                "latency_seconds"
            )

            estimated_cost = result.get(
                "estimated_cost_usd"
            )

            st.markdown(
                f"""
                <div class="meta-row">

                    <div class="meta-item">
                        <div class="field-label">MODEL</div>
                        <div class="field-value">
                            {safe_value(model_name, "Unknown")}
                        </div>
                    </div>

                    <div class="meta-item">
                        <div class="field-label">LATENCY</div>
                        <div class="field-value">
                            {format_latency(latency)}
                        </div>
                    </div>

                    <div class="meta-item">
                        <div class="field-label">ESTIMATED COST</div>
                        <div class="field-value">
                            {format_cost(estimated_cost)}
                        </div>
                    </div>

                    <div class="meta-item">
                        <div class="field-label">FALLBACK</div>
                        <div class="field-value">
                            {"Yes — Groq" if fallback_used else "No"}
                        </div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------------------
            # Pipeline
            # ----------------------------------------------------------------

            st.markdown(
                '<div class="section-heading">Processing pipeline</div>',
                unsafe_allow_html=True,
            )

            fallback_text = (
                "Fallback activated — Groq"
                if fallback_used
                else "Primary model completed extraction"
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="pipeline">

                        <div class="checklist-row">
                            <span class="check-yes">✓</span>
                            Invoice text submitted to FastAPI
                        </div>

                        <div class="checklist-row">
                            <span class="check-yes">✓</span>
                            Job queued through arq
                        </div>

                        <div class="checklist-row">
                            <span class="check-yes">✓</span>
                            Worker processed extraction request
                        </div>

                        <div class="checklist-row">
                            <span class="check-yes">✓</span>
                            {fallback_text}
                        </div>

                        <div class="checklist-row">
                            <span class="check-yes">✓</span>
                            Pydantic schema validation completed
                        </div>

                        <div class="checklist-row">
                            <span class="check-yes">✓</span>
                            Result persisted to PostgreSQL
                        </div>

                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------------------
            # Raw JSON
            # ----------------------------------------------------------------

            with st.expander(
                "View raw API result"
            ):
                st.json(
                    result
                )

        elif status == "timeout":

            st.warning(
                "The job did not complete within the UI polling window. "
                "The worker may still be processing it."
            )

            job_id = job.get(
                "job_id"
            )

            if job_id:

                st.caption(
                    f"Job ID: {job_id}"
                )

        else:

            st.info(
                f"Job status: {status}"
            )


# ===========================================================================
# BENCHMARK
# ===========================================================================

elif page == "Benchmark":

    render_page_header(
        "Evaluation",
        "Model benchmark",
        "Compare the prompted LLM baseline with the fine-tuned Qwen model on synthetic and real invoice data.",
    )

    # -----------------------------------------------------------------------
    # Evaluation setup
    # -----------------------------------------------------------------------

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Evaluation setup</div>
            <div class="card-subtitle">
                The benchmark separates synthetic evaluation from the private
                real-receipt holdout.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_col1, setup_col2, setup_col3 = st.columns(3)

    with setup_col1:

        render_metric(
            "Synthetic Test",
            "28",
            "Synthetic examples",
        )

    with setup_col2:

        render_metric(
            "Real Holdout",
            "13",
            "Private Malaysian receipts",
        )

    with setup_col3:

        render_metric(
            "Fine-tuning Data",
            "205",
            "Synthetic training examples",
        )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Benchmark results</div>',
        unsafe_allow_html=True,
    )

    display_df = BENCHMARK_DF.copy()

    for column in [
        "Valid JSON",
        "Field Accuracy",
        "Item Recall",
        "Item Precision",
        "Complete Item",
    ]:

        display_df[column] = display_df[column].map(
            lambda value: f"{value:.2f}%"
        )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Real holdout comparison
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Real holdout comparison</div>',
        unsafe_allow_html=True,
    )

    real_df = BENCHMARK_DF[
        BENCHMARK_DF["Dataset"] == "Real"
    ].copy()

    metric_options = [
        "Field Accuracy",
        "Item Recall",
        "Item Precision",
        "Complete Item",
        "Valid JSON",
    ]

    selected_metric = st.selectbox(
        "Metric",
        metric_options,
    )

    chart_df = real_df[
        ["Model", selected_metric]
    ].copy()

    model_order = [
        "Groq",
        "LoRA Qwen v3",
        "Base Qwen",
    ]

    comparison_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "Model:N",
                sort=model_order,
                title=None,
            ),
            x=alt.X(
                f"{selected_metric}:Q",
                title=f"{selected_metric} (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "Model:N",
                sort=model_order,
                legend=None,
                scale=alt.Scale(
                    domain=model_order,
                    range=[
                        "#4f46e5",
                        "#039855",
                        "#98a2b3",
                    ],
                ),
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(
                    f"{selected_metric}:Q",
                    format=".2f",
                    title=selected_metric,
                ),
            ],
        )
        .properties(height=220)
    )

    st.altair_chart(
        comparison_chart,
        width="stretch",
    )

    # -----------------------------------------------------------------------
    # LoRA improvement
    # -----------------------------------------------------------------------

    lora_real = BENCHMARK_DF[
        (BENCHMARK_DF["Model"] == "LoRA Qwen v3")
        & (BENCHMARK_DF["Dataset"] == "Real")
    ].iloc[0]

    base_real = BENCHMARK_DF[
        (BENCHMARK_DF["Model"] == "Base Qwen")
        & (BENCHMARK_DF["Dataset"] == "Real")
    ].iloc[0]

    st.markdown(
        '<div class="section-heading">Fine-tuning effect</div>',
        unsafe_allow_html=True,
    )

    improve_col1, improve_col2 = st.columns(2)

    with improve_col1:

        render_improvement(
            "Field Accuracy",
            base_real["Field Accuracy"],
            lora_real["Field Accuracy"],
        )

    with improve_col2:

        render_improvement(
            "Complete Item Accuracy",
            base_real["Complete Item"],
            lora_real["Complete Item"],
        )

    # -----------------------------------------------------------------------
    # Findings
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Key findings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="finding-box">
            <div class="finding-title">
                LoRA improves the base Qwen model
            </div>
            <div class="finding-text">
                On the real holdout, LoRA Qwen v3 improves field accuracy
                from 46.15% to 54.95%, an 8.80 percentage-point increase.
                Complete-item accuracy increases from 24.39% to 29.27%.
            </div>
        </div>

        <div class="finding-box">
            <div class="finding-title">
                The real-world gap remains significant
            </div>
            <div class="finding-text">
                The fine-tuned Qwen model remains below the Groq baseline on
                the real holdout. This indicates that the synthetic training
                data does not fully capture the variability of real receipts.
            </div>
        </div>

        <div class="finding-box">
            <div class="finding-title">
                Synthetic and real results should not be treated as one dataset
            </div>
            <div class="finding-text">
                The Groq synthetic score is reported separately from the real
                holdout results. The real holdout is the more relevant measure
                for testing generalization to unseen receipt formats.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Detailed metric comparison
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Detailed real-holdout metrics</div>',
        unsafe_allow_html=True,
    )

    detailed_df = real_df[
        [
            "Model",
            "Valid JSON",
            "Field Accuracy",
            "Item Recall",
            "Item Precision",
            "Complete Item",
        ]
    ].copy()

    st.dataframe(
        detailed_df.style.format(
            {
                "Valid JSON": "{:.2f}%",
                "Field Accuracy": "{:.2f}%",
                "Item Recall": "{:.2f}%",
                "Item Precision": "{:.2f}%",
                "Complete Item": "{:.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Dataset notes
    # -----------------------------------------------------------------------

    with st.expander(
        "Evaluation notes"
    ):

        st.markdown(
            """
            **Synthetic evaluation**

            - 28 synthetic test examples
            - Groq valid JSON: 100%
            - Groq field accuracy: 89.80%
            - Line-item recall: 94.92%
            - Line-item precision: 100%
            - Complete-item accuracy: 83.05%

            **Real holdout**

            - 13 private Malaysian receipts
            - Groq valid JSON: 100%
            - LoRA Qwen v3 valid JSON: 92.31%
            - Base Qwen valid JSON: 92.31%

            **Important interpretation**

            The fine-tuning experiment demonstrates measurable improvement
            over the base Qwen model, while also showing a substantial
            synthetic-to-real generalization gap.
            """
        )