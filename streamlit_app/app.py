"""
Streamlit demo — a thin UI over the Invoice Extractor API.

Flow: paste invoice text -> submit -> poll for the job to finish ->
display the extracted JSON. A sidebar shows aggregate stats pulled
straight from /stats.

Run with: streamlit run streamlit_app/app.py
"""

import streamlit as st

from api_client import submit_extraction, poll_job, get_stats

st.set_page_config(page_title="Invoice Extractor", page_icon="🧾")

st.title("🧾 Invoice Extractor")
st.caption("Paste raw invoice or receipt text below and extract structured data.")

# --- Sidebar: aggregate stats ---
with st.sidebar:
    st.header("📊 Stats")
    stats = get_stats()
    if stats is None:
        st.warning("API not reachable. Is the FastAPI server running?")
    elif stats["total_requests"] == 0:
        st.info("No requests yet — try extracting an invoice.")
    else:
        st.metric("Total requests", stats["total_requests"])
        st.metric("Success rate", f"{stats['success_rate'] * 100:.1f}%")
        if stats["average_latency_seconds"] is not None:
            st.metric("Avg latency", f"{stats['average_latency_seconds']:.2f}s")
        st.metric("Total est. cost", f"${stats['total_estimated_cost_usd']:.4f}")

# --- Main: extraction form ---
example_text = """ACME Office Supplies
Invoice #: INV-2024-0091
Date: 2024-03-15

Printer Paper (Box) x3 - $74.97
Stapler x1 - $8.50

Subtotal: $83.47
Tax: $6.68
Total Due: $90.15"""

raw_text = st.text_area(
    "Invoice / receipt text",
    value="",
    placeholder=example_text,
    height=220,
)

if st.button("Extract", type="primary", disabled=not raw_text.strip()):
    with st.spinner("Submitting job..."):
        try:
            job_id = submit_extraction(raw_text)
        except Exception as e:
            st.error(f"Failed to reach the API: {e}")
            st.stop()

    with st.spinner("Extracting... (this calls a real LLM, may take a few seconds)"):
        job = poll_job(job_id)

    if job["status"] == "timeout":
        st.error("Timed out waiting for extraction. The service may be under load — try again.")
    elif job["result"]["success"]:
        st.success("Extraction succeeded")
        col1, col2 = st.columns(2)
        col1.metric("Latency", f"{job['result']['latency_seconds']:.2f}s")
        col2.metric("Est. cost", f"${job['result']['estimated_cost_usd']:.6f}")
        st.json(job["result"]["invoice"])
    else:
        st.error(f"Extraction failed: {job['result']['error']}")