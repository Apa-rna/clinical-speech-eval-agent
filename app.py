import json
import subprocess
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CATEGORIES = ["NICU", "CVICU", "PICU", "ICU", "HUC", "ED"]
WER_RANGES = ["clean", "borderline", "failed"]

st.set_page_config(
    page_title="Clinical Speech Evaluation Agent",
    page_icon="🏥",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def get_reports() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("run_*.json"), reverse=True)


def load_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_datasets() -> list[Path]:
    return sorted(DATA_DIR.glob("*.json"))


def load_dataset(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_dataset(name: str, pairs: list[dict], kb: list[dict]) -> Path:
    if not name.endswith(".json"):
        name += ".json"
    path = DATA_DIR / name
    with open(path, "w") as f:
        json.dump({"transcript_pairs": pairs, "knowledge_base": kb}, f, indent=2)
    return path


def run_pipeline(data_path: str, thread_id: str = "session-ui") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", "main.py", "--data", data_path, "--thread-id", thread_id],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )


# ── sidebar nav ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 Clinical Speech\nEvaluation Agent")
    st.markdown("---")
    page = st.radio("Navigate", ["Dashboard", "Dataset Builder"], label_visibility="collapsed")
    st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET BUILDER
# ═════════════════════════════════════════════════════════════════════════════

if page == "Dataset Builder":
    st.title("Dataset Builder")
    st.caption("Create or edit transcript pair datasets, then run the pipeline.")

    # ── session state init ────────────────────────────────────────────────────
    if "ds_pairs" not in st.session_state:
        st.session_state.ds_pairs: list[dict] = []
    if "ds_kb" not in st.session_state:
        st.session_state.ds_kb: list[dict] = []

    # ── load existing dataset ─────────────────────────────────────────────────
    with st.expander("Load existing dataset", expanded=False):
        datasets = get_datasets()
        if datasets:
            sel_ds = st.selectbox("Choose file", [d.name for d in datasets])
            col_load, col_upload = st.columns(2)
            if col_load.button("Load", use_container_width=True):
                loaded = load_dataset(DATA_DIR / sel_ds)
                st.session_state.ds_pairs = loaded.get("transcript_pairs", [])
                st.session_state.ds_kb = loaded.get("knowledge_base", [])
                st.success(f"Loaded {len(st.session_state.ds_pairs)} pairs from {sel_ds}")
                st.rerun()
        else:
            st.info("No datasets found in data/ yet.")

        st.markdown("**Or upload a JSON file**")
        uploaded = st.file_uploader("Upload dataset JSON", type="json", label_visibility="collapsed")
        if uploaded:
            data = json.load(uploaded)
            st.session_state.ds_pairs = data.get("transcript_pairs", [])
            st.session_state.ds_kb = data.get("knowledge_base", [])
            st.success(f"Imported {len(st.session_state.ds_pairs)} pairs from upload.")
            st.rerun()

    st.markdown("---")

    # ── add new pair form ─────────────────────────────────────────────────────
    st.subheader("Add transcript pair")
    with st.form("add_pair", clear_on_submit=True):
        fc1, fc2 = st.columns([1, 3])
        category = fc1.selectbox("Category", CATEGORIES)
        pair_id = fc2.text_input("ID (leave blank to auto-generate)", placeholder="pair_001")

        ref = st.text_area("Reference transcript (what was said)", height=80)
        rec = st.text_area("Recognized transcript (what the model heard)", height=80)

        fc3, fc4 = st.columns(2)
        expected = fc3.selectbox("Expected WER range", WER_RANGES)
        audio_file = fc4.text_input("Audio file name (optional)", placeholder="recording_001.wav")

        submitted = st.form_submit_button("Add Pair", type="primary", use_container_width=True)

    if submitted:
        if not ref.strip() or not rec.strip():
            st.error("Reference and Recognized transcripts are required.")
        else:
            auto_id = pair_id.strip() or f"pair_{str(uuid.uuid4())[:8]}"
            auto_audio = audio_file.strip() or f"audio_{auto_id}.wav"
            existing_ids = {p["id"] for p in st.session_state.ds_pairs}
            if auto_id in existing_ids:
                st.error(f"ID '{auto_id}' already exists. Choose a different ID.")
            else:
                st.session_state.ds_pairs.append({
                    "id": auto_id,
                    "audio_file": auto_audio,
                    "category": category,
                    "reference": ref.strip(),
                    "recognized": rec.strip(),
                    "expected_wer_range": expected,
                })
                st.success(f"Added pair `{auto_id}` ({category})")
                st.rerun()

    st.markdown("---")

    # ── current dataset table ─────────────────────────────────────────────────
    pairs = st.session_state.ds_pairs
    st.subheader(f"Current dataset — {len(pairs)} pair(s)")

    if pairs:
        editor_df = pd.DataFrame(pairs)[["id", "category", "reference", "recognized", "expected_wer_range"]]
        edited = st.data_editor(
            editor_df,
            use_container_width=True,
            num_rows="dynamic",
            height=300,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, width="small"),
                "expected_wer_range": st.column_config.SelectboxColumn("Expected", options=WER_RANGES, width="small"),
                "reference": st.column_config.TextColumn("Reference", width="large"),
                "recognized": st.column_config.TextColumn("Recognized", width="large"),
            },
        )

        if st.button("Apply edits", use_container_width=False):
            updated = edited.to_dict(orient="records")
            for i, row in enumerate(updated):
                orig = st.session_state.ds_pairs[i] if i < len(st.session_state.ds_pairs) else {}
                row["audio_file"] = orig.get("audio_file", f"audio_{row['id']}.wav")
            st.session_state.ds_pairs = updated
            st.success("Edits applied.")
            st.rerun()

        # Per-row delete
        with st.expander("Delete a pair"):
            ids = [p["id"] for p in st.session_state.ds_pairs]
            del_id = st.selectbox("Select pair to delete", ids)
            if st.button("Delete", type="secondary"):
                st.session_state.ds_pairs = [p for p in st.session_state.ds_pairs if p["id"] != del_id]
                st.success(f"Deleted {del_id}")
                st.rerun()

        # Dataset stats preview
        preview_df = pd.DataFrame(st.session_state.ds_pairs)
        cats = preview_df["category"].value_counts()
        st.caption("Category breakdown: " + " · ".join(f"{k}: {v}" for k, v in cats.items()))

    else:
        st.info("No pairs yet. Add one above or load an existing dataset.")

    st.markdown("---")

    # ── save & run ────────────────────────────────────────────────────────────
    st.subheader("Save & Run")
    sc1, sc2 = st.columns([2, 1])
    ds_name = sc1.text_input("Dataset filename", value="my_dataset.json")
    use_default_kb = sc2.checkbox("Use default KB", value=True, help="Use the synthetic knowledge base (15 clinical docs)")

    c_save, c_run = st.columns(2)

    if c_save.button("Save dataset", use_container_width=True):
        if not pairs:
            st.error("Add at least one pair before saving.")
        else:
            kb_docs = []
            if use_default_kb:
                default_ds = DATA_DIR / "synthetic_transcripts.json"
                if default_ds.exists():
                    kb_docs = load_dataset(default_ds).get("knowledge_base", [])
            path = save_dataset(ds_name, st.session_state.ds_pairs, kb_docs)
            st.success(f"Saved to `{path.name}` ({len(pairs)} pairs)")

    if c_run.button("Save & Run pipeline", type="primary", use_container_width=True):
        if not pairs:
            st.error("Add at least one pair before running.")
        else:
            kb_docs = []
            if use_default_kb:
                default_ds = DATA_DIR / "synthetic_transcripts.json"
                if default_ds.exists():
                    kb_docs = load_dataset(default_ds).get("knowledge_base", [])
            path = save_dataset(ds_name, st.session_state.ds_pairs, kb_docs)
            with st.spinner(f"Running pipeline on {path.name}…"):
                result = run_pipeline(str(path), thread_id=f"ui-{path.stem}")
            if result.returncode == 0:
                st.success("Pipeline complete! Switch to **Dashboard** to view results.")
                st.code(result.stdout[-3000:])
            else:
                st.error("Pipeline failed.")
                st.code(result.stderr[-2000:])

    if pairs:
        with st.expander("Preview JSON"):
            preview = {"transcript_pairs": st.session_state.ds_pairs[:3], "knowledge_base": []}
            st.code(json.dumps(preview, indent=2), language="json")
            st.caption(f"Showing first 3 of {len(pairs)} pairs")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

else:
    with st.sidebar:
        if st.button("▶ Run Pipeline (synthetic data)", type="primary", use_container_width=True):
            with st.spinner("Running pipeline…"):
                proc = run_pipeline("data/synthetic_transcripts.json")
            if proc.returncode == 0:
                st.success("Pipeline complete!")
                st.rerun()
            else:
                st.error("Pipeline failed")
                st.code(proc.stderr[-2000:])

        st.markdown("---")
        reports = get_reports()
        if not reports:
            st.warning("No reports yet.\nClick **Run Pipeline** to generate one.")
            st.stop()

        selected_name = st.selectbox(
            "Select report",
            [r.stem for r in reports],
            format_func=lambda n: n.replace("run_", ""),
        )
        report = load_report(OUTPUT_DIR / f"{selected_name}.json")

    summary = report["summary"]
    results = report["results"]
    df = pd.DataFrame(results)

    st.title("Clinical Speech Evaluation Dashboard")
    st.caption(f"**{report['run_id']}** · {report['timestamp'][:19].replace('T', ' ')}")
    st.markdown("---")

    tab_overview, tab_results, tab_review, tab_history = st.tabs(
        ["Overview", "All Results", "Borderline & Failed", "Run History"]
    )

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_overview:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total", summary["total_pairs"])
        c2.metric("Clean", summary["clean"])
        c3.metric("Borderline", summary["borderline"])
        c4.metric("Failed", summary["failed"])
        c5.metric("Avg WER", f"{summary['avg_wer']:.1%}")
        c6.metric("Exact Match", f"{summary['exact_match_rate']:.1%}")

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Classification breakdown")
            pie_df = pd.DataFrame({
                "Classification": ["Clean", "Borderline", "Failed"],
                "Count": [summary["clean"], summary["borderline"], summary["failed"]],
            })
            fig = px.pie(
                pie_df, values="Count", names="Classification",
                color="Classification",
                color_discrete_map={"Clean": "#22c55e", "Borderline": "#f59e0b", "Failed": "#ef4444"},
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=10, b=10), legend_orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Avg WER by category")
            cat_df = df.groupby("category")["wer"].mean().reset_index()
            cat_df["wer_pct"] = (cat_df["wer"] * 100).round(2)
            fig = px.bar(
                cat_df.sort_values("wer_pct", ascending=False),
                x="category", y="wer_pct",
                labels={"wer_pct": "Avg WER (%)", "category": ""},
                color="wer_pct",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 50],
            )
            fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("WER distribution")
        fig = px.histogram(
            df, x=df["wer"] * 100, nbins=20,
            labels={"x": "WER (%)"},
            color_discrete_sequence=["#6366f1"],
        )
        fig.add_vline(x=20, line_dash="dash", line_color="#22c55e", annotation_text="clean threshold")
        fig.add_vline(x=50, line_dash="dash", line_color="#ef4444", annotation_text="fail threshold")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if report.get("run_comparison"):
            rc = report["run_comparison"]
            st.markdown("---")
            st.subheader("vs Previous run")
            pc1, pc2, pc3 = st.columns(3)
            delta = rc["avg_wer_delta"]
            pc1.metric("Prev run", rc["previous_run_id"].replace("run_", ""))
            pc2.metric("WER delta", f"{delta:+.2%}", delta_color="inverse")
            pc3.metric("Regressions", len(rc["regressions"]))
            if rc["regressions"]:
                st.warning("Regressions: " + ", ".join(rc["regressions"]))
            if rc["improvements"]:
                st.success("Improvements: " + ", ".join(rc["improvements"]))

    # ── All Results ───────────────────────────────────────────────────────────
    with tab_results:
        st.subheader("All Results")

        f1, f2, f3 = st.columns([2, 2, 1])
        cat_opts = sorted(df["category"].unique())
        sel_cats = f1.multiselect("Category", cat_opts, default=cat_opts)
        sel_cls = f2.multiselect("Classification", ["clean", "borderline", "failed"], default=["clean", "borderline", "failed"])
        search = f3.text_input("Search ID", "")

        mask = df["category"].isin(sel_cats) & df["classification"].isin(sel_cls)
        if search:
            mask &= df["id"].str.contains(search, case=False)
        filtered = df[mask].copy()

        filtered["WER"] = (filtered["wer"] * 100).round(1).astype(str) + "%"
        filtered["CER"] = (filtered["cer"] * 100).round(1).astype(str) + "%"
        filtered["exact"] = filtered["exact_match"].map({True: "✓", False: "✗"})
        filtered["decision"] = filtered["human_decision"].fillna("—")

        display = filtered[["id", "category", "classification", "WER", "CER", "exact", "decision", "reference", "recognized"]]
        display = display.rename(columns={"id": "ID", "category": "Cat", "classification": "Class",
                                          "exact": "Exact", "decision": "Decision",
                                          "reference": "Reference", "recognized": "Recognized"})

        def _row_color(row):
            c = {"clean": "#f0fdf4", "borderline": "#fffbeb", "failed": "#fef2f2"}.get(row["Class"], "")
            return [f"background-color: {c}" for _ in row]

        st.dataframe(display.style.apply(_row_color, axis=1), use_container_width=True, height=500)
        st.caption(f"Showing {len(filtered)} of {len(df)} pairs")

    # ── Borderline & Failed ───────────────────────────────────────────────────
    with tab_review:
        borderline = [r for r in results if r["classification"] == "borderline"]
        failed_cases = [r for r in results if r["classification"] == "failed"]

        if borderline:
            st.subheader(f"Borderline cases — {len(borderline)}")
            for case in borderline:
                wer_pct = case["wer"] * 100
                label = f"**{case['id']}** · {case['category']} · WER {wer_pct:.1f}%"
                if case.get("human_decision"):
                    label += f" · ✅ {case['human_decision']}"
                with st.expander(label):
                    r_col, g_col = st.columns(2)
                    r_col.markdown(f"**Reference**\n\n> {case['reference']}")
                    g_col.markdown(f"**Recognized**\n\n> {case['recognized']}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("WER", f"{case['wer']:.2%}")
                    m2.metric("CER", f"{case['cer']:.2%}")
                    m3.metric("Exact match", "Yes" if case["exact_match"] else "No")
                    if case.get("clinical_context"):
                        st.info(f"📋 {case['clinical_context']}")

        if failed_cases:
            st.subheader(f"Failed cases — {len(failed_cases)}")
            for case in failed_cases:
                with st.expander(f"**{case['id']}** · {case['category']} · WER {case['wer']:.1%}", expanded=True):
                    r_col, g_col = st.columns(2)
                    r_col.markdown(f"**Reference**\n\n> {case['reference']}")
                    g_col.markdown(f"**Recognized**\n\n> {case['recognized']}")
                    m1, m2 = st.columns(2)
                    m1.metric("WER", f"{case['wer']:.2%}")
                    m2.metric("CER", f"{case['cer']:.2%}")
                    if case.get("clinical_context"):
                        st.info(f"📋 {case['clinical_context']}")

        if not borderline and not failed_cases:
            st.success("No borderline or failed cases in this run.")

    # ── Run History ───────────────────────────────────────────────────────────
    with tab_history:
        history_path = OUTPUT_DIR / "runs_history.json"
        if not history_path.exists():
            st.info("No run history yet. Run the pipeline more than once to see trends.")
        else:
            with open(history_path) as f:
                history = json.load(f)

            if len(history) < 2:
                st.info("Run the pipeline at least twice to see trend comparisons.")
            else:
                rows = []
                for run in history:
                    s = run.get("summary", {})
                    rows.append({
                        "run_id": run["run_id"].replace("run_", ""),
                        "timestamp": run["timestamp"][:16].replace("T", " "),
                        "total": s.get("total_pairs", 0),
                        "clean": s.get("clean", 0),
                        "borderline": s.get("borderline", 0),
                        "failed": s.get("failed", 0),
                        "avg_wer": s.get("avg_wer", 0),
                        "exact_match_rate": s.get("exact_match_rate", 0),
                    })
                hist_df = pd.DataFrame(rows)

                st.subheader("WER trend across runs")
                fig = px.line(hist_df, x="timestamp", y="avg_wer", markers=True,
                              labels={"avg_wer": "Avg WER", "timestamp": "Run"},
                              color_discrete_sequence=["#6366f1"])
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Classification trend")
                trend = hist_df[["timestamp", "clean", "borderline", "failed"]].melt(
                    id_vars="timestamp", var_name="Class", value_name="Count"
                )
                fig = px.bar(trend, x="timestamp", y="Count", color="Class", barmode="stack",
                             color_discrete_map={"clean": "#22c55e", "borderline": "#f59e0b", "failed": "#ef4444"})
                fig.update_layout(margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Run history table")
                display_hist = hist_df.copy()
                display_hist["avg_wer"] = (display_hist["avg_wer"] * 100).round(2).astype(str) + "%"
                display_hist["exact_match_rate"] = (display_hist["exact_match_rate"] * 100).round(1).astype(str) + "%"
                st.dataframe(display_hist, use_container_width=True)
