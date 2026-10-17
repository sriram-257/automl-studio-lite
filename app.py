"""AutoML Studio Lite: a focused local workspace for model comparison."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_diabetes

from src.automl import train_and_compare

ARTIFACTS_DIR = "artifacts"
MINIMUM_ROWS = 10
CLASSIFICATION_METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
)
REGRESSION_METRICS = ("mae", "rmse", "r2")


def configure_page() -> None:
    """Set page configuration and keep the visual system in one place."""
    st.set_page_config(
        page_title="AutoML Studio Lite",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --ink: #102a43;
            --muted: #627d98;
            --line: #d9e2ec;
            --surface: #ffffff;
            --canvas: #f5f8fb;
            --teal: #0f766e;
            --blue: #0b5cad;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #102a43;
            border-right: 1px solid #243b53;
        }
        [data-testid="stSidebar"] * { color: #f0f4f8; }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] small { color: #bcccdc !important; }
        [data-testid="stSidebar"] hr { border-color: #334e68; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #243b53;
            border-color: #486581;
        }
        .block-container { max-width: 1400px; padding: 2rem 3rem 4rem; }
        .product-mark {
            display: flex; align-items: center; gap: .7rem; margin-bottom: 1.8rem;
        }
        .product-symbol {
            display: grid; place-items: center; width: 2.2rem; height: 2.2rem;
            border-radius: .6rem; background: #0f766e; color: white;
            font-weight: 800; letter-spacing: -.04em;
        }
        .product-name { color: var(--ink); font-size: 1.15rem; font-weight: 750; }
        .product-kicker {
            color: var(--muted); font-size: .76rem; letter-spacing: .08em;
            text-transform: uppercase; margin-top: .1rem;
        }
        .page-title { color: var(--ink); font-size: 2rem; font-weight: 760; margin: 0; }
        .page-subtitle { color: var(--muted); font-size: 1rem; margin: .35rem 0 1.8rem; }
        .env-badge {
            display: inline-flex; align-items: center; gap: .4rem; padding: .32rem .65rem;
            border: 1px solid #b8e0dc; border-radius: 999px; color: #0f766e;
            background: #effcf9; font-size: .75rem; font-weight: 700;
        }
        .section-title {
            color: var(--ink); font-size: 1.2rem; font-weight: 730;
            margin: 1.75rem 0 .25rem;
        }
        .section-caption { color: var(--muted); font-size: .9rem; margin-bottom: 1rem; }
        .metric-card {
            background: var(--surface); border: 1px solid var(--line);
            border-radius: .65rem; padding: 1rem 1.1rem; min-height: 6.5rem;
            box-shadow: 0 1px 2px rgba(16, 42, 67, .03);
        }
        .metric-label {
            color: var(--muted); font-size: .73rem; font-weight: 700;
            letter-spacing: .06em; text-transform: uppercase;
        }
        .metric-value {
            color: var(--ink); font-size: 1.45rem; font-weight: 760;
            line-height: 1.25; margin-top: .55rem; overflow-wrap: anywhere;
        }
        .metric-note { color: var(--muted); font-size: .78rem; margin-top: .2rem; }
        .callout {
            background: #eef7ff; border: 1px solid #c8e1f7; border-left: 4px solid var(--blue);
            border-radius: .5rem; color: #234e70; padding: .8rem 1rem; font-size: .88rem;
        }
        .winner-card {
            background: var(--ink); border-radius: .75rem; color: white;
            padding: 1.25rem 1.4rem; min-height: 9.5rem;
        }
        .winner-card .metric-label { color: #9fb3c8; }
        .winner-card .metric-value { color: white; }
        .winner-card .metric-note { color: #bcccdc; }
        .empty-state {
            background: var(--surface); border: 1px dashed #9fb3c8;
            border-radius: .75rem; padding: 2rem; text-align: center; margin-top: 1.25rem;
        }
        .empty-state h3 { color: var(--ink); margin: 0 0 .4rem; }
        .empty-state p { color: var(--muted); margin: 0; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: .5rem; }
        .stButton > button[kind="primary"] {
            background: var(--teal); border-color: var(--teal); min-height: 2.8rem;
            font-weight: 700; box-shadow: 0 2px 4px rgba(15, 118, 110, .16);
        }
        .stButton > button[kind="primary"]:hover {
            background: #0b5f59; border-color: #0b5f59;
        }
        .stDownloadButton > button {
            border-color: #9fb3c8; color: var(--ink); font-weight: 650;
        }
        @media (max-width: 900px) {
            .block-container { padding: 1.5rem 1rem 3rem; }
            .page-title { font-size: 1.7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_builtin_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a built-in dataset from scikit-learn."""
    if dataset_name == "Breast Cancer Classification":
        return load_breast_cancer(as_frame=True).frame
    if dataset_name == "Diabetes Regression":
        return load_diabetes(as_frame=True).frame
    raise ValueError("Unknown built-in dataset.")


def read_uploaded_csv(uploaded_file: Any) -> tuple[pd.DataFrame | None, bytes]:
    """Read uploaded bytes and return them for stable session-state identity."""
    if uploaded_file is None:
        return None, b""
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        return pd.DataFrame(), raw_bytes
    return pd.read_csv(io.BytesIO(raw_bytes)), raw_bytes


def validate_dataset(data: pd.DataFrame, target: str, task_type: str) -> list[str]:
    """Return concise, actionable errors without changing user data."""
    if data.empty:
        return ["The file is empty or contains no data rows."]

    errors: list[str] = []
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        errors.append(
            "Duplicate column names are not supported: "
            + ", ".join(map(str, duplicates))
            + "."
        )
    if not target or target not in data.columns:
        errors.append(f"Target column '{target}' was not found.")
        return errors
    if data.columns.duplicated().any():
        return errors

    target_values = data[target]
    if task_type == "regression":
        numeric_target = pd.to_numeric(target_values, errors="coerce")
        valid_target = numeric_target.notna() & np.isfinite(numeric_target)
        if not valid_target.all():
            invalid_count = int((~valid_target).sum())
            errors.append(
                f"Regression target contains {invalid_count} non-numeric "
                "or non-finite value(s)."
            )
        usable_target = numeric_target[valid_target]
    else:
        usable_target = target_values.dropna()
        if usable_target.empty:
            errors.append("Classification target has no valid values.")

    if len(usable_target) < MINIMUM_ROWS:
        errors.append(
            f"At least {MINIMUM_ROWS} rows with valid target values are required."
        )
    if task_type == "classification" and usable_target.nunique() < 2:
        errors.append("Classification requires at least two distinct target classes.")
    if len(data.columns) < 2:
        errors.append("Add at least one feature column alongside the target.")
    return errors


def dataset_signature(
    source_name: str,
    data: pd.DataFrame,
    raw_upload: bytes,
    target: str,
    task_type: str,
) -> str:
    """Identify a training configuration so stale results cannot persist."""
    source_identity = (
        hashlib.sha256(raw_upload).hexdigest()
        if source_name == "Upload a CSV file"
        else source_name
    )
    return "|".join((source_identity, str(data.shape), target, task_type))


def format_metric_name(metric: str) -> str:
    """Turn engine metric keys into human-readable labels."""
    return metric.replace("_", " ").title()


def metric_card(
    label: str,
    value: str,
    note: str = "",
    dark: bool = False,
) -> None:
    """Render a reusable metric card."""
    card_class = "winner-card" if dark else "metric-card"
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, caption: str = "") -> None:
    """Render consistent section headings."""
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="section-title">{title}</div>{caption_html}',
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render compact product identity and environment context."""
    st.markdown(
        """
        <div class="product-mark">
            <div class="product-symbol">A</div>
            <div>
                <div class="product-name">AutoML Studio Lite</div>
                <div class="product-kicker">Model intelligence workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    title_column, badge_column = st.columns([4.2, 1.2])
    with title_column:
        st.markdown('<h1 class="page-title">Build confidence in your next model</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-subtitle">Profile data, compare reliable baselines, and package the strongest candidate for your next experiment.</div>',
            unsafe_allow_html=True,
        )
    with badge_column:
        st.markdown(
            '<div style="text-align:right"><span class="env-badge">● Local demo environment</span></div>',
            unsafe_allow_html=True,
        )


def render_dataset_overview(data: pd.DataFrame, target: str) -> None:
    """Render dataset health cards and profile tables."""
    section_heading(
        "Dataset workspace",
        "Review the shape and quality of your data before starting a training run.",
    )
    cards = st.columns(4)
    values = (
        ("Rows", f"{len(data):,}", "observations"),
        ("Features", f"{max(len(data.columns) - 1, 0):,}", "input columns"),
        ("Missing values", f"{int(data.isna().sum().sum()):,}", "across all columns"),
        ("Target", target or "Not selected", "selected outcome"),
    )
    for column, (label, value, note) in zip(cards, values):
        with column:
            metric_card(label, value, note)

    table_column, preview_column = st.columns([1, 1.65])
    with table_column:
        st.markdown("**Column profile**")
        details = pd.DataFrame(
            {
                "Column": data.columns,
                "Type": [str(dtype) for dtype in data.dtypes],
                "Missing": data.isna().sum().values,
                "Unique": data.nunique(dropna=True).values,
            }
        )
        st.dataframe(details, use_container_width=True, hide_index=True, height=370)
    with preview_column:
        st.markdown("**Data preview**")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True, height=370)


def render_results(results: dict[str, Any], task_type: str) -> None:
    """Render the leaderboard, winner, directional metrics, and downloads."""
    primary_metric = results["primary_metric"]
    leaderboard = results["leaderboard"]
    rows = [{"Model": item["model"], **item["metrics"]} for item in leaderboard]
    comparison = pd.DataFrame(rows)
    metric_columns = (
        CLASSIFICATION_METRICS
        if task_type == "classification"
        else REGRESSION_METRICS
    )
    display_columns = ["Model"] + [
        metric for metric in metric_columns if metric in comparison.columns
    ]

    section_heading("Model results", "Evaluation on the held-out test split; scores are for comparison, not a production guarantee.")
    winner_columns = st.columns([1.5, 1, 1])
    with winner_columns[0]:
        metric_card("Selected model", results["best_model"], "highest primary score", dark=True)
    with winner_columns[1]:
        metric_card(
            format_metric_name(primary_metric),
            f"{results['best_metrics'][primary_metric]:.4f}",
            "primary comparison metric",
            dark=True,
        )
    with winner_columns[2]:
        metric_card("Candidates evaluated", str(len(leaderboard)), "successful runs", dark=True)

    st.markdown("")
    table_column, chart_column = st.columns([1.25, 1])
    with table_column:
        st.markdown("**Leaderboard**")
        st.dataframe(
            comparison[display_columns].style.format(
                {metric: "{:.4f}" for metric in display_columns[1:]}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with chart_column:
        st.markdown(f"**{format_metric_name(primary_metric)} by model**")
        chart_data = comparison.set_index("Model")[[primary_metric]]
        chart_data = chart_data.rename(columns={primary_metric: format_metric_name(primary_metric)})
        st.bar_chart(chart_data, height=270, use_container_width=True)

    if task_type == "classification":
        st.caption("Classification metrics: accuracy, precision, recall, and F1. Higher is better for each metric.")
    else:
        st.caption("Regression metrics: lower MAE/RMSE is better; higher R² is better.")

    failed_models = results.get("failed_models", [])
    if failed_models:
        with st.expander(f"Candidate models with issues ({len(failed_models)})"):
            st.dataframe(pd.DataFrame(failed_models), use_container_width=True, hide_index=True)

    section_heading("Export artifacts", "Download the selected pipeline and the metadata generated by the ML engine.")
    download_columns = st.columns(2)
    with download_columns[0]:
        model_path = Path(results["artifact_path"])
        st.download_button(
            "Download selected model",
            data=model_path.read_bytes(),
            file_name="best_model.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with download_columns[1]:
        metadata_path = Path(results["metadata_path"])
        st.download_button(
            "Download model metadata",
            data=metadata_path.read_text(encoding="utf-8"),
            file_name="model_metadata.json",
            mime="application/json",
            use_container_width=True,
        )


def render_sidebar() -> tuple[str, str, Any]:
    """Render navigation and return the selected workspace configuration."""
    with st.sidebar:
        st.markdown("### Workspace")
        section = st.radio(
            "Workspace view",
            ("Data preparation", "Model results"),
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Dataset**")
        source_name = st.selectbox(
            "Choose a source",
            (
                "Breast Cancer Classification",
                "Diabetes Regression",
                "Upload a CSV file",
            ),
            label_visibility="collapsed",
        )
        uploaded_file = None
        if source_name == "Upload a CSV file":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.divider()
        st.caption("Configuration")
    return section, source_name, uploaded_file


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    render_header()
    section, source_name, uploaded_file = render_sidebar()

    raw_upload = b""
    if source_name == "Upload a CSV file":
        try:
            data, raw_upload = read_uploaded_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
            data = None
            st.error("This CSV could not be read. Check the file format and try again.")
    else:
        data = load_builtin_dataset(source_name)

    if data is None:
        st.markdown(
            '<div class="empty-state"><h3>Bring a dataset into the workspace</h3><p>Choose a built-in example or upload a CSV from the sidebar to begin.</p></div>',
            unsafe_allow_html=True,
        )
        return
    if data.empty:
        st.error("The selected file is empty or contains no data rows.")
        return

    if source_name == "Breast Cancer Classification":
        task_type = "classification"
        st.sidebar.caption("Task type · Classification")
    elif source_name == "Diabetes Regression":
        task_type = "regression"
        st.sidebar.caption("Task type · Regression")
    else:
        task_type = st.sidebar.selectbox(
            "Task type",
            ("classification", "regression"),
            format_func=str.title,
            key="task_type_upload",
        )
    default_target = "target" if "target" in data.columns else data.columns[-1]
    target = st.sidebar.selectbox(
        "Target column",
        list(data.columns),
        index=list(data.columns).index(default_target),
        key=f"target_column_{source_name}",
    )

    signature = dataset_signature(source_name, data, raw_upload, target, task_type)
    if st.session_state.get("training_signature") != signature:
        st.session_state.pop("training_results", None)
        st.session_state["training_signature"] = signature

    validation_errors = validate_dataset(data, target, task_type)
    if section == "Data preparation":
        render_dataset_overview(data, target)
        feedback_column, action_column = st.columns([1.5, 1])
        with feedback_column:
            if validation_errors:
                st.error("Data quality checks need attention.")
                for error in validation_errors:
                    st.warning(error)
            else:
                st.success("Data quality checks passed. Ready for a training run.")
        with action_column:
            st.markdown("**Training controls**")
            st.caption("Training starts only when you initiate it.")
            if st.button(
                "Train and Compare Models",
                type="primary",
                disabled=bool(validation_errors),
                use_container_width=True,
            ):
                with st.spinner("Training candidate pipelines and evaluating results..."):
                    try:
                        st.session_state["training_results"] = train_and_compare(
                            data=data,
                            target=target,
                            task_type=task_type,
                            artifacts_dir=ARTIFACTS_DIR,
                        )
                        st.success("Training complete. Results are ready to review.")
                    except (ValueError, TypeError, RuntimeError, OSError) as exc:
                        st.session_state.pop("training_results", None)
                        st.error(f"Training could not be completed: {exc}")
                    except Exception:  # noqa: BLE001
                        st.session_state.pop("training_results", None)
                        st.error(
                            "Training could not be completed due to an unexpected application error. "
                            "Review the data and try again."
                        )
    results = st.session_state.get("training_results")
    if section == "Model results":
        if results is None:
            st.markdown(
                '<div class="empty-state"><h3>No model run for this configuration</h3><p>Return to Data preparation, review the checks, and start a training run.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            render_results(results, task_type)
    elif results is not None:
        st.divider()
        render_results(results, task_type)


if __name__ == "__main__":
    main()
