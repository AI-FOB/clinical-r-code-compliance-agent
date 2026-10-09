import streamlit as st
import asyncio
import os
import tempfile
from pathlib import Path
from src.compliance_agent.agent import ComplianceAgent
from src.compliance_agent.tools.reporter import generate_reports

st.set_page_config(page_title="Clinical R Code Compliance Agent", page_icon="🛡️", layout="wide")

st.sidebar.title("Configuration")
use_llm = st.sidebar.checkbox("Enable LLM (gemma4:e4b)", value=True, help="Toggle AI-powered review and auto-fixing.")
max_concurrency = st.sidebar.slider("Max Concurrency", min_value=1, max_value=10, value=4, help="Number of concurrent LLM requests.")

st.title("🛡️ Clinical R Code Compliance Agent")
st.markdown("""
Welcome to the interactive web frontend. Upload your clinical R scripts here to run a comprehensive compliance scan.
The agent will automatically flag violations, suggest fixes using an AI model, and generate validation reports.
""")

uploaded_file = st.file_uploader("Upload an R script (.R)", type=["R", "r"])

if uploaded_file is not None:
    if st.button("Run Compliance Scan", type="primary"):
        with st.spinner("Analyzing code compliance..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".R") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Initialize the agent
                agent = ComplianceAgent(
                    model_name="gemma4:e4b",
                    temperature=1.0,
                    top_k=3,
                    max_concurrency=max_concurrency,
                    use_llm=use_llm
                )

                # Run the asynchronous analysis
                findings = asyncio.run(agent.aanalyze(tmp_file_path))

                # Generate reports
                output_dir = Path(tempfile.gettempdir()) / "compliance_reports"
                output_dir.mkdir(exist_ok=True)
                
                excel_path, html_path, updated_script_path = generate_reports(
                    findings=findings,
                    original_file_path=tmp_file_path,
                    output_dir=str(output_dir)
                )

                st.success(f"Analysis complete! Found {len(findings)} compliance issue(s).")

                # Metrics Dashboard
                st.subheader("Results Summary")
                high_count = sum(1 for f in findings if f.severity == "HIGH")
                medium_count = sum(1 for f in findings if f.severity == "MEDIUM")
                low_count = sum(1 for f in findings if f.severity == "LOW")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Issues", len(findings))
                col2.metric("High Severity", high_count)
                col3.metric("Medium Severity", medium_count)
                col4.metric("Low Severity", low_count)

                # Detailed Findings
                st.subheader("Detailed Findings & Code Diffs")
                for issue in findings:
                    with st.expander(f"[{issue.severity}] {issue.rule_id} (Line {issue.line_number})"):
                        st.markdown(f"**Finding:** {issue.finding}")
                        st.markdown(f"**Recommendation:** {issue.recommendation}")
                        
                        st.markdown("**Original Code (Evidence):**")
                        # Show original code
                        st.code(issue.evidence, language="r")
                        
                        if issue.suggested_fix:
                            st.markdown("**Suggested Fix:**")
                            st.code(issue.suggested_fix, language="r")

                # Download Buttons
                st.subheader("Download Reports & Updated Code")
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                
                with open(updated_script_path, "rb") as f:
                    dl_col1.download_button(
                        label="Download Updated .R Script",
                        data=f,
                        file_name=f"{uploaded_file.name.replace('.R', '')}_updated.R",
                        mime="text/plain",
                        type="primary"
                    )
                
                with open(excel_path, "rb") as f:
                    dl_col2.download_button(
                        label="Download Excel Report",
                        data=f,
                        file_name=f"{uploaded_file.name.replace('.R', '')}_compliance_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with open(html_path, "rb") as f:
                    dl_col3.download_button(
                        label="Download HTML Report",
                        data=f,
                        file_name=f"{uploaded_file.name.replace('.R', '')}_compliance_report.html",
                        mime="text/html"
                    )

            finally:
                # Clean up the original uploaded temp file
                os.unlink(tmp_file_path)
