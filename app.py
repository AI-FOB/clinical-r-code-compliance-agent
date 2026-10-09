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
    # Clear session state if file changes
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.pop("findings", None)
        st.session_state.pop("excel_path", None)
        st.session_state.pop("html_path", None)
        st.session_state.pop("updated_script_path", None)
        st.session_state.last_uploaded_file = uploaded_file.name

    if st.button("Run Compliance Scan", type="primary"):
        with st.spinner("Analyzing code compliance..."):
            # Save uploaded file to a temporary directory with its original name
            tmp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                if use_llm:
                    # Initialize the agent
                    agent = ComplianceAgent(
                        model="gemma4:e4b",
                        temperature=1.0,
                        top_k=3,
                        max_concurrency=max_concurrency
                    )

                    # Run the asynchronous analysis
                    findings = asyncio.run(agent.aanalyze(tmp_file_path))
                else:
                    from src.compliance_agent.tools.code_analyzer import BasicRCodeAnalyzer
                    from src.compliance_agent.models.schemas import ComplianceFinding
                    analyzer = BasicRCodeAnalyzer()
                    raw_findings = analyzer.analyze_file(tmp_file_path)
                    findings = []
                    for raw in raw_findings:
                        findings.append(ComplianceFinding(
                            rule_id=raw["rule_id"],
                            line_number=raw["line_number"],
                            severity=raw["severity"],
                            finding=f"Deterministic match for {raw['rule_name']}",
                            evidence=raw["line_content"],
                            recommendation=raw["recommendation"],
                            confidence=1.0,
                            suggested_fix=""
                        ))

                # Generate reports
                output_dir = Path(tempfile.gettempdir()) / "compliance_reports"
                output_dir.mkdir(exist_ok=True)
                
                excel_path, html_path, updated_script_path = generate_reports(
                    findings=findings,
                    file_name=tmp_file_path,
                    output_dir=str(output_dir),
                    run_metadata={
                        "mode": "LLM Agent" if use_llm else "Deterministic",
                        "model": "gemma4:e4b (T=1.0)" if use_llm else "N/A"
                    }
                )

                st.session_state.findings = findings
                st.session_state.excel_path = excel_path
                st.session_state.html_path = html_path
                st.session_state.updated_script_path = updated_script_path
                
            finally:
                # Clean up the original uploaded temp file
                try:
                    os.unlink(tmp_file_path)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

    # Display results if available in session state
    if "findings" in st.session_state:
        findings = st.session_state.findings
        
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
        
        with open(st.session_state.updated_script_path, "rb") as f:
            dl_col1.download_button(
                label="Download Updated .R Script",
                data=f,
                file_name=f"{uploaded_file.name.replace('.R', '')}_updated.R",
                mime="text/plain",
                type="primary"
            )
        
        with open(st.session_state.excel_path, "rb") as f:
            dl_col2.download_button(
                label="Download Excel Report",
                data=f,
                file_name=f"{uploaded_file.name.replace('.R', '')}_compliance_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with open(st.session_state.html_path, "rb") as f:
            dl_col3.download_button(
                label="Download HTML Report",
                data=f,
                file_name=f"{uploaded_file.name.replace('.R', '')}_compliance_report.html",
                mime="text/html"
            )
