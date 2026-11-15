"""
Interactive Streamlit Web Dashboard:
Stage-Adaptive Stacked Ensemble for Early Student Performance Prediction.

Simulates predictions on:
1. Real CodeWorkout Cohort Data
2. Custom User Uploads (CSV/Excel batch simulation) with built-in instant demo datasets
3. Direct Student Form Entry
4. Interactive What-If Sandbox
"""

import os
import sys
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

# Streamlit page config
st.set_page_config(
    page_title="Stage-Adaptive Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_stage_data():
    """Load all 5 cumulative stage parquet files."""
    data = {}
    for stage in range(1, 6):
        file_path = os.path.join(config.PROCESSED_DATA_DIR, f"stage_{stage}_features.parquet")
        data[stage] = pd.read_parquet(file_path)
    return data


@st.cache_data
def load_weights_and_metrics():
    """Load learned weights and experimental benchmark metrics."""
    weights_path = os.path.join(config.RESULTS_DIR, "weights", "stage_adaptive_weights.csv")
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "consolidated_metrics_summary.csv")
    w_df = pd.read_csv(weights_path)
    m_df = pd.read_csv(metrics_path)
    return w_df, m_df


@st.cache_resource
def load_models(stage):
    """Load trained models and scaler for a given stage."""
    scaler = joblib.load(f"models/saved/scaler_stage_{stage}.joblib")
    knn = joblib.load(f"models/saved/knn_stage_{stage}.joblib")
    svr = joblib.load(f"models/saved/svr_stage_{stage}.joblib")
    xgb = joblib.load(f"models/saved/xgb_stage_{stage}.joblib")
    return scaler, knn, svr, xgb


def predict_batch(df_input: pd.DataFrame, stage: int, weights: tuple) -> pd.DataFrame:
    """Predicts performance for an input DataFrame with the 7 features."""
    w1, w2, w3 = weights
    scaler, knn, svr, xgb = load_models(stage)

    # Ensure all 7 features exist
    missing_cols = [col for col in config.FEATURE_NAMES if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = df_input[config.FEATURE_NAMES].values.astype(float)
    X_scaled = scaler.transform(X)

    preds_knn = knn.predict(X_scaled)
    preds_svr = svr.predict(X_scaled)
    preds_xgb = xgb.predict(X)

    preds_adaptive = (w1 * preds_knn) + (w2 * preds_svr) + (w3 * preds_xgb)
    preds_equal = (preds_knn + preds_svr + preds_xgb) / 3.0

    res = df_input.copy()
    res['Pred_KNN'] = np.round(preds_knn, 1)
    res['Pred_SVR'] = np.round(preds_svr, 1)
    res['Pred_XGBoost'] = np.round(preds_xgb, 1)
    res['Pred_EqualWeight'] = np.round(preds_equal, 1)
    res['Pred_StageAdaptive'] = np.round(preds_adaptive, 1)

    # Risk category
    def assign_risk(score):
        if score < 50:
            return "🔴 At Risk (<50%)"
        elif score < 70:
            return "🟡 Moderate (50-70%)"
        else:
            return "🟢 On Track (>70%)"

    res['PerformanceRisk'] = res['Pred_StageAdaptive'].apply(assign_risk)

    if 'FinalGrade' in res.columns:
        res['Error_Adaptive'] = np.round(np.abs(res['FinalGrade'] - res['Pred_StageAdaptive']), 1)

    return res


def main():
    st.markdown('<div class="main-title">🎓 Stage-Adaptive Student Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive Simulation & Real-Data Inference Engine across 5 Assignment Stages</div>', unsafe_allow_html=True)

    stage_data = load_all_stage_data()
    weights_df, metrics_df = load_weights_and_metrics()

    # Sidebar controls
    st.sidebar.header("🕹️ Simulation Settings")

    sim_mode = st.sidebar.radio(
        "Choose Input Method:",
        [
            "📂 Test with Sample / Uploaded CSV",
            "✍️ Enter Single New Student",
            "👤 Explore CodeWorkout Cohort",
            "📊 Benchmark & Learned Weights"
        ]
    )

    stage_selected = st.sidebar.select_slider(
        "Select Assignment Stage for Simulation:",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"Stage {x} (Assignments 1–{x})" if x > 1 else "Stage 1 (Assignment 1)"
    )

    # Average learned weights for this stage
    stage_w = weights_df[weights_df['stage'] == stage_selected][['w_knn', 'w_svr', 'w_xgb']].mean()
    w1, w2, w3 = stage_w['w_knn'], stage_w['w_svr'], stage_w['w_xgb']

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Stage {stage_selected} Learned Model Weights:**")
    st.sidebar.markdown(f"- 🟦 **KNN ($w_1$):** `{w1:.1%}`")
    st.sidebar.markdown(f"- 🟩 **SVR ($w_2$):** `{w2:.1%}`")
    st.sidebar.markdown(f"- 🟧 **XGBoost ($w_3$):** `{w3:.1%}`")

    # ==========================================
    # MODE 1: Test with Sample / Uploaded CSV
    # ==========================================
    if sim_mode == "📂 Test with Sample / Uploaded CSV":
        st.subheader(f"📂 Batch Performance Simulation (Stage {stage_selected})")
        st.write("You can test new datasets either by **selecting one of the ready-made generated sample CSVs** below with 1 click, or by **uploading your own file**.")

        # Preset Sample Picker
        sample_options = {
            "⚡ Quick 1-Click: Mixed Cohort (High, Average & At-Risk Students)": "sample_data/mixed_cohort_case_study.csv",
            "🌱 Sample: 25 New Students at Stage 1 (Assignment 1)": "sample_data/new_students_stage_1.csv",
            "🌿 Sample: 25 New Students at Stage 3 (Assignments 1-3)": "sample_data/new_students_stage_3.csv",
            "🌳 Sample: 25 New Students at Stage 5 (Full Course)": "sample_data/new_students_stage_5.csv",
            "📁 Upload My Own File from Computer": None
        }

        chosen_option = st.selectbox("Select Data Source:", list(sample_options.keys()), index=0)

        active_file_path = sample_options[chosen_option]
        user_df = None

        if active_file_path is not None:
            user_df = pd.read_csv(active_file_path)
            st.success(f"Loaded generated dataset: `{os.path.basename(active_file_path)}` ({len(user_df)} student records)")
            with st.expander("👀 View Loaded Data Preview"):
                st.dataframe(user_df.head(6), use_container_width=True)
        else:
            uploaded_file = st.file_uploader("Upload CSV or Excel file:", type=["csv", "xlsx", "xls"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)
                st.success(f"Successfully uploaded `{uploaded_file.name}` with {len(user_df)} students.")

        if user_df is not None:
            # Run Batch Prediction
            with st.spinner(f"Running predictions using Stage {stage_selected} ensemble..."):
                results_df = predict_batch(user_df, stage_selected, (w1, w2, w3))

            # Summary Statistics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                avg_pred = results_df['Pred_StageAdaptive'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Cohort Mean Predicted Grade</div>
                    <div class="metric-val">{avg_pred:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                at_risk_cnt = (results_df['PerformanceRisk'].str.contains("At Risk")).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Students At Risk (<50%)</div>
                    <div class="metric-val" style="color: #EF4444;">{at_risk_cnt}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                moderate_cnt = (results_df['PerformanceRisk'].str.contains("Moderate")).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Moderate Tier (50-70%)</div>
                    <div class="metric-val" style="color: #F59E0B;">{moderate_cnt}</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                on_track_cnt = (results_df['PerformanceRisk'].str.contains("On Track")).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">On Track (>70%)</div>
                    <div class="metric-val" style="color: #10B981;">{on_track_cnt}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📋 Detailed Predictions Table")
            st.dataframe(results_df, use_container_width=True)

            # Export Results
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                csv_out = io.StringIO()
                results_df.to_csv(csv_out, index=False)
                st.download_button(
                    label="💾 Export Predictions CSV",
                    data=csv_out.getvalue(),
                    file_name=f"predictions_stage_{stage_selected}_results.csv",
                    mime="text/csv"
                )

            # Interactive Bar Chart for Student Comparison
            st.markdown("### 📊 Individual Student Predictions Breakdown")
            fig_bar = px.bar(
                results_df, x="GlobalStudentID", y="Pred_StageAdaptive", color="PerformanceRisk",
                title=f"Predicted Final Grade by Student (Stage {stage_selected})",
                color_discrete_map={"🔴 At Risk (<50%)": "#EF4444", "🟡 Moderate (50-70%)": "#F59E0B", "🟢 On Track (>70%)": "#10B981"},
                text="Pred_StageAdaptive"
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(yaxis_range=[0, 115], xaxis_title="Student ID", yaxis_title="Predicted Grade (%)")
            st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # MODE 2: Enter Single New Student
    # ==========================================
    elif sim_mode == "✍️ Enter Single New Student":
        st.subheader(f"✍️ Single Student Prediction Form (Stage {stage_selected})")
        st.write("Input a new student's programming activity profile up to this stage to simulate the model's prediction.")

        with st.form("single_student_form"):
            col_id, col_ground = st.columns(2)
            with col_id:
                stu_id = st.text_input("Student Identifier / Name:", value="New_Student_001")
            with col_ground:
                actual_val = st.number_input("Actual Final Grade (Optional, for error check):", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

            st.markdown("##### Cumulative Programming Features")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                f_time = st.number_input("TimeSpent (cumulative seconds active):", min_value=0.0, value=6500.0, step=100.0, help="Sum of consecutive active coding time, idle capped at 600s")
                f_valid = st.number_input("Valid Submissions (successful compilations):", min_value=0, value=48, step=1)
                f_correct = st.number_input("Correct Submissions (passed all tests):", min_value=0, value=20, step=1)
                f_incorrect = st.number_input("Incorrect Submissions (valid but failed tests):", min_value=0, value=28, step=1)
            with col_f2:
                f_err = st.number_input("Compile Errors (syntax / compilation failures):", min_value=0, value=35, step=1)
                f_scores = st.slider("Scores (mean problem score achieved, 0.0 - 1.0):", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
                f_edit = st.number_input("EditDistance (cumulative normalized Levenshtein diff):", min_value=0.0, value=3.8, step=0.1)

            submit_btn = st.form_submit_button("🚀 Run Stage-Adaptive Prediction")

        if submit_btn:
            single_df = pd.DataFrame([{
                'GlobalStudentID': stu_id,
                'TimeSpent': f_time,
                'Valid': f_valid,
                'CorrectSub': f_correct,
                'IncorrectSub': f_incorrect,
                'CompileError': f_err,
                'Scores': f_scores,
                'EditDistance': f_edit
            }])

            res_single = predict_batch(single_df, stage_selected, (w1, w2, w3)).iloc[0]

            pred_ens = res_single['Pred_StageAdaptive']
            pred_eq = res_single['Pred_EqualWeight']
            pred_k = res_single['Pred_KNN']
            pred_s = res_single['Pred_SVR']
            pred_x = res_single['Pred_XGBoost']

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Stage-Adaptive Forecast</div>
                    <div class="metric-val" style="color: #2563EB;">{pred_ens:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                risk_tag = res_single['PerformanceRisk']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Academic Risk Tier</div>
                    <div class="metric-val" style="font-size: 1.3rem;">{risk_tag}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Equal-Weight Prediction</div>
                    <div class="metric-val" style="color: #8B5CF6;">{pred_eq:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                err_diff = abs(actual_val - pred_ens)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Difference from Ground Truth</div>
                    <div class="metric-val" style="color: #10B981;">{err_diff:.1f} pts</div>
                </div>
                """, unsafe_allow_html=True)

            # Model breakdown comparison chart
            fig_single = go.Figure()
            fig_single.add_trace(go.Bar(
                x=['KNN', 'SVR (RBF)', 'XGBoost', 'Stage-Adaptive Ensemble'],
                y=[pred_k, pred_s, pred_x, pred_ens],
                text=[f"{pred_k:.1f}%", f"{pred_s:.1f}%", f"{pred_x:.1f}%", f"{pred_ens:.1f}%"],
                textposition='outside',
                marker_color=['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
            ))
            fig_single.add_hline(y=actual_val, line_dash="dot", line_color="#EF4444", annotation_text=f"Ground Truth ({actual_val}%)")
            fig_single.update_layout(title="Prediction Breakdown Across Base Learners and Ensemble", yaxis_range=[0, 110])
            st.plotly_chart(fig_single, use_container_width=True)

    # ==========================================
    # MODE 3: Explore CodeWorkout Cohort
    # ==========================================
    elif sim_mode == "👤 Explore CodeWorkout Cohort":
        st.subheader(f"👤 Real CodeWorkout Cohort Inspector (Stage {stage_selected})")
        df_curr = stage_data[stage_selected]
        all_students = df_curr['GlobalStudentID'].tolist()

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_student = st.selectbox("Select Student from Real Dataset:", all_students, index=0)
        with col_sel2:
            st.info(f"Cohort Size: **{len(all_students)}** Students")

        # Fetch student row
        student_row = df_curr[df_curr['GlobalStudentID'] == selected_student].iloc[0]
        actual_grade = student_row['FinalGrade']
        feat_vals = student_row[config.FEATURE_NAMES].values.reshape(1, -1)

        scaler, knn, svr, xgb = load_models(stage_selected)
        feat_scaled = scaler.transform(feat_vals)
        pred_knn = float(knn.predict(feat_scaled)[0])
        pred_svr = float(svr.predict(feat_scaled)[0])
        pred_xgb = float(xgb.predict(feat_vals)[0])

        pred_equal = (pred_knn + pred_svr + pred_xgb) / 3.0
        pred_adaptive = (w1 * pred_knn) + (w2 * pred_svr) + (w3 * pred_xgb)

        # Top metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Actual Final Grade</div>
                <div class="metric-val" style="color: #10B981;">{actual_grade:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Stage-Adaptive Prediction</div>
                <div class="metric-val" style="color: #2563EB;">{pred_adaptive:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            err = abs(actual_grade - pred_adaptive)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Absolute Error</div>
                <div class="metric-val" style="color: #F59E0B;">{err:.1f} pts</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Equal-Weight Prediction</div>
                <div class="metric-val" style="color: #8B5CF6;">{pred_equal:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Longitudinal Trajectory
        prog_data = []
        for s in range(1, 6):
            s_row = stage_data[s][stage_data[s]['GlobalStudentID'] == selected_student].iloc[0]
            s_feats = s_row[config.FEATURE_NAMES].values.reshape(1, -1)
            sc, k_m, sv_m, x_m = load_models(s)
            s_w = weights_df[weights_df['stage'] == s][['w_knn', 'w_svr', 'w_xgb']].mean()

            p_k = float(k_m.predict(sc.transform(s_feats))[0])
            p_s = float(sv_m.predict(sc.transform(s_feats))[0])
            p_x = float(x_m.predict(s_feats)[0])
            p_ad = (s_w['w_knn'] * p_k) + (s_w['w_svr'] * p_s) + (s_w['w_xgb'] * p_x)
            p_eq = (p_k + p_s + p_x) / 3.0

            prog_data.append({
                'Stage': f"Stage {s}",
                'Adaptive Pred': p_ad,
                'Equal-Weight Pred': p_eq,
                'Actual Grade': actual_grade
            })

        prog_df = pd.DataFrame(prog_data)

        fig_prog = go.Figure()
        fig_prog.add_trace(go.Scatter(x=prog_df['Stage'], y=prog_df['Adaptive Pred'], mode='lines+markers', name='Stage-Adaptive Ensemble', line=dict(color='#2563EB', width=3)))
        fig_prog.add_trace(go.Scatter(x=prog_df['Stage'], y=prog_df['Equal-Weight Pred'], mode='lines+markers', name='Equal-Weight Ensemble', line=dict(color='#8B5CF6', dash='dash')))
        fig_prog.add_trace(go.Scatter(x=prog_df['Stage'], y=prog_df['Actual Grade'], mode='lines', name='Actual Final Grade (Ground Truth)', line=dict(color='#10B981', width=3, dash='dot')))
        fig_prog.update_layout(title=f"Prediction Convergence for '{selected_student[:15]}...' Across Cumulative Stages", yaxis_range=[0, 105], hovermode="x unified")
        st.plotly_chart(fig_prog, use_container_width=True)

        # Features Table
        st.markdown(f"#### Programming Features Recorded for Student up to Stage {stage_selected}")
        feat_display = pd.DataFrame({
            'Feature': config.FEATURE_NAMES,
            'Value': [
                f"{student_row['TimeSpent']:.1f} sec (~{student_row['TimeSpent']/60:.1f} min)",
                f"{int(student_row['Valid'])}",
                f"{int(student_row['CorrectSub'])}",
                f"{int(student_row['IncorrectSub'])}",
                f"{int(student_row['CompileError'])}",
                f"{student_row['Scores']:.2%}",
                f"{student_row['EditDistance']:.2f}"
            ],
            'Description': [
                "Cumulative active problem-solving time",
                "Syntactically valid executions",
                "Submissions passing all test cases",
                "Submissions failing test cases",
                "Syntax or compilation errors encountered",
                "Mean maximum problem score attained",
                "Normalized Levenshtein edit distance"
            ]
        })
        st.table(feat_display)

    # ==========================================
    # MODE 4: Benchmark & Learned Weights
    # ==========================================
    else:
        st.subheader("📊 Experimental Results & Weights Analysis across 10 Seeds")

        st.markdown("#### Dynamic Evolution of Learned Weights Across All 5 Stages")
        w_summary = weights_df.groupby('stage')[['w_knn', 'w_svr', 'w_xgb']].mean().reset_index()

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(name='KNN Weight (w1)', x=[f"Stage {s}" for s in w_summary['stage']], y=w_summary['w_knn'], marker_color='#95a5a6'))
        fig_w.add_trace(go.Bar(name='SVR Weight (w2)', x=[f"Stage {s}" for s in w_summary['stage']], y=w_summary['w_svr'], marker_color='#3498db'))
        fig_w.add_trace(go.Bar(name='XGBoost Weight (w3)', x=[f"Stage {s}" for s in w_summary['stage']], y=w_summary['w_xgb'], marker_color='#e67e22'))
        fig_w.update_layout(barmode='stack', yaxis_title="Ensemble Contribution", yaxis_range=[0, 1.05])
        st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("#### Test Performance Metrics Summary Table (Mean ± Std Across 10 Seeds)")
        st.dataframe(metrics_df, use_container_width=True)


if __name__ == '__main__':
    main()
