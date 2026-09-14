import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import streamlit_authenticator as stauth
from google import genai

# 1. Authentication System Integration
# Fetch credentials securely from Streamlit secrets management structures
authenticator = stauth.Authenticate(
    st.secrets['credentials'],
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days']
)

# Render the authentication panel interface
login_result = authenticator.login()

# Retrieve user validation variables out of state parameters
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
elif authentication_status:
    # --- MAIN APPLICATION MODULE EXECUTION (ONLY IF AUTHENTICATED SUCCESSFUL) ---

    # 1. Page Configuration & Professional Styling
    st.set_page_config(
        page_title="Decision Coach Framework",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main .block-container {padding-top: 2rem; padding-bottom: 5rem;}
        h1 {color: #1E3A8A; font-weight: 700;}
        h2 {color: #2563EB; font-weight: 600; margin-top: 1.5rem;}
        .stButton button {font-weight: 600;}
        div[data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: bold;}
        .sticky-progress {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f8f9fa;
            padding: 10px 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 999;
        }
        </style>
        """, unsafe_allow_html=True)

    # Function to clear session state / reset scenario
    def reset_scenario():
        for key in list(st.session_state.keys()):
            # Keep auth keys alive so resetting framework doesn't force a user logout
            if key not in ['authentication_status', 'username', 'logout']:
                del st.session_state[key]
        st.rerun()

    # 2. Session State Initialization
    if 'decision_metadata' not in st.session_state:
        st.session_state.decision_metadata = {'name': "", 'owner': "", 'description': ""}

    if 'dq_scores' not in st.session_state:
        st.session_state.dq_scores = {
            'Appropriate Frame': 10, 'Creative Alternatives': 20, 'Reliable Information': 30,
            'Clear Values & Trade-offs': 40, 'Logically Sound Reasoning': 30, 'Commitment to Action': 35
        }

    if 'framing_data' not in st.session_state:
        st.session_state.framing_data = {
            'problem_statement': "", 'in_scope': "", 'out_scope': "", 'constraints': ""
        }

    if 'alternatives_data' not in st.session_state:
        st.session_state.alternatives_data = {
            'alt1_name': "", 'alt1_desc': "",
            'alt2_name': "", 'alt2_desc': "",
            'alt3_name': "", 'alt3_desc': ""
        }

    if 'adaptive_inputs' not in st.session_state:
        st.session_state.adaptive_inputs = []

    if 'adaptive_values' not in st.session_state:
        st.session_state.adaptive_values = {}

    if 'decision_formulas' not in st.session_state:
        st.session_state.decision_formulas = {
            'cost_delta_formula': "",
            'adjusted_surplus_formula': ""
        }

    if 'evaluation_dimensions' not in st.session_state:
        st.session_state.evaluation_dimensions = {
            'financial_label': "Financial Score",
            'non_financial_label': "Strategic Return"
        }

    if 'info_data' not in st.session_state:
        st.session_state.info_data = {
            'metric_name': "Primary Value Impact", 
            'knowledge_gaps': "",
            'custom_formula_rule': ""
        }

    if 'values_data' not in st.session_state:
        st.session_state.values_data = {'cost_weight': 50, 'human_weight': 50, 'value_tradeoff_notes': ""}

    if 'reasoning_data' not in st.session_state:
        st.session_state.reasoning_data = {}

    if 'ai_chat_history' not in st.session_state:
        st.session_state.ai_chat_history = [
            {
                "role": "assistant",
                "content": "Welcome - I am your decision coach. Tell me about the core problem or project scenario you wish to investigate so we can map your financial and non-financial metrics transparently."
            }
        ]

    if 'api_error_hold' not in st.session_state:
        st.session_state['api_error_hold'] = None

    api_key = os.environ.get("GEMINI_API_KEY")

    def execute_safe_local_calculation(input_kv_pairs, formula_str):
        if not formula_str:
            return 0.0, "No formula available."
        try:
            working_expression = formula_str.strip()
            for key in sorted(input_kv_pairs.keys(), key=len, reverse=True):
                val = float(input_kv_pairs[key])
                working_expression = working_expression.replace(key, f"({val})")
            allowed_chars = "0123456789+-*/(). "
            clean_expression = "".join([c for c in working_expression if c in allowed_chars])
            result = float(eval(clean_expression))
            return result, working_expression
        except Exception as e:
            return 0.0, f"Error tracking math substitution: {str(e)}"

    def auto_extract_data():
        if not st.session_state.ai_chat_history or not api_key:
            return False
        chat_log = ""
        for msg in st.session_state.ai_chat_history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                chat_log += f"{msg['role'].upper()}: {str(msg['content'])}\n\n"
            
        extraction_prompt = """
        Analyze the decision framework interview history below. Extract values to populate this dynamic schema structure.
        
        CRITICAL TRANSLATION ARCHITECTURE RULES:
        1. Parse the financial variables and output an array of 2 to 4 baseline properties inside "numeric_variables".
        2. Build algebraic standard formulas inside "formulas" using ONLY the keys established inside "numeric_variables".
        3. Detect custom evaluative criteria labels based on text details. Produce optimized strategic title names for "financial_label" and "non_financial_label" matching the context.
        
        Respond ONLY with a valid, clean JSON object matching this exact structure. Do not append trailing comments or markdown code ticks.
        {
            "problem_statement": "Core problem statement",
            "in_scope": "Explicit scope rules",
            "out_scope": "Deferred factors",
            "constraints": "System criteria boundaries",
            "alt1_name": "Alternative 1 summary text", "alt1_desc": "Alternative 1 context",
            "alt2_name": "Alternative 2 summary text", "alt2_desc": "Alternative 2 context",
            "alt3_name": "Alternative 3 summary text", "alt3_desc": "Alternative 3 context",
            "metric_name": "Primary analytical metric name with currency marker",
            "numeric_variables": [
                {"key": "unique_alphanumeric_key_1", "label": "Descriptive Field Name (€)", "default": 500.0, "min_val": 0.0, "max_val": 5000.0, "step": 50.0}
            ],
            "formulas": {
                "cost_delta_formula": "Algebraic operations pattern using defined keys",
                "adjusted_surplus_formula": "Algebraic balance calculation using defined keys"
            },
            "evaluation_dimensions": {
                "financial_label": "Tailored Financial Parameter Title",
                "non_financial_label": "Tailored Qualitative Human/Risk Parameter Title"
            }
        }
        INTERVIEW LOG TO PARSE:
        """ + chat_log
        
        try:
            extractor_client = genai.Client(api_key=api_key)
            response = extractor_client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[{"role": "user", "parts": [{"text": extraction_prompt}]}]
            )
            clean_json_text = response.text.strip().replace("```json", "").replace("```", "")
            extracted_dict = json.loads(clean_json_text)
            
            for k in ["problem_statement", "in_scope", "out_scope", "constraints"]:
                if k in extracted_dict and extracted_dict[k]:
                    st.session_state.framing_data[k] = extracted_dict[k]
                    
            for alt_idx in ["alt1", "alt2", "alt3"]:
                for attr in ["name", "desc"]:
                    k = f"{alt_idx}_{attr}"
                    if k in extracted_dict and extracted_dict[k]:
                        st.session_state.alternatives_data[k] = extracted_dict[k]
                    
            if "metric_name" in extracted_dict and extracted_dict["metric_name"]:
                st.session_state.info_data["metric_name"] = extracted_dict["metric_name"]
                
            if "numeric_variables" in extracted_dict and isinstance(extracted_dict["numeric_variables"], list) and len(extracted_dict["numeric_variables"]) > 0:
                st.session_state.adaptive_inputs = extracted_dict["numeric_variables"]
                new_vals = {}
                for item in extracted_dict["numeric_variables"]:
                    new_vals[item["key"]] = float(item.get("default", 0.0))
                st.session_state.adaptive_values = new_vals
                
            if "formulas" in extracted_dict:
                st.session_state.decision_formulas = extracted_dict["formulas"]
                
            if "evaluation_dimensions" in extracted_dict:
                st.session_state.evaluation_dimensions = extracted_dict["evaluation_dimensions"]
                
            return True
        except Exception as e:
            st.session_state['api_error_hold'] = f"Data Extraction Notice: {str(e)}"
            return False

    def trigger_elite_gemini_call(prompt_text):
        conversation_context = ""
        for msg in st.session_state.ai_chat_history[-8:]:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                if msg['content'] == prompt_text and msg['role'] == 'user': continue
                conversation_context += f"{msg['role'].upper()}: {str(msg['content'])}\n\n"
        system_instruction = """You are an elite decision coach.
    Your strict objective is helping the user systematically complete framing boundaries across any arbitrary domain.
    Keep responses highly concise (under 2 paragraphs).
    Acknowledge discovered facts, link them directly to clear qualitative and quantitative parameters, and end your response by asking exactly one targeted question to pin down parameters still unrefined."""
        flattened_prompt = f"{system_instruction}\n\nCONVERSATION LOG DATA:\n{conversation_context}\n\nUSER'S LATEST STATEMENT:\n{prompt_text}"
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model='gemini-3.6-flash',contents=[{"role": "user", "parts": [{"text": flattened_prompt}]}])
            return response.text
        except Exception as e:
            st.session_state['api_error_hold'] = f"Model Communication Error: {str(e)}"
            return None

    # --- DYNAMIC QUALITY MATURITY LOGIC CALCULATOR ---
    def compute_framework_maturity():
        score_breakdown = {
            'Appropriate Frame': 40 if st.session_state.framing_data.get('problem_statement') else 10,
            'Creative Alternatives': 50 if st.session_state.alternatives_data.get('alt1_name') and st.session_state.alternatives_data.get('alt2_name') else 20,
            'Reliable Information': 60 if len(st.session_state.adaptive_inputs) >= 2 else 30,
            'Clear Values & Trade-offs': 70 if st.session_state.values_data.get('value_tradeoff_notes') else 40,
            'Logically Sound Reasoning': 80 if st.session_state.reasoning_data else 30,
            'Commitment to Action': 65 if st.session_state.decision_metadata.get('owner') else 35
        }
        
        # Dynamic AI Scaling Boost if active setup talk exists
        if len(st.session_state.ai_chat_history) > 2:
            for key in score_breakdown:
                score_breakdown[key] = min(100, score_breakdown[key] + 15)
                
        st.session_state.dq_scores = score_breakdown
        total_maturity = sum(score_breakdown.values()) / 6.0
        return round(total_maturity, 1)

    current_maturity = compute_framework_maturity()

    # 3. SIDEBAR NAVIGATION & PERSISTENCE BACKUP CONTROLS
    with st.sidebar:
        st.title("🏛️ Decision Coach Engine")
        st.markdown(f"**Welcome, {username}!**")
        authenticator.logout("Logout", "sidebar")
        st.markdown("---")
        app_mode = st.radio("Workflow Steps", ["1. 🤖 AI Framing & Discovery Coach","2. 🎯 Document Frame Boundaries","3. 🎨 Define Strategy Alternatives","4. 📊 Model Information Ranges & Benchmarks","5. ⚖️ Calibrate Values & Rate Analysis","6. 📊 Executive Dashboard"])
        st.markdown("---")
        st.subheader("Decision Context")
        st.session_state.decision_metadata['name'] = st.text_input("Decision Title", value=st.session_state.decision_metadata['name'])
        st.session_state.decision_metadata['owner'] = st.text_input("Decision Owner", value=st.session_state.decision_metadata['owner'])
        
        # LIVE TRANSACTION LEDGER BLUEPRINT MONITOR
        st.markdown("---")
        st.subheader("🔎 Active Ledger Monitor")
        st.caption("Live structural values and evaluation templates running in local session memory:")
        if 'adaptive_inputs' in st.session_state and 'adaptive_values' in st.session_state:
            ledger_list = []
            for item in st.session_state.adaptive_inputs:
                k_id = item["key"]
                ledger_list.append({"Parameter Metric": item["label"],"Value": f"{st.session_state.adaptive_values.get(k_id, 0.0):,.2f}"})
            if ledger_list:
                st.table(pd.DataFrame(ledger_list))
            formulas = st.session_state.decision_formulas
            c_val, _ = execute_safe_local_calculation(st.session_state.adaptive_values, formulas.get('cost_delta_formula', ""))
            a_val, _ = execute_safe_local_calculation(st.session_state.adaptive_values, formulas.get('adjusted_surplus_formula', ""))
            st.caption(f"Target Score Forecast: {c_val:,.2f}")
            st.caption(f"Residual Balance Pool: {a_val:,.2f}")
        st.markdown("---")
        st.subheader("⚙️ Workspace Controls")
        
        if st.button("🔄 Reset Framework Scenario", use_container_width=True, type="primary"):
            reset_scenario()
            
        if st.session_state.get('api_error_hold'):
            st.info(st.session_state['api_error_hold'])
        if st.button("🗑️ Clear Log Notice", use_container_width=True):
            st.session_state['api_error_hold'] = None
            st.rerun()
        output_state = {key: st.session_state[key] for key in st.session_state.keys() if key not in ['api_error_hold', 'authentication_status', 'username', 'logout']}
        chat_dump = json.dumps(output_state, indent=2)
        st.download_button(label="📥 Download Workspace Save (.json)",data=chat_dump,file_name="decision_framework_backup.json",mime="application/json",use_container_width=True)
        uploaded_backup = st.file_uploader("📂 Reload Saved File", type=["json"])
        if uploaded_backup is not None:
            try:
                loaded_json = json.load(uploaded_backup)
                for key, val in loaded_json.items():
                    if key not in ['authentication_status', 'username', 'logout']:
                        st.session_state[key] = val
                st.sidebar.success("🎉 Framework State Synchronized!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Structure mismatch error: {str(e)}")

    # SECTION 1: COACH INTERFACE
    if app_mode == "1. 🤖 AI Framing & Discovery Coach":
        st.title("🤖 Step 1: AI Framing & Discovery Coach")
        st.markdown("Converse with the decision coach to adjust fields, add contextual elements, or expand metrics based on your strategic constraints.")
        if not api_key:
            st.warning("⚠️ GEMINI_API_KEY missing. Operating in fallback descriptive mock state.")
        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if user_input := st.chat_input("Provide details to your decision matrix..."):
            st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.spinner("Processing framework tracking adjustment..."):
                if api_key:
                    ai_response = trigger_elite_gemini_call(user_input)
                    if ai_response:
                        st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})
                    auto_extract_data()
                else:
                    fallback_response = f"Recorded info update: '{user_input}'. Core state variables tracked locally."
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": fallback_response})
            st.rerun()

    # SECTION 2: BOUNDARIES
    elif app_mode == "2. 🎯 Document Frame Boundaries":
        st.title("🎯 Step 2: Document Frame Boundaries")
        st.markdown("Define specific properties contained within this evaluation trace.")
        st.session_state.framing_data['problem_statement'] = st.text_area("Problem Statement / Core Objectives", st.session_state.framing_data['problem_statement'])
        st.session_state.framing_data['in_scope'] = st.text_area("In Scope Variables & Assets", st.session_state.framing_data['in_scope'])
        st.session_state.framing_data['out_scope'] = st.text_area("Out of Scope / Deferred Factors", st.session_state.framing_data['out_scope'])
        st.session_state.framing_data['constraints'] = st.text_area("Hard Boundaries & System Constraints", st.session_state.framing_data['constraints'])

    # SECTION 3: ALTERNATIVES
    elif app_mode == "3. 🎨 Define Strategy Alternatives":
        st.title("🎨 Step 3: Define Strategy Alternatives")
        st.markdown("Ensure options are structured as distinct strategic paths.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Option Alpha")
            st.session_state.alternatives_data['alt1_name'] = st.text_input("Alternative 1 Name", st.session_state.alternatives_data.get('alt1_name', "Alternative 1"))
            st.session_state.alternatives_data['alt1_desc'] = st.text_area("Alternative 1 Description", st.session_state.alternatives_data.get('alt1_desc', ""))
        with col2:
            st.subheader("Option Beta")
            st.session_state.alternatives_data['alt2_name'] = st.text_input("Alternative 2 Name", st.session_state.alternatives_data.get('alt2_name', "Alternative 2"))
            st.session_state.alternatives_data['alt2_desc'] = st.text_area("Alternative 2 Description", st.session_state.alternatives_data.get('alt2_desc', ""))
        with col3:
            st.subheader("Option Gamma")
            st.session_state.alternatives_data['alt3_name'] = st.text_input("Alternative 3 Name", st.session_state.alternatives_data.get('alt3_name', "Alternative 3"))
            st.session_state.alternatives_data['alt3_desc'] = st.text_area("Alternative 3 Description", st.session_state.alternatives_data.get('alt3_desc', ""))

    # SECTION 4: INFORMATION RANGES
    elif app_mode == "4. 📊 Model Information Ranges & Benchmarks":
        st.title("📊 Step 4: Model Information Ranges & Trace Benchmarks")
        st.markdown("Calibrate input variable attributes and review the symbolic equations extracted by the setup routine.")
        st.session_state.info_data['metric_name'] = st.text_input("Primary Evaluative Metric Label", st.session_state.info_data.get('metric_name', "Primary Value Impact"))
        st.session_state.info_data['knowledge_gaps'] = st.text_area("Critical Knowledge Gaps & Uncertainties", st.session_state.info_data.get('knowledge_gaps', ""))
        st.markdown("### 🔢 Live Mathematical Formula Blueprint")
        st.caption("These equations match your custom scenario parameters and resolve strictly via local deterministic execution engine:")
        formulas = st.session_state.decision_formulas
        st.code(f"Primary Cost Delta Target String:  {formulas.get('cost_delta_formula', 'None')}\nResidual Balance Target String:    {formulas.get('adjusted_surplus_formula', 'None')}", language="python")
        st.markdown("### 🛠️ Domain Adaptive Variables")
        columns_list = st.columns(min(3, len(st.session_state.adaptive_inputs)))
        for idx, item in enumerate(st.session_state.adaptive_inputs):
            col_target = columns_list[idx % len(columns_list)]
            v_key = item["key"]
            v_label = item["label"]
            v_def = float(st.session_state.adaptive_values.get(v_key, item.get("default", 0.0)))
            v_min = float(item.get("min_val", 0.0))
            v_max = float(item.get("max_val", 1000000000.0))
            v_step = float(item.get("step", 1.0))
            if v_def < v_min: v_def = v_min
            if v_def > v_max: v_def = v_max
            with col_target:
                st.session_state.adaptive_values[v_key] = st.number_input(label=v_label, min_value=v_min, max_value=v_max, value=v_def, step=v_step, key=f"dyn_input_{v_key}")

    # SECTION 5: VALUE CALIBRATION & LABELING
    elif app_mode == "5. ⚖️ Calibrate Values & Rate Analysis":
        st.title("⚖️ Step 5: Calibrate Strategic Values & Strategy Performance")
        st.markdown("Configure multi-criteria priority weight properties across financial and non-financial indices.")
        labels = st.session_state.evaluation_dimensions
        st.markdown("### 🏷️ Dynamic Criteria Dimensions")
        st.caption("Adjust evaluation header names or modify context elements in Step 1 to prompt automatic structure variations.")
        col_lbl1, col_lbl2 = st.columns(2)
        with col_lbl1:
            labels['financial_label'] = st.text_input("Financial Performance Parameter Title:", value=labels.get('financial_label', "Financial Score"))
        with col_lbl2:
            labels['non_financial_label'] = st.text_input("Non-Financial/Qualitative Parameter Title:", value=labels.get('non_financial_label', "Strategic Return"))
        st.markdown("---")
        st.markdown("### 🎯 Part A: Multi-Criteria Value Weighting Configuration")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.session_state.values_data['cost_weight'] = st.slider(f"{labels['financial_label']} Target Weight", 0, 100, int(st.session_state.values_data.get('cost_weight', 50)))
        with col_w2:
            st.session_state.values_data['human_weight'] = st.slider(f"{labels['non_financial_label']} Target Weight", 0, 100, int(st.session_state.values_data.get('human_weight', 50)))
        st.session_state.values_data['value_tradeoff_notes'] = st.text_area("Value Alignment Evaluation Notes", st.session_state.values_data.get('value_tradeoff_notes', ""))
        st.markdown("---")
        st.markdown(f"### 🧠 Part B: Strategy Performance Configuration Profiles (1 to 10 Scale)")
        for alt_idx in ["alt1", "alt2", "alt3"]:
            name_key = f"{alt_idx}_name"
            name_disp = st.session_state.alternatives_data.get(name_key, alt_idx.upper())
            st.markdown(f"#### 🌟 Performance Assignment for Alternative: {name_disp}")
            col_f, col_h = st.columns(2)
            with col_f:
                k_cost = f"{alt_idx}_cost_perf"
                st.session_state.reasoning_data[k_cost] = st.slider(f"{labels['financial_label']}: {name_disp}", 1, 10, int(st.session_state.reasoning_data.get(k_cost, 5)), key=f"w{k_cost}")
            with col_h:
                k_human = f"{alt_idx}_risk_perf"
                st.session_state.reasoning_data[k_human] = st.slider(f"{labels['non_financial_label']}: {name_disp}", 1, 10, int(st.session_state.reasoning_data.get(k_human, 5)), key=f"w{k_human}")

    # SECTION 6: EXECUTIVE DASHBOARD
    elif app_mode == "6. 📊 Executive Dashboard":
        st.title("📊 Executive Decision Quality Dashboard")
        metric_label = st.session_state.info_data.get('metric_name', "Primary Value Impact")
        formulas = st.session_state.decision_formulas
        labels = st.session_state.evaluation_dimensions
        
        # Run deterministic computations safely
        cost_delta, cost_audit = execute_safe_local_calculation(st.session_state.adaptive_values, formulas.get('cost_delta_formula', ""))
        adjusted_surplus, surplus_audit = execute_safe_local_calculation(st.session_state.adaptive_values, formulas.get('adjusted_surplus_formula', ""))
        
        st.markdown("### 📊 Glass-Box Model Diagnostics")
        m1, m2 = st.columns(2)
        with m1:
            st.metric(f"Net Analytical Computation ({metric_label})", f"€{cost_delta:,.2f}")
        with m2:
            st.metric("Adjusted Residual Boundary Reserve Pool Balance", f"€{adjusted_surplus:,.2f}",
            delta="Sufficient Strategic Surplus Margin" if adjusted_surplus >= 0 else "Negative Residual Margin Alert",
            delta_color="normal" if adjusted_surplus >= 0 else "inverse")
        
        # TRANSPARENT AUDIT LEDGER TRAIL PANEL
        with st.expander("🔎 View Live Deterministic Equation Trace", expanded=False):
            st.markdown(f"Calculated Score Target Formula: {formulas.get('cost_delta_formula')}")
            st.code(f"Substituted Values Trace Evaluation: {cost_audit} = {cost_delta:,.2f}")
            st.markdown(f"Residual Boundary Balance Formula: {formulas.get('adjusted_surplus_formula')}")
            st.code(f"Substituted Values Trace Evaluation: {surplus_audit} = {adjusted_surplus:,.2f}")
            
        chart_records = []
        best_strategy, best_score = None, -1
        for alt_idx in ["alt1", "alt2", "alt3"]:
            alt_name = st.session_state.alternatives_data.get(f"{alt_idx}_name", alt_idx.upper())
            c_perf = st.session_state.reasoning_data.get(f"{alt_idx}_cost_perf", 5)
            h_perf = st.session_state.reasoning_data.get(f"{alt_idx}_risk_perf", 5)
            c_weight = st.session_state.values_data.get('cost_weight', 50)
            h_weight = st.session_state.values_data.get('human_weight', 50)
            composite_score = ((c_perf * c_weight) + (h_perf * h_weight)) / float(max(1, c_weight + h_weight))
            composite_score = round(composite_score, 2)
            chart_records.append({"Strategy Path": alt_name, labels['financial_label']: c_perf, labels['non_financial_label']: h_perf, "Aggregated Performance Score": composite_score})
            if composite_score > best_score:
                best_score = composite_score
                best_strategy = alt_name
        df_scores = pd.DataFrame(chart_records)
        
        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("### 🏆 Multi-Criteria Strategy Optimization Profile")
            fig_bar = px.bar(df_scores, x="Strategy Path", y="Aggregated Performance Score", text="Aggregated Performance Score", color="Strategy Path", range_y=[0, 10])
            fig_bar.update_traces(textposition='outside', texttemplate='%{text:.2f}')
            fig_bar.update_layout(margin=dict(l=40, r=40, t=40, b=80), xaxis_tickangle=-15)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_chart2:
            st.markdown(f"### 🔀 Trade-Off Analysis: {labels['non_financial_label']} vs. {labels['financial_label']}")
            fig_scatter = px.scatter(df_scores, x=labels['financial_label'], y=labels['non_financial_label'], text="Strategy Path", color="Strategy Path", size=[15]*len(df_scores), range_x=[0, 11], range_y=[0, 11])
            fig_scatter.update_traces(textposition='top center', textfont=dict(size=10, color='black'))
            fig_scatter.update_layout(margin=dict(l=40, r=60, t=40, b=80))
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=11, y1=11, line=dict(color="Gray", dash="dash"))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        # --- RADAR SPIDER PLOT ---
        st.markdown("---")
        col_radar, col_rec = st.columns(2)
        with col_radar:
            st.markdown("### 🕸️ 6 Dimensions of Decision Quality")
            categories = list(st.session_state.dq_scores.keys())
            values = list(st.session_state.dq_scores.values())
            
            # Close the loop for the path drawing
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name='Current Framework Score',
                line_color='#2563EB'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=60, r=60, t=30, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_rec:
            st.markdown("### 🏛️ Framework Recommendation Summary")
            if best_strategy and best_score >= 0:
                st.success(f"Optimal Strategy Path Assignment: Based on active performance scores and weight assignments, **{best_strategy}** yields the maximum evaluated outcome with a comprehensive rating of {best_score}/10.")
            st.markdown("""⚠️ Standard Professional Disclaimer
    This decision tool operates deterministically based entirely on specific structural variables, user preference sliders, and substitution templates.
    The generated metrics summarize value trade-offs and do not constitute legal or statutory public sector pension advisory.""")

    # 4. GLOBAL STICKY PROGRESS FOOTER PANEL
    progress_color = "#10B981" if current_maturity >= 70 else "#F59E0B" if current_maturity >= 40 else "#EF4444"
    st.markdown(f"""
        <div class="sticky-progress">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 600; font-size: 0.9rem; color: #374151;">📋 Framework Setup Information Maturity Score</span>
                <span style="font-weight: bold; font-size: 0.9rem; color: {progress_color};">{current_maturity}% Complete</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.progress(current_maturity / 100.0)
