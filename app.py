"""AmazonHelp Autonomous AI Support Console — Streamlit Frontend.

Production-grade interactive console for evaluators and engineers to:
- Conduct multi-turn customer support dialogues.
- Inspect structured decision signals (Intent, State, Action, Escalation, Safety).
- View dense historical retrieval evidence (Train-only K=5 exemplars).
- Monitor execution latency telemetry and component breakdown.
- Step through pre-configured multi-turn demo scenarios.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
import streamlit as st

# Ensure project root is in sys.path for robust resolution across run configurations
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.conversation_manager import (
    ConversationManager,
    TurnDecisionRecord,
)
from src.agent.demo_scenarios import DEMO_SCENARIOS, DemoScenario
from src.llm.model_client import OllamaClient

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AmazonHelp Autonomous AI Support Console",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise support console CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
    .status-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .badge-mock {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        border: 1px solid #bae6fd;
    }
    .badge-live {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        border: 1px solid #bbf7d0;
    }
    .badge-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid #bbf7d0;
    }
    .badge-escalate {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid #fecaca;
    }
    .badge-tag {
        background-color: #f3f4f6;
        color: #374151;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #e5e7eb;
    }
    .decision-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 16px;
    }
    .decision-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 10px;
    }
    .decision-cell {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 8px 12px;
    }
    .decision-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .decision-val {
        font-size: 0.92rem;
        font-weight: 600;
        color: #111827;
        word-break: break-word;
    }
    .decision-sub {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 2px;
    }
    .system-ready-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 16px;
    }
    .exemplar-featured {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-left: 4px solid #f59e0b;
        border-radius: 0 6px 6px 0;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .exemplar-item {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .welcome-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

if "use_mock" not in st.session_state:
    st.session_state.use_mock = True

if "manager" not in st.session_state:
    st.session_state.manager = ConversationManager(
        conversation_id=st.session_state.conversation_id,
        use_mock_llm=st.session_state.use_mock,
        model_name="llama3.2:1b",
        default_k=5,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_record" not in st.session_state:
    st.session_state.latest_record = None

if "customer_input_text" not in st.session_state:
    st.session_state.customer_input_text = ""

# -----------------------------------------------------------------------------
# 3. SIDEBAR: IDENTITY, CONTROLS & MULTI-TURN DEMO STEPPER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 AmazonHelp Agent")
    st.caption("Autonomous Customer Support Decision & Response Console")
    st.markdown("---")

    # Execution Backend Selector
    st.subheader("⚙️ Execution Backend")
    mode_selection = st.radio(
        "Inference Engine",
        options=["Offline Mock Simulation", "Live Ollama (llama3.2:1b)"],
        index=0 if st.session_state.use_mock else 1,
        help="Select offline deterministic mock simulation or local live Ollama model server.",
    )

    ollama_client = OllamaClient()
    ollama_ready = ollama_client.is_available()

    desired_mock = (mode_selection == "Offline Mock Simulation")
    if not desired_mock and not ollama_ready:
        st.warning(
            "⚠️ **Local Ollama is offline.**\n\n"
            "To use live model inference, run `ollama serve` and `ollama pull llama3.2:1b`.\n\n"
            "Falling back safely to **Offline Mock Simulation**."
        )
        desired_mock = True

    # Re-instantiate manager if execution mode changed
    if desired_mock != st.session_state.use_mock:
        st.session_state.use_mock = desired_mock
        st.session_state.manager = ConversationManager(
            conversation_id=st.session_state.conversation_id,
            use_mock_llm=desired_mock,
            model_name="llama3.2:1b",
            default_k=5,
        )
        st.toast(f"Backend switched to: {'Mock Simulation' if desired_mock else 'Live Ollama'}")

    mode_badge_html = (
        '<span class="badge-mock">OFFLINE MOCK SIMULATION</span>'
        if st.session_state.use_mock
        else '<span class="badge-live">LIVE OLLAMA (1B)</span>'
    )
    st.markdown(f"**Engine Status:** {mode_badge_html}", unsafe_allow_html=True)
    st.caption(f"Session ID: `{st.session_state.conversation_id}`")
    st.markdown("---")

    # Multi-Turn Demo Scenarios Stepper
    st.subheader("📋 Pre-Configured Demo Scenarios")
    scenario_titles = ["-- Select a Pre-Built Scenario --"] + [
        f"{s.scenario_id}: {s.title}" for s in DEMO_SCENARIOS
    ]
    selected_scenario_label = st.selectbox(
        "Load Evaluator Scenario",
        options=scenario_titles,
        index=0,
        help="Choose a pre-defined multi-turn scenario to inspect individual dialogue turns.",
    )

    if selected_scenario_label != "-- Select a Pre-Built Scenario --":
        scen_id = selected_scenario_label.split(":")[0].strip()
        scenario_obj: DemoScenario = next((s for s in DEMO_SCENARIOS if s.scenario_id == scen_id), None)
        if scenario_obj and scenario_obj.turns:
            st.markdown(f"**Category:** `{scenario_obj.category}`")
            st.write(scenario_obj.description)

            # Scenario property badges
            badge_htmls = []
            if scenario_obj.is_multi_turn:
                badge_htmls.append('<span class="badge-tag">🔄 Multi-Turn</span>')
            if scenario_obj.demonstrates_safety:
                badge_htmls.append('<span class="badge-tag">🛡️ Safety Guardrail</span>')
            if scenario_obj.demonstrates_low_confidence:
                badge_htmls.append('<span class="badge-tag">📉 Low-Confidence</span>')
            if any(t.expected_escalation for t in scenario_obj.turns):
                badge_htmls.append('<span class="badge-tag">🚨 Escalation</span>')
            if badge_htmls:
                st.markdown(" ".join(badge_htmls), unsafe_allow_html=True)

            st.markdown("##### Scenario Dialogue Turns")
            total_turns = len(scenario_obj.turns)

            for t_idx, turn in enumerate(scenario_obj.turns):
                with st.container():
                    st.caption(f"**Turn {turn.turn_num} of {total_turns}** — *{turn.expected_action}*")
                    if turn.highlights:
                        st.caption(f"💡 {turn.highlights}")
                    if st.button(
                        f"📥 Load Turn {turn.turn_num}",
                        key=f"btn_load_{scenario_obj.scenario_id}_{turn.turn_num}",
                        use_container_width=True,
                    ):
                        st.session_state.customer_input_text = turn.customer_message
                        st.rerun()

    st.markdown("---")

    # Reset Conversation Button
    if st.button("🔄 Reset Conversation", use_container_width=True, type="secondary"):
        st.session_state.manager.reset()
        st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        st.session_state.messages.clear()
        st.session_state.latest_record = None
        st.session_state.customer_input_text = ""
        st.rerun()

# -----------------------------------------------------------------------------
# 4. MAIN HEADER & STATUS BAR
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">AmazonHelp Autonomous AI Support Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Multi-turn Customer Support Agent • Deterministic Policy • Retrieval-Grounded Responses"
    "</div>",
    unsafe_allow_html=True,
)

status_html = (
    f'<div class="status-bar">'
    f'<span><strong>Backend:</strong> {mode_badge_html}</span>'
    f'<span class="badge-tag">Session: {st.session_state.conversation_id}</span>'
    f'<span class="badge-tag">Corpus: 5,502 Train Exemplars</span>'
    f'<span class="badge-tag">Frozen Benchmark: 200 Golden Checkpoints</span>'
    f'</div>'
)
st.markdown(status_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN 2-COLUMN LAYOUT: [58, 42]
# -----------------------------------------------------------------------------
col_chat, col_inspector = st.columns([58, 42])

# =============================================================================
# LEFT COLUMN: LIVE CONVERSATION & CHAT INPUT CONSOLE
# =============================================================================
with col_chat:
    st.subheader("💬 Live Conversation")

    # Conversation History / Welcome State
    if not st.session_state.messages:
        # Welcome Card with 3 Quick-Start Scenarios
        st.markdown(
            """
            <div class="welcome-box">
                <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b; margin-bottom: 4px;">
                    👋 Ready to analyze a customer message
                </div>
                <div style="font-size: 0.88rem; color: #64748b; margin-bottom: 12px;">
                    Select a quick-start scenario below or enter a customer message in the console to evaluate the tri-layer decision pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Quick-Start Evaluation Examples:**")
        q1_col, q2_col, q3_col = st.columns(3)
        with q1_col:
            if st.button("📦 Delivery Tracking", use_container_width=True):
                st.session_state.customer_input_text = (
                    "Hi @AmazonHelp, my package was supposed to arrive yesterday by 8pm but it's still showing in transit. "
                    "Can you check where it is?"
                )
                st.rerun()
            st.caption("Standard delayed parcel status inquiry.")

        with q2_col:
            if st.button("💳 Refund & Return", use_container_width=True):
                st.session_state.customer_input_text = (
                    "I dropped off my return at the Post Office 5 days ago with the prepaid Royal Mail label. "
                    "When will my refund be processed back to my bank account?"
                )
                st.rerun()
            st.caption("Post-office drop-off refund timeframe.")

        with q3_col:
            if st.button("🚨 Security / Credential", use_container_width=True):
                st.session_state.customer_input_text = (
                    "Someone just changed the email on my account and ordered an iPad! "
                    "My old password was hunter2 and my OTP was 998231. Can you log into my account and block this fraud?"
                )
                st.rerun()
            st.caption("Credential suppression & fraud escalation.")

    else:
        # Display conversation history chronologically
        for turn_idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "customer":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**Customer:** {msg['text']}")
            else:
                rec: TurnDecisionRecord = msg.get("record")
                with st.chat_message("assistant", avatar="🛒"):
                    tag = (
                        '<span class="badge-mock">MOCK</span>'
                        if (rec and rec.is_mock)
                        else '<span class="badge-live">OLLAMA 1B</span>'
                    )
                    st.markdown(f"**AmazonHelp Agent** {tag}:", unsafe_allow_html=True)
                    st.markdown(f"{msg['text']}")
                    if rec:
                        st.caption(
                            f"⚡ **Action Taken:** `{rec.action}` | "
                            f"**State:** `{rec.state}` | "
                            f"*{rec.reasoning_summary or 'Policy-aligned response'}*"
                        )

    # Chat Input Console AFTER Conversation History
    st.markdown("---")
    customer_input = st.text_area(
        "Inbound Customer Message:",
        value=st.session_state.customer_input_text,
        placeholder="Enter customer tweet or question (e.g. 'Where is my order? It was supposed to arrive yesterday.')",
        height=75,
        key="customer_input_widget",
    )

    btn_send_col, btn_clear_col = st.columns([1, 1])
    with btn_send_col:
        send_pressed = st.button("🚀 Send Message", type="primary", use_container_width=True)
    with btn_clear_col:
        clear_pressed = st.button("🔄 Clear Input", type="secondary", use_container_width=True)

    if clear_pressed:
        st.session_state.customer_input_text = ""
        st.rerun()

    if send_pressed:
        message_text = customer_input.strip()
        if not message_text:
            st.error("Please enter a customer message before sending.")
        else:
            with st.spinner("Processing turn through decision pipeline..."):
                try:
                    record: TurnDecisionRecord = st.session_state.manager.process_turn(message_text)
                    st.session_state.messages.append({"role": "customer", "text": message_text, "record": None})
                    st.session_state.messages.append({"role": "agent", "text": record.final_response, "record": record})
                    st.session_state.latest_record = record
                    st.session_state.customer_input_text = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing customer turn: {e}")

# =============================================================================
# RIGHT COLUMN: TURN DECISION INSPECTOR
# =============================================================================
with col_inspector:
    st.subheader("🔍 Turn Decision Inspector")

    latest_rec: TurnDecisionRecord = st.session_state.latest_record

    # -------------------------------------------------------------------------
    # INITIAL EMPTY STATE: SYSTEM READY CARD
    # -------------------------------------------------------------------------
    if latest_rec is None:
        st.markdown(
            f"""
            <div class="system-ready-card">
                <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 6px;">
                    🟢 System Ready
                </div>
                <div style="font-size: 0.85rem; color: #15803d; line-height: 1.5;">
                    • <strong>Execution Mode:</strong> {'Offline Mock Simulation' if st.session_state.use_mock else 'Live Ollama'}<br>
                    • <strong>Intent Classifier:</strong> TF-IDF + Logistic Regression (Frozen Baseline)<br>
                    • <strong>Retrieval Engine:</strong> MiniLM-L6-v2 embeddings, K=5<br>
                    • <strong>Decision Layer:</strong> Deterministic State Machine & Action Policy<br>
                    • <strong>Safety Engine:</strong> Deterministic Credential & Action Claim Validator<br>
                    • <strong>Retrieval Corpus:</strong> 5,502 Train-only dialogues (Zero Leakage)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Submit a customer message or load a scenario turn to inspect structured signals.")

    else:
        # ---------------------------------------------------------------------
        # PRIMARY DECISION CARD (ALWAYS VISIBLE - NO TAB SWITCHING REQUIRED)
        # ---------------------------------------------------------------------
        esc_badge = (
            f'<span class="badge-escalate">🚨 ESCALATE ({latest_rec.escalation_reason})</span>'
            if latest_rec.escalate
            else '<span class="badge-safe">✅ AUTO-HANDLE</span>'
        )
        safety_badge = (
            '<span class="badge-safe">🛡️ SAFE</span>'
            if latest_rec.is_safe
            else '<span class="badge-escalate">⚠️ VIOLATION BLOCKED</span>'
        )

        card_html = f"""
        <div class="decision-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.82rem; font-weight: 700; color: #374151; text-transform: uppercase; letter-spacing: 0.05em;">
                    Turn {latest_rec.turn_number} Primary Decisions
                </span>
                <span>{esc_badge} &nbsp; {safety_badge}</span>
            </div>
            <div class="decision-grid">
                <div class="decision-cell">
                    <div class="decision-label">Predicted Intent</div>
                    <div class="decision-val">{latest_rec.intent}</div>
                    <div class="decision-sub">Confidence: {latest_rec.confidence * 100:.1f}%</div>
                </div>
                <div class="decision-cell">
                    <div class="decision-label">Dialogue State</div>
                    <div class="decision-val">{latest_rec.state}</div>
                    <div class="decision-sub">Conversation Depth: {latest_rec.turn_number}</div>
                </div>
                <div class="decision-cell">
                    <div class="decision-label">Selected Action</div>
                    <div class="decision-val">{latest_rec.action}</div>
                    <div class="decision-sub">Policy-Compliant Mapping</div>
                </div>
                <div class="decision-cell">
                    <div class="decision-label">Turn Latency</div>
                    <div class="decision-val">{latest_rec.total_latency_ms:.1f} ms</div>
                    <div class="decision-sub">Mode: {'Mock' if latest_rec.is_mock else 'Ollama 1B'}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Dynamic Safety Warning Alerts (Displayed only when guardrails trigger)
        if latest_rec.safety_violations:
            st.error(f"🚨 **Safety Violations Intercepted ({len(latest_rec.safety_violations)}):**")
            for v in latest_rec.safety_violations:
                st.markdown(f"- `{v}`")

        if latest_rec.safety_overrides:
            st.warning(f"⚠️ **Deterministic Safety Overrides Enforced ({len(latest_rec.safety_overrides)}):**")
            for o in latest_rec.safety_overrides:
                st.markdown(f"- `{o}`")

        # ---------------------------------------------------------------------
        # EXACTLY TWO TABS: EVIDENCE & TELEMETRY
        # ---------------------------------------------------------------------
        tab_evidence, tab_telemetry = st.tabs(
            ["📚 Historical Evidence (K=5)", "⏱️ Diagnostics & Telemetry"]
        )

        # TAB 1: HISTORICAL RETRIEVAL EVIDENCE (K=5 ACCESSIBILITY)
        with tab_evidence:
            st.markdown(
                f"**Top-1 Similarity:** `{latest_rec.retrieval_top1_similarity:.3f}` | "
                f"**Retrieval Quality:** `Evidence-Supported & Policy-Compliant` | "
                f"**Confidence:** `{latest_rec.retrieval_confidence}`"
            )
            st.caption("Historical source: 5,502 Train-only dialogues (MiniLM-L6-v2 embeddings, K=5)")

            exemplars = latest_rec.retrieval_exemplars or []
            if not exemplars:
                st.warning("No historical retrieval exemplars recorded for this turn.")
            else:
                # Exemplar #1 prominently displayed
                ex1 = exemplars[0]
                sim1 = getattr(ex1, "similarity_score", None) or getattr(ex1, "similarity", 0.0)
                prob1 = getattr(ex1, "customer_problem_summary", None) or getattr(ex1, "customer_text", "(N/A)")
                resp1 = getattr(ex1, "support_response", None) or getattr(ex1, "support_reply", "(N/A)")
                intent1 = getattr(ex1, "intent", "UNKNOWN")
                action1 = getattr(ex1, "action", "UNKNOWN")

                st.markdown(
                    f"""
                    <div class="exemplar-featured">
                        <div style="font-weight: 700; font-size: 0.88rem; color: #92400e; margin-bottom: 6px;">
                            ⭐ Top Match (#1) — Similarity: {sim1:.3f} | Intent: {intent1}
                        </div>
                        <div style="font-size: 0.85rem; color: #1f2937; margin-bottom: 6px;">
                            <strong>Customer Problem:</strong> <em>"{prob1}"</em>
                        </div>
                        <div style="font-size: 0.85rem; color: #1f2937; margin-bottom: 4px;">
                            <strong>Historical Support Response:</strong> <em>"{resp1}"</em>
                        </div>
                        <div style="font-size: 0.78rem; color: #6b7280;">
                            Derived Exemplar Action: <code>{action1}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Exemplars #2–#5 accessible in a single expander
                if len(exemplars) > 1:
                    with st.expander(f"View Additional Historical Matches (#2–#{len(exemplars)})", expanded=False):
                        for idx, ex in enumerate(exemplars[1:5], 2):
                            sim = getattr(ex, "similarity_score", None) or getattr(ex, "similarity", 0.0)
                            prob = getattr(ex, "customer_problem_summary", None) or getattr(ex, "customer_text", "(N/A)")
                            resp = getattr(ex, "support_response", None) or getattr(ex, "support_reply", "(N/A)")
                            ex_intent = getattr(ex, "intent", "UNKNOWN")
                            ex_action = getattr(ex, "action", "UNKNOWN")

                            st.markdown(
                                f"""
                                <div class="exemplar-item">
                                    <div style="font-weight: 600; font-size: 0.82rem; color: #374151; margin-bottom: 4px;">
                                        Exemplar #{idx} — Similarity: {sim:.3f} | Intent: {ex_intent}
                                    </div>
                                    <div style="font-size: 0.82rem; color: #4b5563; margin-bottom: 4px;">
                                        <strong>Customer:</strong> "{prob}"
                                    </div>
                                    <div style="font-size: 0.82rem; color: #4b5563; margin-bottom: 2px;">
                                        <strong>Response:</strong> "{resp}"
                                    </div>
                                    <div style="font-size: 0.75rem; color: #6b7280;">
                                        Action: <code>{ex_action}</code>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # TAB 2: DIAGNOSTICS & TELEMETRY
        with tab_telemetry:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Latency", f"{latest_rec.total_latency_ms:.1f} ms")
            with m2:
                st.metric("Retrieval Sim", f"{latest_rec.retrieval_top1_similarity:.3f}")
            with m3:
                st.metric("Ret Confidence", latest_rec.retrieval_confidence)

            st.markdown("##### Component Execution Breakdown")
            breakdown = latest_rec.latency_breakdown_ms or {}
            if breakdown:
                for phase, duration in breakdown.items():
                    col_p, col_d = st.columns([3, 1])
                    with col_p:
                        st.write(f"• {phase.replace('_', ' ').title()}:")
                    with col_d:
                        st.write(f"**{duration:.1f} ms**")
            else:
                st.caption("Sub-millisecond phase breakdown unavailable.")

            st.markdown(f"**Execution Mode:** {'Mock Simulation' if latest_rec.is_mock else 'Live Ollama 1B'}")
            st.markdown(f"**Session Identifier:** `{st.session_state.conversation_id}`")

            # Collapsed Raw JSON Record Viewer
            with st.expander("View Raw Decision Record (JSON)", expanded=False):
                st.json(latest_rec.to_dict())

            # Collapsible Safety Guardrails Reference
            with st.expander("🛡️ Active Safety Guardrails Policy", expanded=False):
                st.markdown(
                    "- **Zero Credential Solicitation:** Automated suppression of passwords, OTPs, PINs, CVVs.\n"
                    "- **Secure Channel Handoff:** Inbound account actions routed to authenticated Direct Message (DM).\n"
                    "- **Unsupported Action Prohibition:** Prohibits fabricating direct financial refunds or cancellations on Twitter."
                )

# -----------------------------------------------------------------------------
# 6. FOOTER / SYSTEM BENCHMARK SPECIFICATION
# -----------------------------------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("AmazonHelp Autonomous AI Support Console | SDE Evaluation Build")
with footer_col2:
    st.caption("Frozen Benchmark: 200 Golden Dev Checkpoints | 5,502 Train Retrieval Corpus")
with footer_col3:
    st.caption(f"Server Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
