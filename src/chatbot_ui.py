
import streamlit as st
from typing import Optional
from src.chatbot_context import InferenceContext
from src.chatbot_engine import ChatbotEngine


def get_chatbot_engine() -> ChatbotEngine:
    """Retrieve or initialize the ChatbotEngine in st.session_state."""
    if "chatbot_engine" not in st.session_state:
        st.session_state.chatbot_engine = ChatbotEngine()
    return st.session_state.chatbot_engine


def update_chatbot_context(context: InferenceContext):
    """Update the engine's inference context in st.session_state."""
    try:
        engine = get_chatbot_engine()
        engine.set_inference_context(context)
    except Exception as e:
        pass


def render_chatbot():
    """Render the floating chatbot widget at bottom-right of the Streamlit dashboard."""
    try:
        engine = get_chatbot_engine()

        if "chatbot_open" not in st.session_state:
            st.session_state.chatbot_open = False
        if "chatbot_messages" not in st.session_state:
            st.session_state.chatbot_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I am the **ABIDE ASD Assistant**. "
                        "Once a scan is uploaded and processed, I can explain the real-time "
                        "predictions, probabilities, pivotal features, and the RBF-SVM pipeline. "
                        "How can I help you today?"
                    ),
                }
            ]

        # Inject CSS for floating button & modern floating chat window
        custom_css = """
        <style>
        /* Floating Chatbot Container */
        .floating-chat-container {
            position: fixed;
            bottom: 25px;
            right: 25px;
            z-index: 999999;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Floating Toggle Button */
        .chat-toggle-btn {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            color: white !important;
            border-radius: 50px;
            padding: 12px 24px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            font-weight: 600;
            cursor: pointer;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease-in-out;
        }
        .chat-toggle-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        }

        /* Chat Panel Card */
        .chat-panel-box {
            background-color: #ffffff;
            border-radius: 16px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
            padding: 16px;
            margin-top: 10px;
            color: #212121;
        }

        /* Message Bubbles */
        .user-bubble {
            background-color: #E3F2FD;
            color: #0D47A1;
            padding: 10px 14px;
            border-radius: 14px 14px 2px 14px;
            margin: 6px 0 6px auto;
            max-width: 85%;
            font-size: 0.92rem;
            word-wrap: break-word;
            border: 1px solid #BBDEFB;
        }

        .bot-bubble {
            background-color: #F5F5F5;
            color: #212121;
            padding: 12px 16px;
            border-radius: 14px 14px 14px 2px;
            margin: 6px auto 6px 0;
            max-width: 90%;
            font-size: 0.92rem;
            line-height: 1.45;
            word-wrap: break-word;
            border: 1px solid #E0E0E0;
        }

        .chat-badge {
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 3px;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

        # Place the chatbot UI in an expandable container on the page
        st.markdown("---")
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col2:
                # Toggle button state
                toggle_label = "💬 Close NeuroLab AI Bot" if st.session_state.chatbot_open else "🤖 NeuroLab AI Bot"
                if st.button(toggle_label, key="chatbot_toggle_trigger"):
                    st.session_state.chatbot_open = not st.session_state.chatbot_open
                    st.rerun()

            if st.session_state.chatbot_open:
                st.markdown("### 🤖 ABIDE ASD Intelligence & Explanation Assistant")
                st.caption("Deterministic reasoning engine grounded in actual pipeline inference results.")

                # Quick Question Chips
                st.write("**Quick Inquiries:**")
                quick_cols = st.columns(3)
                if quick_cols[0].button("📊 Prediction?", key="quick_pred"):
                    _submit_message("What did the model predict and why?", engine)
                if quick_cols[1].button("📈 Probability?", key="quick_prob"):
                    _submit_message("What are the exact ASD and Control probabilities?", engine)
                if quick_cols[2].button("🧠 Features?", key="quick_feat"):
                    _submit_message("How were the 600 pivotal connectivity edges selected?", engine)

                # Message History Box
                chat_history_container = st.container(height=380)
                with chat_history_container:
                    for msg in st.session_state.chatbot_messages:
                        if msg["role"] == "user":
                            st.markdown(
                                f"<div class='user-bubble'><div class='chat-badge' style='color:#1565C0;'>You</div>{msg['content']}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='bot-bubble'><div class='chat-badge' style='color:#2E7D32;'>Assistant</div>{msg['content']}</div>",
                                unsafe_allow_html=True,
                            )

                # Input controls
                with st.form(key="chatbot_input_form", clear_on_submit=True):
                    input_col, send_col, clear_col = st.columns([5, 1, 1])
                    user_query = input_col.text_input(
                        "Ask the assistant a question...",
                        placeholder="e.g., Why ASD? What is C=10? What is the false positive rate?",
                        label_visibility="collapsed",
                    )
                    submit_clicked = send_col.form_submit_button("Send", use_container_width=True)
                    clear_clicked = clear_col.form_submit_button("Reset", use_container_width=True)

                    if clear_clicked:
                        engine.reset_conversation()
                        st.session_state.chatbot_messages = [
                            {
                                "role": "assistant",
                                "content": "Conversation reset. Feel free to ask any question about the model, prediction, or pipeline.",
                            }
                        ]
                        st.rerun()

                    if submit_clicked and user_query:
                        _submit_message(user_query, engine)

    except Exception:
        # Failsafe: Never break the main application
        pass


def _submit_message(text: str, engine: ChatbotEngine):
    """Process user message and update chat history."""
    try:
        # Record user message
        st.session_state.chatbot_messages.append({"role": "user", "content": text})

        # Process through rule-based deterministic engine
        response = engine.process_message(text)

        # Record assistant reply
        st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
        st.rerun()
    except Exception as e:
        st.session_state.chatbot_messages.append({
            "role": "assistant",
            "content": "I encountered an error generating an answer. Please check your query or try rephrasing."
        })
        st.rerun()
