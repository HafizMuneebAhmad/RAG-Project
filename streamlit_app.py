import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #151922;
        border-right: 1px solid #292e39;
    }

    /* Main content width */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Header */
    .app-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 30px;
    }

    /* Welcome box */
    .welcome-box {
        padding: 25px;
        border-radius: 16px;
        background: #151922;
        border: 1px solid #292e39;
        margin-bottom: 25px;
    }

    /* Status */
    .status {
        padding: 10px 14px;
        border-radius: 10px;
        background: #17221b;
        color: #7ee787;
        border: 1px solid #263d2c;
        font-size: 14px;
    }

    /* Small info */
    .info-card {
        padding: 14px;
        border-radius: 12px;
        background: #1a1f29;
        border: 1px solid #292e39;
        margin-bottom: 10px;
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-top: 1px solid #292e39;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🤖 RAG Assistant"
    )

    st.caption(
        "Your document-aware AI assistant"
    )

    st.divider()


    # New chat
    if st.button(
        "＋  New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    st.markdown("### 📚 Knowledge Base")

    st.markdown(
        """
        <div class="info-card">
        📄 <b>Document RAG</b><br>
        Ask questions from your uploaded knowledge.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### ⚙️ System")

    st.markdown(
        """
        <div class="status">
        ● RAG System Online
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 20px;
            color: #6b7280;
            font-size: 12px;
        ">
        RAG Assistant • LangChain
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="app-title">🤖 RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Ask questions and get answers from your documents.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-box">

        ### 👋 Welcome

        I can answer questions using the information
        available in your document knowledge base.

        <br>

        **Try asking:**

        - What is RAG?
        - How does vector search work?
        - Explain embeddings.
        - What does the document say about ...?

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Message your RAG Assistant..."
)


if question:

    # -------------------------
    # User message
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------
    # Assistant
    # -------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""


        try:

            response = requests.post(
                f"{API_URL}/ask/stream",
                json={
                    "question": question
                },
                stream=True,
                timeout=120
            )


            response.raise_for_status()


            for chunk in response.iter_content(
                chunk_size=None,
                decode_unicode=True
            ):

                if chunk:

                    full_response += chunk

                    response_placeholder.markdown(
                        full_response + "▌"
                    )


            response_placeholder.markdown(
                full_response
            )


            # Save response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )


        except requests.RequestException as e:

            error_message = (
                "⚠️ Unable to connect to the RAG server."
            )

            response_placeholder.error(
                error_message
            )