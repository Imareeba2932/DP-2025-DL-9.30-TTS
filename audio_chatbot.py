
import streamlit as st
from groq import Groq
from streamlit_mic_recorder import speech_to_text
import time

# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Multi-Domain Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Domain AI Chatbot")
st.caption("Powered by Groq + Llama | Voice & Text Assistant")

# -----------------------------------
# 2. GROQ CONFIGURATION
# -----------------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

MODEL_NAME = "llama-3.3-70b-versatile"

# -----------------------------------
# 3. DOMAIN OPTIONS
# -----------------------------------

domains = {

    "Education": """
    You are an educational assistant.
    ONLY answer questions related to education,
    Python, programming, technology, study tips,
    and learning methods.
    """,

    "Medical": """
    You are a healthcare information assistant.
    ONLY answer questions related to healthcare,
    common diseases, treatments and healthy lifestyle.
    Provide general information, not a medical diagnosis.
    Encourage consulting a qualified doctor when needed.
    """,

    "Travel": """
    You are a travel assistant.
    ONLY answer travel-related questions,
    destinations, tourism, food, packing and travel tips.
    """,

    "Fitness Coach": """
    You are a professional fitness coach.
    ONLY answer questions related to workouts,
    diet, bodybuilding, fat loss, yoga and fitness motivation.
    """
}

# -----------------------------------
# 4. DOMAIN SELECTOR
# -----------------------------------

selected_domain = st.selectbox(
    "Choose your chatbot domain:",
    list(domains.keys())
)

# -----------------------------------
# 5. SPEECH LANGUAGE
# -----------------------------------

speech_language = st.selectbox(
    "Select your speaking language:",
    ["en-IN", "hi-IN"]
)

# -----------------------------------
# 6. SESSION STATE
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

# -----------------------------------
# 7. CHAT HISTORY
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------
# 8. SPEECH TO TEXT
# -----------------------------------

st.markdown("### 🎤 Voice Input")

spoken_text = speech_to_text(
    language=speech_language,
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    key="speech_input"
)

if spoken_text:

    st.session_state.message_input = spoken_text

    st.success("Speech converted into text!")

# -----------------------------------
# 9. USER INPUT
# -----------------------------------

with st.form("chat_form", clear_on_submit=True):

    user_input = st.text_input(
        "Your message:",
        placeholder="Type or speak your question...",
        key="message_input"
    )

    send_button = st.form_submit_button(
        "🚀 Send Message",
        use_container_width=True
    )

# -----------------------------------
# 10. GROQ RESPONSE GENERATION
# -----------------------------------

if send_button and user_input.strip():

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("🤖 Groq is thinking..."):

        system_prompt = f"""
        {domains[selected_domain]}

        If the user asks anything unrelated to
        {selected_domain}, politely decline by saying:

        Sorry, I can only answer
        {selected_domain}-related questions.

        Respond in the same language as the user's question.
        """

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],

                temperature=0.7,
                max_tokens=1024
            )

            bot_reply = response.choices[0].message.content

        except Exception as e:

            bot_reply = f"Error: {str(e)}"

    # -----------------------------------
    # 11. DISPLAY RESPONSE
    # -----------------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()
        reply_text = ""

        for char in bot_reply:

            reply_text += char

            placeholder.markdown(reply_text)

            time.sleep(0.01)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()

# -----------------------------------
# 12. CLEAR CHAT
# -----------------------------------

if st.session_state.messages:

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.rerun()
