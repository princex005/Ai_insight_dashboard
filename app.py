# -------------------------------
# Modes: Prompt mode (old) or Chat mode (new)
# -------------------------------
mode = st.radio("Choose mode", ["Prompt mode", "Chat mode"], horizontal=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role":"user"/"assistant", "text": str}

user_messages = []  # will collect prompts to turn into charts

if mode == "Prompt mode":
    prompt = st.text_input(
        "Describe what you want to see (e.g., 'bar average age by gender' or "
        "'pie percentage by Investment_Avenues')"
    )
    if prompt:
        user_messages = [prompt]  # existing flow supports ';' too (kept below)

else:  # Chat mode
    st.subheader("Ask DataLens AI 💬")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            st.chat_message("assistant").write(msg["text"])

    chat_in = st.chat_input("Ask a question like 'forecast sales' or 'pie share by category'...")
    if chat_in:
        st.session_state.chat_history.append({"role": "user", "text": chat_in})
        user_messages = [chat_in]  # we’ll convert this to charts below
