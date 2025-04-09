import streamlit as st
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Token limits by model
model_token_limits = {
    "GPT-3.5": 4096,
    "GPT-4 (8K)": 8192,
    "Claude 2": 100000,
    "Claude 3": 200000,
}

# Page config
st.set_page_config(
    page_title="Token Counter 🧠",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox("Choose a Model", list(model_token_limits.keys()))
    token_limit = model_token_limits[model_choice]
    st.markdown("---")
    st.caption("🔒 Powered by GPT2 tokenizer (approximation)")

# Main UI
st.title("🧮 Token Counter")
st.caption("Check how many tokens your prompt will use. Useful for LLM context windows!")

# Input text
text = st.text_area("📋 Paste your input here:", height=200, label_visibility="visible")

# Separate Enter Button
if st.button("🚀 Count Tokens"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        token_count = len(token_ids)

        st.markdown(f"### 🔢 Total Tokens: `{token_count}`")

        usage_percent = (token_count / token_limit) * 100
        st.progress(min(usage_percent / 100, 1.0))
        st.caption(f"Using **{usage_percent:.2f}%** of the `{token_limit}` token limit for `{model_choice}`.")

        with st.expander("🔍 View Tokens"):
            st.write(tokens)
else:
    st.info("👈 Paste some text above and press **Count Tokens**.")
