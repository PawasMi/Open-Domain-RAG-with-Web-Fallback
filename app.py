```python
import joblib
import streamlit as st
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from ddgs import DDGS


# ==========================================================
# STREAMLIT CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_history" not in st.session_state:
    st.session_state.show_history = False


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🎓 AI Teaching Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your lecture transcript using Hybrid RAG + Groq'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style='text-align:center; color:#2563EB;'>
            🎓 AI Assistant
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💬 Conversation")

    # ------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------

    if st.button(
        "💬 Chat History",
        use_container_width=True
    ):

        st.session_state.show_history = (
            not st.session_state.show_history
        )

    # ------------------------------------------------------
    # DISPLAY HISTORY
    # ------------------------------------------------------

    if st.session_state.show_history:

        if len(st.session_state.messages) == 0:

            st.info("No chat history available.")

        else:

            st.markdown("### Previous Chats")

            for msg in st.session_state.messages:

                if msg["role"] == "user":

                    st.markdown(
                        f"**👤 You:** {msg['content']}"
                    )

                else:

                    st.markdown(
                        f"**🤖 AI:** {msg['content']}"
                    )

                st.divider()

    # ------------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------------

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages.clear()

        st.rerun()

    st.markdown("---")

    st.caption(
        "💡 Ask anything from your lecture.\n\n"
        "If the answer is not found in the transcript, "
        "the assistant automatically searches the web."
    )


# ==========================================================
# GROQ CLIENT
# ==========================================================

if "GROQ_API_KEY" not in st.secrets:

    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it in Streamlit Cloud Secrets."
    )

    st.stop()


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ==========================================================
# LOAD EMBEDDING MODEL + DATA
# ==========================================================

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    dataframe = joblib.load(
        "data/embeddings.joblib"
    )

    return embedding_model, dataframe


embedding_model, df = load_resources()


# ==========================================================
# DATA
# ==========================================================

chunks = df["text"].tolist()

embeddings = np.vstack(
    df["embedding"].values
)


# ==========================================================
# WEB SEARCH
# ==========================================================

def web_search(query):

    try:

        with DDGS() as ddgs:

            results = list(
                ddgs.text(
                    query,
                    max_results=5
                )
            )

        if not results:

            return "No external information found."

        formatted_results = []

        for result in results:

            title = result.get(
                "title",
                ""
            )

            body = result.get(
                "body",
                ""
            )

            href = result.get(
                "href",
                ""
            )

            formatted_results.append(
                f"Title: {title}\n"
                f"Content: {body}\n"
                f"Source: {href}"
            )

        return "\n\n".join(
            formatted_results
        )[:4000]

    except Exception:

        return "No external information found."


# ==========================================================
# HYBRID RAG RETRIEVAL
# ==========================================================

def retrieve_context(question):

    # ------------------------------------------------------
    # QUESTION EMBEDDING
    # ------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    # ------------------------------------------------------
    # COSINE SIMILARITY
    # ------------------------------------------------------

    similarity_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    # ------------------------------------------------------
    # TOP 3 CHUNKS
    # ------------------------------------------------------

    top_indices = np.argsort(
        similarity_scores
    )[-3:][::-1]

    top_score = similarity_scores[
        top_indices[0]
    ]

    # ------------------------------------------------------
    # LECTURE CONTEXT
    # ------------------------------------------------------

    lecture_context = "\n\n".join(

        df.iloc[i]["text"]

        for i in top_indices

    )

    lecture_chunks = df.iloc[
        top_indices
    ][
        [
            "chunk_id",
            "start",
            "end",
            "text"
        ]
    ]

    # ------------------------------------------------------
    # SIMILARITY THRESHOLD
    # ------------------------------------------------------

    SIMILARITY_THRESHOLD = 0.45

    # ------------------------------------------------------
    # LECTURE MATCH
    # ------------------------------------------------------

    if top_score >= SIMILARITY_THRESHOLD:

        return {
            "context": lecture_context,
            "source": "lecture",
            "score": float(top_score),
            "chunks": lecture_chunks
        }

    # ------------------------------------------------------
    # WEB FALLBACK
    # ------------------------------------------------------

    web_context = web_search(question)

    return {
        "context": web_context,
        "source": "web",
        "score": float(top_score),
        "chunks": lecture_chunks
    }


# ==========================================================
# SOURCE BADGE
# ==========================================================

def source_badge(source):

    if source == "lecture":

        return "📚 Lecture Notes"

    return "🌐 External Knowledge"


# ==========================================================
# SIMILARITY LABEL
# ==========================================================

def similarity_color(score):

    if score >= 0.75:

        return "🟢 Excellent Match"

    elif score >= 0.60:

        return "🟡 Good Match"

    elif score >= 0.45:

        return "🟠 Weak Match"

    else:

        return "🔴 Using Web Search"


# ==========================================================
# PROMPT
# ==========================================================

def build_prompt(
    question,
    context,
    source
):

    if source == "lecture":

        system_prompt = f"""
You are an AI Teaching Assistant.

Answer ONLY from the lecture context below.

If the lecture contains the answer,
explain it clearly in simple language.

Do NOT invent information.

Lecture Context:

{context}
"""

    else:

        system_prompt = f"""
You are an AI Teaching Assistant.

The lecture does not contain the answer.

Use the following external information
to answer accurately.

External Context:

{context}

Provide a student-friendly explanation.

Do not invent facts that are not supported
by the external context.
"""

    return system_prompt


# ==========================================================
# GROQ GENERATION
# ==========================================================

def generate_answer(
    question,
    context,
    source
):

    prompt = build_prompt(
        question,
        context,
        source
    )

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.2,

        max_tokens=700,

        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[
        0
    ].message.content.strip()


# ==========================================================
# ASK AI
# ==========================================================

def ask_ai(question):

    retrieval = retrieve_context(
        question
    )

    context = retrieval["context"]

    source = retrieval["source"]

    score = retrieval["score"]

    chunks = retrieval["chunks"]

    answer = generate_answer(
        question,
        context,
        source
    )

    return {
        "answer": answer,
        "context": context,
        "source": source,
        "score": score,
        "chunks": chunks
    }


# ==========================================================
# DISPLAY PREVIOUS CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================================
# CHAT INPUT
# ==========================================================

question = st.chat_input(
    "💬 Ask anything..."
)


# ==========================================================
# PROCESS QUESTION
# ==========================================================

if question:

    # ------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ------------------------------------------------------
    # AI RESPONSE
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🔍 Thinking..."):

            result = ask_ai(question)

            answer = result["answer"]

            source = result["source"]

            score = result["score"]

            retrieved_chunks = result["chunks"]

        # --------------------------------------------------
        # ANSWER
        # --------------------------------------------------

        st.markdown(answer)

        # --------------------------------------------------
        # SAVE ANSWER
        # --------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # --------------------------------------------------
        # SOURCE
        # --------------------------------------------------

        if source == "lecture":

            st.success(
                "📚 Answer generated from Lecture"
            )

        else:

            st.info(
                "🌐 Answer generated using External Knowledge"
            )

        # --------------------------------------------------
        # SIMILARITY
        # --------------------------------------------------

        st.progress(
            min(
                max(score, 0.0),
                1.0
            )
        )

        st.caption(
            f"Match Similarity: "
            f"{score * 100:.1f}%"
        )

        st.caption(
            similarity_color(score)
        )

        # --------------------------------------------------
        # TRANSCRIPT CHUNKS
        # --------------------------------------------------

        if source == "lecture":

            with st.expander(
                "📚 Relevant Transcript Chunks"
            ):

                for _, row in retrieved_chunks.iterrows():

                    st.markdown(
                        f"""
### Chunk {row['chunk_id']}

**Time**

{row['start']} sec → {row['end']} sec

**Transcript**

{row['text']}
"""
                    )

                    st.divider()

        # --------------------------------------------------
        # WEB SEARCH INFO
        # --------------------------------------------------

        else:

            with st.expander(
                "🌐 Search Information"
            ):

                st.write(
                    "The lecture did not contain "
                    "sufficient information."
                )

                st.write(
                    "The assistant used external "
                    "knowledge to answer."
                )
```

