import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/chat")


st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide",
)


def render_agent_trace(agent_trace):
    if not agent_trace:
        return

    st.subheader("Agent Trace")

    router = agent_trace.get("router_agent", {})
    with st.expander("Router Agent", expanded=True):
        st.write(f"Route: {router.get('route')}")
        st.write(f"Expected category: {router.get('expected_category')}")
        st.write(f"Document keyword: {router.get('document_keyword')}")

    patient_context = agent_trace.get("patient_context_agent", {})
    with st.expander("Patient Context Agent", expanded=True):
        st.write(f"Has patient context: {patient_context.get('has_patient_context')}")
        st.write(
            f"Applies to patient context: "
            f"{patient_context.get('applies_to_patient_context')}"
        )
        st.write(f"Subject ID: {patient_context.get('subject_id')}")
        st.write(f"HADM ID: {patient_context.get('hadm_id')}")

        warning = patient_context.get("warning")
        if warning:
            st.warning(warning)

    retrieval = agent_trace.get("retrieval_agent", {})
    with st.expander("Retrieval Agent", expanded=True):
        st.write(f"Retrieved documents: {retrieval.get('retrieved_documents')}")
        st.write(f"Retrieved categories: {retrieval.get('retrieved_categories')}")

    treatment_progress = agent_trace.get("treatment_progress_agent", {})
    with st.expander("Treatment Progress Agent", expanded=True):
        st.write(f"Progress category: {treatment_progress.get('progress_category')}")
        description = treatment_progress.get("description")
        if description:
            st.write(description)

    safety_triage = agent_trace.get("safety_triage_agent", {})
    if safety_triage:
        safety_level = safety_triage.get("safety_level")

        with st.expander("Safety / Triage Agent", expanded=True):
            if safety_level == "high":
                st.error(f"Safety level: {safety_level}")
            elif safety_level == "medium":
                st.warning(f"Safety level: {safety_level}")
            else:
                st.success(f"Safety level: {safety_level}")

            st.write(f"Safety reason: {safety_triage.get('safety_reason')}")
            st.write(f"Recommended action: {safety_triage.get('recommended_action')}")

            description = safety_triage.get("description")
            if description:
                st.write(description)

    generation = agent_trace.get("generation_agent", {})
    with st.expander("Generation Agent", expanded=True):
        st.write(f"Answer generated: {generation.get('answer_generated')}")

    verification = agent_trace.get("verification_agent", {})
    with st.expander("Verification Agent", expanded=True):
        verified = verification.get("verified")

        if verified:
            st.success(f"Verified: {verified}")
        else:
            st.error(f"Verified: {verified}")

        warning = verification.get("warning")
        if warning:
            st.warning(warning)


def render_sources(sources):
    if not sources:
        return

    st.subheader("Retrieved medical sources")

    for index, source in enumerate(sources, start=1):
        with st.expander(f"Source {index}", expanded=True):
            st.write(f"Document Source: {source.get('document_source')}")
            st.write(f"File: {source.get('document_url')}")
            st.write(f"Category: {source.get('category')}")
            st.write(source.get("content", ""))


def send_chat_request(query, subject_id=None, hadm_id=None):
    payload = {
        "query": query,
        "session_id": "streamlit-ui",
    }

    if subject_id:
        payload["subject_id"] = subject_id

    if hadm_id:
        payload["hadm_id"] = hadm_id

    response = requests.post(API_URL, json=payload, timeout=300)
    response.raise_for_status()

    return response.json()


st.title("Medical RAG Chatbot")
st.write(
    "Ask questions about pediatric oncology knowledge, patient timeline records, "
    "or clinical QA examples."
)

with st.sidebar:
    st.header("Patient Context")

    st.write("Use these fields only for EHR clinical QA questions.")

    subject_id = st.text_input(
        "Subject ID",
        value="",
        placeholder="Example: 10010655",
    )

    hadm_id = st.text_input(
        "HADM ID",
        value="",
        placeholder="Example: 20421864",
    )

    st.divider()

    st.write("For synthetic timeline questions, use patient IDs like:")

    st.code("PONC-001")

    st.write("Example:")

    st.code("What is the risk level of patient PONC-001 on day 21?")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


show_technical_details = st.checkbox(
    "Show technical details",
    value=False,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and show_technical_details:
            render_sources(message.get("sources", []))
            render_agent_trace(message.get("agent_trace", {}))


user_query = st.chat_input("Ask a medical question...")

if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving medical context and running agents..."):
            try:
                result = send_chat_request(
                    query=user_query,
                    subject_id=subject_id.strip() or None,
                    hadm_id=hadm_id.strip() or None,
                )

                answer = result.get("message", "")

                st.markdown(answer)

                sources = result.get("sources", [])
                agent_trace = result.get("agent_trace", {})

                if show_technical_details:
                    render_sources(sources)
                    render_agent_trace(agent_trace)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "agent_trace": agent_trace,
                    }
                )

            except requests.exceptions.ConnectionError:
                error_message = (
                    "Could not connect to the FastAPI backend. "
                    "Please make sure the backend is running on port 8000."
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "agent_trace": {},
                    }
                )

            except Exception as e:
                error_message = f"Error: {e}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "agent_trace": {},
                    }
                )