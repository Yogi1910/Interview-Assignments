import streamlit as st
from classifier_openai import classify_message

# Page config
st.set_page_config(
    page_title="Patient Message Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for white theme
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stTextArea > div > div > textarea {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        color: #1a1a1a;
    }
    .stButton > button {
        background-color: #0066cc;
        color: black;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    h1 {
        color: #1a1a1a;
        font-weight: 600;
    }
    h2, h3 {
        color: #333333;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Main container
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Title
    st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>Patient Message Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Automatically route patient messages to the appropriate department</p>", unsafe_allow_html=True)
    
    # Input section
    st.markdown("### Enter Patient Message")
    
    # Example selector
    example_messages = {
        "Select an example...": "",
        "Billing": "I need to check my bill",
        "Clinical": "I have chest pain and need to see a doctor",
        "Scheduling": "Can I reschedule my appointment for next week?",
        "Technical": "The app won't load on my phone"
    }
    
    selected_example = st.selectbox("Quick examples:", list(example_messages.keys()), label_visibility="collapsed")
    default_message = example_messages[selected_example] if selected_example != "Select an example..." else ""
    
    message = st.text_area(
        "",
        value=default_message,
        placeholder="Enter patient message here...",
        height=120,
        label_visibility="collapsed"
    )
    
    # Classify button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        classify_clicked = st.button("Classify Message", type="primary", use_container_width=True)
    
    # Results section
    if classify_clicked:
        if not message.strip():
            st.warning("Please enter a message to classify.")
        else:
            with st.spinner("Analyzing message..."):
                result = classify_message(message)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Response message
                chat_message = result.get("message", "")
                if chat_message:
                    st.markdown(f"""
                        <div class="response-box">
                            <p style="margin: 0; font-size: 1.1rem; color: #1a1a1a;">{chat_message}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Classification details
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    st.markdown("**Category**")
                    category_display = result["category"].replace("_", " ").title()
                    st.markdown(f"<p style='font-size: 1.2rem; color: #0066cc; margin-top: 0.5rem;'>{category_display}</p>", unsafe_allow_html=True)
                
                with col_b:
                    st.markdown("**Confidence**")
                    st.markdown(f"<p style='font-size: 1.2rem; color: #0066cc; margin-top: 0.5rem;'>{result.get('confidence', 'N/A').title()}</p>", unsafe_allow_html=True)
                
                with col_c:
                    st.markdown("**Status**")
                    st.markdown(f"<p style='font-size: 1.2rem; color: #28a745; margin-top: 0.5rem;'>✓ Classified</p>", unsafe_allow_html=True)
                
                # Expandable sections
                with st.expander("View Reasoning"):
                    st.write(result.get("reasoning", "No reasoning provided"))
                
                with st.expander("View Raw Response"):
                    st.code(result.get("raw_response", "N/A"), language="json")
