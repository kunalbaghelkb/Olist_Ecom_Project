import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from deep_translator import GoogleTranslator
from src.preprocess import clean_text

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Olist Enterprise Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. GLOBAL CSS
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #1F2937 !important;
        line-height: 1.6;
    }
    
    /* Force dark text on all elements */
    p, span, div, label {
        color: #1F2937 !important;
    }

    /* Main Background */
    .stApp {
        background-color: #F5F7F9;
    }
    
    /* Markdown text */
    .stMarkdown p {
        color: #1F2937 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #E0E0E0 !important;
    }

    /* Sidebar - Professional Navy Blue */
    [data-testid="stSidebar"] {
        background-color: #0F2942;
    }
    [data-testid="stSidebar"] * {
        color: #E0E0E0 !important;
    }

    /* KPI Cards (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        color: #0F2942 !important;
        font-weight: 700;
        font-size: 1.8rem;
    }

    /* Headers - Better Spacing */
    h1 {
        color: #0F2942;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2 {
        color: #0F2942;
        font-weight: 600;
    }
    h3 {
        color: #0F2942;
        font-weight: 600;
    }

    /* Subheaders */
    .stMarkdown h3 {
        padding-bottom: 10px;
        border-bottom: 2px solid #E5E7EB;
    }

    /* Buttons - Solid Corporate Blue */
    div.stButton > button {
        background-color: #0056B3;
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        width: 100%;
        transition: background-color 0.2s;
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        background-color: #004494;
        color: white !important;
        border: none;
        box-shadow: none;
    }
    div.stButton > button p {
        color: white !important;
    }

    /* Input Fields - Better Spacing */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea, 
    .stNumberInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        color: #1F2937 !important;
        border-radius: 4px;
        padding: 0.6rem;
    }
    
    /* Placeholder text */
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #1F2937 !important;
        opacity: 0.6;
    }

    .stTextArea > label, 
    .stTextInput > label,
    .stNumberInput > label {
        font-weight: 500 !important;
        color: #1F2937 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Result Cards - Enhanced */
    .result-card {
        background-color: white;
        padding: 7px 30px;
        border-radius: 6px;
        border-left: 5px solid #0056B3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    .result-card h3 {
        margin: 0 0 15px 0 !important;
        font-size: 1.5rem;
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }

    .result-card p {
        margin: 8px 0;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Info Box Styling */
    .stAlert {
        padding: 1rem 1.2rem;
    }

    /* Section Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #E5E7EB;
    }

    /* Column Spacing */
    [data-testid="column"] {
        padding: 0 1rem;
    }

    /* Chart Containers */
    .js-plotly-plot {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Spinner Text */
    .stSpinner > div {
        font-size: 1rem;
        color: #1F2937 !important;
    }
    
    /* Alert/Info boxes text */
    .stAlert p {
        color: #1F2937 !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #1F2937 !important;
    }
    
    /* Caption text */
    .stCaptionContainer p {
        color: #6B7280 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. MODEL LOADING LOGIC
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, 'models')
    
    nlp_model = joblib.load(os.path.join(base_path, 'sentiment_model.pkl'))
    vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.pkl'))
    
    kmeans_model = joblib.load(os.path.join(base_path, 'customer_segmentation.pkl'))
    scaler_model = joblib.load(os.path.join(base_path, 'rfm_scaler.pkl'))
    
    return nlp_model, vectorizer, kmeans_model, scaler_model

try:
    nlp_model, tfidf, kmeans, scaler = load_models()
except FileNotFoundError:
    st.error("System Error: Model files not found in 'models/' directory.")
    st.stop()

# 4. SIDEBAR NAVIGATION
st.sidebar.title("Olist Analytics")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Module", ["Executive Dashboard", "Sentiment Analysis", "Customer Segmentation"])
st.sidebar.markdown("---")
st.sidebar.caption("v1.1.0 | Enterprise Edition")

# 5. PAGE: EXECUTIVE DASHBOARD
if app_mode == 'Executive Dashboard':
    st.title("Executive Overview")
    st.markdown("Supply Chain & Customer Experience Metrics")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", "$1.2M", "+12%")
    with col2:
        st.metric("Active Customers", "4,250", "+5%")
    with col3:
        st.metric("Avg. Sentiment", "4.2/5.0", "+0.3")
    with col4:
        st.metric("On-Time Delivery", "94.5%", "-1.2%")

    st.markdown("### Performance Trends")
    
    # Dummy Data for Visualization
    chart_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='ME'),
        'Revenue': [100, 120, 110, 130, 140, 135, 150, 160, 170, 165, 180, 190],
        'Satisfaction': [80, 82, 81, 85, 84, 86, 88, 87, 89, 90, 91, 92]
    })

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_data['Date'], y=chart_data['Revenue'], name='Revenue (k)', marker_color='#0F2942'))
    fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['Satisfaction'], name='CSAT Score', yaxis='y2', line=dict(color='#0056B3', width=3)))
    
    fig.update_layout(
        title="Revenue vs Customer Satisfaction",
        template="plotly_white",
        yaxis=dict(title="Revenue (Thousands USD)"),
        yaxis2=dict(title="CSAT Index", overlaying='y', side='right'),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# 6. PAGE: NLP SENTIMENT ANALYSIS (Improved Spacing)
elif app_mode == "Sentiment Analysis":
    st.title("Review Sentiment Classifier")
    st.markdown("Natural Language Processing (NLP) Engine")
    st.subheader("Input Data")
    
    user_text = st.text_area(
        "Customer Review Text", 
        height=220, 
        placeholder="Enter review content here...",
        help="Enter customer feedback in any language"
    )
    
    if st.button("Analyze Text"):
        if user_text:
            try:
                # Translation
                with st.spinner("Processing your review..."):
                    translated_text = GoogleTranslator(source='auto', target='pt').translate(user_text)
                    cleaned_text = clean_text(translated_text)
                    
                    # Prediction
                    vec_text = tfidf.transform([cleaned_text])
                    pred = nlp_model.predict(vec_text)
                    prob = nlp_model.predict_proba(vec_text)
                    
                    # Logic
                    sentiment_score = prob[0][1] * 100
                    is_positive = pred[0] == 1
                    confidence = np.max(prob) * 100
                    
                    # Display Result in Card
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 5px solid {'#28a745' if is_positive else '#dc3545'};">
                        <h3 style="color: {'#28a745' if is_positive else '#dc3545'};">
                            Classification: {'POSITIVE' if is_positive else 'NEGATIVE'}
                        </h3>
                        <p style="color: #6B7280; font-size: 1.1rem; margin-top: 12px;">
                            <strong>Confidence Score:</strong> {confidence:.2f}%
                        </p>
                        <p style="color: #6B7280; margin-top: 8px;">
                            <strong>Sentiment Strength:</strong> {'High' if confidence > 80 else 'Moderate' if confidence > 60 else 'Low'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"**Processing Error:** {str(e)}")
                st.caption("Please check your input and try again.")
        else:
            st.warning("⚠️ Please enter valid text input to analyze.")


# 7. PAGE: CUSTOMER SEGMENTATION
elif app_mode == "Customer Segmentation":
    st.title("Customer Segmentation Engine")
    st.markdown("RFM (Recency, Frequency, Monetary) Clustering Model")

    st.markdown("### Enter Customer Metrics")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        recency = st.number_input(
            "Recency (Days since last order)", 
            min_value=0, 
            value=30,
            help="Number of days since the customer's last purchase"
        )
    with col2:
        frequency = st.number_input(
            "Frequency (Total Orders)", 
            min_value=1, 
            value=5,
            help="Total number of orders placed by the customer"
        )
    with col3:
        monetary = st.number_input(
            "Monetary Value (Total Spent $)", 
            min_value=0.0, 
            value=500.0,
            help="Total amount spent by the customer"
        )
    
    if st.button("Generate Segment"):
        # Prediction Logic
        user_data = np.array([[recency, frequency, monetary]])
        scaled_data = scaler.transform(user_data)
        cluster_id = kmeans.predict(scaled_data)[0]
        
        # Segment Mapping
        cluster_map = {
            0: {'name': 'Active Customer', 'desc': 'Consistent buyer with standard engagement levels.', 'color': '#17a2b8'},
            1: {'name': 'Dormant/Lost', 'desc': 'High recency, low frequency. Requires reactivation campaign.', 'color': '#dc3545'},
            2: {'name': 'Loyalist', 'desc': 'Frequent buyer with solid purchase history.', 'color': '#28a745'},
            3: {'name': 'Whale (VIP)', 'desc': 'High monetary value customer. Premium support required.', 'color': '#ffc107'}
        }
        
        segment = cluster_map.get(cluster_id)
        
        # 1. Result Display
        st.markdown(f"""
        <div class="result-card" style="border-left: 5px solid {segment['color']};">
            <h2 style="margin:0; color: {segment['color']}; font-size: 2rem;">
                Segment: {segment['name'].upper()}
            </h2>
            <p style="font-size: 1.15rem; margin-top: 15px; color: #4B5563; line-height: 1.7;">
                {segment['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        # 2. Strategic Recommendation
        st.subheader("Recommended Strategy")
        
        if cluster_id == 1:
            st.warning("**Action Required:** Initiate Win-back Campaign via Email and Promotional Offers.")
        elif cluster_id == 3:
            st.success("**Action Required:** Assign Dedicated Account Manager & Offer Exclusive Loyalty Perks.")
        elif cluster_id == 2:
            st.success("**Action Required:** Maintain Engagement with Loyalty Program.")
        else:
            st.info("**Action Required:** Standard Retention Flow and Regular Communication.")

        # 3. Radar Chart (Visualization)
        st.subheader("Behavioral Profile")
        
        # Normalization for visualization
        norm_r = min(100, (1 - (recency/365)) * 100)
        norm_f = min(100, (frequency/20) * 100)
        norm_m = min(100, (monetary/2000) * 100)
        
        categories = ['Recency Score', 'Frequency Score', 'Monetary Score']
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[norm_r, norm_f, norm_m],
            theta=categories,
            fill='toself',
            name=segment['name'],
            line_color=segment['color'],
            fillcolor=segment['color'],
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100],
                    showticklabels=True,
                    ticks='outside'
                )
            ),
            showlegend=False,
            template="plotly_white",
            title="Normalized RFM Profile",
            height=450,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)