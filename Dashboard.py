import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Interactive Dashboard for Adults(65+)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# CUSTOM CSS FOR SENIOR ACCESSIBILITY
# ============================================================================
st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 20px;
            background-color: #F5F5F5;
        }
        
        /* Headers - Large and clear */
        h1 {
            font-size: 48px !important;
            font-weight: 700 !important;
            color: #1B5E20 !important;
            margin-bottom: 10px !important;
        }
        
        h2 {
            font-size: 32px !important;
            font-weight: 700 !important;
            color: #1B5E20 !important;
            margin-top: 30px !important;
            margin-bottom: 15px !important;
            padding-bottom: 12px !important;
            border-bottom: 3px solid #E8F5E9 !important;
        }
        
        h3 {
            font-size: 24px !important;
            font-weight: 700 !important;
            color: #1B5E20 !important;
        }
        
        /* Body text - Large and readable */
        body, p, div, span, li {
            font-size: 18px !important;
            font-family: "Segoe UI", sans-serif !important;
            line-height: 1.8 !important;
            color: #333 !important;
        }
        
        /* High contrast backgrounds */
        .stMetric {
            background-color: #FAFAFA !important;
            border-left: 5px solid #2E7D32 !important;
            padding: 25px !important;
            border-radius: 8px !important;
        }
        
        /* Tab buttons - Large and accessible */
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 18px !important;
            padding: 16px 28px !important;
            min-width: 180px !important;
            font-weight: 600 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #2E7D32 !important;
            color: white !important;
        }
        
        /* Insight boxes */
        .insight-box {
            background-color: #E8F5E9 !important;
            border: 2px solid #A5D6A7 !important;
            border-radius: 8px !important;
            padding: 25px !important;
            margin: 20px 0 !important;
        }
        
        /* Capability boxes */
        .capability-box {
            background-color: #F0F7F0 !important;
            border: 2px solid #A5D6A7 !important;
            border-radius: 8px !important;
            padding: 25px !important;
            text-align: center !important;
        }
        
        /* Summary box */
        .summary-box {
            background-color: #E3F2FD !important;
            border: 3px solid #2E7D32 !important;
            border-radius: 8px !important;
            padding: 30px !important;
            text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and prepare the market basket dataset"""

    BillNo = [...]
    Itemname = [...]
    Quantity = [...]
    Price = [...]
    CustomerID = [...]

    # --- FIX: ensure all lists are equal length ---
    assert len(BillNo) == len(Itemname) == len(Quantity) == len(Price) == len(CustomerID), \
        f"Length mismatch: {len(BillNo)}, {len(Itemname)}, {len(Quantity)}, {len(Price)}, {len(CustomerID)}"

    data = {
        "BillNo": BillNo,
        "Itemname": Itemname,
        "Quantity": Quantity,
        "Price": Price,
        "CustomerID": CustomerID
    }

    return pd.DataFrame(data)
