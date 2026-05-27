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


# ============================================================================
# DATA - Market Basket Dataset
# ============================================================================
@st.cache_data
def load_data():
    """Load and prepare the market basket dataset"""
    data = {
        'BillNo': [1000, 1000, 1000, 1000, 1004, 1005, 1005, 1005, 1008, 1008, 1008, 1011, 1012, 1013, 1013, 1013, 1013, 1013, 1013, 1013, 1021, 1021, 1021, 1024, 1024, 1024, 1027, 1027, 1027, 1027, 1027, 1027, 1033, 1033, 1035, 1035, 1035, 1038, 1038, 1038, 1038, 1042, 1042, 1044, 1044, 1044, 1044, 1044, 1049, 1049, 1049, 1049, 1049, 1054, 1054, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1063, 1063, 1063, 1066, 1066, 1066, 1066, 1066, 1066, 1066, 1073, 1073, 1075, 1076, 1076, 1076, 1076, 1076, 1076, 1082, 1082, 1082, 1082, 1086, 1086, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088],
        
        'Itemname': ['Apples', 'Butter', 'Eggs', 'Potatoes', 'Oranges', 'Milk', 'Onions', 'Cereal', 'Tomatoes', 'Potatoes', 'Cereal', 'Bananas', 'Tomatoes', 'Pasta', 'Onions', 'Bread', 'Bananas', 'Coffee', 'Sugar', 'Potatoes', 'Oranges', 'Bananas', 'Potatoes', 'Chicken', 'Cereal', 'Bananas', 'Cheese', 'Pasta', 'Cereal', 'Onions', 'Bananas', 'Chicken', 'Sugar', 'Eggs', 'Onions', 'Cereal', 'Cheese', 'Cereal', 'Coffee', 'Bread', 'Onions', 'Chicken', 'Pasta', 'Eggs', 'Butter', 'Bananas', 'Chicken', 'Tomatoes', 'Tea', 'Bananas', 'Pasta', 'Eggs', 'Cereal', 'Sugar', 'Coffee', 'Apples', 'Tomatoes', 'Chicken', 'Pasta', 'Bread', 'Sugar', 'Potatoes', 'Yogurt', 'Butter', 'Bananas', 'Milk', 'Eggs', 'Sugar', 'Juice', 'Tomatoes', 'Butter', 'Onions', 'Coffee', 'Bananas', 'Onions', 'Coffee', 'Tea', 'Bananas', 'Onions', 'Juice', 'Pasta', 'Tea', 'Juice', 'Cereal', 'Butter', 'Yogurt', 'Juice', 'Pasta', 'Onions', 'Chicken', 'Tea', 'Juice', 'Potatoes', 'Sugar', 'Coffee', 'Milk', 'Tomatoes', 'Apples'],
        
        'Quantity': [5, 4, 4, 4, 2, 3, 3, 2, 5, 4, 3, 3, 1, 5, 1, 4, 2, 4, 5, 2, 3, 2, 4, 3, 1, 2, 5, 1, 5, 4, 3, 4, 4, 3, 4, 2, 2, 1, 2, 5, 1, 4, 1, 2, 5, 3, 1, 3, 5, 4, 3, 3, 5, 1, 5, 4, 3, 4, 1, 5, 1, 1, 3, 5, 2, 5, 2, 4, 1, 2, 3, 3, 5, 2, 1, 3, 4, 1, 4, 1, 5, 4, 1, 2, 4, 4, 4, 1, 4, 2, 4, 1, 1, 5, 3, 2],
        
        'Price': [8.3, 6.06, 2.66, 8.1, 7.26, 5.29, 8.75, 6.91, 2.62, 8.4, 8.68, 7.67, 8.45, 7.35, 7.97, 4.94, 3.4, 3.0, 4.93, 7.12, 6.47, 3.85, 2.58, 9.66, 9.7, 9.94, 9.26, 7.66, 3.98, 6.85, 8.84, 8.84, 9.11, 9.14, 8.41, 9.48, 7.82, 6.13, 1.96, 2.63, 8.41, 2.25, 9.13, 9.62, 5.32, 9.79, 5.39, 8.15, 4.91, 8.33, 2.55, 1.37, 3.69, 3.9, 9.93, 1.41, 4.78, 9.76, 2.04, 4.66, 2.81, 4.43, 7.41, 3.35, 2.67, 7.92, 4.5, 3.45, 4.14, 4.21, 8.92, 4.99, 1.73, 9.72, 2.63, 8.04, 1.21, 2.59, 2.93, 8.6, 6.16, 4.78, 3.67, 4.88, 1.04, 4.08, 1.9, 3.22, 7.43, 5.79, 5.76, 6.98, 7.5, 4.6, 3.63, 1.48, 5.24, 8.96],
        
        'CustomerID': [
            52299, 11752, 16415, 22889, 52255, 54358, 73266, 46399, 22590, 47387, 26046, 20072, 79964, 83712, 68417, 25656, 13413, 71236, 91572, 59237, 32140, 63813, 30570, 12759, 29871, 33260, 20321, 99122, 49195, 72279, 11113, 71938, 72262, 68313, 64324, 84568, 53719, 32332, 42831, 63032, 15876, 37152, 12550, 17772, 35384, 40946, 54206, 34679, 26139, 76666, 63008, 26307, 78522, 86475, 16469, 42281, 42602, 46419, 10504, 50407, 45015, 77640, 50919, 73789, 33668, 45531, 38643, 85960, 23412, 30936, 71628, 26627, 46419, 34870, 84010, 65291, 43275, 74271, 93698, 94737, 51682, 38405, 62431, 52194, 53527, 53486, 74277, 55421, 41923, 58613, 85393, 73544, 38536, 49110, 43704, 56067, 60271, 98942,
            99999  # <-- added missing value to match list lengths
        ]
    }

    return pd.DataFrame(data)

# Load data
df = load_data()
