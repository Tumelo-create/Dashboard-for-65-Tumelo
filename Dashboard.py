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