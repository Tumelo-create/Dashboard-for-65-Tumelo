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
        .main {
            padding: 20px;
            background-color: #F5F5F5;
        }
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
        body, p, div, span, li {
            font-size: 18px !important;
            font-family: "Segoe UI", sans-serif !important;
            line-height: 1.8 !important;
            color: #333 !important;
        }
        .stMetric {
            background-color: #FAFAFA !important;
            border-left: 5px solid #2E7D32 !important;
            padding: 25px !important;
            border-radius: 8px !important;
        }
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
        .insight-box {
            background-color: #E8F5E9 !important;
            border: 2px solid #A5D6A7 !important;
            border-radius: 8px !important;
            padding: 25px !important;
            margin: 20px 0 !important;
        }
        .capability-box {
            background-color: #F0F7F0 !important;
            border: 2px solid #A5D6A7 !important;
            border-radius: 8px !important;
            padding: 25px !important;
            text-align: center !important;
        }
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
    """Load and prepare the market basket dataset (robust to length mismatches)."""
    data = {
        'BillNo': [1000, 1000, 1000, 1000, 1004, 1005, 1005, 1005, 1008, 1008, 1008, 1011, 1012, 1013, 1013, 1013, 1013, 1013, 1013, 1013, 1021, 1021, 1021, 1024, 1024, 1024, 1027, 1027, 1027, 1027, 1027, 1027, 1033, 1033, 1035, 1035, 1035, 1038, 1038, 1038, 1038, 1042, 1042, 1044, 1044, 1044, 1044, 1044, 1049, 1049, 1049, 1049, 1049, 1054, 1054, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1063, 1063, 1063, 1066, 1066, 1066, 1066, 1066, 1066, 1066, 1073, 1073, 1075, 1076, 1076, 1076, 1076, 1076, 1076, 1082, 1082, 1082, 1082, 1086, 1086, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088],
        'Itemname': ['Apples', 'Butter', 'Eggs', 'Potatoes', 'Oranges', 'Milk', 'Onions', 'Cereal', 'Tomatoes', 'Potatoes', 'Cereal', 'Bananas', 'Tomatoes', 'Pasta', 'Onions', 'Bread', 'Bananas', 'Coffee', 'Sugar', 'Potatoes', 'Oranges', 'Bananas', 'Potatoes', 'Chicken', 'Cereal', 'Bananas', 'Cheese', 'Pasta', 'Cereal', 'Onions', 'Bananas', 'Chicken', 'Sugar', 'Eggs', 'Onions', 'Cereal', 'Cheese', 'Cereal', 'Coffee', 'Bread', 'Onions', 'Chicken', 'Pasta', 'Eggs', 'Butter', 'Bananas', 'Chicken', 'Tomatoes', 'Tea', 'Bananas', 'Pasta', 'Eggs', 'Cereal', 'Sugar', 'Coffee', 'Apples', 'Tomatoes', 'Chicken', 'Pasta', 'Bread', 'Sugar', 'Potatoes', 'Yogurt', 'Butter', 'Bananas', 'Milk', 'Eggs', 'Sugar', 'Juice', 'Tomatoes', 'Butter', 'Onions', 'Coffee', 'Bananas', 'Onions', 'Coffee', 'Tea', 'Bananas', 'Onions', 'Juice', 'Pasta', 'Tea', 'Juice', 'Cereal', 'Butter', 'Yogurt', 'Juice', 'Pasta', 'Onions', 'Chicken', 'Tea', 'Juice', 'Potatoes', 'Sugar', 'Coffee', 'Milk', 'Tomatoes', 'Apples'],
        'Quantity': [5, 4, 4, 4, 2, 3, 3, 2, 5, 4, 3, 3, 1, 5, 1, 4, 2, 4, 5, 2, 3, 2, 4, 3, 1, 2, 5, 1, 5, 4, 3, 4, 4, 3, 4, 2, 2, 1, 2, 5, 1, 4, 1, 2, 5, 3, 1, 3, 5, 4, 3, 3, 5, 1, 5, 4, 3, 4, 1, 5, 1, 1, 3, 5, 2, 5, 2, 4, 1, 2, 3, 3, 5, 2, 1, 3, 4, 1, 4, 1, 5, 4, 1, 2, 4, 4, 4, 1, 4, 2, 4, 1, 1, 5, 3, 2],
        'Price': [8.3, 6.06, 2.66, 8.1, 7.26, 5.29, 8.75, 6.91, 2.62, 8.4, 8.68, 7.67, 8.45, 7.35, 7.97, 4.94, 3.4, 3.0, 4.93, 7.12, 6.47, 3.85, 2.58, 9.66, 9.7, 9.94, 9.26, 7.66, 3.98, 6.85, 8.84, 8.84, 9.11, 9.14, 8.41, 9.48, 7.82, 6.13, 1.96, 2.63, 8.41, 2.25, 9.13, 9.62, 5.32, 9.79, 5.39, 8.15, 4.91, 8.33, 2.55, 1.37, 3.69, 3.9, 9.93, 1.41, 4.78, 9.76, 2.04, 4.66, 2.81, 4.43, 7.41, 3.35, 2.67, 7.92, 4.5, 3.45, 4.14, 4.21, 8.92, 4.99, 1.73, 9.72, 2.63, 8.04, 1.21, 2.59, 2.93, 8.6, 6.16, 4.78, 3.67, 4.88, 1.04, 4.08, 1.9, 3.22, 7.43, 5.79, 5.76, 6.98, 7.5, 4.6, 3.63, 1.48, 5.24, 8.96],
        'CustomerID': [52299, 11752, 16415, 22889, 52255, 54358, 73266, 46399, 22590, 47387, 26046, 20072, 79964, 83712, 68417, 25656, 13413, 71236, 91572, 59237, 32140, 63813, 30570, 12759, 29871, 33260, 20321, 99122, 49195, 72279, 11113, 71938, 72262, 68313, 64324, 84568, 53719, 32332, 42831, 63032, 15876, 37152, 12550, 17772, 35384, 40946, 54206, 34679, 26139, 76666, 63008, 26307, 78522, 86475, 16469, 42281, 42602, 46419, 10504, 50407, 45015, 77640, 50919, 73789, 33668, 45531, 38643, 85960, 23412, 30936, 71628, 26627, 46419, 34870, 84010, 65291, 43275, 74271, 93698, 94737, 51682, 38405, 62431, 52194, 53527, 53486, 74277, 55421, 41923, 58613, 85393, 73544, 38536, 49110, 43704, 56067, 60271, 98942, 43945]
    }

    # Make all columns the same length (robust against small mismatches)
    min_len = min(len(col) for col in data.values())
    for k in data:
        data[k] = data[k][:min_len]

    df = pd.DataFrame(data)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df

# Load data
df = load_data()

# ============================================================================
# METRICS
# ============================================================================
total_transactions = df['BillNo'].nunique()
unique_customers = df['CustomerID'].nunique()
unique_products = df['Itemname'].nunique()
avg_items_per_transaction = df.groupby('BillNo')['Itemname'].count().mean()
avg_transaction_value = df.groupby('BillNo')['Price'].sum().mean()
total_revenue = df.groupby('BillNo')['Price'].sum().sum()

product_popularity = df['Itemname'].value_counts().head(8)

basket_sizes = df.groupby('BillNo')['Itemname'].count()
small_baskets = (basket_sizes == 1).sum()
medium_baskets = (basket_sizes == 2).sum()
large_baskets = (basket_sizes >= 3).sum()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div style="background-color: white; padding: 40px; margin-bottom: 30px; border-radius: 12px; border-left: 6px solid #2E7D32; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
        <h1>📊 Market Basket Analysis</h1>
        <p style="font-size: 20px; color: #555; margin-top: 10px;">Understanding Your Online Grocery Business</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR FILTERS (INTERACTIVE)
# ============================================================================
st.sidebar.header("🔎 Filter Your Data")

product_filter = st.sidebar.multiselect(
    "Filter by Product",
    options=sorted(df["Itemname"].unique()),
)

customer_filter = st.sidebar.multiselect(
    "Filter by Customer ID",
    options=sorted(df["CustomerID"].unique()),
)

price_min, price_max = st.sidebar.slider(
    "Filter by Price",
    float(df["Price"].min()),
    float(df["Price"].max()),
    (float(df["Price"].min()), float(df["Price"].max()))
)

filtered_df = df.copy()
if product_filter:
    filtered_df = filtered_df[filtered_df["Itemname"].isin(product_filter)]
if customer_filter:
    filtered_df = filtered_df[filtered_df["CustomerID"].isin(customer_filter)]
filtered_df = filtered_df[(filtered_df["Price"] >= price_min) & (filtered_df["Price"] <= price_max)]

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🔍 KEY INSIGHTS", "🛒 PRODUCTS", "🤖 WHY AI/ML WORKS HERE", "🎯 FILTERS & RECOMMENDER"])

# ============================================================================
# TAB 1: KEY INSIGHTS
# ============================================================================
with tab1:
    st.markdown("## Business at a Glance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"{total_transactions}", "transactions")
    with col2:
        st.metric("Customer Count", f"{unique_customers}", "unique shoppers")
    with col3:
        st.metric("Product Range", f"{unique_products}", "different items")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Average Sale", f"${avg_transaction_value:.2f}", "per transaction")
    with col5:
        st.metric("Items Per Sale", f"{avg_items_per_transaction:.1f}", "on average")
    with col6:
        st.metric("Total Revenue", f"${total_revenue:.2f}", "from sales")

    st.markdown("""
        <div class="insight-box">
            <h3 style="margin-top: 0; color: #1B5E20;">What This Means</h3>
            <p style="font-size: 18px; color: #1B5E20;"><strong>✓ Strong Customer Base:</strong> With many customers, you have enough people shopping to identify real shopping patterns.</p>
            <p style="font-size: 18px; color: #1B5E20;"><strong>✓ Consistent Shopping Habits:</strong> Customers buy multiple items per visit, showing predictable behavior patterns.</p>
            <p style="font-size: 18px; color: #1B5E20;"><strong>✓ Good Product Mix:</strong> A variety of products means customers have choices, and we can see what they prefer.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## Customer Shopping Patterns")
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [large_baskets, medium_baskets, small_baskets]
    labels = ['Frequent Shoppers\n(3+ items)', 'Moderate Shoppers\n(2 items)', 'Quick Shoppers\n(1 item)']
    colors = ['#2E7D32', '#558B2F', '#9CCC65']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 16, 'weight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(18)
        autotext.set_weight('bold')
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 2: PRODUCTS
# ============================================================================
with tab2:
    st.markdown("## Most Popular Products")
    fig, ax = plt.subplots(figsize=(12, 6))
    products = product_popularity.index.tolist()
    purchases = product_popularity.values.tolist()
    bars = ax.bar(products, purchases, color='#2E7D32', edgecolor='#1B5E20', linewidth=2)
    ax.set_ylabel('Number of Purchases', fontsize=18, fontweight='bold')
    ax.set_xlabel('Product', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 3: WHY AI/ML WORKS HERE
# ============================================================================
with tab3:
    st.markdown("## Why This Data is Perfect for AI and Machine Learning")
    st.markdown("Machine Learning models learn from data patterns. Your dataset has all the right ingredients for AI to help your business grow.")

    factors_data = [
        ("Transaction Volume", 85, "Many transactions provide sufficient data for pattern detection"),
        ("Product Variety", 95, "Distinct products enable pattern discovery"),
        ("Customer Base", 80, "A solid number of customers supports segmentation models"),
        ("Purchase Patterns", 75, "Variable basket sizes reveal preferences"),
        ("Price Diversity", 88, "A wide price range for prediction")
    ]
    for factor_name, strength, description in factors_data:
        c1, c2, c3 = st.columns([2, 3, 1])
        with c1:
            st.markdown(f"<h4 style='color: #1B5E20; margin: 0;'>{factor_name}</h4>", unsafe_allow_html=True)
        with c2:
            st.progress(strength / 100)
        with c3:
            st.markdown(f"<h4 style='color: #333; margin: 0;'>{strength}%</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 16px; color: #666; margin-top: -10px;'>{description}</p>", unsafe_allow_html=True)
        st.divider()

# ============================================================================
# TAB 4: FILTERS & RECOMMENDER
# ============================================================================
with tab4:
    st.markdown("## 📄 Filtered Transactions")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("## 🤖 Product Recommendation Engine")

    selected_product = st.selectbox(
        "Choose a product to see what customers often buy with it:",
        sorted(df["Itemname"].unique())
    )

    # Build co-occurrence counts
    co_occurrence = {}
    for bill, items in df.groupby("BillNo")["Itemname"]:
        items = list(items)
        for item in items:
            co_occurrence.setdefault(item, {})
            for other in items:
                if item != other:
                    co_occurrence[item][other] = co_occurrence[item].get(other, 0) + 1

    if selected_product in co_occurrence and co_occurrence[selected_product]:
        recommendations = sorted(
            co_occurrence[selected_product].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        st.markdown(f"### 🛒 Customers who buy **{selected_product}** also buy:")
        for item, score in recommendations:
            st.markdown(f"- **{item}** (together {score} times)")
    else:
        st.info("Not enough data for recommendations yet.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
    <div style="background-color: white; padding: 25px 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-top: 40px; text-align: center;">
        <p style="font-size: 16px; color: #888; font-style: italic; margin: 0;">
            This dashboard is designed for clarity and ease of use. All information is presented in clear language with large, readable text.
        </p>
    </div>
""", unsafe_allow_html=True)