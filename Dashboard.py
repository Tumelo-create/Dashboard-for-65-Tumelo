# -*- coding: utf-8 -*-
"""
Market Basket Analysis Dashboard – Senior‑Friendly Design (65+)
---------------------------------------------------------------
This interactive dashboard:
- Loads the online retail transaction dataset.
- Performs Apriori association rule mining.
- Allows the user to adjust support / confidence thresholds.
- Displays top items, co‑occurrence heatmap, and association rules.
- Explains why this dataset is suitable for Machine Learning.

Design for adults 65+:
- Large fonts, high contrast, simple words.
- Sliders with clear labels and tooltips.
- Only essential controls (2 sliders).
- Tooltips repeat instructions on every control.
- Hover information on all charts.
- No small click targets; use buttons / sliders with big handles.
- Explanatory text in plain English, no jargon.
"""

import streamlit as st
import pandas as pd
import numpy as np
# ------------------------------
# 1. PAGE CONFIGURATION (senior-friendly)
# ------------------------------
st.set_page_config(
    page_title="Market Basket Analysis (65+ friendly)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to increase font sizes, button sizes, and improve readability
st.markdown(
    """
    <style>
    /* Main font size */
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    /* Headers */
    h1 {
        font-size: 42px !important;
        color: #1f4f6e;
    }
    h2 {
        font-size: 34px !important;
    }
    h3 {
        font-size: 28px !important;
    }
    /* Slider label */
    .stSlider label {
        font-size: 20px !important;
        font-weight: bold;
    }
    /* Tooltips / help text */
    .stMarkdown p {
        font-size: 18px;
    }
    /* Buttons */
    .stButton button {
        font-size: 18px;
        padding: 0.5rem 1rem;
    }
    /* Dataframes */
    .dataframe {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# 2. LOAD AND PREPROCESS DATA (cached)
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("market_basket_dataset.csv")
    return df

@st.cache_data
def preprocess_transactions(df):
    """Group by BillNo to get list of unique items per transaction."""
    baskets = df.groupby("BillNo")["Itemname"].apply(lambda x: list(set(x))).reset_index()
    transactions = baskets["Itemname"].tolist()
    return transactions

@st.cache_data
def get_item_frequencies(transactions):
    """Count how many transactions contain each item."""
    all_items = [item for tx in transactions for item in tx]
    freq = pd.Series(all_items).value_counts().reset_index()
    freq.columns = ["Item", "Frequency"]
    return freq

@st.cache_data
def encode_transactions(transactions):
    """One‑hot encoding for mlxtend."""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded

@st.cache_data
def get_co_occurrence_matrix(df_encoded, top_n=10):
    """Co‑occurrence matrix for the top N items."""
    # Get top N items by frequency
    item_counts = df_encoded.sum().sort_values(ascending=False)
    top_items = item_counts.head(top_n).index.tolist()
    # Subset encoded df to those items
    df_top = df_encoded[top_items]
    co_occur = df_top.T.dot(df_top)
    return co_occur, top_items

# Load data
df = load_data()
transactions = preprocess_transactions(df)
df_encoded = encode_transactions(transactions)
item_freq = get_item_frequencies(transactions)
co_occur_matrix, top_items = get_co_occurrence_matrix(df_encoded, top_n=10)

# ------------------------------
# 3. SIDEBAR – SENIOR FRIENDLY CONTROLS
# ------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1995/1995572.png", width=80)  # shopping cart icon (optional)
st.sidebar.title("⚙️ Adjust the rules")
st.sidebar.markdown(
    "**How to use:** Move the sliders to change the **support** (how often items appear together) "
    "and **confidence** (how reliable the rule is). Lower values give more rules."
)

min_support = st.sidebar.slider(
    "📊 Minimum Support (0.01 = 1% of transactions)",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.01,
    help="Support = how many transactions contain the item set. Higher values keep only very frequent patterns.",
)

min_confidence = st.sidebar.slider(
    "🎯 Minimum Confidence",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Confidence = how often the rule is true. Example: If someone buys Apples, how often do they also buy Butter?",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** Start with lower support (0.02-0.03) and medium confidence (0.5) to see interesting rules. "
    "If you see too many rules, increase support."
)

# ------------------------------
# 4. MAIN DASHBOARD
# ------------------------------
st.title("🛒 Market Basket Analysis Dashboard")
st.markdown("### For adults 65+ – easy to read and simple to use")
st.markdown(
    "This dashboard helps a **online retail business** understand what products people often buy together. "
    "Using **Machine Learning** (Apriori algorithm), we find hidden patterns like: *'If a customer buys bread, they often buy butter'*."
)

# ----- TABS for better organisation -----
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "🔗 Item Co‑occurrence", "📜 Association Rules", "🧠 Why ML?"]
)

# ----- TAB 1: DATA OVERVIEW -----
with tab1:
    st.header("Dataset summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total transactions (bills)", len(transactions))
    col2.metric("Unique products", len(item_freq))
    col3.metric("Most frequent item", item_freq.iloc[0]["Item"])

    st.subheader("Top 10 best‑selling products (by number of baskets)")
    fig_bar = px.bar(
        item_freq.head(10),
        x="Item",
        y="Frequency",
        title="Frequency of items in customer baskets",
        labels={"Frequency": "Number of baskets containing the item"},
        color="Frequency",
        color_continuous_scale="blues",
    )
    fig_bar.update_layout(font_size=16, title_font_size=24, showlegend=False)
    fig_bar.update_traces(texttemplate="%{y}", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info(
        "**Observation:** Bananas, Onions, Potatoes, and Butter are very common. "
        "This suggests that a Machine Learning model could learn typical shopping habits and recommend these items."
    )

# ----- TAB 2: CO‑OCCURRENCE HEATMAP -----
with tab2:
    st.header("How often do two products appear together?")
    st.markdown(
        "The heatmap below shows **pairwise co‑occurrence** for the top 10 products. "
        "Darker blue means they are often bought in the same basket."
    )
    # Prepare heatmap
    heatmap_fig = px.imshow(
        co_occur_matrix.values,
        x=co_occur_matrix.columns,
        y=co_occur_matrix.index,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",
        title="Co‑occurrence matrix (top 10 items)",
    )
    heatmap_fig.update_layout(
        font_size=16,
        title_font_size=24,
        xaxis_title="Product",
        yaxis_title="Product",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.success(
        "✅ **Why this is good for ML:** Strong co‑occurrence patterns (e.g., Butter & Bread) are exactly what "
        "association rule learning (Apriori / FP-Growth) can discover automatically. Those rules become the basis for "
        "product recommendation engines."
    )

# ----- TAB 3: ASSOCIATION RULES (Apriori) -----
with tab3:
    st.header("Association rules (Apriori algorithm)")
    st.markdown(
        f"Using **minimum support = {min_support}** and **minimum confidence = {min_confidence}**. "
        "Rules with **Lift > 1** are interesting (products appear together more often than by chance)."
    )

    # Run Apriori (cached based on min_support)
    @st.cache_data
    def get_frequent_itemsets_and_rules(min_sup, min_conf):
        freq_items = apriori(df_encoded, min_support=min_sup, use_colnames=True)
        if len(freq_items) == 0:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        # Add length of antecedents for display
        rules["antecedent_len"] = rules["antecedents"].apply(len)
        rules["consequent_len"] = rules["consequents"].apply(len)
        # Sort by lift descending
        rules = rules.sort_values("lift", ascending=False)
        return freq_items, rules

    with st.spinner("🔍 Mining association rules..."):
        freq_items, rules = get_frequent_itemsets_and_rules(min_support, min_confidence)

    if rules.empty:
        st.warning(
            f"No rules found with support ≥ {min_support} and confidence ≥ {min_confidence}. "
            "Please try lower values on the left sidebar."
        )
    else:
        st.subheader(f"🔍 {len(rules)} association rules discovered")
        # Display top 15 rules
        top_rules = rules.head(15).copy()
        # Convert frozensets to readable strings
        top_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        top_rules["consequents"] = top_rules["consequents"].apply(lambda x: ", ".join(list(x)))

        display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        st.dataframe(
            top_rules[display_cols],
            use_container_width=True,
            column_config={
                "support": st.column_config.NumberColumn(format="%.4f"),
                "confidence": st.column_config.NumberColumn(format="%.3f"),
                "lift": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        # Visualisation: support vs confidence coloured by lift
        fig_scatter = px.scatter(
            top_rules,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_name="antecedents",
            text="consequents",
            title="Rule Quality: Support vs Confidence (circle size = lift)",
            labels={"support": "Support (frequency)", "confidence": "Confidence (reliability)"},
            color_continuous_scale="viridis",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(font_size=16, title_font_size=24)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown(
            "**Interpretation:** Higher lift values (darker circles) mean the rule is much better than random chance. "
            "For example, *'Butter → Bread'* has high lift → great candidate for a 'frequently bought together' recommendation."
        )

# ----- TAB 4: WHY THIS DATASET IS SUITABLE FOR MACHINE LEARNING -----
with tab4:
    st.header("🧠 Why is this dataset suitable for Machine Learning?")
    st.markdown(
        """
        This online retail dataset has the three key ingredients needed for successful Machine Learning:

        1. **Transactional structure** – Each bill is a set of items purchased together.  
           ML algorithms like **Apriori** and **FP‑Growth** are designed exactly for such data.

        2. **Clear patterns** – We already see that `Butter → Bread`, `Onions → Potatoes`, etc., appear often.  
           Models can learn these patterns automatically and predict future purchases.

        3. **Actionable business value** – From the rules we discovered, an online store can:  
           - Show *“Customers who bought X also bought Y”* (recommendation system).  
           - Create bundle offers (e.g., discount on Coffee + Sugar).  
           - Optimise product placement in digital storefronts.

        ### What ML models can do with this dataset?
        - **Association rule mining** (Apriori, FP‑Growth) → discover hidden item relationships.  
        - **Collaborative filtering** → recommend items based on similar customers’ baskets.  
        - **Next‑basket prediction** (Markov chains / RNNs) → predict what a customer will buy next.  
        - **Customer segmentation** (clustering) → group shoppers by their basket composition.

        **In this dashboard** you have already seen how the Apriori algorithm (a classic ML model for market baskets) extracts meaningful rules.  
        The interactive sliders let you explore how changing support/confidence affects the rules – a process used by data scientists to tune models for real‑world deployment.
        """
    )
    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*FjXUONp-FdTYnCHJB04hHw.png",
        caption="Example: Machine learning recommendation system based on market basket analysis",
        use_column_width=True,
    )
    st.info(
        "**In summary:** Because this dataset contains many real customer transactions with varied item combinations, "
        "it is an ideal candidate for training recommendation engines and cross‑selling models."
    )

# ------------------------------
# 5. FOOTER (accessibility note)
# ------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>✅ Dashboard designed for adults 65+ – large fonts, simple controls, clear language. Hover over any slider or chart for extra help.</p>",
    unsafe_allow_html=True,
)