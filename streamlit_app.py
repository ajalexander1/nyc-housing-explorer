"""
Name: Adam Alexander
CS230: Section 4
Data: New York Housing Market
URL: (add Streamlit Cloud URL once deployed)

Description:
This program is an interactive NYC Housing Market Explorer built with Streamlit.
It allows users to filter NYC housing listings by city/borough, property type,
price, beds, and square footage. The program generates charts, pivot tables,
summary statistics, and an interactive PyDeck map visualization. It demonstrates
Pandas data wrangling, Streamlit UI components, Plotly charts, PyDeck mapping,
and multi-page navigation.
"""

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression

# ---------------------------
# Global style / constants
# ---------------------------

st.set_page_config(
    page_title="NYC Housing Market Explorer",
    layout="wide",
)

# Consistent colors for property types
PROPERTY_COLOR_MAP = {
    "Condo for sale": "#1f77b4",
    "Condominium for sale": "#1f77b4",
    "Co-op for sale": "#ff7f0e",
    "House for sale": "#2ca02c",
    "Single family home for sale": "#2ca02c",
    "Townhouse for sale": "#9467bd",
    "Multi-family home for sale": "#8c564b",
    "Land for sale": "#e377c2",
    "Pending": "#7f7f7f",
    "Contingent": "#bcbd22",
    "Foreclosure": "#d62728",
}

BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


def render_header(subtitle: str):
    """
    Shared header used on every page.
    """
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align:center;">
                    <h1 style="margin-bottom:0;">🏙️ NYC Housing Market Explorer</h1>
                    <p style="color:#555; margin-top:0.25rem;">Interactive NYC housing analytics dashboard</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(f"### {subtitle}")
        st.markdown("---")


# ---------- Insight generator ----------

def generate_insights(filtered_df: pd.DataFrame, full_df: pd.DataFrame) -> str:
    """
    Produce short, human-readable insights for the current filtered slice.
    (We intentionally skip the median-vs-overall sentence to avoid formatting issues.)
    """
    if filtered_df.empty:
        return "No listings match the current filters."

    lines = []

    # --- Cheapest borough in this slice ---
    if "BOROUGH" in filtered_df.columns:
        by_boro = (
            filtered_df[filtered_df["BOROUGH"].notna()]
            .groupby("BOROUGH")["PRICE"]
            .median()
            .sort_values()
        )
        if not by_boro.empty:
            cheapest_boro = by_boro.index[0]
            cheap_price = by_boro.iloc[0]
            cheap_str = "${:,.0f}".format(float(cheap_price))
            lines.append(
                f"- {cheapest_boro} is the most affordable borough in this view, "
                f"with a median price of about {cheap_str}."
            )

    # --- Price per sqft insight ---
    if "PRICE_PER_SQFT" in filtered_df.columns:
        ppsf = filtered_df["PRICE_PER_SQFT"].median()
        if pd.notna(ppsf):
            ppsf_str = "${:,.0f}".format(float(ppsf))
            lines.append(
                f"- Typical price per square foot here is around {ppsf_str}."
            )

    # --- Inventory insight ---
    lines.append(
        f"- There are {len(filtered_df):,} active listings that match your filters."
    )

    return "\n".join(lines)

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------

@st.cache_data
def load_data():
    """
    Load and clean the NYC housing dataset.
    Includes:
      - standard CITY_STD column
      - derived PRICE_PER_SQFT
      - normalized BOROUGH column (only 5 boroughs or None)
    """
    df = pd.read_csv("ny_housing.csv")

    # Clean PRICE column (remove any commas, spaces, unicode separators)
    df["PRICE"] = (
        df["PRICE"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
    )

    # Clean SQFT if needed
    df["PROPERTYSQFT"] = (
        df["PROPERTYSQFT"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
    )

    # Fix bathroom values
    df["BATH"] = (
        pd.to_numeric(df["BATH"], errors="coerce")  # convert to numeric
        .fillna(1)  # fallback
    )
    df["BATH"] = (df["BATH"] * 2).round() / 2  # round to nearest 0.5
    # #[LAMBDA] using a lambda to clean and format bathroom counts into readable numeric values
    df["BATH"] = df["BATH"].apply(lambda x: float(f"{x:.2f}".rstrip("0").rstrip(".")))

    # Clean column names to UPPER
    df.columns = [c.strip().upper() for c in df.columns]

    # ---- Location standardization: create CITY_STD ----
    location_col = None
    # #[ITERLOOP] iterating through possible location columns
    for cand in ["CITY", "LOCALITY", "COUNTY", "STATE"]:
        if cand in df.columns:
            location_col = cand
            break

    if location_col is None:
        df["CITY_STD"] = "Unknown"
    else:
        df["CITY_STD"] = df[location_col].fillna("Unknown")

    # #[COLUMNS] creating derived price per square foot column
    df["PRICE_PER_SQFT"] = np.where(
        df["PROPERTYSQFT"] > 0,
        df["PRICE"] / df["PROPERTYSQFT"],
        np.nan,
    )

    # #[LISTCOMP] extracting first token from CITY name using list comprehension
    df["AREA_SIMPLE"] = [
        str(city).split(",")[0] if isinstance(city, str) else ""
        for city in df["CITY_STD"]
    ]

    # ---- Borough normalization (THIS is the important part) ----
    borough_map = {
        "bronx": "Bronx",
        "bronx county": "Bronx",
        "the bronx": "Bronx",

        "kings county": "Brooklyn",
        "brooklyn": "Brooklyn",

        "new york": "Manhattan",
        "new york county": "Manhattan",
        "manhattan": "Manhattan",

        "queens": "Queens",
        "queens county": "Queens",

        "staten island": "Staten Island",
        "richmond county": "Staten Island",
    }

    def normalize_borough(name):
        raw = str(name).strip().lower()
        # ONLY return a value if it’s a true borough; everything else -> None
        return borough_map.get(raw, None)

    df["BOROUGH"] = df["CITY_STD"].apply(normalize_borough)

    return df


df_all = load_data()

# Dictionary for color mapping
PROPERTY_COLORS = {
    "Condo for sale": "blue",
    "House for sale": "green",
    "Townhouse for sale": "orange",
    "Multi-family home for sale": "purple",
    "Co-op for sale": "red",
}

# #[DICTMETHOD] using .get() to retrieve default color from dictionary
DEFAULT_TYPE_COLOR = PROPERTY_COLORS.get("Condo for sale", "gray")
PROPERTY_TYPES_LIST = list(PROPERTY_COLORS.keys())


# ---------------------------------------
# Utility Functions (Filtering / Summary)
# ---------------------------------------

# #[FUNC2P] Function with params + default param
def filter_listings(
    data,
    city=None,
    property_types=None,
    price_range=(0, 300000000),
    beds_range=(0, 10),
    sqft_range=(0, 50000),
):
    """
    Apply all filters to listing data.
    Returns a filtered DataFrame.
    """
    df = data.copy()

    # #[FILTER1] filtering out listings with invalid prices
    df = df[df["PRICE"] > 0]

    if city and city != "All":
        df = df[df["BOROUGH"] == city]

    if property_types:
        df = df[df["TYPE"].isin(property_types)]

    min_price, max_price = price_range
    min_beds, max_beds = beds_range
    min_sqft, max_sqft = sqft_range

    # #[FILTER2] Multi-condition filter
    df = df[
        (df["PRICE"] >= min_price)
        & (df["PRICE"] <= max_price)
        & (df["BEDS"] >= min_beds)
        & (df["BEDS"] <= max_beds)
        & (df["PROPERTYSQFT"] >= min_sqft)
        & (df["PROPERTYSQFT"] <= max_sqft)
    ]

    # #[SORT] Sort by price
    df = df.sort_values(by="PRICE", ascending=True)

    return df


# #[FUNCRETURN2] returns two dictionaries of stats
def compute_summary_stats(filtered_df):
    """
    Compute stats for price and price_per_sqft.
    Returns two dictionaries.
    """
    price_stats = {
        "count": int(filtered_df["PRICE"].count()),
        "min": float(filtered_df["PRICE"].min()) if not filtered_df.empty else None,
        "median": float(filtered_df["PRICE"].median()) if not filtered_df.empty else None,
        "mean": float(filtered_df["PRICE"].mean()) if not filtered_df.empty else None,
        "max": float(filtered_df["PRICE"].max()) if not filtered_df.empty else None,
    }

    ppsf_stats = {
        "median_ppsf": float(filtered_df["PRICE_PER_SQFT"].median())
        if not filtered_df.empty else None,
        "mean_ppsf": float(filtered_df["PRICE_PER_SQFT"].mean())
        if not filtered_df.empty else None,
    }

    return price_stats, ppsf_stats

@st.cache_data
def train_price_model(df: pd.DataFrame):
    """
    Train a simple linear regression model to predict PRICE
    based on beds, baths, sqft, borough, and property type.

    Returns:
      - model: trained LinearRegression model
      - feature_columns: list of feature column names used in training
      - mae: mean absolute error (float) for rough error band
      Or (None, None, None) if model can't be trained.
    """
    required_cols = ["PRICE", "BEDS", "BATH", "PROPERTYSQFT", "BOROUGH", "TYPE"]
    if not all(col in df.columns for col in required_cols):
        return None, None, None

    # Drop rows with missing values in required columns and non-positive prices
    model_df = df[required_cols].dropna().copy()
    model_df = model_df[model_df["PRICE"] > 0]

    if model_df.empty or len(model_df) < 50:
        # Not enough data to train a meaningful model
        return None, None, None

    # Clip ultra-luxury outliers for training robustness (keep cheapest 99%)
    cap = model_df["PRICE"].quantile(0.99)
    model_df = model_df[model_df["PRICE"] <= cap]

    # Base numeric features
    X = model_df[["BEDS", "BATH", "PROPERTYSQFT"]].copy()

    # One-hot encode borough and property type
    borough_dummies = pd.get_dummies(model_df["BOROUGH"], prefix="BOROUGH")
    type_dummies = pd.get_dummies(model_df["TYPE"], prefix="TYPE")

    X = pd.concat([X, borough_dummies, type_dummies], axis=1)
    y = model_df["PRICE"]

    if X.empty:
        return None, None, None

    model = LinearRegression()
    model.fit(X, y)

    # Rough in-sample MAE for error band
    preds = model.predict(X)
    mae = float(np.mean(np.abs(preds - y)))

    return model, list(X.columns), mae

# --------------
# Sidebar Filter UI
# --------------

def sidebar_filters(df):
    st.sidebar.markdown("### Filters")

    # --- options & ranges ---
    borough_values = [b for b in df["BOROUGH"].unique().tolist() if isinstance(b, str)]
    cities = ["All"] + sorted(borough_values)

    type_options = sorted(df["TYPE"].dropna().unique().tolist())
    core_types = [t for t in type_options if "for sale" in t.lower() or "house" in t.lower()]

    min_price = int(df["PRICE"].min())
    max_price = int(df["PRICE"].max())
    max_beds = int(df["BEDS"].max())
    max_sqft = int(df["PROPERTYSQFT"].max())

    # --- Reset filters button ---
    if st.sidebar.button("Reset filters"):
        st.session_state["city_choice"] = "All"
        st.session_state["property_types"] = type_options
        st.session_state["price_range"] = (min_price, max_price)
        st.session_state["beds_range"] = (0, max_beds)
        st.session_state["sqft_range"] = (0, max_sqft)

    # --- widgets using session_state keys ---
    # #[ST1] Streamlit dropdown + multiselect filters
    city_choice = st.sidebar.selectbox(
        "City / Borough",
        cities,
        key="city_choice",
        index=cities.index(st.session_state.get("city_choice", "All"))
        if "city_choice" in st.session_state and st.session_state["city_choice"] in cities
        else 0,
    )

    property_types = st.sidebar.multiselect(
        "Property type(s)",
        options=type_options,
        default=st.session_state.get("property_types", core_types or type_options),
        key="property_types",
    )

    # #[ST2] Slider for selecting price range
    price_range = st.sidebar.slider(
        "Price range ($)",
        min_value=min_price,
        max_value=max_price,
        value=st.session_state.get("price_range", (min_price, max_price)),
        step=50000,
        key="price_range",
    )

    beds_range = st.sidebar.slider(
        "Number of bedrooms",
        min_value=0,
        max_value=max_beds,
        value=st.session_state.get("beds_range", (0, max_beds)),
        key="beds_range",
    )

    sqft_range = st.sidebar.slider(
        "Property size (sqft)",
        min_value=0,
        max_value=max_sqft,
        value=st.session_state.get("sqft_range", (0, max_sqft)),
        key="sqft_range",
    )

    return city_choice, property_types, price_range, beds_range, sqft_range

def generate_text_report(
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    city_choice,
    property_types,
    price_range,
    beds_range,
    sqft_range,
) -> str:
    """
    Build a short, plain-text mini report for the current filtered view.
    This is meant for copying or downloading as a .txt file.
    """
    lines = []

    # --- Header ---
    lines.append("NYC Housing Market Mini Report")
    lines.append("--------------------------------")
    lines.append("")

    # --- Filter Summary ---
    lines.append("Filter Summary:")

    if city_choice and city_choice != "All":
        lines.append(f"- Borough: {city_choice}")
    else:
        lines.append("- Borough: All NYC")

    if property_types:
        if len(property_types) <= 6:
            type_str = ", ".join(sorted(property_types))
        else:
            type_str = f"{len(property_types)} property types selected"
    else:
        type_str = "None selected"

    lines.append(f"- Property types: {type_str}")
    lines.append(
        f"- Price range: ${price_range[0]:,} – ${price_range[1]:,}"
    )
    lines.append(
        f"- Bedrooms: {beds_range[0]} – {beds_range[1]}"
    )
    lines.append(
        f"- Square footage: {sqft_range[0]:,} – {sqft_range[1]:,} sqft"
    )
    lines.append("")

    if filtered_df.empty:
        lines.append("No listings match this filter combination.")
        return "\n".join(lines)

    # --- Headline Metrics ---
    price_stats, ppsf_stats = compute_summary_stats(filtered_df)

    lines.append("Headline Metrics (current view):")
    lines.append(f"- Listings: {price_stats['count']:,}")

    if price_stats["median"] is not None:
        lines.append(f"- Median price: ${price_stats['median']:,.0f}")
    if price_stats["mean"] is not None:
        lines.append(f"- Average price: ${price_stats['mean']:,.0f}")
    if ppsf_stats["median_ppsf"] is not None:
        lines.append(f"- Median price per sqft: ${ppsf_stats['median_ppsf']:,.0f}")

    if price_stats["min"] is not None and price_stats["max"] is not None:
        lines.append(
            f"- Price range in this view: "
            f"${price_stats['min']:,.0f} – ${price_stats['max']:,.0f}"
        )

    lines.append("")

    # --- Borough Summary (if available) ---
    if "BOROUGH" in filtered_df.columns:
        boro_summary = (
            filtered_df[filtered_df["BOROUGH"].notna()]
            .groupby("BOROUGH")
            .agg(
                listings=("PRICE", "count"),
                median_price=("PRICE", "median"),
                median_ppsf=("PRICE_PER_SQFT", "median"),
            )
            .reset_index()
        )

        if not boro_summary.empty:
            lines.append("Listings by Borough (current view):")
            for _, row in boro_summary.sort_values("listings", ascending=False).iterrows():
                boro = row["BOROUGH"]
                count = int(row["listings"])
                med_price = row["median_price"]
                med_ppsf = row["median_ppsf"]

                line = f"- {boro}: {count:,} listings"
                if not pd.isna(med_price):
                    line += f", median price ~ ${med_price:,.0f}"
                if not pd.isna(med_ppsf):
                    line += f", median $/sqft ~ ${med_ppsf:,.0f}"

                lines.append(line)

            lines.append("")

    # --- Closing note ---
    lines.append(
        "Note: This report is generated from the current filtered slice of the dataset "
        "and is intended for exploratory analysis, not formal appraisal."
    )

    return "\n".join(lines)

# --------------
# Page: Overview
# --------------

def overview_page(df, full_df=None):
    # Fallback: if full_df not passed, use df
    if full_df is None:
        full_df = df

    # -------- Ensure PRICE is numeric for both slice & overall (defensive) --------
    for frame in (df, full_df):
        if "PRICE" in frame.columns:
            frame["PRICE"] = (
                frame["PRICE"]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)  # strip commas, spaces, unicode
                .replace("", np.nan)
                .astype(float)
            )

    render_header("Overview 📊")

    # -------- Sidebar filters & filtered dataset --------
    city_choice, property_types, price_range, beds_range, sqft_range = sidebar_filters(df)
    # #[FUNCCALL2] calling filter_listings (also called in charts_page)
    filtered_df = filter_listings(
        df,
        city=city_choice,
        property_types=property_types,
        price_range=price_range,
        beds_range=beds_range,
        sqft_range=sqft_range,
    )

    if filtered_df.empty:
        st.warning("No listings match your filters. Try widening the price, size, or bedroom range.")
        return

    # -------- Summary stats / KPIs --------
    price_stats, ppsf_stats = compute_summary_stats(filtered_df)

    avg_beds = float(filtered_df["BEDS"].mean()) if "BEDS" in filtered_df.columns else np.nan
    avg_baths = float(filtered_df["BATH"].mean()) if "BATH" in filtered_df.columns else np.nan
    avg_sqft = float(filtered_df["PROPERTYSQFT"].mean()) if "PROPERTYSQFT" in filtered_df.columns else np.nan

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Listings", f"{price_stats['count']:,}")
    kpi2.metric(
        "Median Price",
        f"${price_stats['median']:,.0f}" if price_stats["median"] is not None else "—",
    )
    kpi3.metric(
        "Median Price/Sqft",
        f"${ppsf_stats['median_ppsf']:,.0f}" if ppsf_stats["median_ppsf"] is not None else "—",
    )
    if not np.isnan(avg_beds) and not np.isnan(avg_baths):
        kpi4.metric("Avg Beds / Baths", f"{avg_beds:.1f} / {avg_baths:.1f}")
    else:
        kpi4.metric("Avg Beds / Baths", "—")

    # -------- Filter snapshot --------
    st.markdown("### Current Filter Snapshot")

    summary_bits = []

    # Borough label
    if city_choice and city_choice != "All":
        summary_bits.append(f"**Borough:** {city_choice}")
    else:
        summary_bits.append("**Borough:** All NYC")

    # Property types
    if property_types:
        if len(property_types) <= 5:
            types_str = ", ".join(sorted(property_types))
        else:
            types_str = f"{len(property_types)} types selected"
    else:
        types_str = "None selected"
    summary_bits.append(f"**Property types:** {types_str}")

    # Ranges
    summary_bits.append(
        f"**Price range:** ${price_range[0]:,} – ${price_range[1]:,}"
    )
    summary_bits.append(
        f"**Bedrooms:** {beds_range[0]} – {beds_range[1]}  |  "
        f"**Size:** {sqft_range[0]:,} – {sqft_range[1]:,} sqft"
    )

    st.markdown("- " + "\n- ".join(summary_bits))

    # -------- Key insights (now with clean numbers) --------
    with st.expander("💡 Key insights for this view", expanded=True):
        st.write(generate_insights(filtered_df, full_df))

    st.markdown("---")

    # ------------------------------------------------
    # Mini text report (copyable / downloadable)
    # ------------------------------------------------
    with st.expander("📄 Export Mini Market Report", expanded=False):
        report_text = generate_text_report(
            filtered_df=filtered_df,
            full_df=full_df,
            city_choice=city_choice,
            property_types=property_types,
            price_range=price_range,
            beds_range=beds_range,
            sqft_range=sqft_range,
        )

        st.text_area(
            "Report preview (you can copy this):",
            value=report_text,
            height=220,
        )

        st.download_button(
            "Download report as .txt",
            data=report_text.encode("utf-8"),
            file_name="nyc_housing_mini_report.txt",
            mime="text/plain",
        )

    # -------- Market snapshot by borough --------
    st.markdown("### Market Snapshot by Borough")

    if "BOROUGH" in filtered_df.columns:
        by_borough = (
            filtered_df[filtered_df["BOROUGH"].notna()]
            .groupby("BOROUGH")
            .agg(
                median_price=("PRICE", "median"),
                median_ppsf=("PRICE_PER_SQFT", "median"),
                listings=("PRICE", "count"),
            )
            .reset_index()
        )
        if not by_borough.empty:
            # Optional: order boroughs nicely
            if "BOROUGH" in by_borough.columns:
                by_borough["BOROUGH"] = pd.Categorical(
                    by_borough["BOROUGH"],
                    BOROUGH_ORDER,
                )
                by_borough = by_borough.sort_values("BOROUGH")

            col_table, col_chart = st.columns([2, 3])

            with col_table:
                st.dataframe(
                    by_borough.style.format(
                        {
                            "median_price": "${:,.0f}",
                            "median_ppsf": "${:,.0f}",
                            "listings": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )

            with col_chart:
                fig_boro = px.bar(
                    by_borough,
                    x="BOROUGH",
                    y="median_price",
                    title="Median Price by Borough (current filters)",
                    labels={"BOROUGH": "Borough", "median_price": "Median price ($)"},
                )
                fig_boro.update_yaxes(tickprefix="$")
                st.plotly_chart(fig_boro, use_container_width=True)
        else:
            st.info("No borough-level data available for the current filters.")
    else:
        st.info("This dataset does not include a BOROUGH field.")

    st.markdown("---")

    # ------------------------------------------------
    # Buyer Persona / Affordability Scenario
    # ------------------------------------------------
    st.markdown("### 🧑‍💼 Buyer Persona: Affordability Check")

    with st.expander("Try a simple buyer scenario", expanded=False):
        st.caption(
            "Example: a first-time buyer or young family looking for homes that fit a specific budget "
            "and bedroom range across NYC."
        )

        if full_df is None or full_df.empty or "PRICE" not in full_df.columns:
            st.info("Buyer persona analysis is unavailable because the full dataset is missing or empty.")
        else:
            # Clean base frame for persona calc
            persona_df = full_df.copy()
            persona_df = persona_df[
                (persona_df["PRICE"].notna()) &
                (persona_df["BEDS"].notna())
                ]

            if persona_df.empty:
                st.info("Not enough data to run the affordability scenario.")
            else:
                # Inputs
                col_a, col_b = st.columns(2)

                with col_a:
                    # Budget: default to median or 750k, whichever is lower
                    overall_median_price = persona_df["PRICE"].median()
                    suggested_budget = int(
                        min(overall_median_price if not np.isnan(overall_median_price) else 750000, 750000))

                    budget = st.number_input(
                        "Maximum budget ($)",
                        min_value=50000,
                        max_value=int(persona_df["PRICE"].max()),
                        value=suggested_budget,
                        step=25000,
                    )

                    beds_min = st.slider(
                        "Minimum bedrooms",
                        min_value=0,
                        max_value=int(persona_df["BEDS"].max()),
                        value=2,
                    )

                with col_b:
                    beds_max = st.slider(
                        "Maximum bedrooms",
                        min_value=beds_min,
                        max_value=int(persona_df["BEDS"].max()),
                        value=min(beds_min + 2, int(persona_df["BEDS"].max())),
                    )

                    # Optional preferred borough
                    if "BOROUGH" in persona_df.columns:
                        boros = [b for b in persona_df["BOROUGH"].dropna().unique().tolist() if isinstance(b, str)]
                        boros = sorted(boros)
                        borough_pref = st.selectbox(
                            "Preferred borough (optional)",
                            options=["Any"] + boros,
                        )
                    else:
                        borough_pref = "Any"

                # Apply persona filters
                persona_filtered = persona_df[
                    (persona_df["PRICE"] <= budget) &
                    (persona_df["BEDS"] >= beds_min) &
                    (persona_df["BEDS"] <= beds_max)
                    ]

                if borough_pref != "Any" and "BOROUGH" in persona_filtered.columns:
                    persona_filtered = persona_filtered[persona_filtered["BOROUGH"] == borough_pref]

                if persona_filtered.empty:
                    st.warning("No listings in the dataset match this buyer profile.")
                else:
                    st.markdown(
                        f"✅ Found **{len(persona_filtered):,}** listings that match this buyer profile."
                    )

                    # Summary by borough for this persona
                    if "BOROUGH" in persona_filtered.columns:
                        boro_summary = (
                            persona_filtered.groupby("BOROUGH")
                            .agg(
                                listings=("PRICE", "count"),
                                median_price=("PRICE", "median"),
                                median_ppsf=("PRICE_PER_SQFT", "median"),
                            )
                            .reset_index()
                            .sort_values("listings", ascending=False)
                        )

                        st.markdown("**Best boroughs for this buyer profile (by listing count):**")
                        st.dataframe(
                            boro_summary.style.format(
                                {
                                    "median_price": "${:,.0f}",
                                    "median_ppsf": "${:,.0f}",
                                    "listings": "{:,.0f}",
                                }
                            ),
                            use_container_width=True,
                        )

                        # Highlight top borough
                        top_row = boro_summary.iloc[0]
                        st.caption(
                            f"📍 Based on this dataset, **{top_row['BOROUGH']}** has the most options "
                            f"for this profile, with a median price around ${top_row['median_price']:,.0f}."
                        )
                    else:
                        st.info(
                            "Borough information is missing, so results are shown for the entire dataset only."
                        )

    # -------- Sample listings + download --------
    st.markdown("### Sample Listings")

    cols_to_show = [
        col for col in ["BROKERTITLE", "TYPE", "PRICE", "BEDS", "BATH", "PROPERTYSQFT", "BOROUGH", "ADDRESS"]
        if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[cols_to_show].head(25).style.format(
            {
                "PRICE": "${:,.0f}",
                "PROPERTYSQFT": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )
    st.caption("Sample shows the first 25 listings that match your filters, sorted by price.")

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered listings as CSV",
        data=csv,
        file_name="nyc_housing_filtered.csv",
        mime="text/csv",
    )

    # -------- Most & least expensive listings --------
    if not filtered_df.empty:
        # #[MAXMIN] identifying most and least expensive listings
        most_expensive = filtered_df.loc[filtered_df["PRICE"].idxmax()]
        least_expensive = filtered_df.loc[filtered_df["PRICE"].idxmin()]

        with st.expander("Most & Least Expensive Listings"):
            st.write(
                f"**Most Expensive**: ${most_expensive['PRICE']:,.0f} — "
                f"{most_expensive.get('ADDRESS', 'Address N/A')} "
                f"({most_expensive.get('BOROUGH', 'Borough N/A')})"
            )
            st.write(
                f"**Least Expensive**: ${least_expensive['PRICE']:,.0f} — "
                f"{least_expensive.get('ADDRESS', 'Address N/A')} "
                f"({least_expensive.get('BOROUGH', 'Borough N/A')})"
            )

 # ------------------------------------------------
    # Price Estimator (beta) using a trained ML model
    # ------------------------------------------------
    st.markdown("---")
    st.subheader("💰 Price Estimator (beta)")

    model, feature_cols, mae = train_price_model(full_df)

    if model is None:
        st.info(
            "Not enough clean data to train a price model, "
            "or required columns are missing."
        )
    else:
        # UI inputs for the user scenario
        st.markdown("Use this estimator to get a rough predicted price for a listing configuration.")

        col_left, col_right = st.columns(2)

        with col_left:
            # Borough choice
            available_boros = [b for b in full_df["BOROUGH"].dropna().unique().tolist() if isinstance(b, str)]
            available_boros = sorted(available_boros)
            borough_input = st.selectbox(
                "Borough",
                options=available_boros if available_boros else ["Manhattan"],
            )

            # Property type choice
            type_options = sorted(full_df["TYPE"].dropna().unique().tolist())
            default_type = type_options[0] if type_options else "Condo for sale"
            type_input = st.selectbox(
                "Property type",
                options=type_options if type_options else [default_type],
                index=0,
            )

            beds_input = st.slider(
                "Bedrooms",
                min_value=0,
                max_value=int(full_df["BEDS"].max()),
                value=2,
            )

        with col_right:
            # Allow half-baths (0.5 increments)
            max_bath = float(full_df["BATH"].max())
            bath_options = [round(x * 0.5, 1) for x in range(0, int(max_bath * 2) + 1)]
            bath_input = st.select_slider(
                "Bathrooms",
                options=bath_options,
                value=1.0,
            )

            sqft_input = st.number_input(
                "Square footage",
                min_value=200,
                max_value=int(full_df["PROPERTYSQFT"].max()),
                value=800,
                step=50,
            )

        if st.button("Estimate price"):
            # Build a one-row feature frame matching training features
            base = pd.DataFrame(
                {
                    "BEDS": [beds_input],
                    "BATH": [bath_input],
                    "PROPERTYSQFT": [sqft_input],
                }
            )

            # Add all dummy columns we used during training, default 0
            for col in feature_cols:
                if col.startswith("BOROUGH_"):
                    base[col] = 1 if col == f"BOROUGH_{borough_input}" else 0
                elif col.startswith("TYPE_"):
                    base[col] = 1 if col == f"TYPE_{type_input}" else 0

            # Ensure the column order matches training
            for col in feature_cols:
                if col not in base.columns:
                    base[col] = 0
            base = base[feature_cols]

            # -------------------------------
            # Predict price + clamp the output
            # -------------------------------
            raw_pred = float(model.predict(base)[0])

            # Use the minimum real price from the dataset as a realistic floor
            min_valid_price = full_df["PRICE"].min()

            pred_price = max(raw_pred, min_valid_price)
            # (If you prefer floor=0, change to:  pred_price = max(raw_pred, 0))

            col_pred, col_band = st.columns(2)
            col_pred.metric("Estimated price", f"${pred_price:,.0f}")

            if mae and mae > 0:
                col_band.caption(
                    f"Typical error (MAE) on training data is about ± ${mae:,.0f}. "
                    f"Treat this as a rough estimate, not an exact appraisal."
                )
            else:
                col_band.caption("This estimate is based on a simple linear regression model trained on the dataset.")


# --------------
# Page: Charts
# --------------

def charts_page(df, full_df=None):
    if full_df is None:
        full_df = df
    st.title("Price & Affordability Charts 📊")

    # --- Filters & data ---
    city_choice, property_types, price_range, beds_range, sqft_range = sidebar_filters(df)
    filtered_df = filter_listings(
        df,
        city=city_choice,
        property_types=property_types,
        price_range=price_range,
        beds_range=beds_range,
        sqft_range=sqft_range,
    )

    if filtered_df.empty:
        st.warning("No results match your filters. Try widening the ranges or adding more property types.")
        return

    # --- Filter summary (quick context for the charts) ---
    st.markdown("### Current Filter Snapshot")

    summary_bits = []

    if city_choice and city_choice != "All":
        summary_bits.append(f"**Borough:** {city_choice}")
    else:
        summary_bits.append("**Borough:** All NYC")

    if property_types and len(property_types) < 6:
        types_str = ", ".join(sorted(property_types))
    else:
        types_str = f"{len(property_types)} types selected"
    summary_bits.append(f"**Property types:** {types_str}")

    summary_bits.append(
        f"**Price range:** ${price_range[0]:,} – ${price_range[1]:,}"
    )
    summary_bits.append(
        f"**Bedrooms:** {beds_range[0]} – {beds_range[1]} | "
        f"**Size:** {sqft_range[0]:,} – {sqft_range[1]:,} sqft"
    )

    st.markdown("- " + "\n- ".join(summary_bits))

    st.markdown("---")

    # --- Chart tabs ---
    tab1, tab2, tab3 = st.tabs(
        ["📐 Median Price by Type / Borough", "📦 Price Distribution", "📏 Size vs Price"]
    )

    # ---------------------------
    # Tab 1: Median price by type / borough
    # ---------------------------
    with tab1:
        st.subheader("Median Listing Price by Property Type")

        # #[PIVOTTABLE] pivot table of median price by borough and type
        pivot_price = filtered_df.pivot_table(
            values="PRICE",
            index="BOROUGH",
            columns="TYPE",
            aggfunc="median",
        )

        st.markdown("**Median price (rows = boroughs, columns = property types):**")
        st.dataframe(
            pivot_price.style.format("${:,.0f}"),
            use_container_width=True,
        )

        st.markdown("#### Bar Chart: Median Price by Property Type")

        by_type = (
            filtered_df.groupby("TYPE", dropna=True)["PRICE"]
            .median()
            .reset_index()
            .sort_values("PRICE", ascending=False)
        )

        # #[CHART1] Bar chart showing average price by property type
        fig_bar = px.bar(
            by_type,
            x="TYPE",
            y="PRICE",
            title="Median Listing Price by Property Type",
            labels={"TYPE": "Property Type", "PRICE": "Median Price"},
        )
        fig_bar.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- NEW: Box plot of price distribution by borough ---
        if "BOROUGH" in filtered_df.columns:
            st.markdown("#### Price Spread by Borough (Box Plot)")  # NEW
            box_fig = px.box(                             # NEW
                filtered_df[filtered_df["PRICE"] > 0],
                x="BOROUGH",
                y="PRICE",
                points="outliers",
                title="Distribution of Listing Prices by Borough",
                labels={"BOROUGH": "Borough", "PRICE": "Listing Price ($)"},
            )
            box_fig.update_yaxes(tickprefix="$", type="log")  # log scale helps show spread
            st.plotly_chart(box_fig, use_container_width=True)

    # ---------------------------
    # Tab 2: Price distribution
    # ---------------------------
    with tab2:
        st.subheader("Price Distribution (Histogram)")

        # Base data: positive prices only
        hist_df = filtered_df[filtered_df["PRICE"] > 0].copy()

        if hist_df.empty:
            st.warning("No valid price data to plot. Try widening your filters.")
        else:
            min_price = hist_df["PRICE"].min()
            max_price = hist_df["PRICE"].max()
            st.caption(f"Current price range in data: ${min_price:,.0f} – ${max_price:,.0f}")

            cap_outliers = st.checkbox(
                "Cap ultra-luxury listings for this chart (top 1%)",
                value=True,
            )

            if cap_outliers:
                cap = hist_df["PRICE"].quantile(0.99)  # keep 99% of listings
                hist_df = hist_df[hist_df["PRICE"] <= cap]

            if hist_df.empty:
                st.warning("After capping outliers, there are no listings left to show.")
            else:
                fig_hist = px.histogram(
                    hist_df,
                    x="PRICE",
                    nbins=40,
                    title="Distribution of Listing Prices (current filters)",
                )
                fig_hist.update_xaxes(
                    title="Price ($)",
                    type="linear"
                )
                fig_hist.update_yaxes(title="Number of listings")

                st.plotly_chart(fig_hist, use_container_width=True)

        st.caption(
            "Ultra-luxury listings (top ~1% of prices) can be capped here so the chart "
            "focuses on the typical price range. The raw data is still preserved elsewhere in the app."
        )


    # ---------------------------
    # Tab 3: Size vs price scatter
    # ---------------------------
    with tab3:
        st.subheader("Property Size vs Price")

        color_choice = st.radio(
            "Color points by:",
            ("Property Type", "Borough"),
            horizontal=True,
        )

        if color_choice == "Property Type":
            color_col = "TYPE"
        else:
            color_col = "BOROUGH"

        log_y = st.checkbox("Use log scale for price (y-axis)", value=True, key="log_y_scatter")

        # #[CHART2] Scatter chart for Price vs Square Footage
        fig_scatter = px.scatter(
            filtered_df,
            x="PROPERTYSQFT",
            y="PRICE",
            color=color_col,
            size="PRICE",
            hover_data=["ADDRESS", "BEDS", "BATH", "BOROUGH"],
            title="Relationship Between Property Size and Price",
            labels={
                "PROPERTYSQFT": "Property Size (sqft)",
                "PRICE": "Listing Price ($)",
                color_col: color_choice,
            },
        )
        fig_scatter.update_yaxes(type="log" if log_y else "linear")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "Each bubble is a listing; bigger bubbles are more expensive. "
            "Using log scale on price makes both starter homes and luxury listings easier to compare."
        )

# --------------
# Page: Map
# --------------

def map_page(df, full_df=None):
    if full_df is None:
        full_df = df
    st.title("NYC Housing Map 🗺️")

    # Borough selector (Map only) - USE BOROUGH, NOT CITY_STD
    boroughs = sorted(df["BOROUGH"].dropna().unique().tolist())
    borough_choice = st.selectbox("Filter by Borough (Map Only):", ["All"] + boroughs)

    tile_url = "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"

    # Regular sidebar filters
    city_choice, property_types, price_range, beds_range, sqft_range = sidebar_filters(df)

    # Base filtered set using existing function
    filtered_df = filter_listings(
        df,
        city=city_choice,
        property_types=property_types,
        price_range=price_range,
        beds_range=beds_range,
        sqft_range=sqft_range,
    )

    # Apply borough filter AFTER filtered_df exists
    if borough_choice != "All":
        filtered_df = filtered_df[filtered_df["BOROUGH"] == borough_choice]

    if filtered_df.empty:
        st.warning("No properties match your filters.")
        return

    map_df = filtered_df.dropna(subset=["LATITUDE", "LONGITUDE"])
    if map_df.empty:
        st.warning("No listings with valid coordinates.")
        return

    # ---------- Visual encoding: size + color ----------

    # Radius: normalize price so bubbles aren't huge
    price_cap = float(map_df["PRICE"].quantile(0.98))
    map_df["PRICE_CLIPPED"] = map_df["PRICE"].clip(upper=price_cap)
    map_df["PRICE_NORM"] = map_df["PRICE_CLIPPED"] / price_cap
    map_df["RADIUS"] = (map_df["PRICE_NORM"] * 320) + 80  # 80–400m approx

    # Color palette by property TYPE
    color_map = {
        "Condo for sale": [66, 135, 245],
        "Co-op for sale": [255, 99, 71],
        "Townhouse for sale": [255, 185, 0],
        "Multi-family home for sale": [142, 68, 173],
        "House for sale": [46, 204, 113],
    }

    def make_color(t):
        base = color_map.get(t, [150, 150, 150])  # default gray
        return base + [140]  # add alpha channel

    map_df["COLOR"] = map_df["TYPE"].apply(make_color)

    # ---------- Map centering / zoom ----------

    midpoint = (np.average(map_df["LATITUDE"]), np.average(map_df["LONGITUDE"]))

    # Dataset likely uses county names (Bronx County, Kings County, etc.)
    borough_centers = {
        "Manhattan": (40.7831, -73.9712),
        "Brooklyn": (40.6782, -73.9442),
        "Queens": (40.7282, -73.7949),
        "Bronx": (40.8448, -73.8648),
        "Staten Island": (40.5795, -74.1502),
    }

    if borough_choice in borough_centers:
        center_lat, center_lon = borough_centers[borough_choice]
        zoom_level = 11.3   # more zoomed in than "All"
    else:
        center_lat, center_lon = midpoint
        zoom_level = 10.0   # slightly more zoomed out for all-city view


    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom_level,
        bearing=0,
        pitch=35,
    )

    # ---------- Layers ----------

    # FREE basemap TileLayer (actual NYC map)
    basemap = pdk.Layer(
        "TileLayer",
        data=tile_url,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
    )

    # #[MAP] PyDeck geospatial visualization of listings
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[LONGITUDE, LATITUDE]",
        get_radius="RADIUS",
        get_fill_color="COLOR",
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": """
        <div style='padding:6px'>
            <b>{ADDRESS}</b><br>
            Price: <b>${PRICE}</b><br>
            Beds: {BEDS} | Baths: {BATH}<br>
            Sqft: {PROPERTYSQFT}<br>
            <i>{TYPE}</i>
        </div>
        """,
        "style": {"backgroundColor": "rgba(30,30,30,0.8)", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[basemap, layer],  # basemap UNDER points
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,  # we're using TileLayer, not Mapbox styles
    )

    st.pydeck_chart(deck)

    # ---------- Legend + caption ----------

    legend_html = """
    <div style='padding: 12px; background-color: #ffffff; border-radius: 10px; width: 260px;
                box-shadow: 0 0 8px rgba(0,0,0,0.1); font-size: 14px; margin-top: 10px;'>
      <b>Property Type Colors</b><br><br>
      <span style='color: rgb(66,135,245);'>●</span> Condo for sale<br>
      <span style='color: rgb(255,99,71);'>●</span> Co-op for sale<br>
      <span style='color: rgb(255,185,0);'>●</span> Townhouse for sale<br>
      <span style='color: rgb(142,68,173);'>●</span> Multi-family home for sale<br>
      <span style='color: rgb(46,204,113);'>●</span> House for sale<br>
      <span style='color: rgb(150,150,150);'>●</span> Other / Misc<br>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    st.caption(
        "Circle size represents listing price (capped at the 98th percentile and normalized), "
        "and colors represent property type. The basemap styles are free (OpenStreetMap / Carto / Stamen) "
        "so the app can be run anywhere without a Mapbox API key."
    )

# --------------
# Page: About
# --------------

def about_page(df, full_df=None):
    if full_df is None:
        full_df = df
    st.title("About This Project ℹ️")

    st.markdown(
        """
        ### Project Overview  
        The **NYC Housing Market Explorer** is my final project for **CS230 (Python)**.  
        It turns a raw New York housing dataset into an interactive web app that lets users
        explore prices, property types, and neighborhood patterns across the five boroughs.
        """
    )

    # --- NEW: quick dataset metrics row ---
    if not full_df.empty:
        rows, cols = full_df.shape
        unique_boros = full_df["BOROUGH"].dropna().nunique() if "BOROUGH" in full_df.columns else 0
        overall_median = full_df["PRICE"].median() if "PRICE" in full_df.columns else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows in dataset", f"{rows:,}")
        m2.metric("Columns", f"{cols}")
        if not np.isnan(overall_median):
            m3.metric("Overall median price", f"${overall_median:,.0f}")
        else:
            m3.metric("Overall median price", "—")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Technical Highlights")
        st.markdown(
            """
            - Built with **Python + Streamlit** as a single-page web app  
            - Uses **Pandas** for:
              - Data cleaning and feature engineering  
              - Creating a `BOROUGH` field from messy location strings  
              - Calculating **price per square foot** and other derived metrics  
            - Interactive filters for:
              - Borough, property type, price range  
              - Bedrooms and square footage  
            - **Caching** (`@st.cache_data`) to speed up data loading
            """
        )

    with col2:
        st.subheader("Data Visualization")
        st.markdown(
            """
            - **Plotly** bar chart of average price by property type  
            - **Plotly** scatterplot of price vs square footage with log scaling  
            - **Pivot tables** to compare median prices by borough & property type  
            - **PyDeck** map with:
              - Free open-source basemap tiles (no API key required)  
              - Bubble size scaled by listing price  
              - Color encoded by property type  
              - Auto-zoom to each NYC borough
            """
        )

    st.markdown("---")

    st.subheader("How to Use This App")  # NEW
    st.markdown(                           # NEW
        """
        1. Start on the **Overview** tab to see key metrics for your current filters.  
        2. Use the **Charts** tab to dig into price distributions and price vs. size relationships.  
        3. Switch to the **Map** tab to see where listings cluster geographically within NYC.  
        4. Download the filtered data as CSV if you want to do additional analysis in Excel or Python.
        """
    )

    st.subheader("What This Demonstrates")
    st.markdown(
        """
        - Ability to work with a **real, messy dataset** and normalize it for analysis  
        - Experience building a **data product**, not just a script:
          - user-friendly layout  
          - responsive filters  
          - explanations and legends for non-technical users  
        - Practical skills in **Python for data analysis**, **data visualization**, and
          **lightweight web development** that transfer directly to:
          - analytics & BI roles  
          - product / business intelligence  
          - fintech & real-estate analytics
        """
    )

    st.subheader("Data Quality & Cleaning Notes")
    st.markdown(
        """
        Real-world housing data is messy, so a few cleaning steps were applied before analysis:
        - **Prices**: stripped out commas, symbols, and strange Unicode characters, then converted to numeric.
        - **Square footage**: cleaned and enforced as numeric for price-per-sqft calculations.
        - **Bathrooms**: rounded to the nearest half-bath (e.g., 2.37 → 2.5) so counts make sense to users.
        - **Boroughs**: normalized from various text forms (e.g., *“Bronx County”*, *“Kings County”*) 
          into the five standard NYC boroughs.
        - **Visualizations**: some charts cap ultra-luxury outliers so typical listings are easier to compare,
          but the underlying data is preserved.
        """
    )

    st.info(
        "If you’re viewing this as an employer or reviewer, imagine plugging in your own "
        "internal real-estate or pricing data to power a similar interactive dashboard."
    )


# --------------
# Main Navigation
# --------------

def main():
    # ---- Top Header UI ----
    st.markdown("""
          <style>
              .top-header {
                  background-color: #f8f9fa;
                  padding: 18px 28px;
                  font-size: 30px;
                  font-weight: 700;
                  border-bottom: 1px solid #e2e2e2;
                  text-align: center;
              }
          </style>
          <div class="top-header">🏙️ NYC Housing Market Explorer</div>
      """, unsafe_allow_html=True)
    # ---- Sidebar Styling ----
    st.markdown("""
           <style>
               section[data-testid="stSidebar"] {
                   background-color: #f0f2f6;
                   padding-top: 20px;
               }
               .sidebar-title {
                   font-size: 20px;
                   font-weight: 700;
                   margin-top: 20px;
               }
           </style>
       """, unsafe_allow_html=True)
    # #[ST3] Sidebar branding + nav
    st.sidebar.image(
        "https://static.streamlit.io/examples/dice.jpg",
        width=80,
    )
    st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Go to:",
        ("Overview", "Charts", "Map", "About"),
    )

    # #[ITERLOOP]
    for _ in range(1):
        if page == "Overview":
            overview_page(df_all)
        elif page == "Charts":
            charts_page(df_all)
        elif page == "Map":
            map_page(df_all)
        else:
            about_page(df_all)


if __name__ == "__main__":
    main()