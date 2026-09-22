import streamlit as st
import community_db
import pandas as pd
import numpy as np
import joblib
import textwrap
import base64
import requests
from pathlib import Path

from club_assets import CLUBS
from match_data import (
    get_latest_match,
    get_starting_xi,
    format_starting_xi,
    normalize_club_name,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Premier League Analytics Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "premier_league_players.csv"
MODEL_PATH = BASE_DIR / "models" / "best_player_value_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    numeric_columns = [
        "age",
        "appearances",
        "minutes",
        "goals",
        "assists",
        "market_value",
        "goals_per_90",
        "assists_per_90",
        "minutes_per_appearance"
    ]

    for col in numeric_columns:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


df = load_data()
model = load_model()


# ============================================================
# HELPERS
# ============================================================

def money(value):

    if pd.isna(value):
        return "€0"

    value = float(value)

    if value >= 1_000_000_000:
        return f"€{value / 1_000_000_000:.1f}B"

    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"€{value / 1_000:.0f}K"

    return f"€{value:,.0f}"


def html(content):

    st.html(
        textwrap.dedent(content)
    )


# ============================================================
# EMBED REMOTE CLUB LOGOS
# ============================================================

@st.cache_data
def get_logo_data(url):
    if not url:
        return ""

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "svg" in content_type or url.lower().endswith(".svg"):
            mime_type = "image/svg+xml"
        elif "webp" in content_type or url.lower().endswith(".webp"):
            mime_type = "image/webp"
        else:
            mime_type = "image/png"

        encoded = base64.b64encode(response.content).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    except Exception:
        return ""

def club_info(club):

    if club in CLUBS:
        info = CLUBS[club].copy()

        # Compatibility with the existing app
        info.setdefault("short_name", info.get("short", club))
        info.setdefault("code", info.get("short", ""))
        info.setdefault("color", info.get("primary", "#42556a"))

        return info

    return {
        "short_name": club,
        "code": "",
        "color": "#42556a",
        "primary": "#42556a",
        "secondary": "#FFFFFF",
        "accent": "#FFFFFF",
        "background": "#080808",
        "muted": "#AAAAAA",
        "nickname": "",
        "logo": ""
    }

def get_photo(row):

    if "image_url" not in row:
        return ""

    photo = row.get(
        "image_url",
        ""
    )

    if pd.isna(photo):
        return ""

    return str(photo)


def navigate(page, **kwargs):

    st.session_state["page"] = page

    for key, value in kwargs.items():
        st.session_state[key] = value

    st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "selected_club" not in st.session_state:
    st.session_state["selected_club"] = None

if "selected_player" not in st.session_state:
    st.session_state["selected_player"] = None

if "community_posts" not in st.session_state:
    st.session_state["community_posts"] = []


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {

    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(30,64,175,0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(124,58,237,0.12),
            transparent 28%
        ),
        #070b12;

    color: #f5f7fb;
}

.block-container {

    max-width: 1450px;

    padding-top: 1.2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   NAVBAR
   ========================================================= */

.navbar {

    display: flex;
    align-items: center;

    padding: 14px 20px;

    margin-bottom: 25px;

    background:
        rgba(15,23,42,0.82);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    backdrop-filter: blur(16px);
}

.brand {

    display: flex;
    align-items: center;

    gap: 12px;
}

.brand-ball {

    width: 40px;
    height: 40px;

    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );

    font-size: 20px;
}

.brand-title {

    font-size: 18px;

    font-weight: 800;

    letter-spacing: -0.5px;
}

.brand-sub {

    font-size: 11px;

    color: #8f9bad;

    margin-top: 2px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    overflow: hidden;

    padding: 55px 48px;

    border-radius: 28px;

    background:

        radial-gradient(
            circle at 80% 30%,
            rgba(124,58,237,0.34),
            transparent 35%
        ),

        radial-gradient(
            circle at 15% 80%,
            rgba(37,99,235,0.22),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #101827,
            #080d16
        );

    border:
        1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.28);
}

.hero-kicker {

    color: #9f7aea;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 2px;

    text-transform: uppercase;

    margin-bottom: 15px;
}

.hero h1 {

    font-size:
        clamp(40px, 5vw, 72px);

    line-height: 0.98;

    letter-spacing: -3px;

    margin: 0;

    max-width: 850px;
}

.hero h1 span {

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #a78bfa,
            #60a5fa
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero p {

    color: #aab4c4;

    font-size: 16px;

    line-height: 1.7;

    max-width: 720px;

    margin-top: 22px;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {

    font-size: 26px;

    font-weight: 800;

    letter-spacing: -1px;

    margin-top: 40px;

    margin-bottom: 5px;
}

.section-subtitle {

    color: #7f8b9e;

    font-size: 14px;

    margin-bottom: 20px;
}


/* =========================================================
   KPI
   ========================================================= */

.kpi {

    padding: 22px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.78);

    border:
        1px solid rgba(255,255,255,0.07);

    min-height: 120px;
}

.kpi-label {

    color: #8490a3;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 1px;

    font-weight: 700;
}

.kpi-value {

    font-size: 30px;

    font-weight: 900;

    margin-top: 8px;

    letter-spacing: -1px;
}

.kpi-small {

    color: #718096;

    font-size: 12px;

    margin-top: 4px;
}


/* =========================================================
   PLAYER CARD
   ========================================================= */

.player-card {

    background:
        linear-gradient(
            145deg,
            rgba(20,29,45,0.95),
            rgba(10,15,24,0.95)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius: 22px;

    overflow: hidden;
}

.player-photo {

    width: 100%;

    height: 170px;

    object-fit: contain;

    object-position: center;

    display: block;

    background: #111827;
}

.player-placeholder {

    width: 100%;

    height: 170px;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #111827;

    font-size: 60px;
}

.player-info {

    padding: 18px;
}

.player-name {

    font-size: 18px;

    font-weight: 800;
}

.player-club {

    color: #8490a3;

    font-size: 12px;

    margin-top: 4px;
}

.player-value {

    margin-top: 15px;

    font-size: 21px;

    font-weight: 900;
}

.player-meta {

    color: #718096;

    font-size: 12px;

    margin-top: 7px;
}


/* =========================================================
   CLUB CARD
   ========================================================= */

.club-card {

    padding: 24px;

    min-height: 220px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(21,30,47,0.95),
            rgba(10,15,24,0.95)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    transition:
        transform .2s ease,
        border-color .2s ease,
        box-shadow .2s ease;

    text-align: center;
}

.club-card:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(255,255,255,0.18);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.25);
}

.club-logo {

    width: 72px;

    height: 72px;

    object-fit: contain;

    margin-bottom: 13px;
}

.club-name {

    font-size: 17px;

    font-weight: 800;
}

.club-code {

    display: inline-block;

    margin-top: 8px;

    padding: 4px 9px;

    border-radius: 999px;

    font-size: 10px;

    font-weight: 800;

    background:
        rgba(255,255,255,0.06);

    color: #aab4c4;
}

.club-meta {

    color: #718096;

    font-size: 12px;

    margin-top: 12px;
}


/* =========================================================
   CLUB HERO
   ========================================================= */

.club-hero {

    display: flex;

    align-items: center;

    gap: 30px;

    padding: 35px;

    border-radius: 26px;

    background:
        radial-gradient(
            circle at 80% 50%,
            var(--club-color),
            transparent 35%
        ),
        #101827;

    border:
        1px solid rgba(255,255,255,0.08);
}

.club-logo-large {

    width: 110px;

    height: 110px;

    object-fit: contain;

    flex-shrink: 0;
}

.club-kicker {

    color: #8f9bad;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 2px;

    font-weight: 800;
}

.club-title {

    font-size: 43px;

    font-weight: 900;

    letter-spacing: -2px;

    margin-top: 5px;
}

.club-description {

    color: #8490a3;

    margin-top: 8px;
}


/* =========================================================
   INFO
   ========================================================= */

.info-box {

    padding: 20px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.72);

    border:
        1px solid rgba(255,255,255,0.07);
}

.info-label {

    color: #718096;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.info-value {

    font-size: 20px;

    font-weight: 800;

    margin-top: 5px;
}


/* =========================================================
   MODEL
   ========================================================= */

.model-box {

    padding: 28px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,0.18),
            rgba(37,99,235,0.10)
        );

    border:
        1px solid rgba(167,139,250,0.22);
}

.model-title {

    font-size: 13px;

    color: #a78bfa;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 1.5px;
}

.model-value {

    font-size: 42px;

    font-weight: 900;

    margin-top: 8px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {

    border-radius: 12px !important;

    border:
        1px solid rgba(255,255,255,0.08) !important;

    background:
        rgba(255,255,255,0.05) !important;

    color: white !important;

    font-weight: 700 !important;

    transition:
        all .2s ease !important;
}

.stButton > button:hover {

    border-color:
        rgba(167,139,250,0.5) !important;

    background:
        rgba(124,58,237,0.15) !important;
}


/* =========================================================
   SELECT
   ========================================================= */

div[data-baseweb="select"] > div {

    background:
        rgba(15,23,42,0.85) !important;

    border-color:
        rgba(255,255,255,0.08) !important;

    border-radius: 12px !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    margin-top: 70px;

    padding-top: 25px;

    border-top:
        1px solid rgba(255,255,255,0.06);

    color: #596579;

    font-size: 12px;

    text-align: center;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(
    [3.0, 1, 1, 1, 1, 1]
)

with nav1:

    html(
        """
        <div class="navbar">

            <div class="brand">

                <div class="brand-ball">
                    ⚽
                </div>

                <div>

                    <div class="brand-title">
                        Premier League Analytics Lab
                    </div>

                    <div class="brand-sub">
                        Football Intelligence Platform
                    </div>

                </div>

            </div>

        </div>
        """
    )

with nav2:

    if st.button(
        "Home",
        use_container_width=True
    ):
        navigate("Home")

with nav3:

    if st.button(
        "Clubs",
        use_container_width=True
    ):
        navigate("Clubs")

with nav4:

    if st.button(
        "Players",
        use_container_width=True
    ):
        navigate("Players")

with nav5:

    if st.button(
        "Analytics",
        use_container_width=True
    ):
        navigate("Model")

with nav6:

    if st.button(
        "Community",
        use_container_width=True
    ):
        navigate("Community")


# ============================================================
# HOME
# ============================================================

# ============================================================
# HOME
# ============================================================

def home_page():

    total_players = len(df)

    total_clubs = df["club"].nunique()

    total_value = df["market_value"].sum()

    highest_value = df["market_value"].max()

    top_player = df.loc[
        df["market_value"].idxmax()
    ]["name"]

    # ========================================================
    # HERO
    # ========================================================

    html(
        """
        <div class="hero">

            <div class="hero-kicker">
                PREMIER LEAGUE ANALYTICS LAB
            </div>

            <h1>
                Football data.<br>
                <span>Turned into intelligence.</span>
            </h1>

            <p>
                Explore players, clubs, performance, market values
                and machine-learning insights through one football
                analytics platform.
            </p>

        </div>
        """
    )

    # ========================================================
    # QUICK ACTIONS
    # ========================================================

    action1, action2, action3 = st.columns(3)

    with action1:

        if st.button(
            "Explore Clubs",
            use_container_width=True,
            key="home_explore_clubs"
        ):

            navigate("Clubs")

    with action2:

        if st.button(
            "Explore Players",
            use_container_width=True,
            key="home_explore_players"
        ):

            navigate("Players")

    with action3:

        if st.button(
            "AI Valuation",
            use_container_width=True,
            key="home_ai_valuation"
        ):

            navigate("Model")

    # ========================================================
    # LEAGUE OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">League Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Premier League player and club intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Players
                </div>

                <div class="kpi-value">
                    {total_players}
                </div>

                <div class="kpi-small">
                    Player records
                </div>

            </div>
            """
        )

    with c2:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Clubs
                </div>

                <div class="kpi-value">
                    {total_clubs}
                </div>

                <div class="kpi-small">
                    Clubs in dataset
                </div>

            </div>
            """
        )

    with c3:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Squad Value
                </div>

                <div class="kpi-value">
                    {money(total_value)}
                </div>

                <div class="kpi-small">
                    Combined listed value
                </div>

            </div>
            """
        )

    with c4:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Highest Value
                </div>

                <div class="kpi-value">
                    {money(highest_value)}
                </div>

                <div class="kpi-small">
                    {top_player}
                </div>

            </div>
            """
        )

    # ========================================================
    # MARKET LEADERS
    # ========================================================

    st.markdown(
        '<div class="section-title">Market Leaders</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Players with the highest listed market values'
        '</div>',
        unsafe_allow_html=True
    )

    top_players = (
        df
        .sort_values(
            "market_value",
            ascending=False
        )
        .head(3)
        .reset_index(drop=True)
    )

    cols = st.columns(3)

    for i, row in top_players.iterrows():

        with cols[i]:

            photo = get_photo(row)

            if photo:

                image_html = f"""
                <img
                    class="player-photo"
                    src="{photo}"
                    alt="{row['name']}"
                >
                """

            else:

                image_html = """
                <div class="player-placeholder">
                    ⚽
                </div>
                """

            age = (
                int(row["age"])
                if pd.notna(row["age"])
                else "-"
            )

            position = row.get(
                "position",
                "Player"
            )

            html(
                f"""
                <div class="player-card">

                    {image_html}

                    <div class="player-info">

                        <div class="player-name">
                            {row['name']}
                        </div>

                        <div class="player-club">
                            {row['club']}
                        </div>

                        <div class="player-value">
                            {money(row['market_value'])}
                        </div>

                        <div class="player-meta">
                            {position} · {age} years
                        </div>

                    </div>

                </div>
                """
            )

            if st.button(
                f"View {row['name']}",
                key=f"home_player_{row['name']}",
                use_container_width=True
            ):

                navigate(
                    "Player",
                    selected_player=row["name"]
                )

    # ========================================================
    # EXPLORE CLUBS
    # ========================================================

    st.markdown(
        '<div class="section-title">Explore Clubs</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore squads, players and club analytics'
        '</div>',
        unsafe_allow_html=True
    )

    clubs = sorted(
        df["club"]
        .dropna()
        .unique()
    )

    for start in range(
        0,
        len(clubs),
        4
    ):

        row_clubs = clubs[
            start:start + 4
        ]

        club_cols = st.columns(4)

        for col, club in zip(
            club_cols,
            row_clubs
        ):

            info = club_info(club)

            club_players = df[
                df["club"] == club
            ]

            club_value = club_players[
                "market_value"
            ].sum()

            logo = info.get(
                "logo",
                ""
            )

            logo_data = get_logo_data(
                logo
            )

            with col:

                if logo_data:

                    logo_html = f"""
                    <img
                        class="club-logo"
                        src="{logo_data}"
                        alt="{club}"
                    >
                    """

                else:

                    logo_html = """
                    <div style="
                        height:72px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:50px;
                    ">
                        ⚽
                    </div>
                    """

                html(
                    f"""
                    <div class="club-card">

                        {logo_html}

                        <div class="club-name">
                            {info.get(
                                'short_name',
                                club
                            )}
                        </div>

                        <div class="club-code">
                            {info.get(
                                'code',
                                ''
                            )}
                        </div>

                        <div class="club-meta">
                            {len(club_players)}
                            players ·
                            {money(club_value)}
                        </div>

                    </div>
                    """
                )

                if st.button(
                    "Open Club",
                    key=f"home_club_{club}",
                    use_container_width=True
                ):

                    navigate(
                        "Club",
                        selected_club=club
                    )

    # ========================================================
    # ANALYTICS LAB
    # ========================================================

    st.markdown(
        '<div class="section-title">Analytics Lab</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Turn football statistics into actionable insights'
        '</div>',
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        html(
            """
            <div class="club-card">

                <div class="club-name">
                    AI Valuation
                </div>

                <div class="club-meta">
                    Estimate player market values
                    using the trained machine-learning model.
                </div>

            </div>
            """
        )

        if st.button(
            "Open Model",
            key="home_analytics_model",
            use_container_width=True
        ):

            navigate("Model")

    with a2:

        html(
            """
            <div class="club-card">

                <div class="club-name">
                    Player Analysis
                </div>

                <div class="club-meta">
                    Compare player performance,
                    appearances, goals and assists.
                </div>

            </div>
            """
        )

        if st.button(
            "Explore Players",
            key="home_analytics_players",
            use_container_width=True
        ):

            navigate("Players")

    with a3:

        html(
            """
            <div class="club-card">

                <div class="club-name">
                    Club Intelligence
                </div>

                <div class="club-meta">
                    Explore squad size, market value
                    and club-level player data.
                </div>

            </div>
            """
        )

        if st.button(
            "Explore Clubs",
            key="home_analytics_clubs",
            use_container_width=True
        ):

            navigate("Clubs")

    # ========================================================
    # FAN COMMUNITY
    # ========================================================

    st.markdown(
        '<div class="section-title">Fan Community</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your future home for club discussions, matchday talk '
        'and football analytics conversations'
        '</div>',
        unsafe_allow_html=True
    )

    html(
        """
        <div class="hero">

            <div class="hero-kicker">
                COMING SOON
            </div>

            <h1>
                Football is more than numbers.
            </h1>

            <p>
                Choose your club, connect with fellow supporters,
                discuss matches, transfers and analytics, and
                build a football community around the data.
            </p>

        </div>
        """
    )

    # ========================================================
    # FOOTER
    # ========================================================

    html(
        """
        <div class="footer">
            Premier League Analytics Lab ·
            Football Analytics ·
            Machine Learning ·
            Fan Community
        </div>
        """
    )


# ============================================================
# CLUBS
# ============================================================

def clubs_page():

    st.markdown(
        '<div class="section-title">'
        'Premier League Clubs'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore clubs, squad sizes and market values from the 2024/25 dataset'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # SEARCH
    # ========================================================

    search = st.text_input(
        "Search clubs",
        placeholder="Search Arsenal, Liverpool, Manchester..."
    )

    clubs = sorted(
        df["club"]
        .dropna()
        .unique()
    )

    if search:

        clubs = [
            club
            for club in clubs
            if search.lower() in club.lower()
        ]

    st.write("")

    if not clubs:

        st.info(
            "No clubs found. Try another search."
        )

        return

    st.markdown(
        f"""
        <div style="
            color:#7f8b9e;
            font-size:13px;
            margin-bottom:18px;
        ">
            Showing <strong>{len(clubs)}</strong> clubs
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # CLUB GRID
    # ========================================================

    for start in range(
        0,
        len(clubs),
        4
    ):

        row_clubs = clubs[
            start:start + 4
        ]

        cols = st.columns(4)

        for col, club in zip(
            cols,
            row_clubs
        ):

            info = club_info(club)

            club_df = df[
                df["club"] == club
            ]

            club_value = club_df[
                "market_value"
            ].sum()

            logo = info.get(
                "logo",
                ""
            )

            logo_data = get_logo_data(
                logo
            )

            if logo_data:

                logo_html = f"""
                <img
                    src="{logo_data}"
                    alt="{club}"
                    style="
                        width:82px;
                        height:82px;
                        object-fit:contain;
                        display:block;
                        margin:0 auto 18px auto;
                    "
                >
                """

            else:

                logo_html = """
                <div style="
                    width:82px;
                    height:82px;
                    margin:0 auto 18px auto;
                    border-radius:50%;
                    background:#111827;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:38px;
                ">
                    ⚽
                </div>
                """

            short_name = info.get(
                "short_name",
                club
            )

            code = info.get(
                "code",
                ""
            )

            with col:

                html(
                    f"""
                    <div style="
                        min-height:260px;
                        padding:24px 18px;
                        margin-bottom:12px;
                        border-radius:22px;
                        background:
                            linear-gradient(
                                145deg,
                                rgba(255,255,255,0.055),
                                rgba(255,255,255,0.018)
                            );
                        border:1px solid
                            rgba(255,255,255,0.08);
                        text-align:center;
                        transition:0.2s ease;
                    ">

                        {logo_html}

                        <div style="
                            font-size:19px;
                            font-weight:850;
                            color:#f5f7fb;
                            line-height:1.2;
                        ">
                            {short_name}
                        </div>

                        <div style="
                            margin-top:5px;
                            color:#737f92;
                            font-size:11px;
                            font-weight:700;
                            letter-spacing:1.5px;
                            text-transform:uppercase;
                        ">
                            {code}
                        </div>

                        <div style="
                            margin-top:20px;
                            padding-top:16px;
                            border-top:1px solid
                                rgba(255,255,255,0.07);
                        ">

                            <div style="
                                color:#7f8b9e;
                                font-size:11px;
                                text-transform:uppercase;
                                letter-spacing:1px;
                            ">
                                Squad
                            </div>

                            <div style="
                                margin-top:4px;
                                color:#f1f4f8;
                                font-size:18px;
                                font-weight:800;
                            ">
                                {len(club_df)} players
                            </div>

                            <div style="
                                margin-top:9px;
                                color:#aab4c4;
                                font-size:13px;
                            ">
                                Squad value
                            </div>

                            <div style="
                                margin-top:2px;
                                color:#ffffff;
                                font-size:16px;
                                font-weight:800;
                            ">
                                {money(club_value)}
                            </div>

                        </div>

                    </div>
                    """
                )

                if st.button(
                    "View Club",
                    key=f"club_page_{club}",
                    use_container_width=True
                ):

                    navigate(
                        "Club",
                        selected_club=club
                    )

# ============================================================
# CLUB PAGE
# ============================================================

def club_page(club):

    if not club or club not in df["club"].unique():
        st.error("Club not found.")
        if st.button("Back to Clubs"):
            navigate("Clubs")
        return

    # ========================================================
    # CLUB DATA
    # ========================================================

    info = club_info(club)

    club_df = (
        df[df["club"] == club]
        .sort_values("market_value", ascending=False)
        .reset_index(drop=True)
    )

    if club_df.empty:
        st.warning("No squad data available for this club.")
        return

    total_value = club_df["market_value"].sum()
    avg_age = club_df["age"].mean()
    top_player = club_df.iloc[0]["name"]
    top_value = club_df.iloc[0]["market_value"]

    logo = info.get("logo", "")
    logo_data = get_logo_data(logo)

    # ========================================================
    # THEME
    # ========================================================

    primary = info.get("primary", info.get("color", "#DA291C"))
    secondary = info.get("secondary", "#FFFFFF")
    accent = info.get("accent", "#FFFFFF")
    background = info.get("background", "#080808")
    muted = info.get("muted", "#A9A9A9")
    nickname = info.get("nickname", "")
    short_name = info.get("short_name", info.get("short", club))
    code = info.get("code", info.get("short", ""))

    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button("← Back to Clubs"):
        navigate("Clubs")

    # ========================================================
    # HERO
    # ========================================================

    if logo_data:
        logo_html = f"""
        <img
            src="{logo_data}"
            alt="{club}"
            style="
                width:150px;
                height:150px;
                object-fit:contain;
                filter:drop-shadow(0 15px 30px rgba(0,0,0,0.45));
            "
        >
        """
    else:
        logo_html = """
        <div style="
            width:150px;
            height:150px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:64px;
        ">
            ⚽
        </div>
        """

    html(
        f"""
        <div style="
            position:relative;
            overflow:hidden;
            margin-top:10px;
            padding:42px;
            border-radius:30px;
            background:
                radial-gradient(
                    circle at 88% 20%,
                    {primary}55 0%,
                    transparent 34%
                ),
                radial-gradient(
                    circle at 70% 100%,
                    {secondary}12 0%,
                    transparent 32%
                ),
                linear-gradient(
                    135deg,
                    {background},
                    #090b10
                );
            border:1px solid {primary}45;
            box-shadow:
                0 25px 70px rgba(0,0,0,0.35),
                inset 0 1px 0 rgba(255,255,255,0.04);
        ">

            <div style="
                position:absolute;
                top:-100px;
                right:-80px;
                width:320px;
                height:320px;
                border-radius:50%;
                background:{primary};
                opacity:0.08;
                filter:blur(30px);
            "></div>

            <div style="
                position:relative;
                display:flex;
                align-items:center;
                gap:32px;
                flex-wrap:wrap;
            ">

                {logo_html}

                <div style="min-width:260px;">

                    <div style="
                        color:{primary};
                        font-size:12px;
                        font-weight:900;
                        letter-spacing:3px;
                        text-transform:uppercase;
                        margin-bottom:10px;
                    ">
                        PREMIER LEAGUE · CLUB PROFILE
                    </div>

                    <div style="
                        color:#ffffff;
                        font-size:46px;
                        line-height:0.98;
                        font-weight:950;
                        letter-spacing:-2px;
                        text-transform:uppercase;
                    ">
                        {club}
                    </div>

                    <div style="
                        margin-top:12px;
                        display:flex;
                        align-items:center;
                        gap:10px;
                        flex-wrap:wrap;
                    ">

                        <span style="
                            color:{secondary};
                            font-size:15px;
                            font-weight:800;
                        ">
                            {nickname}
                        </span>

                        <span style="
                            color:{muted};
                            font-size:13px;
                        ">
                            •
                        </span>

                        <span style="
                            color:{muted};
                            font-size:13px;
                            font-weight:700;
                            letter-spacing:1px;
                        ">
                            {code}
                        </span>

                    </div>

                    <div style="
                        margin-top:16px;
                        color:#b7beca;
                        font-size:14px;
                        line-height:1.6;
                        max-width:650px;
                    ">
                        Squad intelligence, player valuations and performance
                        data for {club}.
                    </div>

                </div>

            </div>

            <div style="
                position:relative;
                margin-top:34px;
                height:4px;
                width:100%;
                border-radius:10px;
                background:
                    linear-gradient(
                        90deg,
                        {primary},
                        {secondary},
                        transparent
                    );
                opacity:0.85;
            "></div>

        </div>
        """
    )

    # ========================================================
    # CLUB SNAPSHOT
    # ========================================================

    html(
        f"""
        <div style="
            margin-top:34px;
            margin-bottom:14px;
        ">
            <div style="
                color:{primary};
                font-size:11px;
                font-weight:900;
                letter-spacing:3px;
                text-transform:uppercase;
            ">
                CLUB INTELLIGENCE
            </div>

            <div style="
                color:#ffffff;
                font-size:30px;
                font-weight:950;
                letter-spacing:-1px;
                text-transform:uppercase;
                margin-top:4px;
            ">
                Squad Snapshot
            </div>
        </div>
        """
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    k1, k2, k3, k4 = st.columns(4)

    kpis = [
        (k1, "SQUAD SIZE", f"{len(club_df)}", "players"),
        (k2, "SQUAD VALUE", money(total_value), "listed value"),
        (k3, "AVERAGE AGE", f"{avg_age:.1f}", "years"),
        (k4, "TOP VALUE", money(top_value), top_player),
    ]

    for col, label, value, small in kpis:
        with col:
            html(
                f"""
                <div style="
                    padding:22px;
                    min-height:128px;
                    border-radius:20px;
                    background:
                        linear-gradient(
                            145deg,
                            {primary}14,
                            rgba(255,255,255,0.025)
                        );
                    border:1px solid {primary}32;
                    box-shadow:0 10px 30px rgba(0,0,0,0.18);
                ">

                    <div style="
                        color:{muted};
                        font-size:10px;
                        font-weight:900;
                        letter-spacing:1.8px;
                    ">
                        {label}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#ffffff;
                        font-size:27px;
                        font-weight:950;
                        letter-spacing:-1px;
                    ">
                        {value}
                    </div>

                    <div style="
                        margin-top:5px;
                        color:{primary};
                        font-size:11px;
                        font-weight:800;
                        white-space:nowrap;
                        overflow:hidden;
                        text-overflow:ellipsis;
                    ">
                        {small}
                    </div>

                </div>
                """
            )

    # ========================================================
    # STAR PLAYER
    # ========================================================

    html(
        f"""
        <div style="
            margin-top:42px;
            margin-bottom:14px;
        ">
            <div style="
                color:{primary};
                font-size:11px;
                font-weight:900;
                letter-spacing:3px;
                text-transform:uppercase;
            ">
                KEY PLAYER
            </div>

            <div style="
                color:#ffffff;
                font-size:30px;
                font-weight:950;
                letter-spacing:-1px;
                text-transform:uppercase;
                margin-top:4px;
            ">
                Star Player
            </div>
        </div>
        """
    )

    leader_row = club_df.iloc[0]
    leader_photo = get_photo(leader_row)

    if leader_photo:
        leader_image = f"""
        <img
            src="{leader_photo}"
            alt="{top_player}"
            style="
                width:140px;
                height:165px;
                object-fit:contain;
                border-radius:18px;
                background:rgba(0,0,0,0.28);
            "
        >
        """
    else:
        leader_image = """
        <div style="
            width:140px;
            height:165px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:18px;
            background:rgba(0,0,0,0.3);
            font-size:52px;
        ">
            ⚽
        </div>
        """

    position = leader_row.get("position", "Player")
    appearances = int(leader_row.get("appearances", 0))
    goals = int(leader_row.get("goals", 0))
    assists = int(leader_row.get("assists", 0))

    html(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:26px;
            flex-wrap:wrap;
            padding:25px;
            border-radius:24px;
            background:
                linear-gradient(
                    120deg,
                    {primary}20,
                    rgba(255,255,255,0.025)
                );
            border:1px solid {primary}38;
            position:relative;
            overflow:hidden;
        ">

            <div style="
                position:absolute;
                right:-50px;
                top:-70px;
                width:220px;
                height:220px;
                border-radius:50%;
                background:{primary};
                opacity:0.08;
                filter:blur(15px);
            "></div>

            {leader_image}

            <div style="position:relative;">

                <div style="
                    color:{primary};
                    font-size:10px;
                    font-weight:900;
                    letter-spacing:2px;
                    text-transform:uppercase;
                ">
                    HIGHEST LISTED MARKET VALUE
                </div>

                <div style="
                    color:#ffffff;
                    font-size:32px;
                    font-weight:950;
                    margin-top:6px;
                    letter-spacing:-1px;
                ">
                    {top_player}
                </div>

                <div style="
                    color:{muted};
                    font-size:14px;
                    margin-top:6px;
                ">
                    {position}
                </div>

                <div style="
                    display:flex;
                    gap:24px;
                    flex-wrap:wrap;
                    margin-top:18px;
                ">

                    <div>
                        <div style="
                            color:#ffffff;
                            font-size:20px;
                            font-weight:900;
                        ">
                            {money(top_value)}
                        </div>
                        <div style="
                            color:{muted};
                            font-size:10px;
                            text-transform:uppercase;
                            letter-spacing:1px;
                        ">
                            Market Value
                        </div>
                    </div>

                    <div>
                        <div style="
                            color:#ffffff;
                            font-size:20px;
                            font-weight:900;
                        ">
                            {appearances}
                        </div>
                        <div style="
                            color:{muted};
                            font-size:10px;
                            text-transform:uppercase;
                            letter-spacing:1px;
                        ">
                            Apps
                        </div>
                    </div>

                    <div>
                        <div style="
                            color:#ffffff;
                            font-size:20px;
                            font-weight:900;
                        ">
                            {goals}
                        </div>
                        <div style="
                            color:{muted};
                            font-size:10px;
                            text-transform:uppercase;
                            letter-spacing:1px;
                        ">
                            Goals
                        </div>
                    </div>

                    <div>
                        <div style="
                            color:#ffffff;
                            font-size:20px;
                            font-weight:900;
                        ">
                            {assists}
                        </div>
                        <div style="
                            color:{muted};
                            font-size:10px;
                            text-transform:uppercase;
                            letter-spacing:1px;
                        ">
                            Assists
                        </div>
                    </div>

                </div>

            </div>

        </div>
        """
    )

    # ========================================================
    # SQUAD
    # ========================================================

    html(
        f"""
        <div style="
            margin-top:42px;
            margin-bottom:6px;
        ">
            <div style="
                color:{primary};
                font-size:11px;
                font-weight:900;
                letter-spacing:3px;
                text-transform:uppercase;
            ">
                THE SQUAD
            </div>

            <div style="
                color:#ffffff;
                font-size:30px;
                font-weight:950;
                letter-spacing:-1px;
                text-transform:uppercase;
                margin-top:4px;
            ">
                First Team
            </div>

            <div style="
                color:{muted};
                font-size:13px;
                margin-top:5px;
            ">
                Players ranked by listed market value.
            </div>
        </div>
        """
    )

    # ========================================================
    # PLAYER CARDS
    # ========================================================

    for start in range(0, len(club_df), 4):

        rows = club_df.iloc[start:start + 4]
        cols = st.columns(4)

        for col, (_, row) in zip(cols, rows.iterrows()):

            photo = get_photo(row)

            with col:

                if photo:
                    image_html = f"""
                    <img
                        src="{photo}"
                        alt="{row['name']}"
                        style="
                            width:100%;
                            height:190px;
                            object-fit:contain;
                            background:
                                linear-gradient(
                                    145deg,
                                    {primary}12,
                                    rgba(255,255,255,0.025)
                                );
                            border-radius:16px;
                        "
                    >
                    """
                else:
                    image_html = f"""
                    <div style="
                        width:100%;
                        height:190px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:
                            linear-gradient(
                                145deg,
                                {primary}12,
                                rgba(255,255,255,0.025)
                            );
                        border-radius:16px;
                        font-size:48px;
                    ">
                        ⚽
                    </div>
                    """

                player_position = row.get("position", "Player")
                player_value = money(row["market_value"])
                player_apps = int(row.get("appearances", 0))
                player_goals = int(row.get("goals", 0))

                html(
                    f"""
                    <div style="
                        margin-top:12px;
                        border-radius:20px;
                        overflow:hidden;
                        background:
                            linear-gradient(
                                145deg,
                                rgba(255,255,255,0.045),
                                rgba(255,255,255,0.018)
                            );
                        border:1px solid rgba(255,255,255,0.07);
                        transition:all 0.2s ease;
                    ">

                        {image_html}

                        <div style="padding:16px;">

                            <div style="
                                color:#ffffff;
                                font-size:16px;
                                font-weight:900;
                                line-height:1.15;
                                min-height:37px;
                            ">
                                {row['name']}
                            </div>

                            <div style="
                                margin-top:6px;
                                color:{primary};
                                font-size:10px;
                                font-weight:900;
                                letter-spacing:1.3px;
                                text-transform:uppercase;
                            ">
                                {player_position}
                            </div>

                            <div style="
                                margin-top:13px;
                                color:#ffffff;
                                font-size:20px;
                                font-weight:950;
                            ">
                                {player_value}
                            </div>

                            <div style="
                                margin-top:5px;
                                color:{muted};
                                font-size:11px;
                            ">
                                {player_apps} appearances
                                &nbsp;·&nbsp;
                                {player_goals} goals
                            </div>

                        </div>

                    </div>
                    """
                )

                if st.button(
                    "View Player",
                    key=f"squad_{club}_{row['name']}",
                    use_container_width=True
                ):
                    navigate(
                        "Player",
                        selected_player=row["name"]
                    )

    # ========================================================
    # LATEST XI — DATA LAYER COMING NEXT
    # ========================================================
    # ========================================================
    # LATEST STARTING XI
    # ========================================================

    # ========================================================
    # LATEST STARTING XI — TACTICAL MATCHDAY VIEW
    # ========================================================

    official_club = normalize_club_name(club)
    latest_match = get_latest_match(official_club)

    if latest_match:

        home_team = latest_match["homeTeam"]
        away_team = latest_match["awayTeam"]

        if official_club == home_team["name"]:
            team_side = "home"
            opponent = away_team["name"]
            club_score = home_team["score"]
            opponent_score = away_team["score"]
        else:
            team_side = "away"
            opponent = home_team["name"]
            club_score = away_team["score"]
            opponent_score = home_team["score"]

        starting_players = get_starting_xi(
            latest_match["matchId"],
            team_side
        )

        starting_players = format_starting_xi(
            starting_players
        )

        match_date = latest_match["kickoff"].split(" ")[0]

        # ----------------------------------------------------
        # GROUP PLAYERS BY POSITION
        # ----------------------------------------------------

        goalkeepers = [
            p for p in starting_players
            if str(p["position"]).lower() == "goalkeeper"
        ]

        defenders = [
            p for p in starting_players
            if str(p["position"]).lower() == "defender"
        ]

        midfielders = [
            p for p in starting_players
            if str(p["position"]).lower() == "midfielder"
        ]

        forwards = [
            p for p in starting_players
            if str(p["position"]).lower() == "forward"
        ]

        def render_player(player, size="normal"):

            captain = ""

            if player["captain"]:
                captain = """
                <div style="
                    position:absolute;
                    top:-7px;
                    right:-7px;
                    width:24px;
                    height:24px;
                    border-radius:50%;
                    background:#ffffff;
                    color:#111111;
                    font-size:11px;
                    font-weight:950;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border:2px solid rgba(0,0,0,0.25);
                ">
                    C
                </div>
                """

            return f"""
            <div style="
                position:relative;
                width:100%;
                max-width:125px;
                text-align:center;
                margin:auto;
            ">

                <div style="
                    position:relative;
                    width:{'68px' if size == 'normal' else '76px'};
                    height:{'68px' if size == 'normal' else '76px'};
                    margin:0 auto;
                    border-radius:50%;
                    background:
                        linear-gradient(
                            145deg,
                            rgba(255,255,255,0.16),
                            rgba(255,255,255,0.045)
                        );
                    border:3px solid {primary};
                    box-shadow:
                        0 8px 25px rgba(0,0,0,0.35),
                        0 0 0 4px rgba(255,255,255,0.04);
                    display:flex;
                    align-items:center;
                    justify-content:center;
                ">

                    <div style="
                        color:#ffffff;
                        font-size:18px;
                        font-weight:950;
                    ">
                        {player["shirt_number"]}
                    </div>

                    {captain}

                </div>

                <div style="
                    margin-top:8px;
                    color:#ffffff;
                    font-size:13px;
                    font-weight:850;
                    line-height:1.2;
                ">
                    {player["name"]}
                </div>

                <div style="
                    margin-top:3px;
                    color:{muted};
                    font-size:9px;
                    font-weight:800;
                    letter-spacing:1px;
                    text-transform:uppercase;
                ">
                    {player["position"]}
                </div>

            </div>
            """

        def render_row(players, min_height=105):

            if not players:
                return ""

            player_html = ""

            for player in players:
                player_html += render_player(player)

            return f"""
            <div style="
                display:grid;
                grid-template-columns:
                    repeat({len(players)}, minmax(70px, 1fr));
                gap:12px;
                align-items:start;
                min-height:{min_height}px;
                width:100%;
                max-width:760px;
                margin:0 auto;
            ">
                {player_html}
            </div>
            """

        # ----------------------------------------------------
        # MATCHDAY HEADER
        # ----------------------------------------------------

        html(
            f"""
            <div style="
                margin-top:52px;
                margin-bottom:18px;
            ">

                <div style="
                    color:{primary};
                    font-size:10px;
                    font-weight:950;
                    letter-spacing:2.8px;
                    text-transform:uppercase;
                ">
                    MATCHDAY
                </div>

                <div style="
                    color:#ffffff;
                    font-size:30px;
                    font-weight:950;
                    text-transform:uppercase;
                    margin-top:5px;
                    letter-spacing:-0.5px;
                ">
                    Latest Starting XI
                </div>

                <div style="
                    margin-top:7px;
                    color:{muted};
                    font-size:13px;
                ">
                    {match_date}
                </div>

            </div>
            """
        )

        # ----------------------------------------------------
        # SCOREBOARD
        # ----------------------------------------------------

        html(
            f"""
            <div style="
                display:grid;
                grid-template-columns:1fr auto 1fr;
                gap:20px;
                align-items:center;
                padding:20px 24px;
                margin-bottom:20px;
                border-radius:20px;
                background:
                    linear-gradient(
                        135deg,
                        {primary}16,
                        rgba(255,255,255,0.025)
                    );
                border:1px solid {primary}30;
            ">

                <div style="
                    text-align:right;
                    color:#ffffff;
                    font-size:17px;
                    font-weight:900;
                ">
                    {official_club}
                </div>

                <div style="
                    text-align:center;
                    min-width:90px;
                ">

                    <div style="
                        color:#ffffff;
                        font-size:25px;
                        font-weight:950;
                        letter-spacing:2px;
                    ">
                        {club_score} — {opponent_score}
                    </div>

                    <div style="
                        margin-top:3px;
                        color:{muted};
                        font-size:9px;
                        font-weight:850;
                        letter-spacing:1.5px;
                        text-transform:uppercase;
                    ">
                        FULL TIME
                    </div>

                </div>

                <div style="
                    text-align:left;
                    color:#ffffff;
                    font-size:17px;
                    font-weight:900;
                ">
                    {opponent}
                </div>

            </div>
            """
        )

        if starting_players:

            # ------------------------------------------------
            # PITCH
            # ------------------------------------------------

            html(
                f"""
                <div style="
                    position:relative;
                    overflow:hidden;
                    min-height:620px;
                    padding:34px 28px 42px 28px;
                    border-radius:28px;
                    background:
                        linear-gradient(
                            180deg,
                            #173f2a 0%,
                            #12502f 48%,
                            #0f4328 100%
                        );
                    border:2px solid rgba(255,255,255,0.14);
                    box-shadow:
                        0 20px 60px rgba(0,0,0,0.30);
                ">

                    <!-- Pitch markings -->

                    <div style="
                        position:absolute;
                        inset:18px;
                        border:2px solid rgba(255,255,255,0.28);
                        border-radius:16px;
                        pointer-events:none;
                    "></div>

                    <div style="
                        position:absolute;
                        left:18px;
                        right:18px;
                        top:50%;
                        height:2px;
                        background:rgba(255,255,255,0.22);
                        pointer-events:none;
                    "></div>

                    <div style="
                        position:absolute;
                        left:50%;
                        top:50%;
                        width:105px;
                        height:105px;
                        transform:translate(-50%,-50%);
                        border:2px solid rgba(255,255,255,0.22);
                        border-radius:50%;
                        pointer-events:none;
                    "></div>

                    <div style="
                        position:absolute;
                        left:50%;
                        top:50%;
                        width:8px;
                        height:8px;
                        transform:translate(-50%,-50%);
                        background:rgba(255,255,255,0.35);
                        border-radius:50%;
                        pointer-events:none;
                    "></div>

                    <!-- Top penalty area -->

                    <div style="
                        position:absolute;
                        left:22%;
                        right:22%;
                        top:18px;
                        height:105px;
                        border:2px solid rgba(255,255,255,0.22);
                        border-top:none;
                        pointer-events:none;
                    "></div>

                    <!-- Bottom penalty area -->

                    <div style="
                        position:absolute;
                        left:22%;
                        right:22%;
                        bottom:18px;
                        height:105px;
                        border:2px solid rgba(255,255,255,0.22);
                        border-bottom:none;
                        pointer-events:none;
                    "></div>

                    <!-- Players -->

                    <div style="
                        position:relative;
                        z-index:2;
                        min-height:540px;
                        display:flex;
                        flex-direction:column;
                        justify-content:space-between;
                        gap:18px;
                    ">

                        <div>
                            {render_row(goalkeepers, 90)}
                        </div>

                        <div>
                            {render_row(defenders, 100)}
                        </div>

                        <div>
                            {render_row(midfielders, 105)}
                        </div>

                        <div>
                            {render_row(forwards, 105)}
                        </div>

                    </div>

                </div>
                """
            )

            # ------------------------------------------------
            # MATCH SUMMARY
            # ------------------------------------------------

            html(
                f"""
                <div style="
                    margin-top:18px;
                    display:grid;
                    grid-template-columns:
                        repeat(3, minmax(0, 1fr));
                    gap:12px;
                ">

                    <div style="
                        padding:16px;
                        border-radius:16px;
                        background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.07);
                    ">
                        <div style="
                            color:{muted};
                            font-size:9px;
                            font-weight:850;
                            letter-spacing:1.4px;
                            text-transform:uppercase;
                        ">
                            Starting XI
                        </div>

                        <div style="
                            color:#ffffff;
                            font-size:21px;
                            font-weight:950;
                            margin-top:4px;
                        ">
                            {len(starting_players)}
                        </div>
                    </div>

                    <div style="
                        padding:16px;
                        border-radius:16px;
                        background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.07);
                    ">
                        <div style="
                            color:{muted};
                            font-size:9px;
                            font-weight:850;
                            letter-spacing:1.4px;
                            text-transform:uppercase;
                        ">
                            Captain
                        </div>

                        <div style="
                            color:#ffffff;
                            font-size:16px;
                            font-weight:900;
                            margin-top:7px;
                        ">
                            {
                                next(
                                    (
                                        p["name"]
                                        for p in starting_players
                                        if p["captain"]
                                    ),
                                    "—"
                                )
                            }
                        </div>
                    </div>

                    <div style="
                        padding:16px;
                        border-radius:16px;
                        background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.07);
                    ">
                        <div style="
                            color:{muted};
                            font-size:9px;
                            font-weight:850;
                            letter-spacing:1.4px;
                            text-transform:uppercase;
                        ">
                            Result
                        </div>

                        <div style="
                            color:#ffffff;
                            font-size:16px;
                            font-weight:900;
                            margin-top:7px;
                        ">
                            {club_score} — {opponent_score}
                        </div>
                    </div>

                </div>
                """
            )

        else:

            html(
                f"""
                <div style="
                    padding:28px;
                    border-radius:20px;
                    border:1px solid rgba(255,255,255,0.08);
                    color:{muted};
                ">
                    Starting XI data is currently unavailable
                    for this match.
                </div>
                """
            )

    else:

        html(
            f"""
            <div style="
                margin-top:48px;
                padding:28px;
                border-radius:24px;
                background:
                    linear-gradient(
                        135deg,
                        {primary}12,
                        rgba(255,255,255,0.02)
                    );
                border:1px solid {primary}28;
            ">

                <div style="
                    color:{primary};
                    font-size:10px;
                    font-weight:900;
                    letter-spacing:2.5px;
                    text-transform:uppercase;
                ">
                    MATCHDAY
                </div>

                <div style="
                    color:#ffffff;
                    font-size:28px;
                    font-weight:950;
                    margin-top:5px;
                ">
                    Latest Starting XI
                </div>

                <div style="
                    margin-top:8px;
                    color:{muted};
                    font-size:13px;
                ">
                    No completed Premier League match was found
                    for this club.
                </div>

            </div>
            """
        )

# ============================================================
# COMMUNITY
# ============================================================

def community_page():

    html(
        """
        <div style="
            margin-top:12px;
            padding:38px;
            border-radius:28px;
            background:
                radial-gradient(circle at 85% 15%, rgba(75,90,255,0.22), transparent 35%),
                linear-gradient(135deg, #111827, #080b12);
            border:1px solid rgba(255,255,255,0.08);
        ">
            <div style="color:#7c8cff;font-size:10px;font-weight:950;letter-spacing:2.5px;">
                FAN COMMUNITY
            </div>
            <div style="margin-top:10px;color:#ffffff;font-size:40px;font-weight:950;letter-spacing:-1.5px;">
                Premier League Community
            </div>
            <div style="margin-top:10px;max-width:680px;color:#8b97a9;font-size:14px;line-height:1.7;">
                A place for club discussions, matchday reactions, transfer talk, tactics and football analytics.
            </div>
        </div>
        """
    )

    st.markdown('<div class="section-title">Community Areas</div>', unsafe_allow_html=True)

    areas = [
        ("🔥", "Matchday", "Live reactions and post-match discussion."),
        ("🔄", "Transfer Talk", "Rumours, signings and squad planning."),
        ("⚽", "Player Discussion", "Form, performances and player debates."),
        ("🧠", "Tactics", "Lineups, formations and tactical analysis."),
        ("📊", "Analytics", "Football data, statistics and models."),
        ("💬", "General", "Everything else football related."),
    ]

    area_columns = st.columns(3)
    for index, (icon, title, description) in enumerate(areas):
        with area_columns[index % 3]:
            html(
                f"""
                <div style="min-height:145px;margin-top:12px;padding:20px;border-radius:18px;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);">
                    <div style="font-size:25px;">{icon}</div>
                    <div style="margin-top:10px;color:#ffffff;font-size:16px;font-weight:900;">{title}</div>
                    <div style="margin-top:7px;color:#7f8b9e;font-size:11px;line-height:1.6;">{description}</div>
                </div>
                """
            )

    st.markdown('<div class="section-title">Club Communities</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#7f8b9e;font-size:13px;margin-bottom:12px;">Choose a club to enter its community hub.</div>',
        unsafe_allow_html=True,
    )

    community_clubs = list(CLUBS.keys())
    club_columns = st.columns(4)

    for index, club in enumerate(community_clubs):
        info = club_info(club)
        primary = info.get("primary", "#42556a")
        nickname = info.get("nickname", "")
        code = info.get("code", "")

        with club_columns[index % 4]:
            html(
                f"""
                <div style="margin-top:12px;min-height:125px;padding:18px;border-radius:18px;background:linear-gradient(145deg,{primary}18,rgba(255,255,255,0.025));border:1px solid {primary}55;">
                    <div style="color:{primary};font-size:10px;font-weight:950;letter-spacing:2px;">{code}</div>
                    <div style="margin-top:9px;color:#ffffff;font-size:16px;font-weight:900;">{club}</div>
                    <div style="margin-top:6px;color:#7f8b9e;font-size:11px;">{nickname}</div>
                </div>
                """
            )
            if st.button(
                "Open Community",
                key=f"club_community_open_{index}",
                use_container_width=True,
            ):
                navigate("ClubCommunity", selected_community_club=club)

    st.markdown('<div class="section-title">Community Roadmap</div>', unsafe_allow_html=True)
    html(
        """
        <div style="padding:22px;border-radius:20px;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);">
            <div style="color:#ffffff;font-size:17px;font-weight:900;">Community foundation</div>
            <div style="margin-top:10px;color:#7f8b9e;font-size:12px;line-height:1.8;">
                Club hubs and persistent discussions are now live. Future versions can add accounts, profiles, replies, polls, predictions, reputation and moderation.
            </div>
        </div>
        """
    )


# ============================================================
# CLUB COMMUNITY HUB
# ============================================================

def club_community_page(club_name):

    if not club_name or club_name not in CLUBS:
        st.error("Club community could not be found.")
        if st.button("← Back to Community"):
            navigate("Community")
        return

    info = club_info(club_name)
    primary = info.get("primary", "#42556a")
    nickname = info.get("nickname", "")
    code = info.get("code", "")

    if st.button("← Back to Community"):
        navigate("Community")

    html(
        f"""
        <div style="margin-top:12px;padding:34px;border-radius:28px;background:radial-gradient(circle at 85% 15%,{primary}55,transparent 38%),linear-gradient(135deg,#111827,#080b12);border:1px solid {primary}55;">
            <div style="color:{primary};font-size:10px;font-weight:950;letter-spacing:2.5px;">{code} • FAN COMMUNITY</div>
            <div style="margin-top:10px;color:#ffffff;font-size:38px;font-weight:950;">{club_name}</div>
            <div style="margin-top:6px;color:#8b97a9;font-size:14px;">{nickname}</div>
            <div style="display:inline-block;margin-top:18px;padding:7px 12px;border-radius:999px;background:{primary}22;border:1px solid {primary}45;color:#ffffff;font-size:11px;font-weight:800;">Club Community</div>
        </div>
        """
    )

    st.markdown('<div class="section-title">Start a Discussion</div>', unsafe_allow_html=True)

    with st.form(f"create_post_{club_name}", clear_on_submit=True):
        category = st.selectbox(
            "Category",
            ["Matchday", "Transfer Talk", "Player Discussion", "Tactics", "Analytics", "General"],
            key=f"category_{club_name}",
        )
        post_text = st.text_area(
            "Your discussion",
            placeholder=f"Share something about {club_name}...",
            height=120,
            key=f"post_text_{club_name}",
        )
        publish = st.form_submit_button("✍️ Publish Discussion", use_container_width=True)

        if publish:
            clean_text = post_text.strip()
            if not clean_text:
                st.warning("Please write something before publishing.")
            else:
                community_db.create_post(
                    club=club_name,
                    category=category,
                    text=clean_text,
                )
                st.success("Discussion published successfully.")
                st.rerun()

    st.markdown('<div class="section-title">Community Feed</div>', unsafe_allow_html=True)

    # Persistent feed: every refresh reads directly from SQLite.
    club_posts = community_db.get_posts(club_name)

    if not club_posts:
        html(
            """
            <div style="padding:28px;border-radius:20px;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);text-align:center;">
                <div style="color:#ffffff;font-size:18px;font-weight:900;">No discussions yet</div>
                <div style="margin-top:8px;color:#7f8b9e;font-size:12px;">Start the first discussion for this club.</div>
            </div>
            """
        )
    else:
        for post_index, post in enumerate(club_posts):
            post_id = post["id"]
            comment_count = len(community_db.get_comments(post_id))

            html(
                f"""
                <div style="margin-top:14px;padding:22px;border-radius:20px;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);">
                    <div style="color:{primary};font-size:10px;font-weight:950;letter-spacing:1.8px;text-transform:uppercase;">{post.get("category", "General")}</div>
                    <div style="margin-top:10px;color:#ffffff;font-size:15px;line-height:1.7;">{post.get("text", "")}</div>
                    <div style="margin-top:12px;color:#667085;font-size:11px;">Discussion #{post_id}</div>
                </div>
                """
            )

            like_col, comment_col = st.columns(2)

            with like_col:
                if st.button(
                    f"❤️ {post.get('likes', 0)}",
                    key=f"like_{club_name}_{post_id}",
                    use_container_width=True,
                ):
                    community_db.like_post(post_id)
                    st.rerun()

            with comment_col:
                show_comments_key = f"show_comments_{post_id}"
                if st.button(
                    f"💬 {comment_count}",
                    key=f"comments_toggle_{club_name}_{post_id}",
                    use_container_width=True,
                ):
                    st.session_state[show_comments_key] = not st.session_state.get(
                        show_comments_key,
                        False,
                    )
                    st.rerun()

            if st.session_state.get(show_comments_key, False):
                comments = community_db.get_comments(post_id)

                if comments:
                    for comment in comments:
                        html(
                            f"""
                            <div style="margin:8px 0 8px 20px;padding:12px 15px;border-left:2px solid {primary};background:rgba(255,255,255,0.02);border-radius:0 12px 12px 0;color:#c4ccd8;font-size:12px;line-height:1.6;">
                                {comment["text"]}
                            </div>
                            """
                        )
                else:
                    st.markdown(
                        '<div style="margin:10px 0 10px 20px;color:#687386;font-size:12px;">No comments yet.</div>',
                        unsafe_allow_html=True,
                    )

                with st.form(
                    f"comment_form_{club_name}_{post_id}",
                    clear_on_submit=True,
                ):
                    comment_text = st.text_input(
                        "Add a comment",
                        key=f"comment_text_{club_name}_{post_id}",
                    )
                    submit_comment = st.form_submit_button("Reply", use_container_width=True)

                    if submit_comment:
                        clean_comment = comment_text.strip()
                        if not clean_comment:
                            st.warning("Please write a comment before replying.")
                        else:
                            community_db.create_comment(post_id, clean_comment)
                            st.rerun()

    html(
        f"""
        <div style="margin-top:24px;padding:20px;border-radius:20px;background:{primary}12;border:1px solid {primary}35;">
            <div style="color:{primary};font-size:10px;font-weight:950;letter-spacing:2px;">COMMUNITY FOUNDATION</div>
            <div style="margin-top:8px;color:#ffffff;font-size:16px;font-weight:900;">Persistent club discussions are live</div>
            <div style="margin-top:7px;color:#7f8b9e;font-size:12px;line-height:1.7;">
                Posts, likes and comments are stored in the local SQLite community database and remain available after app refreshes.
            </div>
        </div>
        """
    )

def players_page():

    st.markdown(
        '<div class="section-title">'
        'Player Database'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Search and explore Premier League players'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        search = st.text_input(
            "Player",
            placeholder="Search player..."
        )

    with c2:

        clubs = [
            "All Clubs"
        ] + sorted(
            df["club"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_club = st.selectbox(
            "Club",
            clubs
        )

    with c3:

        positions = [
            "All Positions"
        ] + sorted(
            df["position"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_position = st.selectbox(
            "Position",
            positions
        )

    filtered = df.copy()

    if search:

        filtered = filtered[
            filtered["name"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if selected_club != "All Clubs":

        filtered = filtered[
            filtered["club"]
            == selected_club
        ]

    if selected_position != "All Positions":

        filtered = filtered[
            filtered["position"]
            == selected_position
        ]

    filtered = (
        filtered
        .sort_values(
            "market_value",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.write(
        f"**{len(filtered)} players found**"
    )


    for start in range(
        0,
        len(filtered),
        4
    ):

        rows = filtered.iloc[
            start:start + 4
        ]

        cols = st.columns(4)

        for col, (_, row) in zip(
            cols,
            rows.iterrows()
        ):

            with col:

                photo = get_photo(row)

                if photo:

                    image_html = f"""
                    <img
                        class="player-photo"
                        src="{photo}"
                        alt="{row['name']}"
                    >
                    """

                else:

                    image_html = """
                    <div class="player-placeholder">
                        ⚽
                    </div>
                    """

                html(
                    f"""
                    <div class="player-card">

                        {image_html}

                        <div class="player-info">

                            <div class="player-name">
                                {row['name']}
                            </div>

                            <div class="player-club">
                                {row['club']}
                            </div>

                            <div class="player-value">
                                {money(
                                    row['market_value']
                                )}
                            </div>

                            <div class="player-meta">
                                {row.get(
                                    'position',
                                    'Player'
                                )}
                                ·
                                {int(row['age'])}
                                years
                            </div>

                        </div>

                    </div>
                    """
                )

    if st.button(
        "Open Community",
        key=f"community_{club}",
        use_container_width=True
    ):
        navigate(
            "ClubCommunity",
            selected_community_club=club
        )
# ============================================================
# PLAYER PAGE
# ============================================================

def player_page(player_name):

    # ========================================================
    # PLAYER LOOKUP
    # ========================================================

    player_rows = df[
        df["name"].astype(str) == str(player_name)
    ]

    if player_rows.empty:

        st.error("Player not found.")

        if st.button("Back to Players"):
            navigate("Players")

        return

    row = player_rows.iloc[0]

    club = row["club"]
    info = club_info(club)

    primary = info.get(
        "primary",
        info.get("color", "#42556a")
    )

    secondary = info.get(
        "secondary",
        "#FFFFFF"
    )

    nickname = info.get(
        "nickname",
        ""
    )

    logo = info.get("logo", "")
    logo_data = get_logo_data(logo)

    # ========================================================
    # SAFE DATA VALUES
    # ========================================================

    name = str(row.get("name", "Unknown Player"))
    position = str(row.get("position", "Player"))
    sub_position = str(row.get("sub_position", ""))

    age = float(row.get("age", 0))
    appearances = int(row.get("appearances", 0))
    minutes = int(row.get("minutes", 0))
    goals = int(row.get("goals", 0))
    assists = int(row.get("assists", 0))

    goals_per_90 = float(
        row.get("goals_per_90", 0)
    )

    assists_per_90 = float(
        row.get("assists_per_90", 0)
    )

    listed_value = float(
        row.get("market_value", 0)
    )

    # ========================================================
    # TOP NAVIGATION
    # ========================================================

    nav_left, nav_right = st.columns(
        [1, 5]
    )

    with nav_left:

        if st.button(
            "← Players",
            use_container_width=True
        ):
            navigate("Players")

    with nav_right:

        if st.button(
            f"← {club}",
            use_container_width=True
        ):
            navigate(
                "Club",
                selected_club=club
            )

    # ========================================================
    # PLAYER PHOTO
    # ========================================================

    photo = get_photo(row)

    if photo:

        photo_html = f"""
        <img
            src="{photo}"
            alt="{name}"
            style="
                width:250px;
                height:310px;
                object-fit:contain;
                display:block;
                margin:auto;
                filter:drop-shadow(
                    0 18px 25px rgba(0,0,0,0.45)
                );
            "
        >
        """

    else:

        photo_html = f"""
        <div style="
            width:250px;
            height:310px;
            display:flex;
            align-items:center;
            justify-content:center;
            margin:auto;
            border-radius:24px;
            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.10),
                    rgba(255,255,255,0.025)
                );
            color:{primary};
            font-size:80px;
            font-weight:950;
        ">
            ⚽
        </div>
        """

    # ========================================================
    # CLUB LOGO
    # ========================================================

    if logo_data:

        club_logo_html = f"""
        <img
            src="{logo_data}"
            alt="{club}"
            style="
                width:44px;
                height:44px;
                object-fit:contain;
                vertical-align:middle;
                margin-right:10px;
            "
        >
        """

    else:

        club_logo_html = ""

    # ========================================================
    # PLAYER HERO
    # ========================================================

    html(
        f"""
        <div style="
            position:relative;
            overflow:hidden;
            margin-top:18px;
            padding:34px;
            border-radius:30px;

            background:
                radial-gradient(
                    circle at 82% 20%,
                    {primary}45,
                    transparent 32%
                ),
                radial-gradient(
                    circle at 65% 90%,
                    {primary}18,
                    transparent 38%
                ),
                #0c111b;

            border:1px solid {primary}45;

            box-shadow:
                0 25px 70px rgba(0,0,0,0.28);
        ">

            <div style="
                position:absolute;
                top:-120px;
                right:-100px;
                width:300px;
                height:300px;
                border-radius:50%;
                background:{primary};
                opacity:0.07;
            "></div>

            <div style="
                position:relative;
                z-index:2;

                display:grid;
                grid-template-columns:
                    minmax(220px, 280px)
                    minmax(0, 1fr);

                gap:42px;
                align-items:center;
            ">

                <div style="
                    display:flex;
                    justify-content:center;
                    align-items:flex-end;
                    min-height:310px;
                ">
                    {photo_html}
                </div>

                <div>

                    <div style="
                        color:{primary};
                        font-size:10px;
                        font-weight:950;
                        letter-spacing:3px;
                        text-transform:uppercase;
                    ">
                        PLAYER CENTRE
                    </div>

                    <div style="
                        margin-top:8px;
                        color:#ffffff;
                        font-size:clamp(
                            38px,
                            5vw,
                            64px
                        );
                        font-weight:950;
                        line-height:0.98;
                        letter-spacing:-2.5px;
                    ">
                        {name}
                    </div>

                    <div style="
                        margin-top:20px;
                        display:flex;
                        align-items:center;
                        flex-wrap:wrap;
                        color:#ffffff;
                        font-size:17px;
                        font-weight:850;
                    ">
                        {club_logo_html}
                        {club}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#9aa6b8;
                        font-size:14px;
                        font-weight:700;
                    ">
                        {position}
                        {" · " + sub_position if sub_position else ""}
                    </div>

                    <div style="
                        display:flex;
                        flex-wrap:wrap;
                        gap:10px;
                        margin-top:24px;
                    ">

                        <div style="
                            padding:9px 14px;
                            border-radius:999px;
                            background:
                                {primary}22;
                            border:1px solid
                                {primary}55;
                            color:#ffffff;
                            font-size:12px;
                            font-weight:850;
                        ">
                            AGE {age:.0f}
                        </div>

                        <div style="
                            padding:9px 14px;
                            border-radius:999px;
                            background:
                                rgba(255,255,255,0.06);
                            border:1px solid
                                rgba(255,255,255,0.10);
                            color:#ffffff;
                            font-size:12px;
                            font-weight:850;
                        ">
                            {appearances} APPEARANCES
                        </div>

                        {
                            f'''
                            <div style="
                                padding:9px 14px;
                                border-radius:999px;
                                background:
                                    rgba(255,255,255,0.06);
                                border:1px solid
                                    rgba(255,255,255,0.10);
                                color:#ffffff;
                                font-size:12px;
                                font-weight:850;
                            ">
                                {nickname}
                            </div>
                            '''
                            if nickname
                            else ""
                        }

                    </div>

                </div>

            </div>

        </div>
        """
    )

    # ========================================================
    # PLAYER OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">Player Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Key competitive and market indicators'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    overview_cards = [
        (
            c1,
            "Market Value",
            money(listed_value),
            "Listed value"
        ),
        (
            c2,
            "Appearances",
            f"{appearances}",
            "League appearances"
        ),
        (
            c3,
            "Goals",
            f"{goals}",
            "League goals"
        ),
        (
            c4,
            "Assists",
            f"{assists}",
            "League assists"
        )
    ]

    for column, label, value, small in overview_cards:

        with column:

            html(
                f"""
                <div style="
                    min-height:135px;
                    padding:22px;
                    border-radius:20px;

                    background:
                        linear-gradient(
                            145deg,
                            {primary}15,
                            rgba(255,255,255,0.035)
                        );

                    border:1px solid
                        rgba(255,255,255,0.08);
                ">

                    <div style="
                        color:#8793a6;
                        font-size:10px;
                        font-weight:900;
                        letter-spacing:1.5px;
                        text-transform:uppercase;
                    ">
                        {label}
                    </div>

                    <div style="
                        margin-top:12px;
                        color:#ffffff;
                        font-size:28px;
                        font-weight:950;
                        letter-spacing:-1px;
                    ">
                        {value}
                    </div>

                    <div style="
                        margin-top:5px;
                        color:#657184;
                        font-size:11px;
                    ">
                        {small}
                    </div>

                </div>
                """
            )

    # ========================================================
    # PERFORMANCE PROFILE
    # ========================================================

    st.markdown(
        '<div class="section-title">Performance Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Per-90 and workload indicators from the available season data'
        '</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    performance_cards = [
        (
            p1,
            "Minutes",
            f"{minutes:,}",
            "League minutes"
        ),
        (
            p2,
            "Goals / 90",
            f"{goals_per_90:.2f}",
            "Scoring rate"
        ),
        (
            p3,
            "Assists / 90",
            f"{assists_per_90:.2f}",
            "Creation rate"
        ),
        (
            p4,
            "Age",
            f"{age:.0f}",
            "Player age"
        )
    ]

    for column, label, value, small in performance_cards:

        with column:

            html(
                f"""
                <div style="
                    min-height:120px;
                    padding:20px;
                    border-radius:18px;

                    background:
                        rgba(255,255,255,0.035);

                    border:1px solid
                        rgba(255,255,255,0.07);
                ">

                    <div style="
                        color:#7f8b9e;
                        font-size:10px;
                        font-weight:900;
                        letter-spacing:1.4px;
                        text-transform:uppercase;
                    ">
                        {label}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#ffffff;
                        font-size:25px;
                        font-weight:950;
                    ">
                        {value}
                    </div>

                    <div style="
                        margin-top:5px;
                        color:#667286;
                        font-size:11px;
                    ">
                        {small}
                    </div>

                </div>
                """
            )

    # ========================================================
    # AI / ML VALUATION
    # ========================================================

    st.markdown(
        '<div class="section-title">AI Valuation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Model-implied value based on player characteristics '
        'and available performance data'
        '</div>',
        unsafe_allow_html=True
    )

    if model is not None:

        try:

            model_input = pd.DataFrame(
                [{
                    "age": row["age"],
                    "appearances": row["appearances"],
                    "minutes": row["minutes"],
                    "goals": row["goals"],
                    "assists": row["assists"],
                    "goals_per_90": row["goals_per_90"],
                    "assists_per_90": row["assists_per_90"],
                    "minutes_per_appearance":
                        row["minutes_per_appearance"],
                    "position": row["position"],
                    "sub_position":
                        row["sub_position"],
                    "club": row["club"]
                }]
            )

            prediction_log = model.predict(
                model_input
            )[0]

            predicted_value = np.expm1(
                prediction_log
            )

            predicted_value = max(
                0,
                float(predicted_value)
            )

            difference = (
                predicted_value
                - listed_value
            )

            if listed_value > 0:

                pct_difference = (
                    difference
                    / listed_value
                ) * 100

            else:

                pct_difference = 0

            html(
                f"""
                <div style="
                    margin-top:8px;
                    padding:30px;
                    border-radius:24px;

                    background:
                        radial-gradient(
                            circle at 85% 15%,
                            {primary}30,
                            transparent 35%
                        ),
                        linear-gradient(
                            135deg,
                            #101827,
                            #0b1018
                        );

                    border:1px solid
                        {primary}45;

                    box-shadow:
                        0 20px 50px rgba(0,0,0,0.20);
                ">

                    <div style="
                        color:{primary};
                        font-size:10px;
                        font-weight:950;
                        letter-spacing:2.3px;
                        text-transform:uppercase;
                    ">
                        MODEL-IMPLIED VALUE
                    </div>

                    <div style="
                        margin-top:8px;
                        color:#ffffff;
                        font-size:42px;
                        font-weight:950;
                        letter-spacing:-1.5px;
                    ">
                        {money(predicted_value)}
                    </div>

                    <div style="
                        margin-top:7px;
                        color:#8995a8;
                        font-size:13px;
                    ">
                        Statistical estimate generated by the
                        trained player valuation model.
                    </div>

                </div>
                """
            )

            v1, v2, v3 = st.columns(3)

            with v1:

                html(
                    f"""
                    <div style="
                        margin-top:14px;
                        padding:20px;
                        border-radius:18px;
                        background:
                            rgba(255,255,255,0.035);
                        border:1px solid
                            rgba(255,255,255,0.07);
                    ">

                        <div style="
                            color:#7f8b9e;
                            font-size:10px;
                            font-weight:900;
                            letter-spacing:1.3px;
                            text-transform:uppercase;
                        ">
                            Listed Value
                        </div>

                        <div style="
                            margin-top:8px;
                            color:#ffffff;
                            font-size:22px;
                            font-weight:950;
                        ">
                            {money(listed_value)}
                        </div>

                    </div>
                    """
                )

            with v2:

                html(
                    f"""
                    <div style="
                        margin-top:14px;
                        padding:20px;
                        border-radius:18px;
                        background:
                            rgba(255,255,255,0.035);
                        border:1px solid
                            rgba(255,255,255,0.07);
                    ">

                        <div style="
                            color:#7f8b9e;
                            font-size:10px;
                            font-weight:900;
                            letter-spacing:1.3px;
                            text-transform:uppercase;
                        ">
                            Model Difference
                        </div>

                        <div style="
                            margin-top:8px;
                            color:#ffffff;
                            font-size:22px;
                            font-weight:950;
                        ">
                            {money(abs(difference))}
                        </div>

                        <div style="
                            margin-top:4px;
                            color:#667286;
                            font-size:11px;
                        ">
                            {
                                "Model above listed value"
                                if difference >= 0
                                else
                                "Model below listed value"
                            }
                        </div>

                    </div>
                    """
                )

            with v3:

                html(
                    f"""
                    <div style="
                        margin-top:14px;
                        padding:20px;
                        border-radius:18px;
                        background:
                            rgba(255,255,255,0.035);
                        border:1px solid
                            rgba(255,255,255,0.07);
                    ">

                        <div style="
                            color:#7f8b9e;
                            font-size:10px;
                            font-weight:900;
                            letter-spacing:1.3px;
                            text-transform:uppercase;
                        ">
                            Relative Difference
                        </div>

                        <div style="
                            margin-top:8px;
                            color:#ffffff;
                            font-size:22px;
                            font-weight:950;
                        ">
                            {pct_difference:+.1f}%
                        </div>

                        <div style="
                            margin-top:4px;
                            color:#667286;
                            font-size:11px;
                        ">
                            Model vs listed value
                        </div>

                    </div>
                    """
                )

            html(
                """
                <div style="
                    margin-top:14px;
                    padding:14px 18px;
                    border-radius:14px;
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.06);
                    color:#687588;
                    font-size:11px;
                    line-height:1.6;
                ">
                    <strong style="color:#8c98aa;">
                        Note:
                    </strong>
                    This is a model-implied statistical estimate,
                    not an objective transfer valuation.
                </div>
                """
            )

        except Exception:

            st.warning(
                "The model prediction could not be generated."
            )

    else:

        st.info(
            "The valuation model is currently unavailable."
        )

    # ========================================================
    # PLAYER CONTEXT
    # ========================================================

    st.markdown(
        '<div class="section-title">Player Context</div>',
        unsafe_allow_html=True
    )

    html(
        f"""
        <div style="
            padding:24px;
            border-radius:22px;

            background:
                linear-gradient(
                    135deg,
                    {primary}10,
                    rgba(255,255,255,0.025)
                );

            border:1px solid
                rgba(255,255,255,0.07);
        ">

            <div style="
                color:#ffffff;
                font-size:18px;
                font-weight:900;
            ">
                {name}
            </div>

            <div style="
                margin-top:7px;
                color:#8995a8;
                font-size:13px;
                line-height:1.7;
            ">
                {position}
                {" · " + sub_position if sub_position else ""}
                · {club}
                · Age {age:.0f}
            </div>

            <div style="
                margin-top:14px;
                color:#697587;
                font-size:12px;
                line-height:1.7;
            ">
                The profile currently combines the available
                player performance dataset with the Premier League
                club identity layer and the player valuation model.
                Additional match-level and advanced analytics will
                be connected in later Player Centre updates.
            </div>

        </div>
        """
    )

# ============================================================
# MODEL PAGE
# ============================================================

def model_page():

    # ========================================================
    # ANALYTICS LAB
    # ========================================================

    st.markdown(
        '<div class="section-title">Analytics Lab</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore player performance, production, market value and model insights'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    analytics_df = df.copy()

    required_columns = [
        "name",
        "club",
        "position",
        "age",
        "appearances",
        "minutes",
        "goals",
        "assists",
        "goals_per_90",
        "assists_per_90",
        "market_value"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in analytics_df.columns
    ]

    if missing_columns:

        st.error(
            "Analytics data is missing: "
            + ", ".join(missing_columns)
        )

        return

    numeric_columns = [
        "age",
        "appearances",
        "minutes",
        "goals",
        "assists",
        "goals_per_90",
        "assists_per_90",
        "market_value"
    ]

    for column in numeric_columns:
        analytics_df[column] = pd.to_numeric(
            analytics_df[column],
            errors="coerce"
        )

    analytics_df = analytics_df.dropna(
        subset=[
            "name",
            "club",
            "position",
            "market_value"
        ]
    ).copy()

    # --------------------------------------------------------
    # LAB OVERVIEW
    # --------------------------------------------------------

    total_players = len(analytics_df)

    total_goals = int(
        analytics_df["goals"].fillna(0).sum()
    )

    total_assists = int(
        analytics_df["assists"].fillna(0).sum()
    )

    total_minutes = int(
        analytics_df["minutes"].fillna(0).sum()
    )

    total_market_value = analytics_df[
        "market_value"
    ].fillna(0).sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        html(
            f"""
            <div class="kpi">
                <div class="kpi-label">
                    Players Analysed
                </div>

                <div class="kpi-value">
                    {total_players}
                </div>

                <div class="kpi-small">
                    Current dataset
                </div>
            </div>
            """
        )

    with c2:

        html(
            f"""
            <div class="kpi">
                <div class="kpi-label">
                    Goals
                </div>

                <div class="kpi-value">
                    {total_goals:,}
                </div>

                <div class="kpi-small">
                    Recorded production
                </div>
            </div>
            """
        )

    with c3:

        html(
            f"""
            <div class="kpi">
                <div class="kpi-label">
                    Assists
                </div>

                <div class="kpi-value">
                    {total_assists:,}
                </div>

                <div class="kpi-small">
                    Recorded production
                </div>
            </div>
            """
        )

    with c4:

        html(
            f"""
            <div class="kpi">
                <div class="kpi-label">
                    Squad Value
                </div>

                <div class="kpi-value">
                    {money(total_market_value)}
                </div>

                <div class="kpi-small">
                    Listed player values
                </div>
            </div>
            """
        )

    # --------------------------------------------------------
    # ANALYTICS CONTROLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Performance Explorer</div>',
        unsafe_allow_html=True
    )

    filter1, filter2, filter3 = st.columns(3)

    with filter1:

        selected_club = st.selectbox(
            "Club",
            ["All Clubs"] +
            sorted(
                analytics_df["club"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with filter2:

        selected_position = st.selectbox(
            "Position",
            ["All Positions"] +
            sorted(
                analytics_df["position"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with filter3:

        minimum_minutes = st.slider(
            "Minimum Minutes",
            min_value=0,
            max_value=int(
                max(
                    1,
                    analytics_df["minutes"]
                    .fillna(0)
                    .max()
                )
            ),
            value=0,
            step=100
        )

    filtered = analytics_df.copy()

    if selected_club != "All Clubs":

        filtered = filtered[
            filtered["club"].astype(str)
            == selected_club
        ]

    if selected_position != "All Positions":

        filtered = filtered[
            filtered["position"].astype(str)
            == selected_position
        ]

    filtered = filtered[
        filtered["minutes"].fillna(0)
        >= minimum_minutes
    ].copy()

    st.caption(
        f"{len(filtered)} players match the current filters."
    )

    # --------------------------------------------------------
    # TOP PRODUCTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Production Leaders</div>',
        unsafe_allow_html=True
    )

    production1, production2 = st.columns(2)

    with production1:

        top_goals = (
            filtered
            .sort_values(
                "goals",
                ascending=False
            )
            .head(10)
            [
                [
                    "name",
                    "club",
                    "goals",
                    "minutes"
                ]
            ]
            .copy()
        )

        top_goals.columns = [
            "Player",
            "Club",
            "Goals",
            "Minutes"
        ]

        st.dataframe(
            top_goals,
            use_container_width=True,
            hide_index=True
        )

    with production2:

        top_assists = (
            filtered
            .sort_values(
                "assists",
                ascending=False
            )
            .head(10)
            [
                [
                    "name",
                    "club",
                    "assists",
                    "minutes"
                ]
            ]
            .copy()
        )

        top_assists.columns = [
            "Player",
            "Club",
            "Assists",
            "Minutes"
        ]

        st.dataframe(
            top_assists,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # GOALS VS ASSISTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Goals / 90 vs Assists / 90'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Compare attacking production rates across the filtered player pool'
        '</div>',
        unsafe_allow_html=True
    )

    rate_data = filtered[
        [
            "name",
            "goals_per_90",
            "assists_per_90",
            "market_value"
        ]
    ].dropna().copy()

    if not rate_data.empty:

        rate_chart = rate_data.rename(
            columns={
                "name": "Player",
                "goals_per_90": "Goals / 90",
                "assists_per_90": "Assists / 90"
            }
        )

        st.scatter_chart(
            rate_chart,
            x="Goals / 90",
            y="Assists / 90",
            use_container_width=True
        )

        st.caption(
            "Each point represents a player. "
            "The chart shows rate-based attacking production; "
            "it does not represent an overall player rating."
        )

    # --------------------------------------------------------
    # MINUTES VS PRODUCTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Minutes vs Production'
        '</div>',
        unsafe_allow_html=True
    )

    minutes_data = filtered[
        [
            "name",
            "minutes",
            "goals"
        ]
    ].dropna().copy()

    if not minutes_data.empty:

        minutes_data = minutes_data.rename(
            columns={
                "name": "Player",
                "minutes": "Minutes",
                "goals": "Goals"
            }
        )

        st.scatter_chart(
            minutes_data,
            x="Minutes",
            y="Goals",
            use_container_width=True
        )

        st.caption(
            "This view helps distinguish raw goal totals "
            "from the amount of playing time available."
        )

    # --------------------------------------------------------
    # MARKET VALUE VS PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Market Value vs Performance'
        '</div>',
        unsafe_allow_html=True
    )

    value_data = filtered[
        [
            "name",
            "market_value",
            "goals",
            "assists"
        ]
    ].dropna().copy()

    if not value_data.empty:

        value_data["Production"] = (
            value_data["goals"]
            + value_data["assists"]
        )

        value_chart = value_data.rename(
            columns={
                "name": "Player",
                "market_value": "Market Value"
            }
        )

        st.scatter_chart(
            value_chart,
            x="Market Value",
            y="Production",
            use_container_width=True
        )

        st.caption(
            "Production here is defined as goals + assists. "
            "Market value is the listed value in the dataset."
        )

    # --------------------------------------------------------
    # PLAYER COMPARISON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Player Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_players = st.multiselect(
        "Choose players to compare",
        sorted(
            analytics_df["name"]
            .astype(str)
            .unique()
            .tolist()
        ),
        max_selections=4
    )

    if comparison_players:

        comparison_df = analytics_df[
            analytics_df["name"].astype(str).isin(
                comparison_players
            )
        ].copy()

        comparison_table = comparison_df[
            [
                "name",
                "club",
                "position",
                "age",
                "appearances",
                "minutes",
                "goals",
                "assists",
                "goals_per_90",
                "assists_per_90",
                "market_value"
            ]
        ].copy()

        comparison_table = comparison_table.rename(
            columns={
                "name": "Player",
                "club": "Club",
                "position": "Position",
                "age": "Age",
                "appearances": "Apps",
                "minutes": "Minutes",
                "goals": "Goals",
                "assists": "Assists",
                "goals_per_90": "Goals / 90",
                "assists_per_90": "Assists / 90",
                "market_value": "Market Value"
            }
        )

        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Select two or more players to build a comparison."
        )

    # --------------------------------------------------------
    # SIMILAR PLAYERS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Similar Player Discovery'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Find players with similar statistical profiles using the available dataset features'
        '</div>',
        unsafe_allow_html=True
    )

    similar_player = st.selectbox(
        "Find similar players to",
        sorted(
            analytics_df["name"]
            .astype(str)
            .unique()
            .tolist()
        ),
        key="similar_player_selector"
    )

    similarity_features = [
        "age",
        "minutes",
        "goals",
        "assists",
        "goals_per_90",
        "assists_per_90"
    ]

    similarity_df = analytics_df[
        ["name"] + similarity_features
    ].copy()

    similarity_df[similarity_features] = (
        similarity_df[similarity_features]
        .apply(pd.to_numeric, errors="coerce")
    )

    target_rows = similarity_df[
        similarity_df["name"].astype(str)
        == str(similar_player)
    ]

    if not target_rows.empty:

        target = target_rows.iloc[0]

        usable = similarity_df.dropna(
            subset=similarity_features
        ).copy()

        if len(usable) > 1:

            means = usable[
                similarity_features
            ].mean()

            stds = usable[
                similarity_features
            ].std().replace(0, 1)

            normalized = (
                usable[similarity_features]
                - means
            ) / stds

            target_vector = (
                pd.DataFrame(
                    [target[similarity_features]]
                )
                - means
            ) / stds

            distances = (
                (
                    normalized
                    - target_vector.iloc[0]
                ) ** 2
            ).sum(axis=1) ** 0.5

            usable["distance"] = distances

            similar_results = (
                usable[
                    usable["name"].astype(str)
                    != str(similar_player)
                ]
                .sort_values("distance")
                .head(5)
                [["name", "distance"]]
                .copy()
            )

            similar_results = similar_results.merge(
                analytics_df[
                    [
                        "name",
                        "club",
                        "position",
                        "market_value"
                    ]
                ],
                on="name",
                how="left"
            )

            similar_results = similar_results[
                [
                    "name",
                    "club",
                    "position",
                    "market_value"
                ]
            ]

            similar_results.columns = [
                "Player",
                "Club",
                "Position",
                "Market Value"
            ]

            st.dataframe(
                similar_results,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Similarity is based on standardized age, minutes, "
                "goals, assists and per-90 production. "
                "It is a statistical similarity measure, not a scouting verdict."
            )

    # --------------------------------------------------------
    # ML VALUATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'AI Valuation Engine'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Machine-learning estimates based on the trained player value model'
        '</div>',
        unsafe_allow_html=True
    )

    if model is not None:

        valuation_player = st.selectbox(
            "Select a player",
            sorted(
                analytics_df["name"]
                .astype(str)
                .unique()
                .tolist()
            ),
            key="analytics_valuation_player"
        )

        valuation_rows = analytics_df[
            analytics_df["name"].astype(str)
            == str(valuation_player)
        ]

        if not valuation_rows.empty:

            valuation_row = valuation_rows.iloc[0]

            try:

                model_input = pd.DataFrame(
                    [{
                        "age": valuation_row["age"],
                        "appearances": valuation_row["appearances"],
                        "minutes": valuation_row["minutes"],
                        "goals": valuation_row["goals"],
                        "assists": valuation_row["assists"],
                        "goals_per_90": valuation_row["goals_per_90"],
                        "assists_per_90": valuation_row["assists_per_90"],
                        "minutes_per_appearance":
                            valuation_row.get(
                                "minutes_per_appearance",
                                0
                            ),
                        "position": valuation_row["position"],
                        "sub_position":
                            valuation_row.get(
                                "sub_position",
                                ""
                            ),
                        "club": valuation_row["club"]
                    }]
                )

                prediction_log = model.predict(
                    model_input
                )[0]

                predicted_value = max(
                    0,
                    float(
                        np.expm1(
                            prediction_log
                        )
                    )
                )

                listed_value = float(
                    valuation_row["market_value"]
                )

                difference = (
                    predicted_value
                    - listed_value
                )

                if listed_value > 0:

                    pct_difference = (
                        difference
                        / listed_value
                    ) * 100

                else:

                    pct_difference = 0

                vc1, vc2, vc3 = st.columns(3)

                with vc1:

                    html(
                        f"""
                        <div class="kpi">
                            <div class="kpi-label">
                                Model Estimate
                            </div>

                            <div class="kpi-value">
                                {money(predicted_value)}
                            </div>

                            <div class="kpi-small">
                                Statistical estimate
                            </div>
                        </div>
                        """
                    )

                with vc2:

                    html(
                        f"""
                        <div class="kpi">
                            <div class="kpi-label">
                                Listed Value
                            </div>

                            <div class="kpi-value">
                                {money(listed_value)}
                            </div>

                            <div class="kpi-small">
                                Dataset value
                            </div>
                        </div>
                        """
                    )

                with vc3:

                    html(
                        f"""
                        <div class="kpi">
                            <div class="kpi-label">
                                Model Difference
                            </div>

                            <div class="kpi-value">
                                {money(abs(difference))}
                            </div>

                            <div class="kpi-small">
                                {pct_difference:+.1f}% vs listed
                            </div>
                        </div>
                        """
                    )

                html(
                    """
                    <div style="
                        margin-top:16px;
                        padding:16px 18px;
                        border-radius:16px;
                        background:rgba(255,255,255,0.025);
                        border:1px solid rgba(255,255,255,0.06);
                        color:#7f8b9e;
                        font-size:12px;
                        line-height:1.7;
                    ">
                        <strong style="color:#9ba7b8;">
                            Model note:
                        </strong>
                        This is a model-implied statistical estimate,
                        not an objective transfer valuation.
                        The model was evaluated using a random test split
                        and should not be interpreted as a future-transfer
                        forecasting system.
                    </div>
                    """
                )

            except Exception as error:

                st.warning(
                    "The valuation model could not generate "
                    "an estimate for this player."
                )

    else:

        st.info(
            "The valuation model is currently unavailable."
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    mc1, mc2, mc3, mc4 = st.columns(4)

    with mc1:

        html(
            """
            <div class="kpi">
                <div class="kpi-label">
                    Model
                </div>

                <div class="kpi-value"
                     style="font-size:20px;">
                    Gradient Boosting
                </div>

                <div class="kpi-small">
                    Selected by MAE
                </div>
            </div>
            """
        )

    with mc2:

        html(
            """
            <div class="kpi">
                <div class="kpi-label">
                    MAE
                </div>

                <div class="kpi-value">
                    €8.23M
                </div>

                <div class="kpi-small">
                    Test split
                </div>
            </div>
            """
        )

    with mc3:

        html(
            """
            <div class="kpi">
                <div class="kpi-label">
                    RMSE
                </div>

                <div class="kpi-value">
                    €15.67M
                </div>

                <div class="kpi-small">
                    Test split
                </div>
            </div>
            """
        )

    with mc4:

        html(
            """
            <div class="kpi">
                <div class="kpi-label">
                    R²
                </div>

                <div class="kpi-value">
                    0.628
                </div>

                <div class="kpi-small">
                    Test split
                </div>
            </div>
            """
        )

    html(
        """
        <div style="
            margin-top:18px;
            padding:18px;
            border-radius:16px;
            background:rgba(255,255,255,0.025);
            border:1px solid rgba(255,255,255,0.06);
            color:#7f8b9e;
            font-size:12px;
            line-height:1.7;
        ">
            <strong style="color:#9ba7b8;">
                Dataset features:
            </strong>
            age, appearances, minutes, goals, assists,
            goals per 90, assists per 90, minutes per appearance,
            position, sub-position and club.
        </div>
        """
    )


# ============================================================
# ROUTER
# ============================================================

page = st.session_state["page"]

if page == "Home":

    home_page()

elif page == "Clubs":

    clubs_page()

elif page == "Club":

    club_page(
        st.session_state.get(
            "selected_club"
        )
    )

elif page == "Players":

    players_page()

elif page == "Player":

    player_page(
        st.session_state.get(
            "selected_player"
        )
    )

elif page == "Model":

    model_page()

elif page == "Community":

    community_page()
elif page == "ClubCommunity":

    club_community_page(
        st.session_state.get(
            "selected_community_club"
        )
    )   
else:

    home_page()
