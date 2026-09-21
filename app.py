import streamlit as st
import pandas as pd
import numpy as np
import joblib
import textwrap
import base64
import requests
from pathlib import Path

from club_assets import CLUBS


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
        return CLUBS[club]

    return {
        "short_name": club,
        "code": "",
        "color": "#42556a",
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
        st.info(
            "Fan Community is coming soon."
        )


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

    if (
        not club
        or club not in df["club"].unique()
    ):

        st.error("Club not found.")

        if st.button("Back to Clubs"):
            navigate("Clubs")

        return

    info = club_info(club)

    club_df = (
        df[df["club"] == club]
        .sort_values(
            "market_value",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_value = club_df["market_value"].sum()

    avg_age = club_df["age"].mean()

    top_player = club_df.iloc[0]["name"]

    top_value = club_df.iloc[0]["market_value"]

    logo = info.get("logo", "")

    logo_data = get_logo_data(logo)

    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button("← Back to Clubs"):

        navigate("Clubs")

    # ========================================================
    # CLUB HERO
    # ========================================================

    if logo_data:

        logo_html = f"""
        <img
            src="{logo_data}"
            alt="{club}"
            style="
                width:120px;
                height:120px;
                object-fit:contain;
                flex-shrink:0;
            "
        >
        """

    else:

        logo_html = """
        <div style="
            width:120px;
            height:120px;
            border-radius:24px;
            background:#111827;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:55px;
            flex-shrink:0;
        ">
            ⚽
        </div>
        """

    club_color = info.get(
        "color",
        "#7c3aed"
    )

    html(
        f"""
        <div style="
            position:relative;
            overflow:hidden;
            padding:34px;
            border-radius:28px;
            background:
                radial-gradient(
                    circle at 85% 15%,
                    {club_color}55,
                    transparent 38%
                ),
                linear-gradient(
                    135deg,
                    #101827,
                    #0c1320
                );
            border:1px solid rgba(255,255,255,0.08);
        ">

            <div style="
                display:flex;
                align-items:center;
                gap:28px;
                flex-wrap:wrap;
            ">

                {logo_html}

                <div>

                    <div style="
                        color:#7f8b9e;
                        font-size:11px;
                        font-weight:800;
                        letter-spacing:2px;
                        text-transform:uppercase;
                        margin-bottom:8px;
                    ">
                        PREMIER LEAGUE · 2024/25
                    </div>

                    <div style="
                        color:#ffffff;
                        font-size:42px;
                        font-weight:900;
                        letter-spacing:-1.5px;
                        line-height:1.05;
                    ">
                        {club}
                    </div>

                    <div style="
                        margin-top:12px;
                        color:#aab4c4;
                        font-size:15px;
                    ">
                        {info.get(
                            'short_name',
                            club
                        )}
                        &nbsp;·&nbsp;
                        {info.get(
                            'code',
                            ''
                        )}
                    </div>

                    <div style="
                        margin-top:12px;
                        color:#7f8b9e;
                        font-size:13px;
                    ">
                        Squad analytics, player values
                        and performance data
                    </div>

                </div>

            </div>

        </div>
        """
    )

    # ========================================================
    # CLUB OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Club Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A quick snapshot of the squad and its market profile'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Squad Size
                </div>

                <div class="kpi-value">
                    {len(club_df)}
                </div>

                <div class="kpi-small">
                    Players
                </div>

            </div>
            """
        )

    with c2:

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
                    Listed values
                </div>

            </div>
            """
        )

    with c3:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Average Age
                </div>

                <div class="kpi-value">
                    {avg_age:.1f}
                </div>

                <div class="kpi-small">
                    Years
                </div>

            </div>
            """
        )

    with c4:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Top Value
                </div>

                <div class="kpi-value">
                    {money(top_value)}
                </div>

                <div class="kpi-small">
                    {top_player}
                </div>

            </div>
            """
        )

    # ========================================================
    # TOP PLAYER
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Squad Leader'
        '</div>',
        unsafe_allow_html=True
    )

    leader_photo = get_photo(
        club_df.iloc[0]
    )

    if leader_photo:

        leader_photo_html = f"""
        <img
            src="{leader_photo}"
            alt="{top_player}"
            style="
                width:110px;
                height:130px;
                object-fit:contain;
                border-radius:18px;
                background:#111827;
            "
        >
        """

    else:

        leader_photo_html = """
        <div style="
            width:110px;
            height:130px;
            border-radius:18px;
            background:#111827;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:48px;
        ">
            ⚽
        </div>
        """

    leader_row = club_df.iloc[0]

    html(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:22px;
            padding:22px;
            border-radius:22px;
            background:rgba(255,255,255,0.035);
            border:1px solid rgba(255,255,255,0.07);
        ">

            {leader_photo_html}

            <div>

                <div style="
                    color:#7f8b9e;
                    font-size:11px;
                    font-weight:800;
                    letter-spacing:1.5px;
                    text-transform:uppercase;
                ">
                    Highest listed market value
                </div>

                <div style="
                    margin-top:7px;
                    color:#ffffff;
                    font-size:27px;
                    font-weight:900;
                ">
                    {top_player}
                </div>

                <div style="
                    margin-top:6px;
                    color:#9aa6b8;
                    font-size:14px;
                ">
                    {leader_row.get(
                        'position',
                        'Player'
                    )}
                    &nbsp;·&nbsp;
                    {int(
                        leader_row['appearances']
                    )}
                    appearances
                    &nbsp;·&nbsp;
                    {int(
                        leader_row['goals']
                    )}
                    goals
                </div>

                <div style="
                    margin-top:12px;
                    color:#ffffff;
                    font-size:22px;
                    font-weight:850;
                ">
                    {money(top_value)}
                </div>

            </div>

        </div>
        """
    )

    # ========================================================
    # SQUAD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Squad'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Players ranked by listed market value'
        '</div>',
        unsafe_allow_html=True
    )

    for start in range(
        0,
        len(club_df),
        4
    ):

        rows = club_df.iloc[
            start:start + 4
        ]

        cols = st.columns(4)

        for col, (_, row) in zip(
            cols,
            rows.iterrows()
        ):

            photo = get_photo(row)

            with col:

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
                                {row.get(
                                    'position',
                                    'Player'
                                )}
                            </div>

                            <div class="player-value">
                                {money(
                                    row['market_value']
                                )}
                            </div>

                            <div class="player-meta">
                                {int(
                                    row['appearances']
                                )}
                                appearances
                                ·
                                {int(
                                    row['goals']
                                )}
                                goals
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

# ============================================================
# PLAYERS PAGE
# ============================================================

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
                    "Open Profile",
                    key=f"player_{row['name']}_{start}",
                    use_container_width=True
                ):

                    navigate(
                        "Player",
                        selected_player=row["name"]
                    )


# ============================================================
# PLAYER PAGE
# ============================================================

def player_page(player_name):

    player_rows = df[
        df["name"].astype(str)
        == str(player_name)
    ]

    if player_rows.empty:

        st.error(
            "Player not found."
        )

        if st.button(
            "Back to Players"
        ):

            navigate("Players")

        return

    row = player_rows.iloc[0]

    club = row["club"]

    info = club_info(club)

    logo = info.get(
        "logo",
        ""
    )

    logo_data = get_logo_data(
        logo
    )


    c1, c2 = st.columns(
        [1, 5]
    )

    with c1:

        if st.button(
            "← Players"
        ):

            navigate("Players")

    with c2:

        if st.button(
            f"← {club}"
        ):

            navigate(
                "Club",
                selected_club=club
            )


    photo = get_photo(row)

    if photo:

        photo_html = f"""
        <img
            src="{photo}"
            style="
                width:220px;
                height:280px;
                object-fit:contain;
                background:#111827;
                border-radius:22px;
            "
        >
        """

    else:

        photo_html = """
        <div style="
            width:220px;
            height:280px;
            border-radius:22px;
            background:#111827;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:90px;
        ">
            ⚽
        </div>
        """


    if logo_data:

        club_logo_html = f"""
        <img
            src="{logo_data}"
            style="
                width:42px;
                height:42px;
                object-fit:contain;
                vertical-align:middle;
                margin-right:8px;
            "
        >
        """

    else:

        club_logo_html = "⚽ "


    html(
        f"""
        <div style="
            display:flex;
            gap:35px;
            align-items:center;
            padding:35px;
            border-radius:28px;
            background:
                radial-gradient(
                    circle at 80% 20%,
                    {info.get('color','#7c3aed')}44,
                    transparent 38%
                ),
                #101827;
            border:1px solid rgba(255,255,255,0.08);
        ">

            {photo_html}

            <div>

                <div class="hero-kicker">
                    PLAYER PROFILE
                </div>

                <div style="
                    font-size:48px;
                    font-weight:900;
                    letter-spacing:-2px;
                    line-height:1;
                ">
                    {row['name']}
                </div>

                <div style="
                    margin-top:18px;
                    color:#aab4c4;
                    font-size:16px;
                ">
                    {club_logo_html}
                    {club}
                </div>

                <div style="
                    margin-top:10px;
                    color:#7f8b9e;
                    font-size:14px;
                ">
                    {row.get(
                        'position',
                        'Player'
                    )}
                    ·
                    {row.get(
                        'sub_position',
                        ''
                    )}
                </div>

            </div>

        </div>
        """
    )


    # STATISTICS

    st.markdown(
        '<div class="section-title">'
        'Player Statistics'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Market Value
                </div>

                <div class="kpi-value">
                    {money(
                        row['market_value']
                    )}
                </div>

                <div class="kpi-small">
                    Listed value
                </div>

            </div>
            """
        )

    with c2:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Appearances
                </div>

                <div class="kpi-value">
                    {int(
                        row['appearances']
                    )}
                </div>

                <div class="kpi-small">
                    2024/25
                </div>

            </div>
            """
        )

    with c3:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Goals
                </div>

                <div class="kpi-value">
                    {int(row['goals'])}
                </div>

                <div class="kpi-small">
                    2024/25
                </div>

            </div>
            """
        )

    with c4:

        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Assists
                </div>

                <div class="kpi-value">
                    {int(row['assists'])}
                </div>

                <div class="kpi-small">
                    2024/25
                </div>

            </div>
            """
        )


    # PERFORMANCE

    st.markdown(
        '<div class="section-title">'
        'Performance Profile'
        '</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        html(
            f"""
            <div class="info-box">

                <div class="info-label">
                    Age
                </div>

                <div class="info-value">
                    {row['age']:.0f}
                </div>

            </div>
            """
        )

    with p2:

        html(
            f"""
            <div class="info-box">

                <div class="info-label">
                    Minutes
                </div>

                <div class="info-value">
                    {int(row['minutes']):,}
                </div>

            </div>
            """
        )

    with p3:

        html(
            f"""
            <div class="info-box">

                <div class="info-label">
                    Goals / 90
                </div>

                <div class="info-value">
                    {row['goals_per_90']:.2f}
                </div>

            </div>
            """
        )

    with p4:

        html(
            f"""
            <div class="info-box">

                <div class="info-label">
                    Assists / 90
                </div>

                <div class="info-value">
                    {row['assists_per_90']:.2f}
                </div>

            </div>
            """
        )


    # ML MODEL

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Valuation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Model-implied market value based on player characteristics and 2024/25 performance'
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

            listed_value = float(
                row["market_value"]
            )

            difference = (
                predicted_value
                - listed_value
            )


            html(
                f"""
                <div class="model-box">

                    <div class="model-title">
                        Model-Implied Value
                    </div>

                    <div class="model-value">
                        {money(predicted_value)}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#8490a3;
                        font-size:13px;
                    ">
                        Listed market value:
                        <strong>
                            {money(listed_value)}
                        </strong>
                    </div>

                </div>
                """
            )


            st.write("")

            c1, c2 = st.columns(2)

            with c1:

                html(
                    f"""
                    <div class="info-box">

                        <div class="info-label">
                            Model Difference
                        </div>

                        <div class="info-value">
                            {money(
                                abs(difference)
                            )}
                        </div>

                        <div style="
                            margin-top:5px;
                            color:#718096;
                            font-size:12px;
                        ">
                            {
                                'Model above listed value'
                                if difference >= 0
                                else
                                'Model below listed value'
                            }
                        </div>

                    </div>
                    """
                )

            with c2:

                if listed_value > 0:

                    pct_difference = (
                        difference
                        / listed_value
                    ) * 100

                else:

                    pct_difference = 0

                html(
                    f"""
                    <div class="info-box">

                        <div class="info-label">
                            Relative Difference
                        </div>

                        <div class="info-value">
                            {pct_difference:+.1f}%
                        </div>

                        <div style="
                            margin-top:5px;
                            color:#718096;
                            font-size:12px;
                        ">
                            Model vs listed value
                        </div>

                    </div>
                    """
                )

        except Exception as e:

            st.warning(
                "The model prediction could not be generated."
            )

            with st.expander(
                "Technical details"
            ):

                st.write(
                    str(e)
                )

    else:

        st.info(
            "ML model not found. Make sure "
            "'models/best_player_value_model.pkl' exists."
        )


# ============================================================
# MODEL PAGE
# ============================================================

def model_page():

    st.markdown(
        '<div class="section-title">'
        'Player Value Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Machine-learning valuation engine trained on Premier League player data'
        '</div>',
        unsafe_allow_html=True
    )


    if model is None:

        st.error(
            "Model file not found: "
            "models/best_player_value_model.pkl"
        )

        return


    c1, c2, c3, c4 = st.columns(4)

    with c1:

        html(
            """
            <div class="kpi">

                <div class="kpi-label">
                    Model
                </div>

                <div class="kpi-value"
                     style="font-size:22px;">
                    Gradient Boosting
                </div>

                <div class="kpi-small">
                    Selected by MAE
                </div>

            </div>
            """
        )

    with c2:

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

    with c3:

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

    with c4:

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


    st.markdown(
        '<div class="section-title">'
        'How It Works'
        '</div>',
        unsafe_allow_html=True
    )

    html(
        """
        <div class="info-box">

            <div style="
                font-size:20px;
                font-weight:800;
                margin-bottom:15px;
            ">
                Player characteristics →
                ML model →
                estimated value
            </div>

            <div style="
                color:#8a96a8;
                line-height:1.8;
                font-size:14px;
            ">

                The model uses player age, appearances,
                minutes, goals, assists, scoring rate,
                assist rate, minutes per appearance,
                position, sub-position and club.

                <br><br>

                The target market value is transformed
                using a logarithmic transformation before
                training.

            </div>

        </div>
        """
    )


    st.markdown(
        '<div class="section-title">'
        'Model Features'
        '</div>',
        unsafe_allow_html=True
    )

    feature_data = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Appearances",
                "Minutes",
                "Goals",
                "Assists",
                "Goals / 90",
                "Assists / 90",
                "Minutes / Appearance",
                "Position",
                "Sub-position",
                "Club"
            ],

            "Type": [
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Categorical",
                "Categorical",
                "Categorical"
            ]
        }
    )

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        '<div class="section-title">'
        'Tested Models'
        '</div>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Gradient Boosting",
                "Linear Regression",
                "Extra Trees",
                "Random Forest"
            ],

            "MAE": [
                "€8.23M",
                "€8.62M",
                "€9.52M",
                "€10.80M"
            ],

            "RMSE": [
                "€15.67M",
                "€14.84M",
                "€17.95M",
                "€19.39M"
            ],

            "R²": [
                "0.628",
                "0.666",
                "0.512",
                "0.430"
            ]
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    html(
        """
        <div style="
            margin-top:18px;
            padding:18px;
            border-radius:16px;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.06);
            color:#7f8b9e;
            font-size:12px;
            line-height:1.7;
        ">

            Note: the model is a statistical estimate
            rather than an objective market valuation.
            The current training setup uses an 80/20
            random split, so it should not be interpreted
            as a true future-transfer forecasting model.

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

else:

    home_page()