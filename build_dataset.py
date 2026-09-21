import duckdb
import pandas as pd
import numpy as np
import requests
import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from datetime import date


# ============================================================
# PREMIER LEAGUE ANALYTICS LAB
# 2026/27 DATASET BUILDER
# ============================================================

print()
print("=" * 60)
print(" PREMIER LEAGUE ANALYTICS LAB")
print(" 2026/27 SQUAD DATASET BUILDER")
print("=" * 60)
print()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "transfermarkt.duckdb"

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "premier_league_players.csv"
)


# ============================================================
# OFFICIAL PREMIER LEAGUE SOURCE
# ============================================================

PL_SQUADS_URL = (
    "https://www.premierleague.com/en/news/"
    "4706139/see-all-the-202627-premier-league-squad-lists"
)


# ============================================================
# OFFICIAL 2026/27 CLUBS
# ============================================================

PREMIER_LEAGUE_CLUBS = [
    "AFC Bournemouth",
    "Arsenal",
    "Aston Villa",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
]


# ============================================================
# CLUB NAME NORMALIZATION
# ============================================================

CLUB_ALIASES = {

    "AFC Bournemouth": [
        "AFC Bournemouth",
        "Bournemouth",
    ],

    "Arsenal": [
        "Arsenal",
        "Arsenal FC",
    ],

    "Aston Villa": [
        "Aston Villa",
        "Aston Villa FC",
    ],

    "Brentford": [
        "Brentford",
        "Brentford FC",
    ],

    "Brighton & Hove Albion": [
        "Brighton & Hove Albion",
        "Brighton",
        "Brighton and Hove Albion",
        "Brighton & Hove Albion FC",
    ],

    "Chelsea": [
        "Chelsea",
        "Chelsea FC",
    ],

    "Coventry City": [
        "Coventry City",
        "Coventry",
    ],

    "Crystal Palace": [
        "Crystal Palace",
        "Crystal Palace FC",
    ],

    "Everton": [
        "Everton",
        "Everton FC",
    ],

    "Fulham": [
        "Fulham",
        "Fulham FC",
    ],

    "Hull City": [
        "Hull City",
        "Hull",
    ],

    "Ipswich Town": [
        "Ipswich Town",
        "Ipswich",
    ],

    "Leeds United": [
        "Leeds United",
        "Leeds",
    ],

    "Liverpool": [
        "Liverpool",
        "Liverpool FC",
    ],

    "Manchester City": [
        "Manchester City",
        "Man City",
    ],

    "Manchester United": [
        "Manchester United",
        "Man Utd",
        "Manchester United FC",
    ],

    "Newcastle United": [
        "Newcastle United",
        "Newcastle",
    ],

    "Nottingham Forest": [
        "Nottingham Forest",
        "Nottingham Forest FC",
    ],

    "Sunderland": [
        "Sunderland",
        "Sunderland AFC",
    ],

    "Tottenham Hotspur": [
        "Tottenham Hotspur",
        "Tottenham",
        "Spurs",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()

    value = value.replace(
        "ø",
        "o"
    )

    value = value.replace(
        "ð",
        "d"
    )

    value = value.replace(
        "þ",
        "th"
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return " ".join(
        value.split()
    )


def clean_player_name(name):

    name = str(name)

    # Remove PL home-grown marker
    name = name.replace("*", "")

    # Remove loan marker
    name = re.sub(
        r"\(Loan\)",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove extra whitespace
    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name


def split_official_name(name):

    """
    Premier League lists often use:
        Saka, Bukayo

    Convert to:
        Bukayo Saka
    """

    name = clean_player_name(name)

    if "," in name:

        parts = [
            x.strip()
            for x in name.split(
                ",",
                1
            )
        ]

        if len(parts) == 2:

            surname = parts[0]
            first_name = parts[1]

            return (
                first_name
                + " "
                + surname
            )

    return name


def similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# DOWNLOAD OFFICIAL PL SQUADS
# ============================================================

def download_squad_page():

    print(
        "Downloading official 2026/27 "
        "Premier League squad lists..."
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0 Safari/537.36"
    }

    response = requests.get(
        PL_SQUADS_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    print(
        "Official Premier League page downloaded."
    )

    return response.text


# ============================================================
# PARSE OFFICIAL SQUADS
# ============================================================

def parse_squads(html):

    print()
    print(
        "Reading official 2026/27 squad lists..."
    )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = soup.get_text(
        "\n"
    )

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if line:
            lines.append(line)

    # --------------------------------------------------------
    # Club detection
    # --------------------------------------------------------

    club_lookup = {}

    for official_club in PREMIER_LEAGUE_CLUBS:

        for alias in CLUB_ALIASES[
            official_club
        ]:

            club_lookup[
                normalize_text(alias)
            ] = official_club


    squads = {
        club: []
        for club in PREMIER_LEAGUE_CLUBS
    }

    current_club = None

    inside_squad = False

    for line in lines:

        normalized = normalize_text(
            line
        )

        # ----------------------------------------------------
        # Detect club
        # ----------------------------------------------------

        if normalized in club_lookup:

            current_club = club_lookup[
                normalized
            ]

            inside_squad = False

            continue


        # ----------------------------------------------------
        # Detect squad section
        # ----------------------------------------------------

        if (
            "squad players" in normalized
            and current_club
        ):

            inside_squad = True

            continue


        # ----------------------------------------------------
        # Detect U21 section
        # ----------------------------------------------------

        if (
            "u21 players" in normalized
            and current_club
        ):

            # We intentionally stop here.
            # U21 players are not part of the
            # 25-man squad list.
            inside_squad = False

            continue


        # ----------------------------------------------------
        # Stop at next article section
        # ----------------------------------------------------

        if (
            "what is a home grown player" in normalized
            or "what constitutes an under 21 player"
            in normalized
        ):

            break


        # ----------------------------------------------------
        # Store player
        # ----------------------------------------------------

        if (
            current_club
            and inside_squad
        ):

            # Ignore obvious non-player text
            if len(line) < 3:
                continue

            if line.lower() in [
                "25 squad players",
                "squad players",
            ]:
                continue

            # Avoid paragraphs
            if len(line) > 100:
                continue

            player_name = split_official_name(
                line
            )

            # Don't accidentally include headings
            if any(
                keyword in normalize_text(
                    player_name
                )
                for keyword in [
                    "see all",
                    "home grown",
                    "players",
                    "contract and scholars",
                    "loan"
                ]
            ):

                # Loan is handled as part of
                # player names, but these other
                # headings should be ignored.
                if "loan" not in normalize_text(
                    player_name
                ):
                    continue

            if player_name:

                squads[
                    current_club
                ].append(
                    player_name
                )


    # --------------------------------------------------------
    # Clean duplicates
    # --------------------------------------------------------

    for club in squads:

        unique = []

        seen = set()

        for player in squads[club]:

            key = normalize_text(
                player
            )

            if key not in seen:

                seen.add(key)

                unique.append(
                    player
                )

        squads[club] = unique


    print()

    for club in PREMIER_LEAGUE_CLUBS:

        print(
            f"{club:<30} "
            f"{len(squads[club]):>2} players"
        )


    return squads


# ============================================================
# FALLBACK OFFICIAL SQUAD DATA
#
# This protects the build if the PL page changes
# its HTML structure.
# ============================================================

def validate_squads(squads):

    valid = True

    for club in PREMIER_LEAGUE_CLUBS:

        count = len(
            squads.get(
                club,
                []
            )
        )

        if count == 0:

            print(
                f"WARNING: No players found for {club}"
            )

            valid = False

        elif count < 15:

            print(
                f"WARNING: Only {count} players "
                f"found for {club}"
            )

            valid = False

    return valid


# ============================================================
# LOAD LOCAL TRANSFERMARKT DATA
# ============================================================

def load_transfermarkt():

    print()
    print(
        "Connecting to Transfermarkt DuckDB..."
    )

    if not DB_PATH.exists():

        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}"
        )

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True
    )

    print(
        "Loading players..."
    )

    players = con.execute(
        """
        SELECT
            player_id,
            first_name,
            last_name,
            name,
            date_of_birth,
            sub_position,
            position,
            foot,
            height_in_cm,
            image_url,
            current_club_name,
            market_value_in_eur,
            highest_market_value_in_eur
        FROM players
        """
    ).fetchdf()

    print(
        "Loading appearances..."
    )

    appearances = con.execute(
        """
        SELECT
            player_id,
            player_name,
            competition_id,
            SUM(minutes_played) AS minutes,
            SUM(goals) AS goals,
            SUM(assists) AS assists,
            COUNT(
                DISTINCT appearance_id
            ) AS appearances
        FROM appearances
        WHERE competition_id = 'GB1'
        GROUP BY
            player_id,
            player_name,
            competition_id
        """
    ).fetchdf()

    print(
        "Loading valuations..."
    )

    valuations = con.execute(
        """
        SELECT
            player_id,
            market_value_in_eur,
            date
        FROM player_valuations
        WHERE date = (
            SELECT MAX(date)
            FROM player_valuations
        )
        """
    ).fetchdf()

    con.close()

    print(
        f"Transfermarkt players loaded: "
        f"{len(players):,}"
    )

    print(
        f"Premier League appearance records: "
        f"{len(appearances):,}"
    )

    return (
        players,
        appearances,
        valuations
    )


# ============================================================
# CREATE PLAYER LOOKUP
# ============================================================
def create_player_lookup(players):
    """
    Build fast indexes for Transfermarkt players.

    Indexes:
      - exact normalized name
      - first + last name
      - last name
    """

    exact_lookup = {}
    first_last_lookup = {}
    last_lookup = {}
    token_lookup = {}

    for _, row in players.iterrows():

        player = row.to_dict()

        raw_names = set()

        if pd.notna(row.get("name")):
            raw_names.add(str(row["name"]))

        first = str(row.get("first_name", "") or "").strip()
        last = str(row.get("last_name", "") or "").strip()

        if first or last:
            raw_names.add(f"{first} {last}".strip())

        for raw_name in raw_names:

            normalized = normalize_text(raw_name)

            if not normalized:
                continue

            exact_lookup.setdefault(normalized, []).append(player)

            parts = normalized.split()

            if len(parts) >= 2:
                first_last_key = (
                    parts[0],
                    parts[-1]
                )
                first_last_lookup.setdefault(
                    first_last_key, []
                ).append(player)

                last_lookup.setdefault(
                    parts[-1], []
                ).append(player)

            # Index every meaningful name token. This is important
            # for official PL legal names vs Transfermarkt common names.
            for token in set(parts):
                if len(token) >= 4:
                    token_lookup.setdefault(
                        token, []
                    ).append(player)

    return {
        "exact": exact_lookup,
        "first_last": first_last_lookup,
        "last": last_lookup,
        "token": token_lookup,
    }


def _first_name_variants(first_name):
    """Return common short/full-name variants."""

    first_name = normalize_text(first_name)

    aliases = {
        "benjamin": {"benjamin", "ben"},
        "christopher": {"christopher", "chris"},
        "joseph": {"joseph", "joe", "joey"},
        "alexander": {"alexander", "alex"},
        "daniel": {"daniel", "dan", "danny"},
        "dominic": {"dominic", "dom"},
        "thomas": {"thomas", "tom", "tommy"},
        "william": {"william", "will"},
        "matthew": {"matthew", "matt"},
        "charles": {"charles", "charlie"},
        "edward": {"edward", "ed", "eddie"},
        "frederick": {"frederick", "fred"},
        "nicholas": {"nicholas", "nick"},
        "jonathan": {"jonathan", "jon"},
        "joshua": {"joshua", "josh"},
        "maximillian": {"maximillian", "max"},
        "manuel": {"manuel", "manu"},
        "alvaro": {"alvaro", "alvaro"},
        "francisco": {"francisco", "fran"},
        "gabriel": {"gabriel", "gabi"},
        "martin": {"martin", "marti"},
        "mikel": {"mikel"},
        "piero": {"piero"},
        "emiliano": {"emiliano", "emi"},
        "kevin": {"kevin"},
        "pascal": {"pascal"},
        "joao": {"joao", "joao"},
        "jose": {"jose", "josé"},
        "moises": {"moises", "moises"},
        "brandon": {"brandon"},
        "ellis": {"ellis"},
        "jack": {"jack"},
        "victor": {"victor"},
        "axel": {"axel"},
        "yeremy": {"yeremy"},
        "carlos": {"carlos"},
        "fraser": {"fraser"},
        "hayden": {"hayden"},
        "gustavo": {"gustavo"},
        "jorge": {"jorge"},
        "rodrigo": {"rodrigo"},
        "terrell": {"terrell"},
        "charles": {"charles", "charlie"},
        "lewie": {"lewie", "lewis"},
        "oluwa": {"oluwa", "semi", "semedo"},
        "anis": {"anis"},
        "emersonn": {"emersonn", "emerson"},
        "florentino": {"florentino", "florentino"},
        "jaden": {"jaden", "jayden"},
        "marcelino": {"marcelino", "marcelo"},
        "mateo": {"mateo", "matteo"},
        "alisson": {"alisson", "alison"},
        "isaac": {"isaac"},
        "ronald": {"ronald"},
        "allan": {"allan", "alan"},
        "matheus": {"matheus", "mathew"},
        "nicolas": {"nicolas", "nico"},
        "valentino": {"valentino", "tino"},
        "aaron": {"aaron"},
        "arnaud": {"arnaud"},
        "igor": {"igor"},
        "murillo": {"murillo"},
        "temitayo": {"temitayo", "tayo"},
        "abdul": {"abdul"},
        "benjamin": {"benjamin", "ben"},
        "mykhailo": {"mykhailo", "mykhailo"},
        "pedro": {"pedro"},
        "jacob": {"jacob", "jake"},
        "marcos": {"marcos", "marc"},
        "mikel": {"mikel"},
        "diego": {"diego"},
        "ferdi": {"ferdi", "ferdi"},
    }

    return aliases.get(
        first_name,
        {first_name}
    )


def _name_similarity(official_norm, candidate_norm):
    """Score names while handling long legal names and nicknames."""

    if not official_norm or not candidate_norm:
        return 0.0

    official_parts = official_norm.split()
    candidate_parts = candidate_norm.split()

    if not official_parts or not candidate_parts:
        return 0.0

    # Strong signal: same surname.
    same_last = official_parts[-1] == candidate_parts[-1]

    # Also allow compound surname endings.
    same_last_two = (
        len(official_parts) >= 2
        and len(candidate_parts) >= 2
        and official_parts[-2:] == candidate_parts[-2:]
    )

    official_first = official_parts[0]
    candidate_first = candidate_parts[0]

    first_variants = _first_name_variants(official_first)

    first_match = candidate_first in first_variants

    # Short/full first-name similarity, e.g. Phil/Philip.
    first_similarity = SequenceMatcher(
        None,
        official_first,
        candidate_first
    ).ratio()

    official_tokens = set(official_parts)
    candidate_tokens = set(candidate_parts)
    common = official_tokens.intersection(candidate_tokens)

    score = SequenceMatcher(
        None,
        official_norm,
        candidate_norm
    ).ratio()

    if same_last:
        score += 0.18

    if same_last_two:
        score += 0.08

    if first_match:
        score += 0.20

    elif first_similarity >= 0.72:
        score += 0.12

    if len(common) >= 2:
        score += 0.08

    if len(common) >= 3:
        score += 0.05

    return min(score, 1.0)


# Verified/common-name aliases for official Premier League legal names.
# These are used BEFORE fuzzy matching, so they do not weaken the
# conservative matcher or create arbitrary player identities.
MANUAL_NAME_ALIASES = {
    "Adam James Smith": "Adam Smith",
    "Djorde Petrovic": "Djordje Petrovic",
    "Francisco Evanilson De Lima Barbosa": "Evanilson",
    "Juan Luis Sanchez Velasco": "Juanlu Sanchez",
    "Julian Vicente Araujo Zuniga": "Julian Araujo",
    "Chukwunonso Azuka Tristan Madueke": "Noni Madueke",
    "Cristhian Andrey Mosquera Ibargüen": "Cristhian Mosquera",
    "Gabriel Dos Santos Magalhaes": "Gabriel",
    "Kepa Arrizabalaga Revuelta": "Kepa Arrizabalaga",
    "Martin Zubimendi Ibanez": "Martin Zubimendi",
    "Piero Martin Hincapie Reyna": "Piero Hincapie",
    "Alejandro Garnacho Ferreyra": "Alejandro Garnacho",
    "Joao Victor Gomes Da Silva": "Joao Gomes",
    "Kevin Oghenetega Tamaraebi Bakumo-Abraham": "Tammy Abraham",
    "Matthew Stuart Cash": "Matty Cash",
    "Callum Eddie Graham Wilson": "Callum Wilson",
    "Igor Thiago Nascimento Rodrigues": "Igor Thiago",
    "Pelenda Joshua Tunga Dasilva": "Josh Dasilva",
    "Yehor Yarmoliuk": "Yegor Yarmolyuk",
    "Diego Alexander Gomez Amarilla": "Diego Gomez",
    "Joao Pedro Loureiro Da Costa": "Joao Pedro",
    "Oluwafemi Javier Azeez Beloso": "Femi Azeez",
    "Pascal Alexander Gross": "Pascal Groß",
    "Joao Pedro Junqueira De Jesus": "Joao Pedro",
    "Josep Maria Chavarria Perez": "Pep Chavarria",
    "Moises Isaac Caicedo Corozo": "Moises Caicedo",
    "Ben Wilson": "Ben Wilson",
    "Ephron Jardell Mason-Clark": "Ephron Mason-Clark",
    "Jack Edward Rudoni": "Jack Rudoni",
    "Joshua Elliot Eccles": "Josh Eccles",
    "Victor Torp Overgaard": "Victor Torp",
    "William James Hughes": "Will Hughes",
    "Yeremy Jesus Pino Santos": "Yeremy Pino",
    "Carlos Jonas Alcaraz Duran": "Carlos Alcaraz",
    "Fraser Paul Barnsley": "Fraser Barnsley",
    "Hayden Rhys Hackney": "Hayden Hackney",
    "Gonzalo Garcia Torres": "Gonzalo Garcia",
    "Kevin Santos Lopes de Macedo": "Kevin",
    "Manuel Ángel Moran Ibanez": "Manuel Morán",
    "Michael Thomas Allen": "Michael Allen",
    "Rodrigo Muniz Carvalho": "Rodrigo Muniz",
    "Terrell Lawrence Isaiah Works": "Terrell Works",
    "Charles Roger Hughes": "Charlie Hughes",
    "Lewie Jacob Coyle": "Lewie Coyle",
    "Oluwasemilogo Adesewo Ibidapo Ajayi": "Semi Ajayi",
    "Patrick James Coleman McNair": "Paddy McNair",
    "Florentino Ibrain Morris Luis": "Florentino",
    "Julio Cesar Enciso Espinola": "Julio Enciso",
    "Marcelino Ignacio Nunez Espinoza": "Marcelino Nunez",
    "Alisson Ramses Becker": "Alisson",
    "Joseph David Gomez": "Joe Gomez",
    "Ronald Federico Araujo da Silva": "Ronald Araujo",
    "Victor Munoz Villanueva": "Victor Munoz",
    "Allan Andrade Elias": "Allan",
    "Manuel Ugarte Ribeiro": "Manuel Ugarte",
    "Matheus Santos Carneiro da Cunha": "Matheus Cunha",
    "Joelinton Cassio Apolinario De Lira": "Joelinton",
    "Nicolas Gonzalez Iglesias": "Nico González",
    "Daniel Munoz Mejia": "Daniel Munoz",
    "John Victor Maciel Furtado": "John Victor",
    "Murillo Santiago Costa Dos Santos": "Murillo",
    "Temitayo Olufisayo Olaoluwa Aina": "Ola Aina",
    "Ajibola Joshua Odunayo Alese": "Ajibola Alese",
    "Alan Browne": "Alan Browne",
    "Mouhamadou Habib Diarra": "Habib Diarra",
    "Nilson David Angulo Ramirez": "Nilson Angulo",
    "Abdul-Nasir Oluwatosin Adarabioyo": "Tosin Adarabioyo",
    "Benjamin Thomas Davies": "Ben Davies",
    "Dominic Ayodele Solanke-Mitchell": "Dominic Solanke",
    "Marcos Nicolas Senesi Baron": "Marcos Senesi",
    "Savio Moreira De Oliveira": "Savinho",
}


def get_manual_alias(official_name):
    key = normalize_text(clean_player_name(official_name))
    for raw, target in MANUAL_NAME_ALIASES.items():
        if normalize_text(raw) == key:
            return target
    return None


def match_player(
    official_name,
    players,
    player_lookup
):
    """
    Fast, conservative player matching.

    Exact and first/last matches are preserved first.
    Fuzzy matching is only attempted for candidates
    sharing a surname, first name, or strong name token.

    Returns:
        (matched_row, score, match_type)
    """

    if not official_name:
        return None, 0.0, "missing"

    official_clean = clean_player_name(official_name)
    official_norm = normalize_text(official_clean)

    if not official_norm:
        return None, 0.0, "missing"

    exact_lookup = player_lookup["exact"]
    first_last_lookup = player_lookup["first_last"]
    last_lookup = player_lookup["last"]
    token_lookup = player_lookup["token"]

    # =========================================================
    # 1. VERIFIED MANUAL / COMMON-NAME ALIAS
    # =========================================================

    alias_name = get_manual_alias(official_clean)

    if alias_name:
        alias_norm = normalize_text(alias_name)
        alias_matches = exact_lookup.get(alias_norm, [])

        if len(alias_matches) == 1:
            return alias_matches[0], 1.0, "manual_alias"

        # If the alias is not an exact DB name, let the normal
        # first/last and candidate matching continue using it.
        if len(alias_matches) == 1:
            return alias_matches[0], 0.99, "manual_alias"

        # Use the alias as the search name for the remaining
        # matching stages. This handles accents/spacing differences.
        official_norm = alias_norm

    # =========================================================
    # 2. EXACT MATCH
    # =========================================================

    exact_matches = exact_lookup.get(
        official_norm,
        []
    )

    if len(exact_matches) == 1:
        return exact_matches[0], 1.0, "exact"

    if exact_matches:

        for player in exact_matches:

            player_name = normalize_text(
                clean_player_name(
                    str(player.get("name", ""))
                )
            )

            if player_name == official_norm:
                return player, 1.0, "exact"

        return exact_matches[0], 0.98, "exact"

    # =========================================================
    # 2. FIRST + LAST MATCH
    # =========================================================

    parts = official_norm.split()

    if len(parts) >= 2:

        key = (
            parts[0],
            parts[-1]
        )

        candidates = first_last_lookup.get(
            key,
            []
        )

        if len(candidates) == 1:
            return candidates[0], 0.95, "first_last"

    # =========================================================
    # 3. LAST NAME CANDIDATES
    # =========================================================

    last_name = parts[-1]

    candidates = list(
        last_lookup.get(
            last_name,
            []
        )
    )

    # If no exact surname, use the indexed name-token pool.
    # This catches cases where the official PL list contains a
    # long legal surname but Transfermarkt stores the common name.
    if not candidates:

        token_pool = {}

        for token in parts:

            if len(token) < 4:
                continue

            for player in token_lookup.get(token, []):
                token_pool[player.get("player_id")] = player

        candidates = list(token_pool.values())

    if not candidates:
        return None, 0.0, "missing"

    # =========================================================
    # 4. SCORE CANDIDATES
    # =========================================================

    scored = []

    for player in candidates:

        candidate_name = normalize_text(
            clean_player_name(
                str(player.get("name", ""))
            )
        )

        if not candidate_name:
            continue

        score = _name_similarity(
            official_norm,
            candidate_name
        )

        scored.append(
            (
                score,
                player
            )
        )

    if not scored:
        return None, 0.0, "missing"

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_player = scored[0]

    second_score = (
        scored[1][0]
        if len(scored) > 1
        else 0.0
    )

    best_name = normalize_text(
        clean_player_name(
            str(best_player.get("name", ""))
        )
    )
    best_parts = best_name.split()

    official_first = parts[0]
    official_last = parts[-1]

    first_ok = (
        best_parts
        and (
            best_parts[0] in _first_name_variants(official_first)
            or official_first in _first_name_variants(best_parts[0])
            or SequenceMatcher(
                None,
                official_first,
                best_parts[0]
            ).ratio() >= 0.82
        )
    )

    last_ok = (
        best_parts
        and (
            best_parts[-1] == official_last
            or official_last in best_parts
            or best_parts[-1] in parts
        )
    )

    common_tokens = set(parts).intersection(best_parts)

    # Very strong long-name match.
    if (
        best_score >= 0.92
        and (
            len(scored) == 1
            or best_score - second_score >= 0.04
        )
    ):
        return best_player, best_score, "fuzzy_strong"

    # First-name + surname evidence.
    if (
        first_ok
        and last_ok
        and (
            len(scored) == 1
            or best_score - second_score >= 0.03
        )
    ):
        return best_player, best_score, "name_parts"

    # Two or more shared meaningful tokens with a clear lead.
    if (
        len(common_tokens) >= 2
        and best_score >= 0.72
        and (
            len(scored) == 1
            or best_score - second_score >= 0.05
        )
    ):
        return best_player, best_score, "token_match"

    # Unique high-quality candidate.
    if (
        best_score >= 0.88
        and len(scored) == 1
    ):
        return best_player, best_score, "fuzzy_unique"

    return None, 0.0, "missing"


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    squads,
    players,
    appearances,
    valuations
):

    print()
    print("=" * 60)
    print(" BUILDING 2026/27 DATASET")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Normalize local names
    # --------------------------------------------------------

    players = players.copy()

    players["_name_normalized"] = (
        players["name"]
        .fillna("")
        .apply(normalize_text)
    )


    # --------------------------------------------------------
    # Appearance aggregation
    # --------------------------------------------------------

    appearances = appearances.copy()

    appearances["minutes"] = pd.to_numeric(
        appearances["minutes"],
        errors="coerce"
    ).fillna(0)

    appearances["goals"] = pd.to_numeric(
        appearances["goals"],
        errors="coerce"
    ).fillna(0)

    appearances["assists"] = pd.to_numeric(
        appearances["assists"],
        errors="coerce"
    ).fillna(0)

    appearances["appearances"] = pd.to_numeric(
        appearances["appearances"],
        errors="coerce"
    ).fillna(0)


    appearance_lookup = (
        appearances
        .groupby("player_id")
        .agg(
            {
                "minutes": "sum",
                "goals": "sum",
                "assists": "sum",
                "appearances": "sum"
            }
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Valuation lookup
    # --------------------------------------------------------

    valuations = valuations.copy()

    valuation_lookup = (
        valuations
        .sort_values(
            "date"
        )
        .drop_duplicates(
            "player_id",
            keep="last"
        )
    )


    # --------------------------------------------------------
    # Merge local information
    # --------------------------------------------------------

    players = players.merge(
        appearance_lookup,
        on="player_id",
        how="left"
    )

    players = players.merge(
        valuation_lookup[
            [
                "player_id",
                "market_value_in_eur"
            ]
        ],
        on="player_id",
        how="left",
        suffixes=(
            "",
            "_valuation"
        )
    )


    # --------------------------------------------------------
    # Use current player value where available
    # --------------------------------------------------------

    players["market_value_final"] = (
        players["market_value_in_eur"]
        .fillna(
            players["market_value_in_eur_valuation"]
            if "market_value_in_eur_valuation"
            in players.columns
            else np.nan
        )
    )


    # --------------------------------------------------------
    # Player lookup
    # --------------------------------------------------------

    player_lookup = create_player_lookup(
        players
    )


    records = []

    matched_count = 0

    missing_count = 0

    fuzzy_count = 0


    # --------------------------------------------------------
    # Build official squads
    # --------------------------------------------------------

    for club in PREMIER_LEAGUE_CLUBS:

        official_players = squads.get(
            club,
            []
        )

        for official_name in official_players:

            matched_row, score, match_type = (
                match_player(
                    official_name,
                    players,
                    player_lookup
                )
            )


            if matched_row is not None:

                matched_count += 1

                if match_type == "fuzzy":
                    fuzzy_count += 1

                player_id = (
                    matched_row["player_id"]
                )

                player_name = (
                    matched_row["name"]
                    if pd.notna(
                        matched_row["name"]
                    )
                    else official_name
                )

                position = (
                    matched_row["position"]
                    if pd.notna(
                        matched_row["position"]
                    )
                    else "Unknown"
                )

                sub_position = (
                    matched_row["sub_position"]
                    if pd.notna(
                        matched_row["sub_position"]
                    )
                    else ""
                )

                image_url = (
                    matched_row["image_url"]
                    if pd.notna(
                        matched_row["image_url"]
                    )
                    else ""
                )

                dob = matched_row[
                    "date_of_birth"
                ]

                market_value = (
                    matched_row[
                        "market_value_final"
                    ]
                )

                minutes = (
                    matched_row["minutes"]
                    if pd.notna(
                        matched_row["minutes"]
                    )
                    else 0
                )

                goals = (
                    matched_row["goals"]
                    if pd.notna(
                        matched_row["goals"]
                    )
                    else 0
                )

                assists = (
                    matched_row["assists"]
                    if pd.notna(
                        matched_row["assists"]
                    )
                    else 0
                )

                appearances_count = (
                    matched_row["appearances"]
                    if pd.notna(
                        matched_row["appearances"]
                    )
                    else 0
                )

            else:

                missing_count += 1

                player_id = np.nan

                player_name = official_name

                position = "Unknown"

                sub_position = ""

                image_url = ""

                dob = pd.NaT

                market_value = np.nan

                minutes = 0

                goals = 0

                assists = 0

                appearances_count = 0


            # ------------------------------------------------
            # Age
            # ------------------------------------------------

            if pd.notna(dob):

                try:

                    dob_date = (
                        pd.Timestamp(dob)
                        .date()
                    )

                    today = date.today()

                    age = (
                        today.year
                        - dob_date.year
                        - (
                            (
                                today.month,
                                today.day
                            )
                            <
                            (
                                dob_date.month,
                                dob_date.day
                            )
                        )
                    )

                except Exception:

                    age = np.nan

            else:

                age = np.nan


            # ------------------------------------------------
            # Rates
            # ------------------------------------------------

            if minutes > 0:

                goals_per_90 = (
                    goals
                    / minutes
                    * 90
                )

                assists_per_90 = (
                    assists
                    / minutes
                    * 90
                )

            else:

                goals_per_90 = 0

                assists_per_90 = 0


            if appearances_count > 0:

                minutes_per_appearance = (
                    minutes
                    / appearances_count
                )

            else:

                minutes_per_appearance = 0


            records.append(
                {
                    "player_id":
                        player_id,

                    "name":
                        player_name,

                    "official_squad_name":
                        official_name,

                    "club":
                        club,

                    "position":
                        position,

                    "sub_position":
                        sub_position,

                    "age":
                        age,

                    "appearances":
                        appearances_count,

                    "minutes":
                        minutes,

                    "goals":
                        goals,

                    "assists":
                        assists,

                    "goals_per_90":
                        goals_per_90,

                    "assists_per_90":
                        assists_per_90,

                    "minutes_per_appearance":
                        minutes_per_appearance,

                    "market_value":
                        market_value,

                    "image_url":
                        image_url,

                    "match_score":
                        score,

                    "match_type":
                        match_type,

                    "season":
                        "2026/27",

                    "data_source":
                        "Official Premier League squad list"
                }
            )


    dataset = pd.DataFrame(
        records
    )


    # --------------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "age",
        "appearances",
        "minutes",
        "goals",
        "assists",
        "goals_per_90",
        "assists_per_90",
        "minutes_per_appearance",
        "market_value",
        "match_score"
    ]

    for column in numeric_columns:

        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    dataset = (
        dataset
        .sort_values(
            [
                "club",
                "market_value",
                "name"
            ],
            ascending=[
                True,
                False,
                True
            ],
            na_position="last"
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print(" DATASET CREATED")
    print("=" * 60)

    print()

    print(
        f"Players: {len(dataset)}"
    )

    print(
        f"Clubs: {dataset['club'].nunique()}"
    )

    print(
        f"Matched to Transfermarkt: "
        f"{matched_count}"
    )

    print(
        f"Fuzzy matches: "
        f"{fuzzy_count}"
    )

    print(
        f"Players not found in local database: "
        f"{missing_count}"
    )

    print()

    print(
        "Club counts:"
    )

    print()

    club_counts = (
        dataset
        .groupby("club")
        .size()
        .sort_index()
    )

    for club, count in club_counts.items():

        print(
            f" - {club:<30} {count:>2}"
        )


    # --------------------------------------------------------
    # Missing players
    # --------------------------------------------------------

    missing = dataset[
        dataset["match_type"]
        == "missing"
    ]

    if not missing.empty:

        print()
        print("=" * 60)
        print(
            " PLAYERS NOT FOUND IN LOCAL DATABASE"
        )
        print("=" * 60)

        print()

        for _, row in missing.iterrows():

            print(
                f" - {row['name']}"
                f" ({row['club']})"
            )


    # --------------------------------------------------------
    # Top values
    # --------------------------------------------------------

    valued = dataset[
        dataset["market_value"]
        .notna()
    ]

    if not valued.empty:

        print()
        print("=" * 60)
        print(" TOP 10 MARKET VALUES")
        print("=" * 60)

        print()

        top10 = (
            valued
            .sort_values(
                "market_value",
                ascending=False
            )
            .head(10)
        )

        for _, row in top10.iterrows():

            print(
                f"{row['name']:<25}"
                f"{row['club']:<25}"
                f"€{row['market_value']/1_000_000:.1f}M"
            )


    print()
    print("=" * 60)
    print(" SAVED")
    print("=" * 60)

    print()

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Season: 2026/27"
    )

    print(
        "Source: Official Premier League squad lists"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Players missing from the local Transfermarkt "
        "snapshot are retained with blank market value/"
        "photo fields instead of being removed."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    html = download_squad_page()


    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    squads = parse_squads(
        html
    )


    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    if not validate_squads(
        squads
    ):

        print()
        print(
            "ERROR:"
        )

        print(
            "The Premier League webpage structure "
            "could not be parsed correctly."
        )

        print()
        print(
            "Your old dataset has NOT been changed."
        )

        return


    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    players, appearances, valuations = (
        load_transfermarkt()
    )


    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    build_dataset(
        squads,
        players,
        appearances,
        valuations
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()