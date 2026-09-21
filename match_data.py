import requests
from datetime import datetime


MATCH_API = (
    "https://sdp-prem-prod.premier-league-prod.pulselive.com"
    "/api/v2/matches"
)

LINEUP_API = (
    "https://sdp-prem-prod.premier-league-prod.pulselive.com"
    "/api/v3/matches/{match_id}/lineups"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.premierleague.com",
    "Referer": "https://www.premierleague.com/",
}


# ---------------------------------------------------------
# CLUB NAME NORMALIZATION
# ---------------------------------------------------------

CLUB_ALIASES = {
    "Manchester United": [
        "Manchester United",
        "Man United",
        "Man Utd",
    ],

    "Manchester City": [
        "Manchester City",
        "Man City",
    ],

    "Tottenham Hotspur": [
        "Tottenham Hotspur",
        "Tottenham",
        "Spurs",
    ],

    "Nottingham Forest": [
        "Nottingham Forest",
        "Nott'm Forest",
        "Nott'm Forest",
    ],

    "Brighton & Hove Albion": [
        "Brighton & Hove Albion",
        "Brighton",
    ],

    "AFC Bournemouth": [
        "AFC Bournemouth",
        "Bournemouth",
    ],

    "Newcastle United": [
        "Newcastle United",
        "Newcastle",
    ],

    "West Ham United": [
        "West Ham United",
        "West Ham",
    ],

    "Crystal Palace": [
        "Crystal Palace",
    ],

    "Leeds United": [
        "Leeds United",
        "Leeds",
    ],

    "Ipswich Town": [
        "Ipswich Town",
        "Ipswich",
    ],

    "Sunderland": [
        "Sunderland",
    ],

    "Coventry City": [
        "Coventry City",
        "Coventry",
    ],

    "Hull City": [
        "Hull City",
        "Hull",
    ],

    "Everton": [
        "Everton",
    ],

    "Liverpool": [
        "Liverpool",
        "Liverpool FC",
    ],

    "Arsenal": [
        "Arsenal",
        "Arsenal FC",
    ],

    "Chelsea": [
        "Chelsea",
        "Chelsea FC",
    ],

    "Aston Villa": [
        "Aston Villa",
    ],

    "Brentford": [
        "Brentford",
    ],

    "Fulham": [
        "Fulham",
    ],
}


def normalize_club_name(club_name):
    """
    Convert a Club Hub name or alias into the
    official Premier League match-feed name.
    """

    if not club_name:
        return ""

    club_name = club_name.strip()

    for official_name, aliases in CLUB_ALIASES.items():

        for alias in aliases:

            if club_name.lower() == alias.lower():
                return official_name

    return club_name


# ---------------------------------------------------------
# MATCH DATA
# ---------------------------------------------------------

def get_matches(limit=100):
    """
    Fetch Premier League matches for the 2026/27 season.
    """

    params = {
        "competition": 8,
        "season": 2026,
        "_limit": limit,
    }

    try:

        response = requests.get(
            MATCH_API,
            params=params,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        print(f"Match API error: {error}")

        return None

    except ValueError:

        print("Match API returned invalid JSON.")

        return None


def get_latest_match(club_name):
    """
    Find the latest completed Premier League match
    for a specific club.
    """

    official_name = normalize_club_name(club_name)

    data = get_matches()

    if not data:
        return None

    matches = data.get("data", [])

    club_matches = []

    for match in matches:

        if match.get("period") != "FullTime":
            continue

        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})

        home_name = home_team.get("name", "")
        away_name = away_team.get("name", "")

        if official_name in (home_name, away_name):

            club_matches.append(match)

    if not club_matches:
        return None

    club_matches.sort(
        key=lambda match: datetime.strptime(
            match["kickoff"],
            "%Y-%m-%d %H:%M:%S",
        ),
        reverse=True,
    )

    return club_matches[0]


# ---------------------------------------------------------
# LINEUP DATA
# ---------------------------------------------------------

def get_match_lineups(match_id):
    """
    Fetch official Premier League lineup data.
    """

    url = LINEUP_API.format(match_id=match_id)

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        print(f"Lineup API error: {error}")

        return None

    except ValueError:

        print("Lineup API returned invalid JSON.")

        return None


def get_starting_xi(match_id, team):
    """
    Return the starting XI for one team.

    team must be:
        home
        away
    """

    data = get_match_lineups(match_id)

    if not data:
        return []

    team_key = f"{team}_team"

    if team_key not in data:
        return []

    players = data[team_key].get("players", [])

    starting_xi = [
        player
        for player in players
        if player.get("position") != "Substitute"
    ]

    return starting_xi


# ---------------------------------------------------------
# PLAYER FORMATTING
# ---------------------------------------------------------

def player_display_name(player):
    """
    Create a clean player display name.
    """

    known_name = player.get("knownName")

    if known_name:
        return known_name

    first_name = player.get("firstName", "")
    last_name = player.get("lastName", "")

    return f"{first_name} {last_name}".strip()


def format_starting_xi(players):
    """
    Convert API player data into Club Hub-friendly data.
    """

    formatted = []

    for player in players:

        formatted.append(
            {
                "name": player_display_name(player),
                "shirt_number": player.get("shirtNum", ""),
                "position": player.get("position", ""),
                "player_id": player.get("id", ""),
                "captain": player.get("isCaptain", False),
            }
        )

    return formatted


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Premier League Match Data Test")
    print("=" * 50)

    test_clubs = [
        "Manchester United",
        "Man United",
        "Spurs",
        "Man City",
        "Nott'm Forest",
    ]

    for club in test_clubs:

        print()
        print(f"Searching: {club}")

        latest = get_latest_match(club)

        if not latest:

            print("No completed match found.")

            continue

        home = latest["homeTeam"]
        away = latest["awayTeam"]

        print(
            f'{home["name"]} '
            f'{home["score"]} - '
            f'{away["score"]} '
            f'{away["name"]}'
        )

        print("Match ID:", latest["matchId"])