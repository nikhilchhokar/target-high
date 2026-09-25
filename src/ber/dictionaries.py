"""Hand-written normalisation dictionaries (domain knowledge, no external data).

All keys/values are lowercase, accent-stripped, punctuation-free tokens, because
they are applied after basic_clean(). Country codes: 'us', 'in', 'fr', 'other'.
Unknown countries only get COMMON rules - the country set is open.
"""

COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us",
    "in": "in", "ind": "in", "india": "in", "bharat": "in",
    "fr": "fr", "fra": "fr", "france": "fr", "republique francaise": "fr",
}

# ---------------------------------------------------------------- names
LEGAL_TOKENS = {
    "common": {"ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
               "co", "company", "plc", "llp", "lp", "pllc", "pc", "gmbh", "bv", "ag"},
    "in": {"pvt", "private", "p", "opc", "pvtltd"},
    "fr": {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl", "scp",
           "societe", "ste", "ets", "etablissements", "etablissement", "cie", "compagnie"},
    "us": set(),
}

NAME_ABBR = {
    "common": {
        "intl": "international", "natl": "national", "mfg": "manufacturing",
        "mfr": "manufacturer", "mfrs": "manufacturers", "svc": "service", "svcs": "services",
        "assoc": "associates", "assn": "association", "bros": "brothers",
        "ent": "enterprises", "ents": "enterprises", "entp": "enterprises",
        "inds": "industries", "indus": "industries", "mgmt": "management",
        "mktg": "marketing", "dept": "department", "ctr": "center", "centre": "center",
        "hosp": "hospital", "govt": "government", "univ": "university", "inst": "institute",
        "tech": "technology", "techs": "technologies", "sys": "systems", "engg": "engineering",
        "engrs": "engineers", "constr": "construction", "devt": "development",
        "grp": "group", "hldgs": "holdings", "hldg": "holding", "med": "medical",
        "labs": "laboratories", "lab": "laboratory", "coop": "cooperative",
        "trdrs": "traders", "agncy": "agency", "agy": "agency", "jewelers": "jewellers",
        "colour": "color",
        # common outputs of translit.indic_to_latin for English loanwords
        "praivet": "private", "piraivet": "private", "prayivet": "private", "limitet": "limited",
        "limited": "limited", "elelapi": "llp", "elelpi": "llp", "kampani": "company",
        "kampni": "company", "kompani": "company",
    },
    "fr": {"st": "saint", "gd": "grand", "gde": "grande"},
    "us": {}, "in": {},
}

NAME_STOPWORDS = {"the", "and", "of", "a", "an", "le", "la", "les", "l", "de", "du",
                  "des", "d", "et", "en", "au", "aux", "ms", "m/s"}

DBA_PATTERN = r"\b(?:d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|trading as|a\s*/\s*k\s*/\s*a|aka|formerly)\b"

# ---------------------------------------------------------------- addresses
ADDR_ABBR = {
    "common": {"bldg": "building", "blg": "building", "apt": "apartment",
               "apts": "apartments", "rm": "room", "flr": "floor"},
    "us": {
        "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
        "rd": "road", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
        "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "pl": "place", "sq": "square",
        "ter": "terrace", "terr": "terrace", "trl": "trail", "ste": "suite", "fl": "floor",
        "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
        "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort",
        "rte": "route", "expy": "expressway", "fwy": "freeway", "plz": "plaza",
        "ctr": "center", "hts": "heights", "jct": "junction", "xing": "crossing",
    },
    "in": {
        "rd": "road", "st": "street", "nr": "near", "opp": "opposite", "opps": "opposite",
        "fl": "floor", "ngr": "nagar", "clny": "colony", "col": "colony", "sec": "sector",
        "sect": "sector", "ph": "phase", "indl": "industrial", "ind": "industrial",
        "estt": "estate", "soc": "society", "mkt": "market", "stn": "station",
        "jn": "junction", "jnc": "junction", "jct": "junction", "bazar": "bazaar",
        "extn": "extension", "ext": "extension", "blk": "block", "distt": "district",
        "dist": "district", "tq": "taluk", "tal": "taluk", "marg": "marg",
        # well-known city renames
        "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
        "bangalore": "bengaluru", "gurgaon": "gurugram", "poona": "pune",
        "baroda": "vadodara", "orissa": "odisha", "pondicherry": "puducherry",
    },
    "fr": {
        "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
        "blvd": "boulevard", "boul": "boulevard", "pl": "place", "rte": "route",
        "ch": "chemin", "chem": "chemin", "imp": "impasse", "all": "allee", "sq": "square",
        "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "crs": "cours", "pas": "passage",
        "psg": "passage", "res": "residence", "resid": "residence", "bat": "batiment",
        "bt": "batiment", "st": "saint", "ste": "sainte", "etg": "etage", "esc": "escalier",
    },
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}

IN_STATES = {  # full name -> single canonical token; plus unambiguous short forms
    "andhra pradesh": "andhrapradesh", "arunachal pradesh": "arunachalpradesh",
    "himachal pradesh": "himachalpradesh", "madhya pradesh": "madhyapradesh",
    "uttar pradesh": "uttarpradesh", "tamil nadu": "tamilnadu", "west bengal": "westbengal",
    "jammu and kashmir": "jammukashmir", "uttaranchal": "uttarakhand",
    "mh": "maharashtra", "ka": "karnataka", "tn": "tamilnadu", "ap": "andhrapradesh",
    "ts": "telangana", "tg": "telangana", "gj": "gujarat", "rj": "rajasthan",
    "hr": "haryana", "pb": "punjab", "kl": "kerala", "wb": "westbengal", "dl": "delhi",
    "mp": "madhyapradesh", "jk": "jammukashmir", "hp": "himachalpradesh",
    "up": "uttarpradesh", "uk": "uttarakhand",
    # transliterated native-script state names (translit.indic_to_latin output)
    "maharashtr": "maharashtra", "dilli": "delhi", "karnatak": "karnataka",
    "pashchim bangal": "westbengal", "telangan": "telangana", "keral": "kerala",
    "hariyana": "haryana", "panjab": "punjab", "tamil natu": "tamilnadu", "tamilnatu": "tamilnadu",
    "rajasthan": "rajasthan", "gujarat": "gujarat", "bihar": "bihar",
}

ADDR_STOPWORDS = {"the", "and", "of", "no", "nos", "number", "de", "la", "du", "des",
                  "le", "les", "l", "d", "india", "usa", "france", "cedex",
                  "bis", "ter", "quater", "null", "na", "none"}

LANDMARK_PATTERN = (r"\b(?:near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj"
                    r"|in front of|close to|pres de|pres du|en face de|en face du|a cote de"
                    r"|derriere)\b[^,]*")

UNIT_PATTERN = (r"\b(?:suite|ste|apt|apartment|unit|floor|fl|flr|shop|office|room|rm|etage"
                r"|bureau|door|flat)\s*(?:no\s*)?#?\s*(\d+[a-z]?)")
