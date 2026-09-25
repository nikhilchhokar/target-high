"""Synthetic dataset in the EXACT challenge format, for testing the pipeline before real data.

    python tools/make_synthetic.py --out data_synth

Mimics the documented noise: legal-suffix changes, abbreviations, typos, word order,
transliteration, missing postcodes/states, landmarks, chains (same name, other branch),
distractor pool records, singletons, multi-matches; train = US+India, test adds France.
Scores on synthetic data mean nothing - it only proves the code runs end to end.
"""
import argparse
import random
from pathlib import Path

R = random.Random(7)

US = dict(
    a=["Golden", "Blue", "Summit", "Pioneer", "Liberty", "Eagle", "Pacific", "Silver", "Green",
       "Sunrise", "Maple", "River", "Harbor", "Metro", "Capital", "Heritage", "Evergreen", "Redwood"],
    b=["Auto", "Dental", "Bakery", "Plumbing", "Logistics", "Consulting", "Pharmacy", "Grill",
       "Hardware", "Realty", "Insurance", "Fitness", "Electric", "Printing", "Motors", "Cafe"],
    c=["Group", "Services", "Center", "Partners", "Associates", "Systems", ""],
    legal=["Inc", "LLC", "Corp", "Ltd", "Corporation", "Incorporated", ""],
    streets=["Main", "Oak", "Maple", "Washington", "Lake", "Hill", "Park", "Pine", "Cedar", "Sunset"],
    stype=[("St", "Street"), ("Ave", "Avenue"), ("Blvd", "Boulevard"), ("Rd", "Road"), ("Dr", "Drive")],
    cities=[("Springfield", "IL", "Illinois", "627"), ("Austin", "TX", "Texas", "787"),
            ("Denver", "CO", "Colorado", "802"), ("Columbus", "OH", "Ohio", "432"),
            ("Seattle", "WA", "Washington", "981"), ("Miami", "FL", "Florida", "331")])
IN = dict(
    a=["Shree", "Laxmi", "Ganesh", "Balaji", "Krishna", "Sai", "Durga", "Bharat", "Surya",
       "Mahalaxmi", "Vinayak", "Shiv", "Kaveri", "Annapurna"],
    b=["Traders", "Enterprises", "Textiles", "Electricals", "Medicals", "Sweets", "Hardware",
       "Motors", "Jewellers", "Agencies", "Builders", "Industries", "Steels", "Garments"],
    legal=["Pvt Ltd", "Private Limited", "(P) Ltd", "LLP", "& Sons", "", ""],
    areas=["Andheri", "Bandra", "Koramangala", "Jayanagar", "Salt Lake", "Sector 14", "Shivaji"],
    atype=["Nagar", "Colony", "Layout", "Market"],
    roads=[("MG Road", "M.G. Rd"), ("Station Road", "Station Rd"), ("Main Road", "Main Rd"),
           ("Link Road", "Link Rd")],
    landmarks=["SBI ATM", "Bus Stand", "City Hospital", "Railway Station", "Hanuman Temple"],
    cities=[("Mumbai", "Maharashtra", "MH", "400"), ("Bengaluru", "Karnataka", "KA", "560"),
            ("Chennai", "Tamil Nadu", "TN", "600"), ("Kolkata", "West Bengal", "WB", "700"),
            ("Pune", "Maharashtra", "MH", "411")])
TRANSLIT = {"Shree": ["Shri", "Sri"], "Laxmi": ["Lakshmi"], "Ganesh": ["Ganesha"],
            "Mahalaxmi": ["Mahalakshmi"], "Vinayak": ["Vinayaka"], "Shiv": ["Shiva"],
            "Kaveri": ["Cauvery"], "Balaji": ["Baalaji"]}
FR = dict(
    a=["Boulangerie", "Pharmacie", "Garage", "Cabinet", "Restaurant", "Librairie", "Fromagerie",
       "Boucherie", "Agence", "Atelier", "Hôtel", "Café"],
    b=["du Centre", "de la Gare", "Saint-Michel", "des Lilas", "Martin", "Dupont", "Le Bon Pain",
       "Les Halles", "Bernard", "Moreau", "L'Étoile"],
    legal=["SARL", "SAS", "SA", "EURL", ""],
    stype=[("rue", "r."), ("avenue", "av."), ("boulevard", "bd"), ("place", "pl."), ("chemin", "ch.")],
    streets=["de la République", "Victor Hugo", "Jean Jaurès", "de la Paix", "Pasteur",
             "Saint-Honoré", "des Écoles", "Gambetta"],
    cities=[("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Lille", "59")])


def typo(s, p=0.25):
    if R.random() > p or len(s) < 5:
        return s
    i = R.randrange(1, len(s) - 1)
    op = R.choice("dsi")
    if op == "d":
        return s[:i] + s[i + 1:]
    if op == "s":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + R.choice("aeiourstn") + s[i:]


def base_entity(country):
    if country == "US":
        name = [R.choice(US["a"]), R.choice(US["b"]), R.choice(US["c"])]
        city, st, stf, zp = R.choice(US["cities"])
        s, _ = R.choice(US["stype"])
        return dict(country="US", core=[w for w in name if w], legal=R.choice(US["legal"]),
                    num=str(R.randint(10, 9999)), street=R.choice(US["streets"]), stype=R.choice(US["stype"]),
                    unit=R.choice(["", "", f"Suite {R.randint(100, 400)}"]), city=city, st=st, stf=stf,
                    postal=zp + f"{R.randint(0, 99):02d}")
    if country == "India":
        name = [R.choice(IN["a"]), R.choice(IN["b"])]
        city, stf, st, pin = R.choice(IN["cities"])
        return dict(country="India", core=name, legal=R.choice(IN["legal"]),
                    num=f"{R.randint(1, 200)}", area=R.choice(IN["areas"]), atype=R.choice(IN["atype"]),
                    road=R.choice(IN["roads"]), lm=R.choice(IN["landmarks"]), city=city, st=st, stf=stf,
                    postal=pin + f"{R.randint(0, 999):03d}")
    name = [R.choice(FR["a"]), R.choice(FR["b"])]
    city, dep = R.choice(FR["cities"])
    return dict(country="France", core=name, legal=R.choice(FR["legal"]), num=str(R.randint(1, 150)),
                stype=R.choice(FR["stype"]), street=R.choice(FR["streets"]), city=city,
                postal=dep + f"{R.randint(0, 999):03d}")


def render(e, noisy):
    core = list(e["core"])
    if noisy and e["country"] == "India":
        core = [R.choice(TRANSLIT[w]) if w in TRANSLIT and R.random() < 0.4 else w for w in core]
    if noisy and R.random() < 0.1 and len(core) > 1:
        core[0], core[1] = core[1], core[0]
    legal = e["legal"]
    if noisy and R.random() < 0.4:
        pool = {"US": US["legal"], "India": IN["legal"], "France": FR["legal"]}[e["country"]]
        legal = R.choice(pool)
    name = " ".join(core + ([legal] if legal else []))
    if e["country"] == "France" and legal and R.random() < 0.5:
        name = f"{legal} {' '.join(core)}"
    if noisy:
        name = typo(name).replace(" and ", " & ") if R.random() < 0.5 else typo(name)
        if R.random() < 0.3:
            name = name.upper()
    if e["country"] == "US":
        s = e["stype"][1 if (noisy and R.random() < 0.5) else 0]
        parts = [f"{e['num']} {e['street']} {s}"]
        if e["unit"]:
            parts.append(e["unit"].replace("Suite", "Ste") if noisy and R.random() < 0.5 else e["unit"])
        parts.append(e["city"])
        st = e["stf"] if noisy and R.random() < 0.3 else e["st"]
        zp = "" if noisy and R.random() < 0.3 else " " + e["postal"]
        parts.append(st + zp)
        addr = ", ".join(parts)
    elif e["country"] == "India":
        road = e["road"][1 if (noisy and R.random() < 0.5) else 0]
        parts = [f"Shop No {e['num']}", f"{e['area']} {e['atype']}", road]
        if not noisy or R.random() < 0.5:
            parts.append(("Nr " if noisy and R.random() < 0.5 else "Near ") + e["lm"])
        parts.append(e["city"])
        if not noisy or R.random() > 0.3:
            parts.append(e["stf"] if R.random() < 0.7 else e["st"])
        addr = ", ".join(parts)
        if not noisy or R.random() > 0.35:
            addr += f" - {e['postal']}"
        if noisy and R.random() < 0.1:
            addr = ", ".join(reversed(addr.split(", ")))
    else:
        st = e["stype"][1 if (noisy and R.random() < 0.5) else 0]
        addr = f"{e['num']}{' bis' if R.random() < 0.05 else ''} {st} {e['street']}, "
        addr += f"{e['postal']} {e['city']}" + (" CEDEX" if noisy and R.random() < 0.1 else "")
        if noisy and R.random() < 0.3:
            import unicodedata
            addr = "".join(c for c in unicodedata.normalize("NFKD", addr) if not unicodedata.combining(c))
    if noisy:
        addr = typo(addr, 0.2)
    return name, addr


def make_split(split, n_s1, countries, out):
    s1, s2, s3, gt = [], [], [], []
    bases = []
    for k in range(n_s1):
        e = base_entity(R.choice(countries))
        if bases and R.random() < 0.08:  # chain: same name, different branch
            other = R.choice(bases)
            if other["country"] == e["country"]:
                e["core"], e["legal"] = list(other["core"]), other["legal"]
        bases.append(e)
    for e in bases:
        sid = len(s1)
        s1.append((sid, *render(e, noisy=False), e["country"]))
        matches = []
        if R.random() > 0.35:  # non-singleton
            n2 = R.choices([0, 1, 2], [0.25, 0.65, 0.10])[0]
            n3 = R.choices([0, 1, 2], [0.25, 0.65, 0.10])[0]
            if n2 + n3 == 0:
                n2 = 1
            for _ in range(n2):
                s2.append((len(s2), *render(e, True), e["country"]))
                matches.append(("S2", len(s2) - 1))
            for _ in range(n3):
                s3.append((len(s3), *render(e, True), e["country"]))
                matches.append(("S3", len(s3) - 1))
        gt.append(matches)
    for _ in range(int(0.3 * (len(s2) + len(s3)))):  # distractors not in S1
        e = base_entity(R.choice(countries))
        (s2 if R.random() < 0.5 else s3).append((len(s2) if R.random() < 0.5 else len(s3), *render(e, True), e["country"]))
    # assign shuffled public ids
    ids = {}
    for tag, rows in (("S1", s1), ("S2", s2), ("S3", s3)):
        perm = list(range(len(rows)))
        R.shuffle(perm)
        ids[tag] = {i: f"{tag}-{perm[i] + 1:05d}" for i in range(len(rows))}
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    for tag, rows, n in (("S1", s1, 1), ("S2", s2, 2), ("S3", s3, 3)):
        with open(d / f"{split}_source{n}.tsv", "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i, (_, name, addr, c) in enumerate(rows):
                f.write(f"{ids[tag][i]}\t{name}\t{addr}\t{c}\n")
    gt_name = "train_ground_truth.tsv" if split == "train" else "_hidden_test_ground_truth.tsv"
    with open(d / gt_name, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i, m in enumerate(gt):
            f.write(f"{ids['S1'][i]}\t{','.join(ids[t][j] for t, j in m)}\n")
    print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    make_split("train", a.n_train, ["US", "India"], a.out)
    make_split("test", a.n_test, ["US", "India", "France"], a.out)
