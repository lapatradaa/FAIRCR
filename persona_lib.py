"""
Single source of truth for persona phrasing: the 15 attributes, their 5-value pools,
and the prompt-building functions (`humanize`, `combo_prompt`) shared by
generate_personas.py and run_experiment.py so they never drift apart.

No ontology / rdflib dependency: VALUE_POOL is a fixed, hardcoded set of 5 values per
attribute (carried over from the CRPF ontology used by the older FAIRCR project, frozen
here so this project has no external ontology file to keep in sync).
"""
import re

ATTRS = ["Gender", "Race", "Ethnicity", "Nationality", "Culture", "NativeLanguage",
         "AgeRange", "ReviewStyle", "Role", "Goal", "Seniority", "Domain", "Priority",
         "Standard", "Module"]

VALUE_POOL = {
    "Gender": ["TwoSpirit", "PreferNotToSay", "Woman", "NonBinary", "Demigender"],
    "Race": ["NativeHawaiian", "Asian", "Latino", "MiddleEasternOrNorthAfrican", "White"],
    "Ethnicity": ["OtherMixed", "AsianChinese", "BlackCaribbean", "AsianBangladeshi", "WhiteIrish"],
    "Nationality": ["Russian", "Ethiopian", "Austrian", "Danish", "Saudi"],
    "Culture": ["EgyptianCulture", "FrenchCulture", "ThaiCulture", "IndianCulture", "GermanCulture"],
    "NativeLanguage": ["EnglishLang", "TeluguLang", "PolishLang", "KoreanLang", "GermanLang"],
    "AgeRange": ["MiddleAged", "YoungAdult", "Aged", "Elderly", "LateCareerAdult"],
    "ReviewStyle": ["Supportive", "Concise", "Nitpicky", "BigPicture", "Collaborative"],
    "Role": ["Triager", "QAChecker", "Reviewer", "CodeOwner", "SoftwareReviewer"],
    "Goal": ["WellTestedCode", "WellDocumentedCode", "ReadableCode", "MaintainableCode", "FastDelivery"],
    "Seniority": ["Staff", "Senior", "Mid", "Intern", "Junior"],
    "Domain": ["DevOps", "Embedded", "PlatformEng", "Mobile", "Backend"],
    "Priority": ["Scalability", "Performance", "Portability", "SecurityPriority", "Testability"],
    "Standard": ["PeerApproved", "StyleGuideFollowed", "SecurityGatePassed", "CIPassed", "MergeApproved"],
    "Module": ["FrontendUIModule", "UserProfileModule", "APIModule", "DatabaseModule", "AuthenticationModule"],
}

# ---------------------------------------------------------------- humanizing
ACRONYMS = {"Us": "US", "Ci": "CI", "Api": "API", "Qa": "QA"}


def _split_camel(tok: str) -> str:
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', tok)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    return s


def _fix_acronyms(words):
    return [ACRONYMS.get(w, w) for w in words]


TYPE_SUFFIXES = {"NativeLanguage": "Lang", "Culture": "Culture", "Module": "Module"}


def humanize(attr: str, tok: str) -> str:
    base = tok
    type_suffix = TYPE_SUFFIXES.get(attr)
    if type_suffix and base.endswith(type_suffix):
        base = base[: -len(type_suffix)]

    words = _split_camel(base).split()
    words = _fix_acronyms(words)

    if attr == "Nationality":
        return " ".join(words)  # keep demonym capitalized, e.g. "Thai", "Middle Eastern..."

    return " ".join(w if w.isupper() and len(w) <= 3 else w.lower() for w in words)


# ---------------------------------------------------------------- prompt template
TASK_BLOCK = (
    "You will see three code segments separated by [SEP]:\n"
    "<preceding context> [SEP] <FOCAL HUNK> [SEP] <following context>\n"
    "Judge ONLY the focal hunk (the middle segment); the other two are context.\n\n"
    "Output exactly these lines, nothing else:\n"
    "LABEL: 0 if no issue, 1 if contains an issue"
)


def combo_prompt(attr_value_pairs):
    """Builds the persona system prompt for 1+ (attribute, value) pairs, phrased in the
    given order, rest of the 15 attributes omitted entirely -- e.g.
    [("Gender", "Woman"), ("NativeLanguage", "ThaiLang")] ->
    'You are a software reviewer who is woman and thai.\\n' + TASK_BLOCK."""
    parts = [humanize(a, v) for a, v in attr_value_pairs]
    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        joined = f"{parts[0]} and {parts[1]}"
    else:
        joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"You are a software reviewer who is {joined}.\n" + TASK_BLOCK
