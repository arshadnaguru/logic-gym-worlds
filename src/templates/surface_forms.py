"""
src/templates/surface_forms.py

Domain vocabularies: entities, unary predicates, binary predicates.
Each domain provides natural-sounding names so generated rules read like English.
"""

DOMAINS = {
    "animals": {
        "entities": ["Tweety", "Opus", "Felix", "Dumbo", "Nemo", "Simba", "Bambi", "Dory"],
        "unary_predicates": [
            ("bird",      "a bird"),
            ("mammal",    "a mammal"),
            ("fish",      "a fish"),
            ("reptile",   "a reptile"),
            ("can_fly",   "able to fly"),
            ("can_swim",  "able to swim"),
            ("large",     "large"),
            ("small",     "small"),
            ("endangered","endangered"),
            ("nocturnal", "nocturnal"),
            ("carnivore", "a carnivore"),
            ("herbivore", "a herbivore"),
        ],
        "binary_predicates": [
            ("eats",       "eats"),
            ("lives_with", "lives with"),
            ("larger_than","larger than"),
            ("preys_on",   "preys on"),
        ],
    },

    "company": {
        "entities": ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"],
        "unary_predicates": [
            ("manager",    "a manager"),
            ("engineer",   "an engineer"),
            ("intern",     "an intern"),
            ("remote",     "working remotely"),
            ("senior",     "senior"),
            ("has_car",    "has a company car"),
            ("has_permit", "has a parking permit"),
            ("full_time",  "full-time"),
            ("on_call",    "on call"),
            ("certified",  "certified"),
        ],
        "binary_predicates": [
            ("reports_to", "reports to"),
            ("works_with", "works with"),
            ("mentors",    "mentors"),
            ("manages",    "manages"),
        ],
    },

    "geography": {
        "entities": ["Paris", "London", "Tokyo", "Sydney", "Cairo", "Oslo", "Lima"],
        "unary_predicates": [
            ("capital",    "a capital city"),
            ("coastal",    "coastal"),
            ("large_city", "a large city"),
            ("in_europe",  "in Europe"),
            ("in_asia",    "in Asia"),
            ("landlocked", "landlocked"),
            ("island",     "an island"),
            ("cold_climate","a cold climate"),
        ],
        "binary_predicates": [
            ("north_of",      "north of"),
            ("connected_to",  "connected to"),
            ("borders",       "borders"),
            ("trade_with",    "trades with"),
        ],
    },

    "academic": {
        "entities": ["Alex", "Jordan", "Morgan", "Casey", "Riley", "Sam", "Taylor"],
        "unary_predicates": [
            ("student",      "a student"),
            ("professor",    "a professor"),
            ("phd",          "a PhD student"),
            ("published",    "published"),
            ("funded",       "funded"),
            ("enrolled",     "enrolled"),
            ("graduated",    "graduated"),
            ("on_probation", "on academic probation"),
        ],
        "binary_predicates": [
            ("advises",    "advises"),
            ("co_authored","co-authored with"),
            ("enrolled_in","is enrolled in"),
            ("teaches",    "teaches"),
        ],
    },
}

# Default domain for quick generation
DEFAULT_DOMAIN = "animals"
