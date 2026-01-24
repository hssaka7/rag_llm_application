

from enum import Enum

class Topic(Enum):
    ECONOMY = "economy"
    POLITICS = "politics"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SPORTS = "sports"
    HEALTH = "health"
    SCIENCE = "science"
    CLIMATE = "climate"
    WORLD = "world"
    OPINION = "opinion"
    OTHER = "other"

class ArticleType(Enum):
    BREAKING = "breaking"
    UPDATE = "update"
    ANALYSIS = "analysis"
    EXPLAINER = "explainer"
    INTERVIEW = "interview"
    OPINION = "opinion"
    FEATURE = "feature"

class Section(Enum):
    HEADLINE = "headline"
    LEDE = "lede"
    FACTS = "facts"
    BACKGROUND = "background"
    QUOTES = "quotes"
    ANALYSIS = "analysis"
    CONTEXT = "context"
    IMPACT = "impact"

class Region(Enum):
    US = "US"
    EU = "EU"
    UK = "UK"
    NEPAL = "NEPAL"
    INDIA = "INDIA"
    CHINA = "CHINA"
    RUSSIA = "RUSSIA"
    EUROPE = "EUROPE"
    ASIA = "ASIA"
    MIDDLE_EAST = "MIDDLE_EAST"
    AFRICA = "AFRICA"
    LATAM = "LATAM"
    GLOBAL = "GLOBAL"

class Language(Enum):
    EN = "en"
    NE = "ne"

class Sentiment(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EventType(Enum):
    POLICY_DECISION = "policy_decision"
    ECONOMIC_REPORT = "economic_report"
    LEGAL_ACTION = "legal_action"
    CORPORATE_ACTION = "corporate_action"
    MARKET_MOVEMENT = "market_movement"
    CONFLICT = "conflict"
    ELECTION = "election"
    NATURAL_DISASTER = "natural_disaster"
    TECHNOLOGY_RELEASE = "technology_release"

