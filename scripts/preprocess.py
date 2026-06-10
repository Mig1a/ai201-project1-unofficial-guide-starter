from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


PROFESSOR_SLUGS = {
    "Ahmed Zaman": "ahmed_zaman",
    "Sanjeev Setia": "sanjeev_setia",
    "Jana Kosecka": "jana_kosecka",
    "Alexander Laufer": "alexander_laufer",
    "Wassim (Wes) Masri": "wes_masri",
}

THEME_KEYWORDS = {
    "lecture clarity": ["clear", "explain", "explains", "understand", "concept", "lecture", "teaching"],
    "difficulty": ["hard", "difficult", "tough", "challenging", "easy", "heavy"],
    "grading style": ["grade", "grading", "curve", "rubric", "points", "feedback"],
    "exams/quizzes": ["exam", "exams", "quiz", "quizzes", "test", "midterm", "final"],
    "office hours/helpfulness": ["office hour", "office hours", "help", "helpful", "available", "email", "questions"],
    "organization": ["organized", "organization", "disorganized", "structure", "confusing", "unclear"],
    "workload": ["homework", "assignment", "project", "workload", "work", "reading"],
}

POSITIVE_WORDS = {
    "amazing", "awesome", "best", "clear", "easy", "excellent", "fair", "great",
    "kind", "nice", "recommend", "understanding",
}

NEGATIVE_WORDS = {
    "avoid", "bad", "boring", "confusing", "disorganized", "hard", "horrible",
    "poor", "rude", "tough", "unclear", "unfair", "worst",
}

NOISE_PATTERNS = [
    r"https?://\S+",
    r"www\.\S+",
    r"Rate My Professors",
    r"Would take again",
    r"Level of Difficulty",
    r"Textbook:",
    r"Attendance:",
    r"For Credit:",
    r"ADVERTISEMENT",
    r"Ad Choices",
    r"Sponsored",
    r"It's one of our biggest sales of the year!",
    r"20 million seats on sale\.",
    r"Four days only\.",
    r"Book yours by 6/11\.",
    r"servedby\.flashtalking\.com\s*>\s*Rate Compare[\s\S]{0,260}?\bat\s+\d+/\d+",
    r"Rate Compare\s+[A-Za-z() ]+\s+Computer Science\s*/\s*\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M\s+[A-Za-z() ]+\s+at\s+\d+/\d+",
    r"Help Site GuidelinesTerms & ConditionsPrivacy PolicyCopyright Compliance Policy.*?All Rights Reserved",
    r"Start Chat to use extension \(Free\)",
    r"Click \"Start Chat\"",
    r"Add GPT for Chrome",
    r"WOW!",
    r"Wow!",
    r"Log In",
    r"Sign Up",
    r"Find a professor",
    r"Find a Professor",
    r"Professor in the Computer Science department",
    r"George Mason University",
]


def infer_professor_name(pdf_path: Path) -> str:
    name = pdf_path.stem.split(" at ")[0].strip()
    if name == "Wassim Masri":
        return "Wassim (Wes) Masri"
    return name


def professor_slug(name: str) -> str:
    return PROFESSOR_SLUGS.get(name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\x00", " ").replace("\u00a0", " ").replace("|", " ")
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\s*(of\s+\d+)?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"servedby\.flashtalking\.com\s*>\s*Rate Compare.*?\bat\s+\d+/\d+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Rate Compare\s+[A-Za-z() ]+\s+Computer Science\s*/\s*\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M\s+[A-Za-z() ]+\s+at\s+\d+/\d+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Help Site GuidelinesTerms & ConditionsPrivacy PolicyCopyright Compliance Policy.*?All Rights Reserved",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(item)
    return result


def pick_evidence(sentences: list[str], keywords: list[str], limit: int = 5) -> list[str]:
    lowered_keywords = [keyword.lower() for keyword in keywords]
    matches = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in lowered_keywords)
    ]
    return dedupe_preserve_order(matches)[:limit]


def sentiment_label(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    counts = Counter(tokens)
    positive = sum(counts[word] for word in POSITIVE_WORDS)
    negative = sum(counts[word] for word in NEGATIVE_WORDS)
    if positive and negative and abs(positive - negative) <= max(3, int(0.25 * max(positive, negative))):
        return "mixed"
    if positive > negative:
        return "mostly positive"
    if negative > positive:
        return "mostly negative"
    return "unclear from extracted text"


def build_summary_document(professor_name: str, cleaned_text: str) -> str:
    sentences = split_sentences(cleaned_text)
    lines = [
        f"# {professor_name} Summary and Review Themes",
        "",
        "This summary is generated only from the extracted Rate My Professors PDF text for this professor.",
        f"Overall student sentiment: {sentiment_label(cleaned_text)}.",
        "",
        "## Themes",
    ]
    for theme, keywords in THEME_KEYWORDS.items():
        evidence = pick_evidence(sentences, keywords)
        lines.append(f"### {theme.title()}")
        if evidence:
            lines.append("Extracted review evidence mentions this theme in the following ways:")
            for sentence in evidence:
                lines.append(f"- {sentence}")
        else:
            lines.append("No direct evidence for this theme was found in the extracted text.")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_reviews_document(professor_name: str, cleaned_text: str) -> str:
    return (
        f"# {professor_name} Cleaned Student Reviews\n\n"
        "Source: manually collected Rate My Professors PDF.\n\n"
        "## Extracted Review Text\n\n"
        f"{cleaned_text}\n"
    )
