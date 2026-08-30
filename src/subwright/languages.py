"""The languages Whisper can recognise.

This is Whisper's own language set, transcribed from the model's tokenizer. It
is data, not configuration: adding a code here does not make the model support
it, and removing one does not stop it working. It exists so the settings page
can offer a real list instead of asking someone to know that Japanese is "ja".

Note `yue` (Cantonese) is only present in large-v3. Selecting it with an older
model gets a clear error from faster-whisper rather than silent nonsense.

Target language is deliberately absent, because there is no such thing: Whisper
translates INTO English and nothing else. See README.
"""

from __future__ import annotations

# Ordered by how likely this application is to meet them, then alphabetically.
# Purely a UI affordance - it puts the realistic choices at the top of a list of
# ninety-nine rather than making someone scroll past Afrikaans to reach Japanese.
COMMON = ["ja", "ko", "zh", "yue", "th", "es", "fr", "de", "ru", "it", "pt", "hi"]

LANGUAGES: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "zh": "Chinese",
}

# Below this, Whisper's guess at the language is not worth trusting. Detection
# reads roughly the first 30 seconds, so a film opening on music, silence or a
# distributor logo can easily be misread - and a wrong language produces fluent,
# confident, completely invented subtitles rather than an obvious failure.
LOW_CONFIDENCE = 0.75


def name(code: str | None) -> str:
    """Human-readable name for a code, falling back to the code itself."""
    if not code:
        return "auto-detect"
    return LANGUAGES.get(code, code)


def is_valid(code: str) -> bool:
    """Blank is valid: it means auto-detect."""
    return code == "" or code in LANGUAGES


def choices() -> list[tuple[str, list[tuple[str, str]]]]:
    """Grouped (code, name) pairs for a <select>, common languages first."""
    common = [(c, LANGUAGES[c]) for c in COMMON if c in LANGUAGES]
    rest = sorted(
        ((c, n) for c, n in LANGUAGES.items() if c not in set(COMMON)),
        key=lambda pair: pair[1],
    )
    return [("Common", common), ("All languages", rest)]
