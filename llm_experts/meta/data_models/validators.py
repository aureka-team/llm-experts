from iso639 import Language, LanguageNotFoundError


def languge_validator(language: str) -> str:
    try:
        return Language.from_name(language).name
    except LanguageNotFoundError:
        raise ValueError(f"{language} isn't an ISO language name")
