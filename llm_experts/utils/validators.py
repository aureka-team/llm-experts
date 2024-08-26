from pycountry import languages


def language_name_validator(language_name: str) -> str:
    language = languages.get(name=language_name)
    if language is not None:
        return language.name

    raise ValueError(f"{language} isn't an ISO language name")
