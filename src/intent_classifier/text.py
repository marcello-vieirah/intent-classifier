import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normaliza o texto:
    - Converte para minúsculas;
    - Remove acentos/diacríticos;
    - Remove pontuações e caracteres especiais mantendo espaços.
    """
    if not text:
        return ""

    # Normalização NFKD para separar caracteres de seus acentos
    text = text.lower().strip()
    nfkd_form = unicodedata.normalize("NFKD", text)
    text_without_accents = "".join(
        [c for c in nfkd_form if not unicodedata.combining(c)]
    )

    # Mantém apenas letras, números e espaços
    clean_text = re.sub(r"[^\w\s]", "", text_without_accents)

    # Colapsa múltiplos espaços em um só
    return re.sub(r"\s+", " ", clean_text).strip()


def extract_tokens(text: str) -> set[str]:
    """Retorna um conjunto de palavras (tokens) normalizados e únicos."""
    normalized = normalize_text(text)
    return set(normalized.split()) if normalized else set()
