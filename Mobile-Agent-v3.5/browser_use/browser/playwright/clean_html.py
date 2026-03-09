from typing import Any, Iterable, List
from bs4 import BeautifulSoup

SALIENT_ATTRIBUTES = (
    "alt",
    "aria-describedby",
    "aria-label",
    "aria-role",
    "aria-controls",
    "input-checked",
    "label",
    "name",
    "option_selected",
    "placeholder",
    "readonly",
    "text-value",
    "title",
    "value",
    "data-gtm-label",
    "href",
    "role",
)

def process_element_tag(element: str, salient_attributes: Iterable[str]) -> str:
    """Clean an HTML element string, keeping only salient_attributes."""
    if not element.endswith(">"):
        element += "'>"

    soup = BeautifulSoup(element, "html.parser")
    for tag in soup.find_all(True):
        # Keep only salient attributes
        filtered_attrs = {k: tag.attrs[k] for k in tag.attrs if k in salient_attributes}
        name_val = filtered_attrs.pop("name", None)
        new_tag = soup.new_tag(tag.name, **filtered_attrs)
        if name_val:
            new_tag["name"] = name_val
        return str(new_tag).split(f"</{tag.name}>")[0]
    return element

if __name__ == "__main__":
    text = '<input type=\"text\" name=\"q\" id=\"mntl-search-form--open__search-input\" class=\"mntl-search-form__input\" placeholder=\"Find a recipe or ingredient\" required=\"required\" value=\"\" autocomplete=\"off\" style=\"\"> -> TYPE beef sirloin'
    print(process_element_tag(text, SALIENT_ATTRIBUTES))