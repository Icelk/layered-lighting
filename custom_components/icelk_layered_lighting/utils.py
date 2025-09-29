from typing import Any
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN


def get_data(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Get data for this entry."""
    return hass.data.setdefault(f"{DOMAIN}.{entry.entry_id}", {})
