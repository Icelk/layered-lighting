from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .utils import get_data


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    """Set up sensors."""
    data = get_data(hass, entry)
    definitions = data["layer_sensors"]

    sensors = [
        ActiveLayerSensor(definition[0], definition[1]) if definition else None
        for definition in definitions
    ]
    data["layer_sensors_data"] = sensors
    async_add_entities([sensor for sensor in sensors if sensor])


class ActiveLayerSensor(SensorEntity):
    """Tells the user which layer this light is attached to."""

    _attr_should_poll = False
    _attr_native_value = ""
    _attr_has_entity_name = True
    _attr_name = "Active light layer"

    def __init__(self, device: DeviceInfo, id: str) -> None:
        """Init layer sensor and attach to device."""
        self._attr_device_info = device
        self._attr_unique_id = id

    def own_update(self, new_value: str):
        """Update with new value."""
        self._attr_native_value = new_value
        self.schedule_update_ha_state()
