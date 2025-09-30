from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .utils import get_data


async def async_setup_entry(hass, entry, async_add_entities: AddEntitiesCallback):
    """Set up sensors."""
    data = get_data(hass, entry)
    definitions = data["override_sensors"]

    sensors = [
        ManualOverrideSensor(definition[0], definition[1]) if definition else None
        for definition in definitions
    ]
    async_add_entities([sensor for sensor in sensors if sensor])
    data["override_sensors_data"] = sensors


class ManualOverrideSensor(BinarySensorEntity):
    """Tells the user which layer this light is attached to."""

    _attr_should_poll = False
    is_on = False
    _attr_has_entity_name = True
    _attr_name = "Manually overridden"

    def __init__(self, device: DeviceInfo, id: str) -> None:
        """Init layer sensor and attach to device."""
        self._attr_device_info = device
        self._attr_unique_id = id

    def own_update(self, new_value: bool):
        """Update with new value."""
        self.is_on = new_value
        self.schedule_update_ha_state()
