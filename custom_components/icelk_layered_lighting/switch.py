from typing import Callable
from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .utils import get_data


async def async_setup_entry(hass, entry, async_add_entities: AddEntitiesCallback):
    """Set up switches."""
    data = get_data(hass, entry)
    definitions = data["layer_switches"]

    switches = [
        LayerActiveSwitch(*definition) if definition else None
        for definition in definitions
    ]
    async_add_entities([switch for switch in switches if switch])
    data["layer_switches_data"] = switches

    definitions = data["override_switches"]

    switches = [
        ManualOverrideSwitch(*definition) if definition else None
        for definition in definitions
    ]
    async_add_entities([sensor for sensor in switches if sensor])
    data["override_switches_data"] = switches


class LayerActiveSwitch(SwitchEntity):
    """Is this layer on?"""

    _attr_should_poll = False
    is_on = False
    _attr_has_entity_name = True

    def __init__(
        self, device: DeviceInfo, id: str, name: str, callback: Callable
    ) -> None:
        """Init layer sensor and attach to device."""
        self._attr_device_info = device
        self._attr_unique_id = id
        self._attr_name = name
        self._callback = callback

    def own_update(self, new_value: bool):
        """Update with new value."""
        self.is_on = new_value
        self.schedule_update_ha_state()

    def turn_on(self, **kwargs):
        """Turn on."""
        self.is_on = True
        self._callback(True)

    def turn_off(self, **kwargs):
        """Turn off."""
        self.is_on = False
        self._callback(False)


class ManualOverrideSwitch(SwitchEntity):
    """If the light is manually overridden."""

    _attr_should_poll = False
    is_on = False
    _attr_has_entity_name = True
    _attr_name = "Manually overridden"

    def __init__(
        self, device: DeviceInfo, id: str, callback: Callable, name: str | None = None
    ) -> None:
        """Init layer sensor and attach to device."""
        self._attr_device_info = device
        self._attr_unique_id = id
        self._callback = callback
        if name is not None:
            self._attr_name = f"{self._attr_name} for {name}"

    def own_update(self, new_value: bool):
        """Update with new value."""
        self.is_on = new_value
        self.schedule_update_ha_state()

    def turn_on(self, **kwargs):
        """Turn on."""
        self.is_on = True
        self._callback(True)

    def turn_off(self, **kwargs):
        """Turn off."""
        self.is_on = False
        self._callback(False)
