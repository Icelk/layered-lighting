"""Config flow for the Layered Lighting integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import voluptuous as vol

from homeassistant.helpers import selector
from homeassistant.helpers.schema_config_entry_flow import (
    SchemaConfigFlowHandler,
    SchemaFlowFormStep,
    SchemaFlowMenuStep,
)

from .const import DOMAIN

OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Optional("manual_override_timeout"): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0, max=600, step="any", unit_of_measurement="minutes"
            )
        ),
        vol.Required(
            "dimming_speed", msg="Speed of light dimming", default=40
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0, max=100, unit_of_measurement="% per second"
            )
        ),
        vol.Required(
            "dimming_delay", msg="Delay before dimming kicks in", default=0.5
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.1, max=5, step="any", unit_of_measurement="seconds"
            )
        ),
        vol.Required(
            "toggle_speed",
            default=0.2,
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0, max=5, step="any", unit_of_measurement="seconds"
            )
        ),
        vol.Required(
            "action_interval",
            default=60,
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=2, unit_of_measurement="seconds")
        ),
        vol.Optional(
            "switch_threshold",
            default=0,
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=100, unit_of_measurement="%")
        ),
        vol.Required(
            "manual_detect_enabled",
            default=True,
        ): selector.BooleanSelector(),
        vol.Optional("lights", msg="Lights"): selector.ObjectSelector(
            selector.ObjectSelectorConfig(
                fields={
                    "entity": selector.ObjectSelectorField(
                        label="Entity ID",
                        required=True,
                        selector={
                            "entity": {
                                "required": True,
                                "multiple": False,
                                "filter": {
                                    "domain": ["light", "switch", "input_boolean"]
                                },
                            }
                        },
                    ),
                    "factor": selector.ObjectSelectorField(
                        label="Light brightness factor (for sun power action) (also controls dim speed)",
                        required=False,
                        selector={
                            "number": {
                                "min": 0.1,
                                "max": 10,
                                "step": "any",
                                "default": 1,
                            }
                        },
                    ),
                    "min_brightness": selector.ObjectSelectorField(
                        label="Minimum brightness while on",
                        required=False,
                        selector={
                            "number": {
                                "min": 1,
                                "max": 100,
                                "step": "any",
                                "default": 30,
                                "unit_of_measurement": "%",
                            }
                        },
                    ),
                    "dimming_btn_down_trigger": selector.ObjectSelectorField(
                        label="Dimming: button down trigger (also available as action)",
                        required=False,
                        selector={"trigger": {}},
                    ),
                    "dimming_btn_up_trigger": selector.ObjectSelectorField(
                        label="Dimming: button up trigger (also available as action)",
                        required=False,
                        selector={"trigger": {}},
                    ),
                    "toggle_trigger": selector.ObjectSelectorField(
                        label="Toggle: trigger (also available as action) (don't use if you have dimming)",
                        required=False,
                        selector={"trigger": {}},
                    ),
                },
                multiple=True,
            )
        ),
        vol.Optional("layers", msg="Layers"): selector.ObjectSelector(
            selector.ObjectSelectorConfig(
                fields={
                    "name": selector.ObjectSelectorField(
                        label="ID",
                        required=True,
                        selector={"text": {}},
                    ),
                    "trigger_enable": selector.ObjectSelectorField(
                        label="Enable layer on (also available as action)",
                        required=False,
                        selector={"trigger": {}},
                    ),
                    "trigger_disable": selector.ObjectSelectorField(
                        label="Disable layer on (also available as action)",
                        required=False,
                        selector={"trigger": {}},
                    ),
                    "lights": selector.ObjectSelectorField(
                        label="Lights (leave empty for all, if action is scene, this is auto-populated)",
                        required=False,
                        selector={
                            "entity": {
                                "required": True,
                                "multiple": True,
                                "filter": {
                                    "domain": ["light", "switch", "input_boolean"]
                                },
                            }
                        },
                    ),
                    "action": selector.ObjectSelectorField(
                        label="Action when layer activates",
                        required=True,
                        selector={"action": {}},
                    ),
                },
                multiple=True,
            )
        ),
    }
)

CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required("name"): selector.TextSelector(),
    }
).extend(OPTIONS_SCHEMA.schema)

CONFIG_FLOW: dict[str, SchemaFlowFormStep | SchemaFlowMenuStep] = {
    "user": SchemaFlowFormStep(CONFIG_SCHEMA)
}

OPTIONS_FLOW: dict[str, SchemaFlowFormStep | SchemaFlowMenuStep] = {
    "init": SchemaFlowFormStep(OPTIONS_SCHEMA)
}


class ConfigFlowHandler(SchemaConfigFlowHandler, domain=DOMAIN):
    """Handle a config or options flow for Layered Lighting."""

    config_flow = CONFIG_FLOW
    # TODO remove the options_flow if the integration does not have an options flow
    options_flow = OPTIONS_FLOW

    def async_config_entry_title(self, options: Mapping[str, Any]) -> str:
        """Return config entry title."""
        return cast(str, options["name"]) if "name" in options else ""
