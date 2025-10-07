"""The Layered Lighting integration."""

from __future__ import annotations
from asyncio import sleep
import asyncio
from datetime import datetime, time
import logging
from math import pi
from random import random
from typing import TYPE_CHECKING, Callable, Coroutine, Literal, TypedDict

from homeassistant.helpers import device_registry as dr, entity_registry

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import ServiceCall, State, callback
from homeassistant.core import HomeAssistant
from homeassistant.helpers.trigger import (
    async_initialize_triggers,
    async_validate_trigger_config,
)
from homeassistant.helpers.condition import (
    async_from_config,
    async_validate_conditions_config,
)
from homeassistant.helpers.service import (
    async_call_from_config,
    async_extract_entity_ids,
)
from suncalc import get_position

if TYPE_CHECKING:
    from homeassistant.helpers.trigger import TriggerConfig
    from homeassistant.helpers.service import (
        ConfigType as ServiceConfig,
    )
    from homeassistant.helpers.typing import ConfigType

from .utils import get_data as raw_get_data
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class _Light(TypedDict):
    entity: str
    factor: float
    min_brightness: float
    dimming_btn_down_trigger: "TriggerConfig"
    dimming_btn_up_trigger: "TriggerConfig"
    toggle_trigger: "TriggerConfig"


class _Layer(TypedDict):
    name: str
    lights: list[str]
    trigger_enable: "TriggerConfig"
    trigger_disable: "TriggerConfig"
    action: "ServiceConfig"


class _SunConfig(TypedDict):
    absolute: bool
    offset: float
    factor: float


# TODO: in 2026: remove hass from async_extract_entity_ids

debounce: dict[str, float] = {}


async def do_debounce(
    hass: HomeAssistant, entry: ConfigEntry, id: str, t: float, cb: Coroutine
):
    global debounce
    rand = random()

    async def task():
        global debounce
        await sleep(t)
        if debounce.get(id) == rand:
            del debounce[id]
            await cb
        else:
            cb.close()

    entry.async_create_background_task(hass, task(), f"debounce {id}")
    debounce[id] = rand


entries = {}


async def get_integration_for_entity(hass: HomeAssistant, entity_id: str) -> str | None:
    ent_reg = entity_registry.async_get(hass)
    dev_reg = dr.async_get(hass)

    ent_entry = ent_reg.async_get(entity_id)
    if not ent_entry or not ent_entry.device_id:
        return None

    device = dev_reg.async_get(ent_entry.device_id)
    if not device or not device.primary_config_entry:
        return None

    entry = hass.config_entries.async_get_entry(device.primary_config_entry)
    domain = entry.domain if entry else None
    return domain if domain != DOMAIN else None


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up integration."""

    def get_data():
        data = hass.data.setdefault(f"{DOMAIN}.global", {"services_to_unload": []})
        if "services_to_unload" not in data:
            data["services_to_unload"] = []
        return data

    device_registry = dr.async_get(hass)

    @callback
    def handle_sun_power(_call: ServiceCall):
        raise Exception("uncallable")

    def register(name: str, callback: Callable):
        hass.services.async_register(DOMAIN, name, callback)
        get_data()["services_to_unload"].append(name)

    def handle_name(name: str):
        @callback
        async def handle_callback(call: ServiceCall):
            if not (entry_config := call.data.get("entry")):
                raise Exception("invalid entry")
            if not (entry_data := entries.get(entry_config)):
                raise Exception("unknown entry")
            if entry_data[name] is None:
                raise Exception(
                    f"somebody who developed {DOMAIN} messed up and didn't register {name}"
                )
            await entry_data[name](call)

        register(name, handle_callback)

    def handle_names(names: list[str]):
        for name in names:
            handle_name(name)

    handle_names(
        [
            "layer_enable",
            "layer_disable",
            "layer_disable_all",
            "layer_toggle",
            "manual_override_enable",
            "manual_override_disable",
            "dimming_down",
            "dimming_up",
            "dimming_toggle",
            "update_internal_state",
        ]
    )
    register("sun_power", handle_sun_power)

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Layered Lighting from a config entry."""

    manual_detect_enabled = (
        f if (f := entry.options.get("manual_detect_enabled")) is not None else True
    )
    manual_override_timeout = entry.options.get("manual_override_timeout") or 0
    action_interval = entry.options.get("action_interval") or 60
    dimming_speed = entry.options.get("dimming_speed") or 40
    dimming_delay = v if (v := entry.options.get("dimming_delay")) is not None else 0.5
    toggle_speed = v if (v := entry.options.get("toggle_speed")) is not None else 0.2
    switch_threshold = (
        v if (v := entry.options.get("switch_threshold")) is not None else 20
    ) / 100
    layers: list[_Layer] = entry.options.get("layers") or []
    layers_id_to_idx = {layer["name"]: idx for (idx, layer) in enumerate(layers)}
    lights: list[_Light] = entry.options.get("lights") or []
    lights_loaded: list[bool] = [False for _ in lights]
    lights_id_to_idx = {light["entity"]: idx for (idx, light) in enumerate(lights)}

    def get_data():
        data = raw_get_data(hass, entry)
        if "triggers" not in data:
            data["triggers"] = []
        if "layer_sensors" not in data:
            data["layer_sensors"] = []
        return data

    # support multiple triggers
    async def add_triggers(trigger, callback):
        triggers = await async_validate_trigger_config(
            hass, _transform_triggers(trigger)
        )
        unsub = await async_initialize_triggers(
            hass,
            triggers,
            callback,
            domain=entry.domain,
            name=entry.title,
            log_cb=_LOGGER.log,
        )
        return get_data()["triggers"].append(unsub)

    async def attach_trigger_manually_handle_unsub(trigger, callback):
        triggers = await async_validate_trigger_config(
            hass, _transform_triggers(trigger)
        )
        return await async_initialize_triggers(
            hass,
            triggers,
            callback,
            domain=entry.domain,
            name=entry.title,
            log_cb=_LOGGER.log,
        )

    async def action(action):
        await async_call_from_config(hass, action, validate_config=True)

    entries[entry.entry_id] = {}
    entry_services = {}
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        name=entry.title,
        identifiers={(DOMAIN, entry.entry_id)},
    )
    get_data()["device"] = device

    device_callbacks: list[dict[str, Callable]] = []

    layers_enabled = [False for _ in layers]
    overrides: list[None | datetime] = [None for _ in lights]
    last_states: list[None | dict[str, int | float | str]] = [None for _ in lights]

    def extract_attributes(
        input: dict[str, str | int | float],
        strict_decode=False,
    ):
        output = {}
        if input.get("brightness") is not None:
            output["brightness"] = input["brightness"]
        elif input.get("brightness_pct") is not None:
            output["brightness_pct"] = input["brightness_pct"]
        elif input.get("brightness_step") is not None:
            output["brightness_step"] = input["brightness_step"]
        elif input.get("brightness_pct") is not None:
            output["brightness_step_pct"] = input["brightness_step_pct"]

        # supported_color_modes: brightness indicates lights don't support color.
        # often those kinds of lights reports colors oddly.
        if (not strict_decode) or (
            input.get("color_mode") != "brightness"
            and input.get("supported_color_modes") != ["brightness"]
        ):
            # trying to extract color state. This is kinda sketchy. At least for Hue lights, effect is `off` when disabled.
            # I don't know how it is for other brands.
            if input.get("effect") is not None and input["effect"] != "off":
                output["effect"] = input["effect"]
            elif input.get("color_temp_kelvin") is not None:
                output["color_temp_kelvin"] = input["color_temp_kelvin"]
            elif input.get("rgbww_color") is not None:
                output["rgbww_color"] = input["rgbww_color"]
            elif input.get("rgbw_color") is not None:
                output["rgbw_color"] = input["rgbw_color"]
            elif input.get("rgb_color") is not None:
                output["rgb_color"] = input["rgb_color"]
            elif input.get("hs_color") is not None:
                output["hs_color"] = input["hs_color"]
            elif input.get("xy_color") is not None:
                output["xy_color"] = input["xy_color"]
            elif input.get("color_name") is not None:
                output["color_name"] = input["color_name"]
        return output

    def extract_full_state(state: State) -> dict[str, float | int | str]:
        if state.state != "on":
            return {"state": state.state}
        else:
            s = extract_attributes(state.attributes, strict_decode=True)
            s["state"] = state.state
            return s

    def full_states_different(
        old: dict[str, str | float | int | list[float]],
        new: dict[str, str | float | int | list[float]],
    ) -> bool:
        numeric_min_diff = {
            "brightness": 3,
            "brightness_pct": 1,
            "color_temp_kelvin": 10,
            "color_temp": 5,
            "_arr_hs_color": 1,
            "_arr_rgb_color": 3,
            "_arr_rgbw_color": 3,
            "_arr_rgbww_color": 3,
            "_arr_xy_color": 0.01,
        }
        if old["state"] == "unavailable":
            return False
        if new.get("state") != old.get("state"):
            return True

        # Asymmetric, which is fine because if we get new props in `new` that's fine and we don't have to trigger manual override
        for key in old:
            if new.get(key) is None:
                return True
            if (numeric_min := numeric_min_diff.get(key)) is not None:
                x = float(old[key])
                y = float(new[key])
                if abs(x - y) > numeric_min:
                    return True
            elif (
                (numeric_min := numeric_min_diff.get(f"_arr_{key}")) is not None
                and isinstance(old[key], list)
                and isinstance(new[key], list)
            ):
                for x, y in zip(old[key], new[key]):
                    x = float(x)
                    y = float(y)
                    if abs(x - y) > numeric_min:
                        return True
            else:
                if old[key] != new[key]:
                    return True

        return False

    async def check_last_state(
        entity: str, idx: int, entity_state: State | None = None, sleep_t=1
    ) -> bool:
        last_state = last_states[idx]
        # get state
        entity_state = entity_state if entity_state else hass.states.get(entity)

        new_state = extract_full_state(entity_state)

        # if last observed state != the observed state, someone's changed!
        if last_state is not None and full_states_different(last_state, new_state):
            set_override(idx, datetime.now())
            return True
        # set this while we try to get the new state
        # because that takes time and possibly this is going to be called when we are trying to find out
        last_states[idx] = None

        return False

    async def update_last_state(entity: str, idx: int, sleep_t=1):
        # apparently groups don't update immediately
        await sleep(sleep_t)
        s = hass.states.get(entity)
        next_state = extract_full_state(s)
        last_states[idx] = {**next_state}

    async def set_entity_state(
        entity: str,
        state: Literal["on", "off"],
        attrs,
        check_override=True,
        entity_state: State | None = None,
        use_switch_threshold=False,
    ):
        match entity.split(".")[0]:
            case "light":
                if state != "on":
                    attrs = {}

                attrs["entity_id"] = entity
                if attrs.get("transition") is None:
                    attrs["transition"] = toggle_speed

                if (
                    check_override
                    and manual_detect_enabled
                    and (idx := lights_id_to_idx.get(entity))
                ):
                    if await check_last_state(entity, idx, entity_state):
                        return

                await hass.services.async_call(
                    "light",
                    "turn_on" if state == "on" else "turn_off",
                    attrs,
                    blocking=True,
                )
                if (
                    check_override
                    and manual_detect_enabled
                    and (idx := lights_id_to_idx.get(entity))
                ):
                    await update_last_state(entity, idx)

            case "switch":
                brightness = attrs.get("brightness")
                idx = lights_id_to_idx.get(entity)
                brightness_factor = (
                    (f if (f := lights[idx].get("factor")) else 1)
                    if idx is not None
                    else 1
                )
                if (
                    check_override
                    and manual_detect_enabled
                    and (idx := lights_id_to_idx.get(entity))
                ):
                    if await check_last_state(entity, idx, entity_state):
                        return

                await hass.services.async_call(
                    "switch",
                    "turn_on"
                    if (
                        state == "on"
                        and (
                            float(brightness * brightness_factor) / 255
                            >= switch_threshold
                            if brightness and use_switch_threshold
                            else True
                        )
                    )
                    else "turn_off",
                    {"entity_id": entity},
                    blocking=True,
                )

                if (
                    check_override
                    and manual_detect_enabled
                    and (idx := lights_id_to_idx.get(entity))
                ):
                    await update_last_state(entity, idx)
            case "input_boolean":
                hass.states.async_set(entity, state, {"entity_id": entity})

    async def toggle_light(entity: str):
        idx = lights_id_to_idx.get(entity)
        if idx is not None:
            if overrides[idx] is not None:
                set_override(idx, None)
                await update_light(idx)
                return
        if idx is not None:
            set_override(idx, datetime.now())
        s = hass.states.get(entity)
        if s.state == "off":
            # special turn on logic (initial brightness)
            # first, try base layer. If that's dark, turn on normally
            # if len(device_callbacks) >= 1 and (cb := device_callbacks[-1].get(entity)):
            if len(device_callbacks) >= 1 and (
                cb := next(
                    (
                        layers[entity]
                        for layers in reversed(device_callbacks)
                        if layers.get(entity)
                    ),
                    None,
                )
            ):
                await cb(check_override=False)
                s = hass.states.get(entity).state
                if s == "off":
                    # normal turn on
                    await set_entity_state(
                        entity,
                        "on",
                        {},
                        entity_state=s,
                        check_override=False,
                    )
            else:
                # normal turn on since we don't have layers
                await set_entity_state(
                    entity,
                    "on",
                    {},
                    entity_state=s,
                    check_override=False,
                )
        else:
            await set_entity_state(
                entity,
                "off",
                {},
                entity_state=s,
                check_override=False,
            )

    async def do_custom_lighting(
        entity_id: str, config: _SunConfig, check_override=True
    ):
        idx = lights_id_to_idx.get(entity_id)
        if idx is None:
            _LOGGER.warning("sun_power failed due to invalid light")
            return
        cfg = lights[idx]
        min_brightness = (
            (v if (v := cfg.get("min_brightness")) is not None else 30) / 100 * 255
        )
        pos = get_position(datetime.now(), hass.config.longitude, hass.config.latitude)
        altitude = float(pos["altitude"]) / pi * 2
        minI = (
            float(
                get_position(
                    datetime.combine(datetime.now().date(), time(hour=0)),
                    hass.config.longitude,
                    hass.config.latitude,
                )["altitude"]
            )
            / pi
            * 2
        )
        if config.get("absolute"):
            altitude /= abs(minI)
            altitude = (altitude + 1) * 0.5
        factor = f if (f := config.get("factor")) is not None else 1
        offset = v if (v := config.get("offset")) is not None else 0.2
        altitude *= factor
        altitude += offset

        # switches are handled in `set_entity_state`
        brightness_factor = (
            f
            if ((f := cfg.get("factor")) is not None and entity_id.startswith("light."))
            else 1
        )

        await set_entity_state(
            entity_id,
            "on",
            {
                "brightness": min(
                    255, max(int(altitude * brightness_factor * 255), min_brightness)
                ),
                "color_temp_kelvin": 2700,
                # "transition": 60,
            },
            check_override=check_override,
            use_switch_threshold=True,
        )

    def _set_layer(idx: int, new_value: bool):
        layers_enabled[idx] = new_value
        if data := get_data().get("layer_switches_data"):
            sensor = data[idx]
            if sensor:
                sensor.own_update(new_value)

    async def set_layer(idx: int, new_value: bool, realize_updates=True):
        should_update = layers_enabled[idx] != new_value
        if should_update:
            _set_layer(idx, new_value)
        if should_update and realize_updates:
            await update_layers()

    def set_override(idx: int, overridden: None | datetime):
        overrides[idx] = overridden
        # reset last states when override done, so it won't start another override
        if overridden is None:
            last_states[idx] = None
        if data := get_data().get("override_switches_data"):
            sensor = data[idx]
            if sensor:
                sensor.own_update(overridden is not None)

    def light_set_sensor(idx: int, layer: str):
        if data := get_data().get("layer_sensors_data"):
            sensor = data[idx]
            if sensor:
                sensor.own_update(layer)

    def light_get_layer(entity_id: str):
        for idx in range(len(layers)):
            if (
                layers_enabled[idx]
                and len(device_callbacks) > idx
                and device_callbacks[idx].get(entity_id)
            ):
                return idx

        return -1

    async def update_lights(indices: list[int]):
        await asyncio.gather(
            *[entry.async_create_task(hass, update_light(idx)) for idx in indices]
        )

    async def update_light(idx: int):
        light = lights[idx]
        entity = light["entity"]

        layer = light_get_layer(entity)
        # above override logic, since we always want to show correct layer
        light_set_sensor(idx, "<none => off>" if layer == -1 else layers[layer]["name"])

        # override
        override = overrides[idx]
        if override is not None:
            if manual_override_timeout != 0 and (
                (datetime.now() - override).total_seconds()
                > manual_override_timeout * 60
            ):
                set_override(idx, None)
            else:
                return

        if layer == -1:
            await set_entity_state(
                entity,
                "off",
                {},
            )
            return
        cb = device_callbacks[layer].get(entity)
        if cb:
            await cb()
        else:
            _LOGGER.warning(
                "entity wasn't found in layer! some internal conflict happened"
            )

    async def update_layers():
        await update_lights(
            [idx for idx, enabled in enumerate(lights_loaded) if enabled]
        )

    resolving_layers = False
    should_re_resolve = False

    async def resolve_layers():
        nonlocal resolving_layers
        nonlocal should_re_resolve

        _LOGGER.info("Resolve layers")

        if resolving_layers:
            should_re_resolve = True
            return

        for idx, light in enumerate(lights):
            lights_loaded[idx] = (
                s := hass.states.get(light["entity"])
            ) is not None and s.state != "unavailable"

        ll = set(
            [light["entity"] for idx, light in enumerate(lights) if lights_loaded[idx]]
        )

        resolving_layers = True
        device_callbacks.clear()

        scenes: list[str] = []

        for i in range(len(overrides)):
            overrides[i] = datetime.now()
        for layer in layers:
            callbacks = {}
            local_lights = layer.get("lights")
            entities = (
                local_lights if local_lights else [light["entity"] for light in lights]
            )
            entities = list(set(entities).intersection(ll))
            if layer.get("action") is None:
                for entity in entities:
                    callbacks[entity] = lambda: ()
                device_callbacks.append(callbacks)
                continue

            if len(layer["action"]) != 1:
                for entity in entities:
                    callbacks[entity] = lambda: ()
                _LOGGER.error(
                    "invalid action in layer %s. Only one action is allowed.",
                    layer["name"],
                )
                device_callbacks.append(callbacks)
                continue

            action_type = layer["action"][0]["action"]
            if action_type == "scene.turn_on":
                # only touch the entities relevant to the scene in this layer
                if not local_lights:
                    entities = set()
                    scene_entities = layer["action"][0]["target"]["entity_id"]
                    if isinstance(scene_entities, list):
                        for scene in scene_entities:
                            scenes.append(scene)
                            s = hass.states.get(scene)
                            light_entities = s.attributes["entity_id"]
                            entities = entities.union(light_entities)
                    else:
                        scenes.append(scene_entities)
                        s = hass.states.get(scene_entities)
                        light_entities = s.attributes["entity_id"]
                        entities = entities.union(light_entities)
                    entities = list(entities)

                entities = list(set(entities).intersection(ll))

                await action(layer["action"][0])
                await sleep(2 + len(entities) * 0.1)
                for entity in entities:
                    state = hass.states.get(entity)
                    if state is not None:
                        s = state.state
                        attrs = extract_attributes(state.attributes)

                        async def cb1(
                            check_override=True,
                            entity=entity,
                            s=s,
                            attrs=attrs,
                        ):
                            await set_entity_state(
                                entity,
                                s,
                                attrs,
                                check_override=check_override,
                            )

                        callbacks[entity] = cb1
                    else:
                        _LOGGER.warning("light with id %s not found!", entity)
            elif action_type == f"{DOMAIN}.sun_power":
                for entity in entities:

                    async def cb2(check_override=True, entity=entity, layer=layer):
                        await do_custom_lighting(
                            entity,
                            layer["action"][0]["data"],
                            check_override=check_override,
                        )

                    callbacks[entity] = cb2
            elif len(entities) <= 1:
                for entity in entities:

                    async def cb3(layer=layer):
                        if manual_detect_enabled and (
                            idx := lights_id_to_idx.get(entity)
                        ):
                            if await check_last_state(entity, idx):
                                return
                        await action(layer["action"][0])
                        if manual_detect_enabled and (
                            idx := lights_id_to_idx.get(entity)
                        ):
                            await update_last_state(entity, idx)

                    callbacks[entity] = cb3
            else:
                for entity in entities:
                    callbacks[entity] = lambda: ()
                _LOGGER.error(
                    "invalid action in layer %s. Please use either `scene.turn_on`, one from this component, or restrict to 1 light",
                    layer["name"],
                )

            device_callbacks.append(callbacks)

        await sleep(1)
        for i in range(len(overrides)):
            overrides[i] = None
            last_states[i] = None

        resolving_layers = False
        await update_layers()

        if should_re_resolve:
            _LOGGER.info(
                "Re-resolving layers because we got requested to do so when we ran last."
            )
            should_re_resolve = False
            await resolve_layers()

    async def handle_update_internal_state(_call: ServiceCall):
        await resolve_layers()

    entry_services["update_internal_state"] = handle_update_internal_state

    for light in lights:

        async def callback(_e=None, _ctx=None):
            await resolve_layers()

        await add_triggers(
            [
                {
                    "platform": "state",
                    "from": "unavailable",
                    "entity_id": [light["entity"]],
                }
            ],
            callback,
        )

    await resolve_layers()

    for idx, layer in enumerate(layers):
        trigger_enable = layer.get("trigger_enable")
        trigger_disable = layer.get("trigger_disable")
        enable_if = layer.get("enable_if")

        async def enable_layer(e=None, ctx=None, idx=idx):
            await set_layer(idx, True)

        async def disable_layer(e=None, ctx=None, idx=idx):
            await set_layer(idx, False)

        if trigger_enable:
            await add_triggers(trigger_enable, enable_layer)
        if trigger_disable:
            await add_triggers(trigger_disable, disable_layer)
        if enable_if:
            transformed_conditions = await async_validate_conditions_config(
                hass, enable_if
            )
            transformed_conditions = _transform_conditions(transformed_conditions)

            async def check_conditions(
                e=None, ctx=None, idx=idx, transformed_conditions=transformed_conditions
            ):
                all_true = True if len(transformed_conditions) > 0 else False
                for condition in transformed_conditions:
                    check = await async_from_config(hass, condition)
                    result = check(hass, {})
                    if not result:
                        all_true = False
                        break

                await set_layer(idx, all_true)

            async def debounced_check_conditions(
                e=None, ctx=None, layer=layer, check_conditions=check_conditions
            ):
                await do_debounce(
                    hass,
                    entry,
                    f"check states for layer {layer['name']}",
                    0.2,
                    check_conditions(),
                )

            triggers = []
            for condition in enable_if:
                _condition_to_triggers(condition, triggers)
            await add_triggers(triggers, debounced_check_conditions)
            await debounced_check_conditions()

    async def runtime():
        while True:
            await sleep(action_interval)
            await update_layers()

    entry.async_create_background_task(hass, runtime(), "update lights")

    async def handle_layer_enable(call: ServiceCall):
        layer = call.data.get("layer_id")
        idx = layers_id_to_idx.get(layer if layer is not None else "")
        if idx is None:
            raise Exception("layer not registered")
        await set_layer(idx, True)

    entry_services["layer_enable"] = handle_layer_enable

    async def handle_layer_disable(call: ServiceCall):
        layer = call.data.get("layer_id")
        idx = layers_id_to_idx.get(layer if layer is not None else "")
        if idx is None:
            raise Exception("layer not registered")
        await set_layer(idx, False)

    entry_services["layer_disable"] = handle_layer_disable

    async def handle_layer_toggle(call: ServiceCall):
        layer = call.data.get("layer_id")
        idx = layers_id_to_idx.get(layer if layer is not None else "")
        if idx is None:
            raise Exception("layer not registered")
        await set_layer(idx, not layers_enabled[idx])

    entry_services["layer_toggle"] = handle_layer_toggle

    async def handle_layer_disable_all(_call: ServiceCall):
        for i in range(len(layers_enabled)):
            await set_layer(i, False, realize_updates=False)

        await update_layers()

    entry_services["layer_disable_all"] = handle_layer_disable_all

    async def handle_enable_manual_override(call: ServiceCall):
        entities = await async_extract_entity_ids(hass, call)
        for entity in entities:
            idx = lights_id_to_idx.get(entity)
            if idx is not None:
                set_override(idx, datetime.now())

    entry_services["manual_override_enable"] = handle_enable_manual_override

    async def handle_disable_manual_override(call: ServiceCall):
        entities = await async_extract_entity_ids(hass, call)
        indices = []
        for entity in entities:
            idx = lights_id_to_idx.get(entity)
            if idx is not None:
                indices.append(idx)
                set_override(idx, None)
        await update_lights(indices)

    entry_services["manual_override_disable"] = handle_disable_manual_override

    dim_direction_up = [False for _ in lights]
    dimming = [False for _ in lights]
    dimming_started = [False for _ in lights]

    async def dim(light: str):
        idx = lights_id_to_idx.get(light)
        if idx is None:
            _LOGGER.warning("invalid light idx for dimming %s", light)
            return
        dim_speed_factor = min(
            v if (v := lights[idx].get("factor")) is not None else 1, 1
        )
        s = hass.states.get(light)
        if s is None:
            _LOGGER.warning("invalid light entity for dimming %s", light)
            return
        if dimming_started[idx]:
            return
        start_brightness: float = s.attributes.get("brightness") or 0
        if s.state == "off":
            start_brightness = 0
        start = datetime.now()
        interval = 0.1
        await sleep(dimming_delay - interval)
        if not dimming[idx]:
            return
        dimming_started[idx] = True
        set_override(idx, start)
        if start_brightness < 10:
            start_brightness = 3
            dim_direction_up[idx] = True
        if start_brightness > 220:
            start_brightness = 250
            dim_direction_up[idx] = False
        iterations = 0
        while True:
            await sleep(interval)
            if not dimming[idx]:
                break
            now = datetime.now()
            current_brightness = (
                start_brightness
                + (1 if dim_direction_up[idx] else -1)
                * (now - start).total_seconds()
                * dimming_speed
                * dim_speed_factor
                / 100
                * 255
            )
            if current_brightness < 3:
                current_brightness = 3
                dimming[idx] = False
            if current_brightness > 254:
                current_brightness = 255
                dimming[idx] = False
            await set_entity_state(
                light,
                "on",
                {
                    "brightness": int(current_brightness),
                    "transition": interval * 2
                    if iterations == 0
                    else (now - start).total_seconds() / iterations,
                },
                check_override=False,
            )
            if not dimming[idx]:
                break
            iterations += 1
        dim_direction_up[idx] = not dim_direction_up[idx]
        dimming_started[idx] = False

    async def handle_dimming_down(call: ServiceCall):
        entities = await async_extract_entity_ids(hass, call)
        for entity in entities:
            idx = lights_id_to_idx.get(entity)
            if idx is None:
                raise Exception("light not registered")
            dimming[idx] = True
            entry.async_create_task(hass, dim(entity), f"dim {entity}")

    entry_services["dimming_down"] = handle_dimming_down

    async def handle_dimming_up(call: ServiceCall):
        entities = await async_extract_entity_ids(hass, call)
        for entity in entities:
            idx = lights_id_to_idx.get(entity)
            if idx is None:
                raise Exception("light not registered")
            if dimming[idx] and not dimming_started[idx]:
                dimming[idx] = False
                await toggle_light(entity)

            dimming[idx] = False
            dimming_started[idx] = False

    entry_services["dimming_up"] = handle_dimming_up

    async def handle_dimming_toggle(call: ServiceCall):
        entities = await async_extract_entity_ids(hass, call)
        for entity in entities:
            await toggle_light(entity)

    entry_services["dimming_toggle"] = handle_dimming_toggle

    for idx, light in enumerate(lights):
        down = light.get("dimming_btn_down_trigger")
        up = light.get("dimming_btn_up_trigger")
        toggle = light.get("toggle_trigger")

        async def dim_down(e=None, ctx=None, idx=idx, light=light):
            dimming[idx] = True
            entry.async_create_task(
                hass, dim(light["entity"]), f"dim {light['entity']}"
            )

        async def dim_up(e=None, ctx=None, idx=idx, light=light):
            if dimming[idx] and not dimming_started[idx]:
                dimming[idx] = False
                await toggle_light(light["entity"])
            dimming[idx] = False
            dimming_started[idx] = False

        async def dim_toggle(e=None, ctx=None, idx=idx, light=light):
            await toggle_light(light["entity"])

        if down:
            await add_triggers(down, dim_down)
        if up:
            await add_triggers(up, dim_up)
        if toggle:
            await add_triggers(toggle, dim_toggle)

    er = entity_registry.async_get(hass)
    layer_sensors = []
    override_switches = []
    for idx, light in enumerate(lights):

        def override_cb(new_bool: bool, idx=idx, light=light):
            set_override(idx, datetime.now() if new_bool else None)
            # since this is called from another thread we have to do this
            # maybe we should not pass the switch around..
            asyncio.run_coroutine_threadsafe(update_light(idx), hass.loop)

        entity = er.async_get(light["entity"])
        if entity:
            if entity.device_id:
                light_dev = device_registry.async_get(entity.device_id)
                device_info = dr.DeviceInfo(
                    identifiers=light_dev.identifiers,
                    name=light_dev.name,
                    connections=light_dev.connections,
                )
                layer_sensors.append(
                    (
                        device_info,
                        f"{DOMAIN}.layer_sensor.{light['entity']}",
                    )
                )
                override_switches.append(
                    (
                        device_info,
                        f"{DOMAIN}.override_sensor.{light['entity']}",
                        override_cb,
                    )
                )
            else:
                device_info = None
                layer_sensors.append(
                    (
                        device_info,
                        f"{DOMAIN}.layer_sensor.{light['entity']}",
                        light["entity"],
                    )
                )
                override_switches.append(
                    (
                        device_info,
                        f"{DOMAIN}.override_sensor.{light['entity']}",
                        override_cb,
                        light["entity"],
                    )
                )
        else:
            layer_sensors.append(None)
            override_switches.append(None)
    get_data()["layer_sensors"] = layer_sensors
    get_data()["override_switches"] = override_switches

    layer_switches = []
    for idx, layer in enumerate(layers):

        def cb(new_bool: bool, idx=idx):
            should_update = new_bool != layers_enabled[idx]
            _set_layer(idx, new_bool)
            # since this is called from another thread we have to do this
            # maybe we should not pass the switch around..
            if should_update:
                asyncio.run_coroutine_threadsafe(update_layers(), hass.loop)

        layer_switches.append(
            (
                dr.DeviceInfo(
                    identifiers=device.identifiers,
                    name=device.name,
                ),
                f"{DOMAIN}.layer_sensor.{layer['name']}",
                layer["name"],
                cb,
            ),
        )
    get_data()["layer_switches"] = layer_switches

    entries[entry.entry_id] = entry_services

    await hass.config_entries.async_forward_entry_setups(entry, ["sensor", "switch"])

    entry.async_on_unload(entry.add_update_listener(config_entry_update_listener))

    return True


# TODO Remove if the integration does not have an options flow
async def config_entry_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update listener, called when the config entry options are changed."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(
        entry, ["sensor", "switch"]
    )

    if data := hass.data.get(f"{DOMAIN}.{entry.entry_id}"):
        if unload_ok:
            data["layer_sensors"] = []
            data["layer_sensors_data"] = []
            data["layer_switches"] = []
            data["layer_switches_data"] = []
            data["override_switches"] = []
            data["override_switches_data"] = []
        if device := data.get("device"):
            device_registry = dr.async_get(hass)
            device_registry.async_remove_device(device.id)

            del data["device"]

        for trigger in data["triggers"]:
            if trigger:
                trigger()
        data["triggers"].clear()

    return True


def _transform_triggers(triggers):
    for trigger in triggers:
        t = trigger.get("trigger")
        if t:
            trigger["platform"] = t
            del trigger["trigger"]
    return triggers


def _transform_conditions(conditions):
    for condition in conditions:
        if e := condition.get("before"):
            try:
                t = time.fromisoformat(e)
                condition["before"] = t
            except Exception:
                pass
        if e := condition.get("after"):
            try:
                t = time.fromisoformat(e)
                condition["after"] = t
            except Exception:
                pass
        if (cs := condition.get("conditions")) and isinstance(cs, list):
            _transform_conditions(cs)
    return conditions


def _condition_to_triggers(condition, triggers=[]):
    def single_list(v: str | list[str]) -> list[str]:
        return v if isinstance(v, list) else [v]

    match condition["condition"]:
        case "time":
            if (before := condition.get("before")) is not None:
                trigger = {"platform": "time", "at": before}
                if wd := condition.get("weekday"):
                    trigger["weekday"] = wd
                triggers.append(trigger)

            if (after := condition.get("after")) is not None:
                trigger = {"platform": "time", "at": after}
                if wd := condition.get("weekday"):
                    trigger["weekday"] = wd
                triggers.append(trigger)
        case "state":
            trigger = {
                "platform": "state",
                "entity_id": single_list(condition["entity_id"]),
                "to": condition["state"],
            }
            if attr := condition.get("attribute"):
                trigger["attribute"] = attr
            if f := condition.get("for"):
                trigger["for"] = f

            triggers.append(trigger)

            trigger = {
                "platform": "state",
                "entity_id": single_list(condition["entity_id"]),
                "from": condition["state"],
            }
            if attr := condition.get("attribute"):
                trigger["attribute"] = attr

            triggers.append(trigger)
        case "numeric_state":
            trigger = {
                "platform": "numeric_state",
                "entity_id": single_list(condition["entity_id"]),
            }
            if above := condition.get("above"):
                trigger["above"] = above
            if below := condition.get("below"):
                trigger["below"] = below
            if attr := condition.get("attribute"):
                trigger["attribute"] = attr
            if vt := condition.get("value_template"):
                trigger["value_template"] = vt

            if condition.get("below") is not None or condition.get("above") is not None:
                triggers.append(trigger)
            if below := condition.get("below"):
                trigger = {
                    "platform": "numeric_state",
                    "entity_id": single_list(condition["entity_id"]),
                    "above": below,
                }
                if attr := condition.get("attribute"):
                    trigger["attribute"] = attr
                if vt := condition.get("value_template"):
                    trigger["value_template"] = vt
                triggers.append(trigger)
            if above := condition.get("above"):
                trigger = {
                    "platform": "numeric_state",
                    "entity_id": single_list(condition["entity_id"]),
                    "below": above,
                }
                if attr := condition.get("attribute"):
                    trigger["attribute"] = attr
                if vt := condition.get("value_template"):
                    trigger["value_template"] = vt
                triggers.append(trigger)
        case "not" | "and" | "or":
            for c in condition.get("conditions") or []:
                _condition_to_triggers(c, triggers)

    return triggers
