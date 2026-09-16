"""Build the dinner-table MuJoCo scene from configs/scene.json and a seed.

Items are simple primitive shapes made for this project (no third-party 3D
assets). The two SO-101 arms come from the pinned Menagerie model.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("left_arm", "right_arm")
UTENSILS = ("fork", "spoon")
SCENE_ITEMS = ("cup",) + UTENSILS


@dataclass(frozen=True)
class SceneParams:
    seed: int
    # item -> (x, y, yaw) resting pose on the table
    poses: dict
    # utensil -> mat slot index; both slot orders occur so the policy must look
    slots: dict
    masses: dict
    frictions: dict
    cup_radius: float
    cup_half_height: float
    utensil_scale: dict
    table_rgb: tuple
    light_diffuse: float
    light_offset: tuple


def load_config(path=None) -> dict:
    return json.loads(Path(path or ROOT / "configs/scene.json").read_text())


def sample_params(config: dict, seed: int) -> SceneParams:
    rng = np.random.default_rng(seed)
    u = lambda pair: float(rng.uniform(pair[0], pair[1]))
    cup, ut = config["cup"], config["utensil"]
    noise = cup["pos_noise"]
    poses = {"cup": (cup["pos"][0] + u((-noise, noise)), cup["pos"][1] + u((-noise, noise)),
                     u((-math.pi, math.pi)))}
    order = list(UTENSILS) if rng.random() < 0.5 else list(reversed(UTENSILS))
    slots = {item: i for i, item in enumerate(order)}
    for item in UTENSILS:
        sx, sy = ut["slots"][slots[item]]
        n = ut["pos_noise"]
        poses[item] = (sx + u((-n, n)), sy + u((-n, n)),
                       math.pi / 2 + u((-ut["yaw_noise"], ut["yaw_noise"])))
    masses = {"cup": u(cup["mass"])} | {item: u(ut["mass"]) for item in UTENSILS}
    frictions = {"cup": u(cup["friction"])} | {item: u(ut["friction"]) for item in UTENSILS}
    light = config["light"]
    off = light["offset"]
    return SceneParams(
        seed=seed, poses=poses, slots=slots, masses=masses, frictions=frictions,
        cup_radius=u(cup["radius"]), cup_half_height=u(cup["half_height"]),
        utensil_scale={item: u(ut["scale"]) for item in UTENSILS},
        table_rgb=_wood_colour(config["table_rgb"], rng),
        light_diffuse=u(light["diffuse"]),
        light_offset=(u((-off, off)), u((-off, off))),
    )


def _wood_colour(cfg: dict, rng) -> tuple:
    """Brightness-scaled wood tone with a little hue jitter (never green like the zones)."""
    t = float(rng.uniform(cfg["brightness"][0], cfg["brightness"][1]))
    j = cfg["jitter"]
    return tuple(float(min(1.0, max(0.0, c * t + rng.uniform(-j, j)))) for c in cfg["base"])


def _v(values) -> str:
    return " ".join(f"{float(x):.6g}" for x in values)


def _box_inertia(mass, half):
    x, y, z = (2 * h for h in half)
    return (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)


def _utensil_xml(item: str, params: SceneParams, config: dict) -> str:
    s = params.utensil_scale[item]
    hx, hy, hz = config["utensil"]["handle_half"]
    hx *= s
    x, y, yaw = params.poses[item]
    mass, fr = params.masses[item], params.frictions[item]
    common = f'friction="{fr:.4g} 0.05 0.002" condim="4" solref="0.01 1"'
    if item == "fork":
        rgba = "0.50 0.52 0.58 1"
        head = [f'<geom name="fork_neck" type="box" size="{0.006*s:.4g} {0.012*s:.4g} {hz*0.7:.4g}" '
                f'pos="{hx + 0.002:.4g} 0 {-hz*0.3:.4g}" rgba="{rgba}" {common} mass="0"/>']
        for i, py in enumerate((-0.009, 0.0, 0.009)):
            head.append(f'<geom name="fork_prong{i}" type="box" size="{0.018*s:.4g} 0.0022 {hz*0.6:.4g}" '
                        f'pos="{hx + 0.026*s:.4g} {py*s:.4g} {-hz*0.4:.4g}" rgba="{rgba}" {common} mass="0"/>')
    else:
        rgba = "0.84 0.85 0.88 1"
        head = [f'<geom name="spoon_bowl" type="ellipsoid" size="{0.026*s:.4g} {0.017*s:.4g} {hz*0.8:.4g}" '
                f'pos="{hx + 0.022*s:.4g} 0 {-hz*0.2:.4g}" rgba="{rgba}" {common} mass="0"/>']
    length = hx + 0.045 * s
    inertia = _box_inertia(mass, (length, 0.012, hz))
    return f"""
    <body name="{item}" pos="{x:.5f} {y:.5f} {hz + 0.0005:.5f}" euler="0 0 {yaw:.5f}">
      <freejoint name="{item}_free"/>
      <inertial pos="{0.01*s:.4g} 0 0" mass="{mass:.6g}" diaginertia="{_v(inertia)}"/>
      <geom name="{item}_handle" type="box" size="{hx:.4g} {hy:.4g} {hz:.4g}" rgba="{rgba}" {common} mass="0"/>
      {"".join(head)}
    </body>"""


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> tuple:
    """MuJoCo camera xyaxes for a camera at `pos` looking at `target` (the camera looks along -z)."""
    forward = np.subtract(target, pos, dtype=float)
    forward /= np.linalg.norm(forward)
    x = np.cross(forward, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, forward)
    return (*x, *y)


# Presentation only: never seen by the policy, never used for training data.
SHOWCASE_CAMERAS = {  # name: (position, look-at point, vertical field of view); arms face +y
    "hero": ((0.55, 0.82, 0.42), (0.0, 0.13, 0.05), 40),
    "handoff": ((0.28, 0.50, 0.17), (0.06, 0.15, 0.07), 46),
    "left_side": ((-0.68, 0.50, 0.32), (0.0, 0.16, 0.05), 40),
    "wide": ((0.0, 1.05, 0.72), (0.0, 0.12, 0.0), 42),
    "cover": ((0.44, 0.66, 0.20), (0.03, 0.13, 0.155), 44),  # scene low in frame, sky free for a title
}


def _showcase_xml():
    """Sky, floor, a shadow-casting light and presentation cameras (no collisions, no bodies)."""
    asset = """<asset>
    <texture name="showcase_sky" type="skybox" builtin="gradient" rgb1="0.30 0.42 0.58" rgb2="0.05 0.07 0.10" width="512" height="512"/>
    <texture name="showcase_floor" type="2d" builtin="checker" rgb1="0.23 0.32 0.42" rgb2="0.17 0.24 0.32" width="512" height="512" mark="edge" markrgb="0.75 0.78 0.82"/>
    <material name="showcase_floor" texture="showcase_floor" texrepeat="12 12" reflectance="0.12"/>
  </asset>"""
    extras = ['<geom name="showcase_floor" type="plane" size="4 4 0.1" pos="0 0 -0.05" material="showcase_floor" '
              'contype="0" conaffinity="0"/>',
              '<light name="showcase_sun" pos="0.8 -0.9 1.8" dir="-0.8 0.9 -1.8" diffuse="0.55 0.55 0.52" '
              'castshadow="true"/>']
    for name, (pos, target, fovy) in SHOWCASE_CAMERAS.items():
        extras.append(f'<camera name="showcase_{name}" pos="{_v(pos)}" '
                      f'xyaxes="{" ".join(f"{v:.5f}" for v in look_at_xyaxes(pos, target))}" fovy="{fovy}"/>')
    return asset, "".join(extras)


def world_xml(params: SceneParams, config: dict, showcase: bool = False) -> str:
    """The MJCF world. showcase=True adds presentation-only extras; the default output is unchanged."""
    tx, ty = config["table"]["center"]
    thx, thy = config["table"]["half_size"]
    plate, mat = config["plate"], config["mat"]
    r, hh = params.cup_radius, params.cup_half_height
    cx, cy, cyaw = params.poses["cup"]
    cup_inertia = (params.masses["cup"] * (3 * r * r + 4 * hh * hh) / 12,) * 2 + (params.masses["cup"] * r * r / 2,)
    zones = "".join(
        f'<geom name="{name}" type="box" size="{_v(z["half_size"])} 0.0004" pos="{_v(z["pos"])} 0.0004" '
        f'contype="0" conaffinity="0" rgba="0.96 0.92 0.70 0.9"/>'
        for name, z in config["zones"].items())
    cameras = "".join(
        f'<camera name="{name}" pos="{_v(c["pos"])}" xyaxes="{_v(c["xyaxes"])}" fovy="{c["fovy"]}"/>'
        for name, c in config["cameras"].items())
    lx, ly = params.light_offset
    d = params.light_diffuse
    asset, off_w, off_h = "", 640, 480
    if showcase:
        asset, extras = _showcase_xml()
        cameras += extras
        off_w, off_h = 1920, 1080
    return f"""<mujoco model="rescuehands_dinner_table">
  <compiler angle="radian"/>{asset}
  <option integrator="implicitfast" timestep="0.005" cone="elliptic" impratio="10" iterations="10" ls_iterations="20"/>
  <visual>
    <global offwidth="{off_w}" offheight="{off_h}"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light name="key" pos="{lx:.4f} {0.2 + ly:.4f} 1.6" dir="{-lx:.4f} {-ly:.4f} -1.6" diffuse="{d:.3f} {d:.3f} {d:.3f}"/>
    <light name="fill" pos="0.6 -0.6 1.2" dir="-0.5 0.5 -1" diffuse="0.25 0.25 0.25"/>
    <geom name="table" type="box" size="{thx} {thy} 0.025" pos="{tx} {ty} -0.025"
          rgba="{_v(params.table_rgb)} 1" friction="0.8 0.01 0.001"/>
    <geom name="plate" type="cylinder" size="{plate["radius"]} {plate["half_height"]}"
          pos="{_v(plate["pos"])} {plate["half_height"]}" rgba="0.95 0.95 0.93 1"/>
    <geom name="plate_rim" type="cylinder" size="{plate["radius"] * 0.72} 0.0002"
          pos="{_v(plate["pos"])} {2 * plate["half_height"] + 0.0002}" contype="0" conaffinity="0"
          rgba="0.85 0.86 0.84 1"/>
    <geom name="mat" type="box" size="{_v(mat["half_size"])} 0.0003" pos="{_v(mat["pos"])} 0.0003"
          contype="0" conaffinity="0" rgba="0.72 0.18 0.16 1"/>
    {zones}
    <body name="cup" pos="{cx:.5f} {cy:.5f} {hh + 0.0005:.5f}" euler="0 0 {cyaw:.5f}">
      <freejoint name="cup_free"/>
      <inertial pos="0 0 0" mass="{params.masses['cup']:.6g}" diaginertia="{_v(cup_inertia)}"/>
      <geom name="cup_body" type="cylinder" size="{r:.5f} {hh:.5f}" rgba="0.15 0.35 0.75 1"
            friction="{params.frictions['cup']:.4g} 0.01 0.001" condim="4" solref="0.01 1" mass="0"/>
      <geom name="cup_top" type="cylinder" size="{r * 0.8:.5f} 0.0005" pos="0 0 {hh:.5f}"
            contype="0" conaffinity="0" rgba="0.05 0.08 0.15 1" mass="0"/>
    </body>
    {"".join(_utensil_xml(item, params, config) for item in UTENSILS)}
    {cameras}
  </worldbody>
</mujoco>"""


def build_model(params: SceneParams, config: dict | None = None, asset_path: Path | None = None,
                showcase: bool = False):
    config = config or load_config()
    if asset_path is None:
        sim_cfg = json.loads((ROOT / "configs/simulation.json").read_text())
        asset_path = (ROOT / sim_cfg["asset_path"]).resolve()
    if not Path(asset_path).is_file():
        raise FileNotFoundError("SO-101 model missing. Follow the asset setup in README.md.")
    spec = mujoco.MjSpec.from_string(world_xml(params, config, showcase=showcase))
    for arm in ARMS:
        mount = config["arms"][arm]
        yaw = mount["yaw"]
        child = mujoco.MjSpec.from_file(str(asset_path))
        child.meshdir = str(Path(asset_path).parent / "assets")
        frame = spec.worldbody.add_frame(name=arm + "_mount", pos=mount["pos"],
                                         quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(child, frame=frame, prefix=arm + "/")
    return spec.compile()
