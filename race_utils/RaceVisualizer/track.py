import numpy as np
import yaml

from race_utils.RaceGenerator.RaceTrack import RaceTrack


def rpy_to_rotation_matrix(rpy):
    r, p, y = np.array(rpy) * np.pi / 180
    cos, sin = np.cos, np.sin
    R = np.eye(3)
    R[0, 0] = cos(y) * cos(p)
    R[1, 0] = sin(y) * cos(p)
    R[2, 0] = -sin(p)
    R[0, 1] = cos(y) * sin(p) * sin(r) - sin(y) * cos(r)
    R[1, 1] = sin(y) * sin(p) * sin(r) + cos(y) * cos(r)
    R[2, 1] = cos(p) * sin(r)
    R[0, 2] = cos(y) * sin(p) * cos(r) + sin(y) * sin(r)
    R[1, 2] = sin(y) * sin(p) * cos(r) - cos(y) * sin(r)
    R[2, 2] = cos(p) * cos(r)
    return R


def plot_track(ax, track_file, set_radius=None, set_width=None, set_height=None, set_margin=0):
    track = track_file.to_dict() if isinstance(track_file, RaceTrack) else yaml.safe_load(open(track_file).read())

    for g in track["orders"]:
        g = track[g]

        if g["type"] == "FreeCorridor":
            raise NotImplementedError("Not support plotting FreeCorridor gate")

        args = {"linewidth": 3.0, "markersize": 5.0, "color": "black"}

        if g["type"] == "SingleBall":
            position = g["position"]
            r = g["radius"] - g["margin"]
            if set_radius is not None:
                r = set_radius - set_margin
            a = np.linspace(0, 2 * np.pi)
            ax.plot(position[0] + r * np.cos(a), position[1] + r * np.sin(a), "-", **args)
            # draw center
            ax.scatter(position[0], position[1], color="black", s=50)

        elif g["type"] == "TrianglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["margin"])
            hh = 0.5 * (g["height"] - g["margin"])
            if set_width is not None:
                hw = 0.5 * (set_width - set_margin)
            if set_height is not None:
                hh = 0.5 * (set_height - set_margin)
            drift = 0.0
            verts = [[-hh, hw, drift], [hh, 0.0, drift], [-hh, -hw, drift], [-hh, hw, drift]] @ R.T + np.array(
                position
            ).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        elif g["type"] == "RectanglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["marginW"])
            hh = 0.5 * (g["height"] - g["marginH"])
            if set_width is not None:
                hw = 0.5 * (set_width - set_margin)
            if set_height is not None:
                hh = 0.5 * (set_height - set_margin)
            drift = 0.0
            verts = [
                [-hh, hw, drift],
                [-hh, -hw, drift],
                [hh, -hw, drift],
                [hh, hw, drift],
                [-hh, hw, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        elif g["type"] == "PentagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            ar = g["radius"] - g["margin"]
            if set_radius is not None:
                ar = set_radius - set_margin
            cos54 = np.cos(0.3 * np.pi)
            sin54 = np.sin(0.3 * np.pi)
            nd, on = ar * cos54, ar * sin54
            bc = 2 * nd
            fc = bc * sin54
            of = ar - bc * cos54
            drift = 0.0
            verts = [
                [-on, nd, drift],
                [of, fc, drift],
                [ar, 0.0, drift],
                [of, -fc, drift],
                [-on, -nd, drift],
                [-on, nd, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        elif g["type"] == "HexagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            aside = g["side"] - g["margin"]
            if set_radius is not None:
                aside = set_radius - set_margin
            hside = 0.5 * aside
            height = hside * np.tan(np.pi / 3.0)
            drift = 0.0
            verts = [
                [-height, hside, drift],
                [0.0, aside, drift],
                [height, hside, drift],
                [height, -hside, drift],
                [0.0, -aside, drift],
                [-height, -hside, drift],
                [-height, hside, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], "o-", **args)

        else:
            raise ValueError("Unrecognized gate: " + g["type"])


def plot_track_3d(
    ax, track_file, set_radius=None, set_width=None, set_height=None, set_margin=0, color=None, gate_alpha=0.1
):
    if color is None:
        color = "r"

    track = track_file.to_dict() if isinstance(track_file, RaceTrack) else yaml.safe_load(open(track_file).read())

    for g in track["orders"]:
        g = track[g]

        if g["type"] == "FreeCorridor":
            raise NotImplementedError("Not support plotting FreeCorridor gate")

        args = {"linewidth": 3.0, "markersize": 5.0, "color": "black"}

        if g["type"] == "SingleBall":
            position = g["position"]
            r = g["radius"] - g["margin"]
            if set_radius is not None:
                r = set_radius - set_margin

            # create sphere
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = position[0] + r * np.cos(u) * np.sin(v)
            y = position[1] + r * np.sin(u) * np.sin(v)
            z = position[2] + r * np.cos(v)
            # draw sphere
            ax.plot_surface(x, y, z, color=color, alpha=gate_alpha, edgecolor="none")
            # draw center
            ax.scatter(position[0], position[1], position[2], color="black", s=50)

        elif g["type"] == "TrianglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["margin"])
            hh = 0.5 * (g["height"] - g["margin"])
            if set_width is not None:
                hw = 0.5 * (set_width - set_margin)
            if set_height is not None:
                hh = 0.5 * (set_height - set_margin)
            drift = 0.0
            verts = [[-hh, hw, drift], [hh, 0.0, drift], [-hh, -hw, drift], [-hh, hw, drift]] @ R.T + np.array(
                position
            ).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)

        elif g["type"] == "RectanglePrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            hw = 0.5 * (g["width"] - g["marginW"])
            hh = 0.5 * (g["height"] - g["marginH"])
            if set_width is not None:
                hw = 0.5 * (set_width - set_margin)
            if set_height is not None:
                hh = 0.5 * (set_height - set_margin)
            drift = 0.0
            verts = [
                [-hh, hw, drift],
                [-hh, -hw, drift],
                [hh, -hw, drift],
                [hh, hw, drift],
                [-hh, hw, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)

        elif g["type"] == "PentagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            ar = g["radius"] - g["margin"]
            if set_radius is not None:
                ar = set_radius - set_margin
            cos54 = np.cos(0.3 * np.pi)
            sin54 = np.sin(0.3 * np.pi)
            nd, on = ar * cos54, ar * sin54
            bc = 2 * nd
            fc = bc * sin54
            of = ar - bc * cos54
            drift = 0.0
            verts = [
                [-on, nd, drift],
                [of, fc, drift],
                [ar, 0.0, drift],
                [of, -fc, drift],
                [-on, -nd, drift],
                [-on, nd, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)

        elif g["type"] == "HexagonPrisma":
            position = g["position"]
            R = rpy_to_rotation_matrix(g["rpy"])
            aside = g["side"] - g["margin"]
            if set_radius is not None:
                aside = set_radius - set_margin
            hside = 0.5 * aside
            height = hside * np.tan(np.pi / 3.0)
            drift = 0.0
            verts = [
                [-height, hside, drift],
                [0.0, aside, drift],
                [height, hside, drift],
                [height, -hside, drift],
                [0.0, -aside, drift],
                [-height, -hside, drift],
                [-height, hside, drift],
            ] @ R.T + np.array(position).reshape((1, 3))
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], "o-", **args)

        else:
            raise ValueError("Unrecognized gate: " + g["type"])
