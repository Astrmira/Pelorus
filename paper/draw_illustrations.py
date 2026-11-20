"""生成 Pelorus 几何与执行引擎插图，保留 SVG 源文件并导出高清 PNG。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
import html
import math
import os
import shutil

import matplotlib
from matplotlib import font_manager, mathtext
import numpy as np
import resvg_py


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "vldb" / "figures"
SUBMISSION = ROOT / "vldb_submission" / "figures"
INK = "#17263F"
NAVY = "#3E4C93"
TEAL = "#148C8B"
ORANGE = "#C98B28"
PURPLE = "#8866AE"
RED = "#C76461"
GRAY = "#778394"
LIGHT = "#E7EBF0"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["svg.fonttype"] = "path"


def font_files() -> list[str]:
    """依次复用系统和论文编译缓存中的字体，其他环境回退到 Matplotlib 字体。"""
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    regular = windows_fonts / "LinBiolinum_R_G.ttf"
    bold = windows_fonts / "LinBiolinum_RB_G.ttf"
    if regular.exists() and bold.exists():
        return [str(regular), str(bold)]
    cache = Path(os.environ.get("LOCALAPPDATA", "")) / "TectonicProject/Tectonic/cache/bundles/data"
    result = []
    for name in ["LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinLibertine_R.otf"]:
        matches = list(cache.glob(f"*/{name}"))
        if matches:
            result.append(str(matches[0]))
    if result:
        return result
    return [font_manager.findfont("DejaVu Sans")]


FONTS = font_files()
FONT = font_manager.FontProperties(fname=FONTS[0]).get_name()


class Drawing:
    """以统一坐标和字体生成可编辑文字与矢量几何。"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = []
        self.formula_count = 0
        self.parts.append('<rect width="100%" height="100%" fill="white"/>')

    def add(self, markup: str):
        self.parts.append(markup)

    def path(self, d, color=INK, width=2, fill="none", opacity=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="{fill}" opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{extra}/>')

    def line(self, x1, y1, x2, y2, color=INK, width=2, dash=None, opacity=1):
        self.path(f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}", color, width, opacity=opacity, dash=dash)

    def polygon(self, points, fill="white", color=INK, width=2, opacity=1):
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polygon points="{coords}" fill="{fill}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round"/>')

    def rect(self, x, y, width, height, fill="white", color=INK, stroke=2, radius=0):
        self.add(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{color}" stroke-width="{stroke}"/>')

    def circle(self, x, y, radius, fill="white", color=INK, width=2):
        self.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{color}" stroke-width="{width}"/>')

    def ellipse(self, x, y, rx, ry, fill="none", color=INK, width=2, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def text(self, x, y, value, size=27, color=INK, weight="normal", anchor="start", rotate=None):
        extra = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(f'<text x="{x}" y="{y}" fill="{color}" font-family="{html.escape(FONT)}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{extra}>{html.escape(value)}</text>')

    def formula(self, x, y, value, size=25, color=INK, anchor="start"):
        """公式由数学排版器生成矢量轮廓，并使用独立标识避免字形引用冲突。"""
        buffer = BytesIO()
        mathtext.math_to_image(f"${value}$", buffer, prop=font_manager.FontProperties(size=size), format="svg", color=color)
        root = ET.fromstring(buffer.getvalue())
        for parent in root.iter():
            for child in list(parent):
                if child.attrib.get("id") == "patch_1":
                    parent.remove(child)
        view_box = root.attrib["viewBox"]
        _, _, width, height = map(float, view_box.split())
        prefix = f"formula{self.formula_count}_"
        self.formula_count += 1
        for element in root.iter():
            if "id" in element.attrib:
                element.attrib["id"] = prefix + element.attrib["id"]
            for key, val in list(element.attrib.items()):
                if key.endswith("href") and val.startswith("#"):
                    element.attrib[key] = "#" + prefix + val[1:]
                elif "url(#" in val:
                    element.attrib[key] = val.replace("url(#", "url(#" + prefix)
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        root.attrib.update({"x": str(x), "y": str(y), "width": str(width), "height": str(height)})
        self.add(ET.tostring(root, encoding="unicode"))

    def arrow(self, points, color=INK, width=2.5, dash=None, head=9):
        self.path("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points), color, width, dash=dash)
        end = np.asarray(points[-1], dtype=float)
        tangent = end - np.asarray(points[-2], dtype=float)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        back = end - head * tangent
        self.polygon([end, back + head * 0.45 * normal, back - head * 0.45 * normal], fill=color, color=color, width=0)

    def star(self, x, y, radius=15, fill="#F0CE81"):
        angles = np.linspace(-math.pi / 2, 3 * math.pi / 2, 11)[:-1]
        points = [(x + (radius if i % 2 == 0 else radius * 0.44) * math.cos(a), y + (radius if i % 2 == 0 else radius * 0.44) * math.sin(a)) for i, a in enumerate(angles)]
        self.polygon(points, fill=fill, color=INK, width=2)

    def save(self, name: str):
        FIGURES.mkdir(exist_ok=True)
        SUBMISSION.mkdir(exist_ok=True)
        svg = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ' + f'width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">' + "\n".join(self.parts) + "</svg>"
        svg_path = FIGURES / f"{name}.svg"
        svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        svg_path.write_text(svg, encoding="utf-8")
        png = resvg_py.svg_to_bytes(svg_string=svg, width=4200, background="white", font_files=FONTS, skip_system_fonts=True, font_family=FONT)
        png_path = FIGURES / f"{name}.png"
        png_path.write_bytes(png)
        for path in [svg_path, png_path]:
            shutil.copy2(path, SUBMISSION / path.name)
        print(f"已生成矢量与高清插图：{name}")


def point(radius, angle):
    return radius * complex(math.cos(math.radians(angle)), math.sin(math.radians(angle)))


def geodesic(a: complex, b: complex, samples=90):
    """用 Möbius 平移后的径向插值生成庞加莱球内的真实测地线。"""
    relative = (b - a) / (1 - a.conjugate() * b)
    norm = abs(relative)
    if norm < 1e-12:
        return np.repeat(a, samples)
    radial = np.tanh(np.linspace(0, math.atanh(min(norm, 1 - 1e-12)), samples)) * relative / norm
    return (a + radial) / (1 + a.conjugate() * radial)


def map_point(z, center, radius, vertical=1):
    return (center[0] + radius * z.real, center[1] + radius * vertical * z.imag)


def draw_geodesic(drawing, a, b, center, radius, color, width=2, opacity=1, dash=None, vertical=1):
    coordinates = [map_point(z, center, radius, vertical) for z in geodesic(a, b)]
    drawing.path("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in coordinates), color, width, opacity=opacity, dash=dash)


def geometry_figure():
    d = Drawing(1800, 590)
    d.text(22, 43, "(a) Radial occlusion", 34, weight="bold")
    d.text(555, 43, "(b) Inside-out QuadriPrune", 34, weight="bold")
    d.text(1280, 43, "(c) Polar + tangent codes", 34, weight="bold")
    d.line(524, 70, 524, 557, LIGHT, 1.5)
    d.line(1252, 70, 1252, 557, LIGHT, 1.5)

    # 左图同时展示中心枢纽的内球遮挡和同壳层幸存角域。
    center, radius = (253, 250), 172
    depth, alpha = 4.0, 1.2
    outer_radius = math.tanh(depth / 2)
    occlusion_radius = math.tanh(depth / (alpha + 1) / 2)
    cap = math.degrees(2 * math.asin(math.sinh(alpha * depth / 2) / math.sinh(depth)))
    u = point(outer_radius, -37)
    d.circle(*center, radius, "#FAFBFC", INK, 3)
    for angle in [-130, -85, -40, 5, 50]:
        draw_geodesic(d, point(0.991, angle), point(0.991, angle + 130), center, radius, "#D5DCE5", 1.2)
    d.circle(*center, radius * occlusion_radius, "#F6E3E0", RED, 1.6)
    for y in range(145, 365, 20):
        for x in range(148, 363, 20):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < (radius * occlusion_radius - 5) ** 2:
                d.circle(x, y, 1, "#D8AEAA", "none", 0)
    outer = [map_point(point(0.99, a), center, radius) for a in np.linspace(-37 - cap, -37 + cap, 35)]
    inner = [map_point(point(0.80, a), center, radius) for a in np.linspace(-37 + cap, -37 - cap, 35)]
    d.polygon(outer + inner, "#BFDFD9", TEAL, 1)
    for angle in [-175, -130, -92, 17, 60, 101, 146]:
        candidate = point(outer_radius, angle)
        draw_geodesic(d, u, candidate, center, radius, RED, 2.1, opacity=0.70, dash="6 5")
        d.circle(*map_point(candidate, center, radius), 5.6, "white", INK, 1.8)
    for angle in [-49, -26]:
        candidate = point(outer_radius, angle)
        draw_geodesic(d, u, candidate, center, radius, NAVY, 2.8)
        d.circle(*map_point(candidate, center, radius), 5.6, "#DCE8EE", INK, 1.8)
    for a, r in [(30, .45), (130, .38), (190, .55), (264, .50)]:
        x, y = map_point(point(r, a), center, radius)
        d.line(x - 4, y - 4, x + 4, y + 4, RED, 2)
        d.line(x - 4, y + 4, x + 4, y - 4, RED, 2)
    d.line(*center, *map_point(u, center, radius), NAVY, 3.8)
    d.circle(*center, 7, RED, INK, 2)
    ux, uy = map_point(u, center, radius)
    d.circle(ux, uy, 8.5, TEAL, INK, 2)
    d.formula(ux + 14, uy - 17, "u", 30)
    d.formula(center[0] + 12, center[1] + 9, "p=0", 28)
    d.text(253, 445, "All-angle occlusion", 28, RED, "bold", "middle")
    d.formula(253, 457, r"R_v\leq R_{\mathrm{occ}}=\frac{R_u-\alpha R_p}{\alpha+1}", 25, anchor="middle")
    d.text(253, 526, "Same-shell angular radius", 25, anchor="middle")
    d.formula(253, 537, r"73.7^\circ\ \longrightarrow\ 23.1^\circ\quad(\rho:0\to4)", 24, anchor="middle")

    # 中图的 32 条出边采用 12、16、2、2 的示例配额，环通道累计到 28。
    center, radius = (889, 258), 217
    u = point(.86, -33)
    d.ellipse(*center, radius, radius * .74, "#FAFBFC", INK, 3)
    for r in [.30, .58, .86]:
        d.ellipse(*center, radius * r, radius * r * .74, "#EEF4F3" if r == .58 else "none", "#B4C5CD", 1.5)
    geodesic_nodes = [point(.24 + .041 * i, -160 + i * 28) for i in range(12)]
    entry_depth = math.atanh(math.tanh(4.0) * .7)
    entry = point(math.tanh(entry_depth / 2), -105)
    geodesic_nodes[0] = entry
    ring_nodes = [point(.86, angle) for angle in np.linspace(-177, 167, 16)]
    inward_nodes = [point(.42, -30), point(.50, -45)]
    outward_nodes = [point(.97, -22), point(.97, -45)]
    groups = [(geodesic_nodes, NAVY), (ring_nodes, TEAL), (inward_nodes, ORANGE), (outward_nodes, PURPLE)]
    assert sum(len(nodes) for nodes, _ in groups) == 32
    for nodes, color in groups:
        for target in nodes:
            draw_geodesic(d, u, target, center, radius, color, 2.2, .78, vertical=.74)
    for nodes, color in [(inward_nodes, ORANGE), (outward_nodes, PURPLE)]:
        for target in nodes:
            tail = geodesic(u, target)[-24:]
            d.arrow([map_point(z, center, radius, .74) for z in tail], color, 2.3, head=10)
    for nodes, color in groups:
        for target in nodes:
            d.circle(*map_point(target, center, radius, .74), 5.4, color, "white", 1.5)
    d.line(*center, *map_point(point(.99, -105), center, radius, .74), GRAY, 1.8, dash="4 5")
    ex, ey = map_point(entry, center, radius, .74)
    d.star(ex, ey, 14)
    d.text(579, 112, "PolarRouter", 25, weight="bold")
    d.formula(580, 123, r"\rho_{\mathrm{sup}}^*(\rho_q,\gamma)", 24)
    d.line(712, 126, ex - 15, ey - 8, GRAY, 1.3)
    island, anchor, old_target = point(.98, 135), geodesic_nodes[4], ring_nodes[9]
    draw_geodesic(d, anchor, old_target, center, radius, GRAY, 1.7, .75, dash="4 5", vertical=.74)
    draw_geodesic(d, anchor, island, center, radius, TEAL, 3.1, dash="6 3", vertical=.74)
    ix, iy = map_point(island, center, radius, .74)
    d.circle(ix, iy, 7.5, "#D1E8D7", TEAL, 2)
    d.text(565, 393, "Chord swap", 24, TEAL, weight="bold")
    d.line(697, 383, ix - 6, iy + 8, TEAL, 1.4)
    ux, uy = map_point(u, center, radius, .74)
    d.circle(ux, uy, 9, "#F2FAF8", INK, 2.5)
    d.formula(ux + 14, uy - 19, "u", 31)
    d.text(889, 434, "32 outgoing slots · cumulative ring cap 28", 25, anchor="middle")
    bar_x, bar_y, unit = 621, 451, 16.8
    offset = 0
    for count, color in [(12, NAVY), (16, TEAL), (2, ORANGE), (2, PURPLE)]:
        d.rect(bar_x + offset * unit, bar_y, count * unit, 41, color, "white", 1)
        for i in range(1, count):
            d.line(bar_x + (offset + i) * unit, bar_y + 3, bar_x + (offset + i) * unit, bar_y + 38, "white", .7, opacity=.3)
        d.text(bar_x + (offset + count / 2) * unit, bar_y + 30, str(count), 29, "white", "bold", "middle")
        offset += count
    for x, label, color in [(621, "geodesic", NAVY), (787, "ring", TEAL), (918, "inward", ORANGE), (1060, "outward", PURPLE)]:
        d.line(x, 526, x + 23, 526, color, 4)
        d.text(x + 31, 534, label, 23)
    d.formula(889, 548, r"|\rho_w-\rho_v|\leq0.60", 24, anchor="middle")

    # 右图的切平面通过球面北极，轴向投影和切向分量保持正交。
    d.path("M1364,285 A146,146 0 0 1 1656,285", INK, 2.6, "#F2F2F8")
    d.ellipse(1510, 285, 146, 32, "#F8FAFC", INK, 2)
    for half_width in [48, 97]:
        d.path(f"M1510,139 C{1510-half_width},152 {1510-half_width},269 1510,317", "#ADB7C8", 1.5)
        d.path(f"M1510,139 C{1510+half_width},152 {1510+half_width},269 1510,317", "#ADB7C8", 1.5)
    d.polygon([(1315,114),(1605,114),(1705,164),(1415,164)], "#D6EAE4", TEAL, 2, .87)
    origin = (1510, 285)
    axial = (1510, 285 - 146 * math.sqrt(1 - .25**2))
    candidate = (1510 + 146 * .25, axial[1])
    d.arrow([origin, (1510, 139)], PURPLE, 2.7)
    d.arrow([origin, candidate], NAVY, 2.7)
    d.arrow([axial, candidate], ORANGE, 3, head=7)
    d.arrow([(1510, 139),(1656, 139)], TEAL, 2.5)
    d.circle(*candidate, 5.5, NAVY, "white", 1)
    d.circle(*axial, 4, PURPLE, "white", 1)
    d.formula(1490, 78, r"\mu_c", 27)
    d.formula(1582, 114, r"\tilde r_v", 27)
    d.formula(1560, 162, r"u_v", 27)
    d.formula(1284, 196, r"T_{\mu_c}\mathbb{S}^{D-1}", 23)
    d.formula(1516, 338, r"u_v=\alpha_v\mu_c+\beta_v\tilde r_v", 27, anchor="middle")
    d.text(1285, 400, "32 B polar header", 26, weight="bold")
    widths = [176,176,117]
    x = 1285
    for width, label, fill in zip(widths, ["radial · 12 B", "SM · 12 B", "tag + α · 8 B"], ["#DCE4F0", "#D6E9E4", "#E7DEF0"]):
        d.rect(x, 411, width, 38, fill, "white", 1)
        d.text(x + width / 2, 438, label, 23, anchor="middle")
        x += width
    d.text(1285, 482, "Same 2-bit slot", 26, weight="bold")
    for i in range(24):
        d.rect(1285 + i * 19.55, 494, 18.5, 29, ["#DCE4F0", "#91C6BE", "#D4B47B", "#BDACD4"][i % 4], "white", .7)
    d.text(1519, 559, "0 B/vertex extra", 30, TEAL, "bold", "middle")
    d.save("pelorus_geometry")


def memory_tile(d, origin, u, v, fill, edge=INK):
    points = [origin, origin + u, origin + u + v, origin + v]
    d.polygon(points, fill, edge, 1.7)


def execution_figure():
    d = Drawing(1800, 570)
    d.text(18, 43, "(a) Single-page layout", 34, weight="bold")
    d.text(620, 43, "(b) Header-first traversal", 34, weight="bold")
    d.text(1217, 43, "(c) Pool refinement", 34, weight="bold")
    d.line(600, 70, 600, 550, LIGHT, 1.5)
    d.line(1196, 70, 1196, 550, LIGHT, 1.5)

    # 左图的记录严格分为 64 B 头和三条各 64 B 的位平面缓存行。
    d.text(117, 92, "SoA", 28, weight="bold", anchor="middle")
    for x, label, fill in [(27, "meta", "#E5E9EF"), (132, "bits", "#DFE8E8")]:
        d.rect(x, 115, 80, 127, fill, INK, 1.8)
        for y in [146,178,210]:
            d.line(x, y, x + 80, y, "#B5BEC9", 1)
        d.text(x + 40, 143, label, 22, anchor="middle")
        d.text(x + 40, 202, "···", 27, anchor="middle")
    d.text(116, 268, "two array mappings", 22, anchor="middle")
    d.text(432, 94, "256 B vertex record", 27, weight="bold", anchor="middle")
    x, y, w, h = 313, 116, 235, 42
    d.rect(x, y, w / 2, h, "#D4E8E3", INK, 1.8)
    d.rect(x + w / 2, y, w / 2, h, "#E4DCEF", INK, 1.8)
    d.text(x + w / 4, y + 27, "32 B meta", 23, anchor="middle")
    d.text(x + 3 * w / 4, y + 27, "32 B sketch", 23, anchor="middle")
    for i in range(1,4):
        d.rect(x, y + i * h, w, h, ["#DCE4F1", "#E8E9F4", "#E2DEF0"][i - 1], INK, 1.8)
        d.text(x + w / 2, y + i * h + 28, "64 B bitplanes", 24, anchor="middle")
    for i in range(4):
        d.text(x - 11, y + i * h + 27, f"CL{i}", 20, anchor="end")
    d.path(f"M555,{y+h} h9 v{3*h} h-9", INK, 1.5)
    d.text(584, 224, "192 B", 22, anchor="middle", rotate=-90)
    d.path("M107,244 C110,285 99,303 115,338", GRAY, 1.6)
    d.path("M212,244 C223,281 236,281 243,311", GRAY, 1.6)

    page_origin = np.array([78.0, 443.0])
    u, v = np.array([67.0,-8.0]), np.array([27.0,-27.0])
    page = [page_origin, page_origin + 4*u, page_origin + 4*u + 4*v, page_origin + 4*v]
    d.polygon([page[0], page[1], page[1] + [0,15], page[0] + [0,15]], "#C9D1DE", INK, 1.8)
    d.polygon([page[1], page[2], page[2] + [0,15], page[1] + [0,15]], "#B9C4D4", INK, 1.8)
    d.polygon(page, "#F4F7FB", INK, 2)
    for i in range(4):
        for j in range(4):
            origin = page_origin + (i + .07)*u + (j + .07)*v
            selected = i == 3 and j == 3
            memory_tile(d, origin, .86*u, .86*v, "#E4D1A3" if selected else "#DFE9EC", ORANGE if selected else "#7C8CA3")
            for stripe in range(1,4):
                a = origin + .86*v*(stripe/4)
                b = a + .86*u
                d.line(*a,*b, "#A3B7BF", .8)
    selected_center = page_origin + 3.5*u + 3.5*v
    d.line(*selected_center, 339, 286, GRAY, 1.4, dash="5 4")
    d.line(selected_center[0]+27, selected_center[1]-17, 526, 286, GRAY, 1.4, dash="5 4")
    d.text(276, 490, "4 KiB HotSmPage · 16 × 256 B", 28, weight="bold", anchor="middle")
    d.text(276, 531, "one page mapping covers the full record", 25, TEAL, anchor="middle")

    # 中图的残差旁路从径向门控之后分出，仅绕过草图门控。
    d.arrow([(550,136),(580,136),(612,210),(634,210)], INK, 2.2)
    for x, title, subtitle, fill in [(637,"Radial","bound", "#D6EAE5"),(808,"256-bit","sketch", "#DCE6EF")]:
        d.polygon([(x,147),(x+127,174),(x+127,277),(x,305)], fill, INK, 2.2)
        d.text(x+62,214,title,28,weight="bold",anchor="middle")
        d.text(x+62,248,subtitle,27,anchor="middle")
    d.text(700, 128, "Stage-1a", 24, anchor="middle")
    d.text(871, 128, "Stage-1b", 24, anchor="middle")
    d.arrow([(766,211),(805,211)], INK, 2.4)
    d.arrow([(936,211),(993,211)], INK, 2.4)
    d.arrow([(786,211),(786,82),(1083,82),(1083,137)], PURPLE, 2.1, dash="7 5")
    d.text(957,110,"residual cone",24,PURPLE,anchor="middle")
    for x in [700,871]:
        d.arrow([(x,294),(x,358)], RED, 2.2)
        for offset in [-20,0,20]:
            d.line(x+offset-4,374,x+offset+4,382,RED,2.5)
            d.line(x+offset-4,382,x+offset+4,374,RED,2.5)
    d.text(788,438,"51.65%",39,RED,"bold","middle")
    d.text(788,471,"of candidates rejected",26,RED,anchor="middle")
    d.rect(997,139,178,155,NAVY,NAVY,2,5)
    for i in range(7):
        d.line(1011+i*24,133,1011+i*24,139,INK,2)
        d.line(1011+i*24,294,1011+i*24,300,INK,2)
    d.text(1086,177,"CSA AVX2",27,"white","bold","middle")
    for start_y, end_y in [(201,218),(224,218),(247,241)]:
        d.line(1014,start_y,1046,start_y,"white",2.2)
        d.line(1046,start_y,1096,end_y,"white",2.2)
    d.arrow([(1096,218),(1153,218)],"white",2.2,head=7)
    d.arrow([(1096,241),(1153,241)],"white",2.2,head=7)
    d.text(1086,278,"3 → 2 fold",23,"white",anchor="middle")
    d.text(1086,336,"15 → 10 shuffles",26,weight="bold",anchor="middle")
    d.text(1086,375,"5 popcounts",24,anchor="middle")
    d.text(1086,413,"<35 ns/candidate",24,anchor="middle")
    d.formula(896,515,r"P_{\mathrm{rad}}>\theta_{\mathrm{beam}}",25,anchor="middle")

    # 右图采用递增代理分数，并把扩大的精排窗口画在两个截断位置之间。
    for i in range(6):
        d.polygon([(1220+i*3,165-i*11),(1271+i*3,165-i*11),(1285+i*3,152-i*11),(1234+i*3,152-i*11)],"#E4DCEF",PURPLE,1.3)
    d.formula(1248,80,"L",30,anchor="middle")
    d.arrow([(1290,142),(1313,142)],INK,2.2)
    d.polygon([(1316,101),(1455,117),(1455,174),(1316,189)],"#D6EAE5",TEAL,2.1)
    d.text(1387,139,"Stage-0.5",28,weight="bold",anchor="middle")
    d.text(1387,168,"projection",24,anchor="middle")
    d.text(1387,217,"1.5 KiB i16",24,anchor="middle")
    d.text(1387,246,"query · 768D",22,anchor="middle")
    d.text(1387,281,"+2.2% / +36.8% QPS",21,TEAL,"bold","middle")
    d.text(1387,308,"at 99% / 99.5% recall",20,TEAL,anchor="middle")
    d.arrow([(1456,145),(1493,145)],INK,2.1)
    d.text(1640,85,"CJ-Gate",30,weight="bold",anchor="middle")
    px, py, pw, ph = 1502, 109, 267, 181
    b0, beff = .38, .76
    d.rect(px+pw*b0,py,pw*(beff-b0),ph,"#F6EBD1","none",0)
    d.arrow([(px,py-5),(px,py+ph),(px+pw+8,py+ph)],INK,1.8,head=7)
    knots_x=np.array([0,.10,.23,.38,.52,.68,.76,.88,1])
    knots_y=np.array([.07,.21,.29,.32,.34,.37,.43,.73,.93])
    ranks=np.linspace(0,1,75)
    scores=np.interp(ranks,knots_x,knots_y)
    curve=[(px+t*pw,py+ph*(1-score)) for t,score in zip(ranks,scores)]
    d.path("M"+" L".join(f"{x:.2f},{y:.2f}" for x,y in curve),NAVY,3)
    for t in knots_x:
        score=float(np.interp(t,knots_x,knots_y))
        d.circle(px+t*pw,py+ph*(1-score),3.3,NAVY,NAVY,1)
    for value,label in [(b0,r"B_0"),(beff,r"B_{\mathrm{eff}}")]:
        x=px+value*pw
        cutoff_score = float(np.interp(value, knots_x, knots_y))
        d.line(x, py+ph, x, py+ph*(1-cutoff_score), GRAY, 1.5, dash="5 4")
        d.formula(x,py+ph+9,label,25,anchor="middle")
    y=py+ph*.65
    d.arrow([(px+pw*b0+3,y),(px+pw*beff-3,y)],ORANGE,2.2,head=7)
    d.text(px-13,py+90,"proxy score",21,anchor="middle",rotate=-90)
    d.formula(1510,345,r"\widehat P_{(B_0)}-p_{\mathrm{ref}}\leq z\,s_{\mathrm{ref}}",25,anchor="middle")
    d.formula(1510,384,r"L\ \longrightarrow\ B_{\mathrm{eff}}\ \longrightarrow\ k",26,anchor="middle")
    d.arrow([(1778,274),(1789,274),(1789,418),(1314,418),(1314,446)],INK,1.8,head=7)
    for i in range(3):
        y=451+i*23
        d.polygon([(1243,y),(1347,y),(1357,y-8),(1253,y-8)],"#CBD6DD",INK,1.5)
    d.text(1298,539,"cold FP32",25,anchor="middle")
    d.arrow([(1360,477),(1420,477)],INK,2.1)
    d.rect(1424,444,163,69,"#E5DCEE",PURPLE,2,4)
    d.text(1505,474,"FP64 SIMD",27,weight="bold",anchor="middle")
    d.text(1505,500,"vcvtps2pd",22,anchor="middle")
    d.arrow([(1591,477),(1641,477)],INK,2.1)
    for i in range(10):
        d.circle(1655+(i%5)*24,465+(i//5)*22,4.8,INK,INK,1)
    d.formula(1703,513,"k=10",28,anchor="middle")
    d.save("pelorus_execution")


if __name__ == "__main__":
    geometry_figure()
    execution_figure()
