# "C:\Program Files\OCAD\OCAD 2018 Viewer\Samples\Forest Orienteering Map Bürenflue.ocd"
# "C:\Users\Alexander\Documents\Orientering\Kartor\Asaklitt.ocd"
# "C:\Users\Alexander\Documents\Orientering\Kartor\TMOK Granby-Tornberget 2018.ocd"
# "C:\Users\Alexander\Downloads\data.xlsx"
# "C:\Program Files\OCAD\OCAD 2018 Viewer\Samples\Sprint Orienteering Map Solothurn.ocd"
# Define the binary format based on the Delphi structure
# Försök att läsa in färger från filen!!!!!!!!!!!!!!!!!!!!!!
import preproces as pp
import pandas as pd
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import mapbox_earcut as earcut
import math
from shapely.geometry import Polygon, Point, LineString, LinearRing, MultiLineString, GeometryCollection
from scipy.interpolate import PchipInterpolator


width, height = 800, 600

# Example usage
file = r"C:\Program Files\OCAD\OCAD 2018 Viewer\Samples\Forest Orienteering Map Bürenflue.ocd"
color_file = r"C:\Users\Alexander\Downloads\data - Blad1.csv"
header = pp.read_header(file)
# for key in header:
#     print(key + ": ", header[key])

entries = pp.read_all_string_index_blocks(file, header['Internal3'] if header['Version'] == 2018 else header['FirstStringIndexBlk'])
string_objects = pp.read_all_strings(file, entries)

# Example usage:
object_list = pp.read_all_object_indices(file, header["ObjectIndexBlock"])
# ocad_objects = sorted(pp.read_all_ocad_objects(file, object_list, header['Version']), reverse=False, key=lambda x: x['Otp'])
ocad_objects = pp.read_all_ocad_objects(file, object_list, header['Version'])

# # Example usage:
symbol_indices = pp.read_symbol_index_blocks(file, header["FirstSymbolIndexBlk"])
symbol = pp.read_symbols(file, symbol_indices) if header['Version'] == 2018 else pp.read_symbols_9(file, symbol_indices)
# for key in symbol:
#     if symbol[key]['Otp'] == 1:
#         print(symbol[key])
#         # if symbol[key]['MainLength']:
#         # print([symbol[key][key2] for key2 in ['Description', 'DistFromStart', 'DistToEnd', 'MainLength', 'EndLength', 'MainGap', 'SecGap', 'EndGap']])
#         # if symbol[key]['LineStyle'] == 2:
#         #     print(symbol[key]['Description'])
# exit()


def read_colors(filename):
    df = pd.read_csv(filename)
    d = df.set_index('ID').to_dict(orient='index')
    return d


def read_colors2(strings):
    c = {}
    c_index = 0
    for s in strings:
        if s[0] == 9:
            c[int(s[2]['n'])] = {'Description': s[1],
                                 'C': float(s[2]['c']) if 'c' in s[2] else 0,
                                 'M': float(s[2]['m']) if 'm' in s[2] else 0,
                                 'Y': float(s[2]['y']) if 'y' in s[2] else 0,
                                 'K': float(s[2]['k']) if 'k' in s[2] else 0,
                                 'O': float(s[2]['o']) if 'o' in s[2] else 0,
                                 'T': float(s[2]['t']) if 't' in s[2] else 0,
                                 'S': s[2]['s'] if 's' in s[2] else 0,
                                 'P': float(s[2]['p']) if 'p' in s[2] else 0,
                                 'Z': 1 - c_index / 100 if header['Version'] >= 12 else 0}
            c_index += 1
            # c[int(s[2]['n'])]['Z'] = 1 if c[float(s[2]['n'])]['O'] else 0
    return c


colors = read_colors2(string_objects)
# for key in colors:
#     print(colors[key])
# exit()
# colors = read_colors(color_file)
# for key in colors:
#     print(colors[key])
# exit()


def read_map_scale(strings):
    for s in strings:
        if s[0] == 1039:
            return float(s[2]['m'])


def get_view(strings):
    for s in strings:
        if s[0] == 1030:
            return float(s[2]['x']), float(s[2]['y']), float(s[2]['z'])
    return 0, 0, 1


def cmyk_to_rgb(cyan, m, y, k, cmyk_scale=100, rgb_scale=255):
    r = (rgb_scale * (1.0 - cyan / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale)))
    g = (rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale)))
    b = (rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale)))
    return r, g, b


pan_x, pan_y, zoom = get_view(string_objects)
dragging = False
last_mouse_x = 0
last_mouse_y = 0
base_bounds = [0, 1, 0, 1]  # will be updated in setup_projection
map_scale = read_map_scale(string_objects)
SCENE_LIST_ID = None
aspect_ratio = 1.0
bezier_curve_resolution = 10


# --- NEW: Function to compile all static geometry into a Display List ---
def compile_scene_list():

    global SCENE_LIST_ID
    SCENE_LIST_ID = glGenLists(1)  # Generate one list ID

    # Start recording commands into the list
    glNewList(SCENE_LIST_ID, GL_COMPILE)

    # This is your existing drawing loop, moved from draw2()
    for ocad in ocad_objects:
        if ocad['Sym'] not in symbol:
            continue
        if ocad['Otp'] == 1:
            draw_point(ocad)
        if ocad['Otp'] == 2:
            draw_line(ocad)
        if ocad['Otp'] == 3:
            draw_area(ocad)
        # if ocad['Otp'] == 7:
        #     draw_rectangle(ocad)

    glEndList()

    print("✅ Scene compiled into a Display List.")


def setup_projection(box):
    global base_bounds
    # Assuming 'box' is a numpy array from your original code
    min_x, max_x = np.min(box[:, 0]), np.max(box[:, 0])  # Simplified from your code
    min_y, max_y = np.min(box[:, 1]), np.max(box[:, 1])  # Simplified from your code
    base_bounds = [min_x, max_x, min_y, max_y]
    apply_view_transform()


def apply_view_transform():
    global base_bounds, zoom, pan_x, pan_y, aspect_ratio
    left, right, bottom, top = base_bounds

    # Calculate initial view dimensions based on zoom
    width_view = (right - left) / zoom
    height_view = (top - bottom) / zoom

    # Adjust view dimensions to match the window's aspect ratio
    view_aspect = width_view / height_view
    if view_aspect > aspect_ratio:
        # View is wider than window: increase height
        height_view = width_view / aspect_ratio
    else:
        # View is taller than window: increase width
        width_view = height_view * aspect_ratio

    # Calculate the panned center
    cx = (left + right) / 2.0 + pan_x
    cy = (bottom + top) / 2.0 + pan_y

    # Set the projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(cx - width_view / 2, cx + width_view / 2,
               cy - height_view / 2, cy + height_view / 2)
    glMatrixMode(GL_MODELVIEW)


# --- NEW: Helper to convert screen coordinates to world coordinates ---
def screen_to_world(x, y):
    """Converts mouse screen coordinates to OpenGL world coordinates."""
    win_w = glutGet(GLUT_WINDOW_WIDTH)
    win_h = glutGet(GLUT_WINDOW_HEIGHT)

    # Get current view parameters
    left, right, bottom, top = base_bounds
    view_width = (right - left) / zoom
    view_height = (top - bottom) / zoom

    cx = (left + right) / 2.0 + pan_x
    cy = (bottom + top) / 2.0 + pan_y

    view_left = cx - view_width / 2.0
    view_bottom = cy - view_height / 2.0

    # Convert screen coordinates (top-left origin) to world coordinates
    world_x = view_left + (x / win_w) * view_width
    world_y = view_bottom + ((win_h - y) / win_h) * view_height

    return world_x, world_y


# --- FIXED: Panning and Zoom-to-Mouse ---
def mouse(button, state, x, y):
    """Handles mouse clicks and scroll wheel for panning and zooming."""
    global dragging, last_mouse_x, last_mouse_y, zoom, pan_x, pan_y

    # Left mouse button for panning
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            dragging = True
            last_mouse_x, last_mouse_y = x, y
        else:
            dragging = False

    # Scroll wheel for zooming (buttons 3 and 4)
    elif button == 3 or button == 4:
        if state == GLUT_UP:
            return # Ignore scroll release events

        # 1. Get world coordinates at mouse position BEFORE zoom
        world_x_before, world_y_before = screen_to_world(x, y)

        # 2. Determine zoom factor and update zoom
        zoom_factor = 1.2 if button == 3 else 1.0 / 1.2
        zoom *= zoom_factor

        # 3. Calculate the required pan adjustment to keep the point under the mouse stationary
        cx = (base_bounds[0] + base_bounds[1]) / 2.0 + pan_x
        cy = (base_bounds[2] + base_bounds[3]) / 2.0 + pan_y

        pan_x += (world_x_before - cx) * (1 - 1 / zoom_factor)
        pan_y += (world_y_before - cy) * (1 - 1 / zoom_factor)

        glutPostRedisplay()


def motion(x, y):
    """Handles mouse dragging for panning."""
    global last_mouse_x, last_mouse_y, pan_x, pan_y, aspect_ratio

    if dragging:
        # Calculate mouse movement delta in pixels
        dx = x - last_mouse_x
        dy = y - last_mouse_y
        last_mouse_x, last_mouse_y = x, y

        # Get window and view dimensions for accurate scaling
        win_w = glutGet(GLUT_WINDOW_WIDTH)
        win_h = glutGet(GLUT_WINDOW_HEIGHT)

        left, right, bottom, top = base_bounds
        view_width = (right - left) / zoom
        view_height = (top - bottom) / zoom

        # Convert screen pixel delta to world coordinate delta
        world_dx = dx * (view_width / win_w)
        world_dy = dy * (view_height / win_h)

        # Update pan. Note the signs for intuitive "drag-the-world" behavior.
        pan_x -= world_dx * aspect_ratio
        pan_y += world_dy  # Y is inverted between screen and OpenGL coords

        glutPostRedisplay()


def rotate(symbol_coords, center, angle):
    if symbol_coords.ndim == 1:
        symbol_coords = symbol_coords.reshape(-1, 2)
    # Translate to origin
    if angle == 0:
        return symbol_coords + center
    angle_rad = angle * np.pi / 1800

    # Rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    # Apply rotation
    rotated = symbol_coords @ rotation_matrix.T

    # Translate back
    return rotated + center


def vector_angle(vector):
    return np.arctan2(vector[1], vector[0]) * 1800 / np.pi


def find_corners(points, threshold_deg=45):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return np.array([0, 1])

    # Vectors between consecutive points
    v1 = points[1:-1] - points[:-2]   # segment before each middle point
    v2 = points[2:] - points[1:-1]    # segment after each middle point

    # Normalize
    v1 /= np.linalg.norm(v1, axis=1)[:, None]
    v2 /= np.linalg.norm(v2, axis=1)[:, None]

    # Compute angle (dot product -> arccos)
    dot = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))

    # Find where angle exceeds threshold
    mask = angles > threshold_deg
    flagged_indices = [0]  # always first point
    flagged_indices.extend((np.where(mask)[0] + 1).tolist())  # shift because v1/v2 skip first
    flagged_indices.append(len(points) - 1)  # always last point

    return np.array(flagged_indices)


def get_point_on_segment(p1, p2, distance_from_p1):
    """Finds a point at a specific distance along a line from p1 to p2."""
    segment_vec = p2 - p1
    segment_len = np.linalg.norm(segment_vec)
    if segment_len == 0:
        return p1
    # Interpolate the coordinates
    return p1 + (segment_vec / segment_len) * distance_from_p1


def cubic_bezier_curve(p0, p1, p2, p3, num_points=5):
    # Estimate the length of the curve by summing the lengths of the segments
    # of the control polygon. This provides a scale-independent measure.
    estimated_length = (np.linalg.norm(p1 - p0) +
                        np.linalg.norm(p2 - p1) +
                        np.linalg.norm(p3 - p2))

    # Calculate the number of points needed based on the density.
    # We ensure at least two points (start and end) are generated.

    # Create the parameter `t` array, which goes from 0 to 1.
    t = np.linspace(0, 1, num_points)

    # For vectorized operations, reshape t to be a column vector.
    # This allows broadcasting with the coordinate points.
    t = t[:, np.newaxis]

    # The cubic Bezier curve formula: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    # This is calculated for all t values at once using NumPy's vectorization.
    points = ((1 - t)**3 * p0 +
              3 * (1 - t)**2 * t * p1 +
              3 * (1 - t) * t**2 * p2 +
              t**3 * p3)

    return points


def expand_bezier(coords, ctrl1_pts, ctrl2_pts):
    global bezier_curve_resolution
    # return coords
    if not ctrl1_pts.any() or not ctrl2_pts.any():
        return coords
    # Identify start and end indices for each cubic
    start_idx = ctrl1_pts - 1
    end_idx = ctrl2_pts + 1

    # Control points for all curves (shape: N_curves x 2)
    P0 = coords[start_idx]
    P1 = coords[ctrl1_pts]
    P2 = coords[ctrl2_pts]
    P3 = coords[end_idx]

    # Precompute blending coefficients for cubic Bezier
    t = np.linspace(0, 1, bezier_curve_resolution, dtype=np.float32)[:, None]  # (num_points, 1)
    omt = 1 - t
    b0 = omt**3
    b1 = 3 * omt**2 * t
    b2 = 3 * omt * t**2
    b3 = t**3

    # Compute all curves in one go -> (N_curves, num_points, 2)
    curves = (
            b0[None] * P0[:, None] +
            b1[None] * P1[:, None] +
            b2[None] * P2[:, None] +
            b3[None] * P3[:, None]
    )

    # Flatten curves into sequence, avoiding duplicate points
    curves = curves.reshape(-1, 2)

    # Now insert straight segments and curves in correct order
    result = []
    last_curve_end = -1
    for si, ei in zip(start_idx, end_idx):
        # Add straight points before this curve
        if si - 1 > last_curve_end:
            result.extend(coords[last_curve_end+1:si])
        # Add curve
        result.extend(curves[:bezier_curve_resolution])
        curves = curves[bezier_curve_resolution:]  # move forward in curves array
        last_curve_end = ei
    # Add remaining straight points
    if last_curve_end + 1 < len(coords):
        result.extend(coords[last_curve_end+1:])

    return np.array(result, dtype=np.float32)


def line_helper(coords, lw, c, z):
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)

    # Segment start and end points
    p0 = coords[:-1]
    p1 = coords[1:]

    # Direction vectors for each segment
    seg_vec = p1 - p0
    # seg_vec = np.vstack([seg_vec, seg_vec[-1]])
    seg_len = np.linalg.norm(seg_vec, axis=1)

    # Filter out zero-length segments
    mask = seg_len > 0
    p0 = p0[mask]
    p1 = p1[mask]
    seg_vec = seg_vec[mask]
    seg_len = seg_len[mask]

    # Unit direction vectors
    dir_vec = seg_vec / seg_len[:, None]

    # Perpendicular normals (scaled by linewidth)
    scale = lw / 2
    normals = np.column_stack((dir_vec[:, 1], -dir_vec[:, 0])) * scale

    # Build the vertex list (triangle strip for each segment)
    # Each segment contributes 4 vertices: p0-left, p0-right, p1-left, p1-right
    v_left0  = np.column_stack((p0 - normals, np.full(len(p0), z, dtype=np.float32)))
    v_right0 = np.column_stack((p0 + normals, np.full(len(p0), z, dtype=np.float32)))
    v_left1  = np.column_stack((p1 - normals, np.full(len(p1), z, dtype=np.float32)))
    v_right1 = np.column_stack((p1 + normals, np.full(len(p1), z, dtype=np.float32)))

    # Interleave them exactly as GL_TRIANGLE_STRIP expects
    verts = np.empty((len(p0) * 4, 3), dtype=np.float32)
    verts[0::4] = v_left0
    verts[1::4] = v_right0
    verts[2::4] = v_left1
    verts[3::4] = v_right1

    # Draw
    glColor3f(*c)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)


def draw_simple_line(line_object):
    coords = line_object['Coordinates']
    s = symbol[line_object['Sym']]

    if s['LineWidth'] == 0:
        return

    control1 = np.where(line_object['x_flags'] & 1)[0]
    control2 = np.where(line_object['x_flags'] & 2)[0]
    coords = expand_bezier(coords, control1, control2)

    color_data = get_color(s['LineColor'])
    color = color_data['Color']
    z_depth = color_data['Z']

    line_helper(coords, s['LineWidth'], color, z_depth)

    if s['LineStyle'] & 1:
        draw_dot({'Coordinates': line_object['Coordinates'][0].reshape(-1, 2), 'Radius': s['LineWidth'] / 2, 'Color': s['LineColor']})
        draw_dot({'Coordinates': line_object['Coordinates'][-1].reshape(-1, 2), 'Radius': s['LineWidth'] / 2, 'Color': s['LineColor']})


def draw_element(element, coordinates, angle=0):
    if element['stType'] == 1:
        center = coordinates
        if center.ndim == 1:
            center = center.reshape(-1, 2)
        coords = element['stPoly']
        rotated_coords = rotate(coords, center, angle)

        color_data = get_color(element['stColor'])
        line_helper(rotated_coords, element['stLineWidth'], color_data['Color'], color_data['Z'])
    if element['stType'] == 2:
        center = coordinates
        coords = element['stPoly']
        rotated_coords = rotate(coords, center, angle)
        flags = np.append(np.where(element['y_flags'] & 2)[0], len(rotated_coords))

        color_data = get_color(element['stColor'])
        area_helper(rotated_coords, flags, color_data['Color'], color_data['Z'])
    if element['stType'] == 3:
        coords = coordinates
        radius = element['stDiameter'] / 2
        color_data = get_color(element['stColor'])
        draw_circle_as_line(coords, radius, element['stLineWidth'], color_data['Color'], color_data['Z'])
    if element['stType'] == 4:
        coords = coordinates
        radius = element['stDiameter'] / 2

        draw_dot({'Coordinates': coords,
                  'Radius': radius,
                  'Color': element['stColor']})


def draw_point(point_object):
    s = symbol[point_object['Sym']]
    control1 = np.where(point_object['x_flags'] & 1)[0]
    control2 = np.where(point_object['x_flags'] & 2)[0]
    coords = expand_bezier(point_object['Coordinates'], control1, control2)
    for e in s['Elements']:
        draw_element(e, coords, point_object['Ang'])


def draw_circle_as_line(coords, radius, lw, c, z):
    res = 10
    # Angles in radians
    angles = np.deg2rad(np.arange(0, 371, res, dtype=np.float32))

    # Center point
    center = np.array([[coords[0, 0], coords[0, 1]]], dtype=np.float32)

    # Circle perimeter points
    verts = np.column_stack((
        coords[0, 0] + np.cos(angles) * radius,
        coords[0, 1] + np.sin(angles) * radius
    ))

    line_helper(verts, lw, c, z)


def draw_dot(dot_object):
    coords = dot_object['Coordinates']
    color_data = get_color(dot_object['Color'])
    color = color_data['Color']
    z_depth = np.float32(color_data['Z'])
    radius = np.float32(dot_object['Radius'])

    # Angles in radians
    angles = np.deg2rad(np.arange(0, 371, 10, dtype=np.float32))

    # Center point (triangle fan start)
    center = np.array([[coords[0, 0], coords[0, 1], z_depth]], dtype=np.float32)

    # Circle perimeter points
    circle_pts = np.column_stack((
        coords[0, 0] + np.cos(angles) * radius,
        coords[0, 1] + np.sin(angles) * radius,
        np.full_like(angles, z_depth)
    ))

    # Combine center and perimeter into one vertex array
    verts = np.vstack((center, circle_pts))

    # Draw
    glColor3f(*color)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glDrawArrays(GL_TRIANGLE_FAN, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)


def draw_double_line(line_object):
    s = symbol[line_object['Sym']]
    points = line_object['Coordinates']

    control1 = np.where(line_object['x_flags'] & 1)[0]
    control2 = np.where(line_object['x_flags'] & 2)[0]
    points = expand_bezier(points, control1, control2)

    lw_fill, lw_left, lw_right = s['DblWidth'], s['DblLeftWidth'], s['DblRightWidth']

    direction = np.diff(points, axis=0)
    direction = np.vstack([direction, direction[-1]])
    dir_len = np.linalg.norm(direction, axis=1, keepdims=True)
    normal = np.column_stack((direction[:, 1], -direction[:, 0])) / dir_len

    if lw_fill and (s['DblFlags'] & 1):
        color_data_fill = get_color(s['DblFillColor'])
        line_helper(points, lw_fill, color_data_fill['Color'], color_data_fill['Z'])
    if lw_left:
        color_data_left = get_color(s['DblLeftColor'])
        line_helper(points + normal * (lw_fill + lw_left) / 2, lw_left, color_data_left['Color'], color_data_left['Z'])
    if lw_right:
        color_data_right = get_color(s['DblRightColor'])
        line_helper(points - normal * (lw_fill + lw_right) / 2, lw_right, color_data_right['Color'], color_data_right['Z'])


def draw_dashed_or_solid_line(line_object):
    s = symbol[line_object['Sym']]
    points = line_object['Coordinates']

    control1 = np.where(line_object['x_flags'] & 1)[0]
    control2 = np.where(line_object['x_flags'] & 2)[0]
    points = expand_bezier(points, control1, control2)

    # 1. Handle the solid line case as before
    gap = s['MainGap'] or s['SecGap']
    if gap == 0:
        draw_simple_line(line_object)
        return

    # 2. Setup for the dashed line calculation
    color_data = get_color(s['LineColor'])
    color = color_data['Color']
    z_depth = color_data['Z']
    line_width = s['LineWidth']

    a = s.get('MainLength', 0)
    b = s.get('EndLength', 0)
    C = s.get('MainGap', 0)
    D = s.get('SecGap', 0)
    E = s.get('EndGap', 0)

    dash_len = (a - D) / 2 if D else a / 2
    pattern = [
        {'length': dash_len, 'is_draw': True},
        {'length': C and D, 'is_draw': False},
        {'length': dash_len, 'is_draw': True},
        {'length': C or D, 'is_draw': False}
    ]

    pattern_index = 0
    dist_into_pattern = 0

    # This will hold the completed dash polylines, e.g., [[p1, p2, p3], [p5, p6]]
    final_dash_polylines = []
    # This holds the points of the dash we are currently building
    current_dash_polyline = []

    # Fast-forward to our starting position in the pattern
    while dist_into_pattern >= pattern[pattern_index]['length']:
        dist_into_pattern -= pattern[pattern_index]['length']
        pattern_index = (pattern_index + 1) % len(pattern)

    # 3. Walk along the polyline to build complete dash geometries
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]

        segment_len = np.linalg.norm(p2 - p1)
        if segment_len == 0:
            continue

        dist_along_segment = 0.0

        while dist_along_segment < segment_len:
            current_part = pattern[pattern_index]
            remaining_in_pattern = current_part['length'] - dist_into_pattern
            remaining_in_segment = segment_len - dist_along_segment
            chunk_len = min(remaining_in_pattern, remaining_in_segment)

            # --- This is the key logic change ---
            if current_part['is_draw']:
                start_pt = get_point_on_segment(p1, p2, dist_along_segment)
                end_pt = get_point_on_segment(p1, p2, dist_along_segment + chunk_len)

                if not current_dash_polyline:
                    # If this is the start of a new dash, add both points
                    current_dash_polyline.extend([start_pt, end_pt])
                else:
                    # If we are continuing a dash, just add the new endpoint
                    current_dash_polyline.append(end_pt)
            else:
                # If we are in a gap, it means any dash we were building is now finished.
                if current_dash_polyline:
                    final_dash_polylines.append(np.array(current_dash_polyline))
                    current_dash_polyline = []

            dist_along_segment += chunk_len
            dist_into_pattern += chunk_len

            if dist_into_pattern >= current_part['length']:
                dist_into_pattern = 0
                pattern_index = (pattern_index + 1) % len(pattern)
                # If we just finished a dash and are moving to a gap, finalize the polyline
                if not pattern[pattern_index]['is_draw'] and current_dash_polyline:
                    final_dash_polylines.append(np.array(current_dash_polyline))
                    current_dash_polyline = []

    # After the loop, check if a dash was still being built
    if current_dash_polyline:
        final_dash_polylines.append(np.array(current_dash_polyline))

    # 4. Draw each completed dash polyline using your helper
    for dash_polyline in final_dash_polylines:
        if len(dash_polyline) > 1:
            line_helper(dash_polyline, line_width, color, z_depth)


def draw_repeating_symbols(points, symbol_data, num_per_group, intra_group_dist, group_dist, end_len):
    # If parameters are invalid, do nothing. group_dist must be positive.
    if num_per_group <= 0 or group_dist <= 0:
        return

    # draw_dot({'Coordinates': points[0], 'Color': 69, 'Radius': 20})
    # draw_dot({'Coordinates': points[-1], 'Color': 69, 'Radius': 20})
    # Calculate the fixed distance/gap between the last symbol of one group
    # and the first symbol of the next group.
    # If a group has only 1 symbol, its length is 0.
    group_len = (num_per_group - 1) * intra_group_dist if num_per_group > 1 else 0
    gap_after_group = group_dist - group_len

    # If the gap is negative, the pattern is impossible, so don't draw.
    if gap_after_group < 0:
        return

    # --- "Walking" algorithm state ---
    segment_len = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    if segment_len < end_len:
        return
    n_sym = 1 + int((segment_len - end_len - group_len) / group_dist)
    offset = (segment_len - (n_sym - 1) * group_dist - group_len) / 2
    distance_walked = 0.0
    # Start by placing the first symbol at the beginning of the line.
    next_symbol_at = offset
    symbols_placed_in_group = 0

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        segment_len = np.linalg.norm(p2 - p1)
        if segment_len == 0:
            continue

        unit_vec = (p2 - p1) / segment_len

        # This loop places all symbols that fall onto the current segment
        while next_symbol_at <= distance_walked + segment_len:
            dist_into_segment = next_symbol_at - distance_walked

            sym_coords = p1 + unit_vec * dist_into_segment
            sym_coords = sym_coords.reshape(-1, 2)
            angle = vector_angle(unit_vec)
            draw_element(symbol_data, sym_coords, angle)

            symbols_placed_in_group += 1

            # --- This is the key logic change ---
            # Determine the location of the NEXT symbol
            if symbols_placed_in_group < num_per_group:
                # We are still inside a group. Use the intra-group distance.
                next_symbol_at += intra_group_dist
            else:
                # We just finished a group. Reset counter and use the inter-group gap.
                symbols_placed_in_group = 0
                next_symbol_at += gap_after_group

        distance_walked += segment_len


def draw_decorating_symbols(line_object):
    global bezier_curve_resolution
    coords = line_object['Coordinates']
    s = symbol[line_object['Sym']]

    control1 = np.where(line_object['x_flags'] & 1)[0]
    control2 = np.where(line_object['x_flags'] & 2)[0]
    coords = expand_bezier(coords, control1, control2)

    if s.get('UseSymbolFlags', 0) > 0 and 'Symbols' in s:
        # Draw Corner Symbols on all intermediate vertices
        if (s['UseSymbolFlags'] & 4) and 'Corner' in s['Symbols']:
            for i in range(0, len(coords)):
                # Orientation is based on the angle bisector for a smooth look
                dir_in = coords[i] - coords[i-1] if i > 0 else np.array([0, 0])
                dir_out = coords[i+1] - coords[i] if i < len(coords) - 1 else np.array([0, 0])
                angle_vec = (dir_in / (np.linalg.norm(dir_in) + 0.01)) + (dir_out / (np.linalg.norm(dir_out) + 0.01))
                draw_element(s['Symbols']['Corner'], coords[i], vector_angle(angle_vec))

        # Draw Start Symbol
        if (s['UseSymbolFlags'] & 2) and 'Start' in s['Symbols']:
            unit_vec = coords[1] - coords[0]
            draw_element(s['Symbols']['Start'], coords[0], vector_angle(unit_vec))

        # Draw End Symbol
        if (s['UseSymbolFlags'] & 1) and 'End' in s['Symbols']:
            unit_vec = coords[-1] - coords[-2]
            draw_element(s['Symbols']['End'], coords[-1], vector_angle(unit_vec))

    # 2. Draw Repeating Primary Symbols with the corrected logic
    num_symbols_per_group = s.get('nPrimSym', 0)

    if num_symbols_per_group > 0 and 'Symbols' in s and 'Main' in s['Symbols']:
        # Get all the necessary parameters for the new helper function
        intra_group_dist = s.get('PrimSymDist', 0)
        group_dist = s.get('MainLength', 0) # This is the key parameter we missed

        # In draw_decorating_symbols:

        # 1. Identify the indices of all break-points (start, corners, end)
        flags = np.where(line_object['y_flags'] & 1)[0]
        if flags.size == 0:
            corners = np.array([0, len(coords) - 1])
        else:
            parts = []
            if control1.any() and control2.any():
                # print(flags)
                flags = shift_flags2(flags, control1 - 1, control2 + 1, bezier_curve_resolution)
                # print(flags)
            if flags[0] != 0:
                parts.append(np.array([0]))
            parts.append(flags)
            if flags[-1] != len(coords) - 1:
                parts.append(np.array([len(coords) - 1]))
            corners = np.concatenate(parts)

        # 2. Loop through each segment defined by the break-points
        for i in range(len(corners) - 1):
            start_index = corners[i]
            end_index = corners[i+1]
            segment_polyline = coords[start_index:end_index + 1]
            draw_repeating_symbols(segment_polyline, s['Symbols']['Main'], num_symbols_per_group, intra_group_dist, group_dist, s['EndLength'])

        # for i in range(len(coords)):
        #     draw_dot({'Coordinates': coords[i].reshape(-1, 2), 'Radius': 40 if i in corners else 20, 'Color': 36})
            # if i in corners and (i in control1 or i in control2):
            #     print(1)


def draw_line(line_object):
    # if len(line_object['Coordinates']) < 2:
    #     return
    s = symbol[line_object['Sym']]
    if s['LineWidth']:
        draw_dashed_or_solid_line(line_object)
    if s['DblMode']:
        draw_double_line(line_object)

    if len(s['Symbols']):
        draw_decorating_symbols(line_object)

    # dash_flags = np.where(line_object['y_flags'] & 8)[0]
    # if dash_flags.size > 0:
    #     for i in range(len(dash_flags)):
    #         draw_dot({'Coordinates': line_object['Coordinates'][i].reshape(-1, 2), 'Radius': 30, 'Color': 36})
    #         # draw_dot({'Coordinates': line_object['Coordinates'][i+1].reshape(-1, 2), 'Radius': 30, 'Color': 36})
    #         # end = dash_flags[i]
    #         # line_helper(coords[start:end+1], s['LineWidth'], color, z_depth)
    #         # start = end + 1


def extract_lines_from_geom(geom):
    """
    Normalize a shapely geometry into a list of LineString objects.
    Handles LineString, MultiLineString, GeometryCollection, LinearRing, and empty geometries.
    """
    if geom is None:
        return []
    # empty geometry
    try:
        if geom.is_empty:
            return []
    except AttributeError:
        # not a geometry object
        return []

    gtype = geom.geom_type

    if gtype == "LineString":
        return [geom]
    if gtype == "LinearRing":
        # convert ring to linestring
        return [LineString(geom.coords)]
    if gtype == "MultiLineString":
        # .geoms is a sequence of LineString parts
        return list(geom.geoms)
    if gtype == "GeometryCollection":
        lines = []
        for part in geom.geoms:
            ptype = part.geom_type
            if ptype == "LineString":
                lines.append(part)
            elif ptype == "LinearRing":
                lines.append(LineString(part.coords))
            elif ptype == "MultiLineString":
                lines.extend(list(part.geoms))
            # ignore Points, Polygons, etc.
        return lines
    # other geometry types (Point, Polygon, MultiPolygon) → empty list for hatch clipping
    return []


def draw_clipped_hatch_segments(poly, hatch_line, lw, c, z):
    """
    poly: shapely Polygon
    hatch_line: shapely LineString (long line spanning bbox)
    line_helper: function expecting a numpy array shape (n,2)
    """
    inter = poly.intersection(hatch_line)
    segments = extract_lines_from_geom(inter)
    for seg in segments:
        coords = np.array(seg.coords, dtype=float)  # ensure numpy array
        if coords.shape[0] >= 2:  # skip degenerate segments
            line_helper(coords, lw, c, z)


def draw_hatch(polygon_coords, hatch_angle, hatch_dist, lw, c, z):
    # Convert OCAD tenths-of-degree to radians
    angle_rad = math.radians(hatch_angle / 10.0)

    # Create Shapely polygon
    poly = Polygon(polygon_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)

    # Get bounding box
    minx, miny, maxx, maxy = poly.bounds
    if math.isnan(minx) or math.isnan(maxx) or math.isnan(miny) or math.isnan(maxy):
        return
    # Create hatch lines
    length = math.hypot(maxx - minx, maxy - miny) * 2
    dx = math.cos(angle_rad) * length
    dy = math.sin(angle_rad) * length

    # Generate lines spaced by hatch_dist
    num_lines = int(((maxx - minx) * math.sin(angle_rad) + (maxy - miny) * math.cos(angle_rad)) / hatch_dist) + 1
    for i in range(-num_lines, num_lines):
        offset = i * hatch_dist
        cx = minx + offset * math.sin(angle_rad)
        cy = miny - offset * math.cos(angle_rad)
        p1 = (cx - dx/2, cy - dy/2)
        p2 = (cx + dx/2, cy + dy/2)
        hatch_line = LineString([p1, p2])

        draw_clipped_hatch_segments(poly, hatch_line, lw, c, z)


def area_helper(coords, flags, c, z):
    # Ensure coords is 2D (N, 2)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)

    # Triangulate (returns flat index array)
    tri_idx = earcut.triangulate_int32(coords, flags)

    # Get the triangle vertices in order, add Z coordinate
    verts = coords[tri_idx]                      # shape (N, 2)
    verts = np.column_stack((verts, np.full(len(verts), z, dtype=np.float32)))  # shape (N, 3)

    # Ensure float32 for OpenGL
    verts = np.asarray(verts, dtype=np.float32)

    # Draw without Python loops
    glColor3f(*c)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glDrawArrays(GL_TRIANGLES, 0, len(verts))
    glDisableClientState(GL_VERTEX_ARRAY)


def area_basic(coords, flags, c, z):
    glColor3f(*c)
    glBegin(GL_POLYGON)

    for i in range(len(coords)):
        glVertex3f(coords[i, 0], coords[i, 1], z)

    glEnd()


def shift_flags(flags, starts, ends, n, total_len=None):
    # --- Pre-calculation (no changes here) ---
    seg_len = (ends - starts) + 1
    inc = n - seg_len
    cum = np.concatenate(([0], np.cumsum(inc)))

    # --- Base mapping (for points outside segments) ---
    idx_end = np.searchsorted(ends, flags, side="right")
    new_idx = flags + cum[idx_end]

    # --- Identify and remap flags INSIDE segments ---
    k = np.searchsorted(starts, flags, side="right") - 1

    # **FIX 1 & 3**: Robustly create the mask for flags within segments.
    # Initialize mask to all False.
    in_seg = np.zeros_like(flags, dtype=bool)
    # Consider only flags that could possibly be in a segment (k >= 0).
    possible_mask = (k >= 0)
    if np.any(possible_mask):
        # Check against the correct end bound (inclusive).
        # This prevents indexing `ends` with k=-1 and uses the correct logic.
        in_seg[possible_mask] = (flags[possible_mask] <= ends[k[possible_mask]])

    if in_seg.any():
        k_in = k[in_seg]
        flags_in = flags[in_seg]

        r = flags_in - starts[k_in]
        seg_len_k = seg_len[k_in]

        # Initialize new relative positions
        r_new = np.zeros_like(r, dtype=int)

        # **FIX 2**: Handle segments with length > 1 to avoid ZeroDivisionError.
        # Create a mask for segments that can be scaled.
        scalable_mask = (seg_len_k > 1)

        if np.any(scalable_mask):
            # Apply scaling formula only where it's safe and necessary.
            r_scalable = r[scalable_mask]
            seg_len_scalable = seg_len_k[scalable_mask]
            r_new[scalable_mask] = (r_scalable * (n - 1)) // (seg_len_scalable - 1)

        # For segments of length 1, r is always 0, so r_new correctly defaults to 0.

        # Calculate final index and update the master array
        new_seg_start = starts[k_in] + cum[k_in]
        new_idx[in_seg] = new_seg_start + r_new

    if total_len:
        new_idx = np.append(new_idx, total_len)

    return new_idx


def shift_flags2(flags, starts, ends, n):
    # Calculate the net change in the number of points for each segment.
    # Original segment length is inclusive, e.g., indices 5 to 8 is 4 points.
    original_len = (ends - starts)
    increment = n - original_len

    # Calculate the cumulative offset introduced by all segments *before* a
    # given segment index. cum_offset[k] is the total shift from segments 0 to k-1.
    cum_offset = np.concatenate(([0], np.cumsum(increment)))

    # Find how many segments have fully completed before each flag's position.
    # `np.searchsorted` is highly efficient for this. `side="right"` correctly
    # handles flags that are exactly at an endpoint.
    completed_segments_count = np.searchsorted(ends, flags, side="right")

    # The new index is the original index plus the total offset from all
    # the completed segments that came before it.
    new_flags = flags + cum_offset[completed_segments_count]

    return new_flags


def draw_structure(area_object):
    coords = area_object['Coordinates']
    s = symbol[area_object['Sym']]

    polygon = Polygon(coords)
    minx, miny, maxx, maxy = polygon.bounds

    # Step 1: Create base grid
    x_range = np.arange(minx - s['StructWidth'], maxx + s['StructWidth'], s['StructWidth'])
    y_range = np.arange(miny - s['StructHeight'], maxy + s['StructHeight'], s['StructHeight'])
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    # Step 2: Apply shifted rows if StructMode=2
    if s['StructMode'] == 2:
        shift = (np.arange(len(y_range)) % 2) * (s['StructWidth'] / 2)
        grid_x += shift[:, None]

    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    if s['StructAngle']:
        centroid = np.array(polygon.centroid.coords[0])
        theta = np.deg2rad(s['StructAngle'] / 10)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
        grid_points = (grid_points - centroid) @ rot_matrix.T + centroid

    for pt in grid_points:
        p = Point(pt)
        if polygon.contains(p):
            draw_element(s['Elements'][0], pt.reshape(-1, 2))


def draw_area(area_object):
    global bezier_curve_resolution
    coords = area_object['Coordinates']
    s = symbol[area_object['Sym']]

    control1 = np.where(area_object['x_flags'] & 1)[0]
    control2 = np.where(area_object['x_flags'] & 2)[0]
    flags = np.where(area_object['y_flags'] & 2)[0]
    coords = expand_bezier(coords, control1, control2)

    starts = control1 - 1
    ends = control2 + 1

    if flags.any() and control1.any() and control2.any():
        flags_shifted = shift_flags(flags, starts, ends, bezier_curve_resolution, len(coords))
    else:
        flags_shifted = np.array([len(coords)])

    if s['FillOn']:
        color_data = get_color(s['FillColor'])
        area_helper(coords, flags_shifted, color_data['Color'], color_data['Z'])

    if s['BorderOn']:
        border_sym = symbol[s['BorderSym']]
        if border_sym['Otp'] == 2:
            lw = border_sym['LineWidth']
            color_data = get_color(border_sym['Colors'][0])
            line_helper(coords, lw, color_data['Color'], color_data['Z'])

    if s['HatchMode'] & 1 and s['HatchDist'] > 0:
        color_data = get_color(s['HatchColor'])
        draw_hatch(coords, s['HatchAngle1'], s['HatchDist'], s['HatchLineWidth'], color_data['Color'], color_data['Z'])

    if s['HatchMode'] & 2 and s['HatchDist'] > 0:
        color_data = get_color(s['HatchColor'])
        draw_hatch(coords, s['HatchAngle2'], s['HatchDist'], s['HatchLineWidth'], color_data['Color'], color_data['Z'])

    if s['StructMode']:
        draw_structure(area_object)

    # for i in range(len(coords)):
    #     if area_object['x_flags'][i] & 3 and s['PreferredDrawingTool'] == 1:
    #         draw_dot({'Coordinates': coords[i], 'Color': 2, 'Z': 0.99, 'Radius': 30})


def draw_rectangle(rectangle):
    coords = rectangle['Coordinates']
    s = symbol[rectangle['Sym']]
    color_data = get_color(s['Colors'][0])

    glColor3f(*color_data['Color'])
    glBegin(GL_POLYGON)

    for i in range(len(coords)):
        glVertex3f(coords[i, 0], coords[i, 1], color_data['Z'])

    glEnd()


def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    apply_view_transform()

    # Execute the pre-compiled list of all objects (this is extremely fast)
    glCallList(SCENE_LIST_ID)

    glutSwapBuffers()


def get_color(s):
    color = colors[s]
    c = cmyk_to_rgb(color['C'], color['M'], color['Y'], color['K'], 100, 1)
    return {'Color': c, 'Z': color['Z']}


def reshape(width, height):
    global aspect_ratio
    if height == 0:
        height = 1  # Avoid division by zero
    aspect_ratio = width / height
    glViewport(0, 0, width, height)
    apply_view_transform()  # Update the projection immediately
    glutPostRedisplay()    # Ensure the scene redraws with the new projection


def create_scene():
    global aspect_ratio

    window_w, window_h = 800, 600
    aspect_ratio = window_w / window_h
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(window_w, window_h)
    glutCreateWindow(b"OpenGL OCAD viewer in Python")
    glEnable(GL_DEPTH_TEST)
    # glDepthFunc(GL_LEQUAL)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    glClearColor(1, 1, 1, 1)  # Background color

    bounding_box = np.array([index['BoundingBox'] for index in object_list])
    setup_projection(bounding_box)

    # --- NEW: Calculate the conversion factor ---
    # This must be done AFTER the window is created and initial projection is set
    win_w = glutGet(GLUT_WINDOW_WIDTH)

    # --- NEW: Compile the scene geometry once at startup ---
    compile_scene_list()

    glutDisplayFunc(draw)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutReshapeFunc(reshape)
    glutMainLoop()


if __name__ == '__main__':
    create_scene()


