import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d.art3d import Line3DCollection, LineCollection
from shapely import (LineString,Polygon,MultiLineString,unary_union,line_merge)
from shapely.ops import linemerge
from scipy.spatial import KDTree, cKDTree
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import math
import psutil
import platform

def find_solid(heirarchy, poly):
    solid_region = []
    dont_fill = []
    for parent_key in heirarchy:
        if parent_key not in dont_fill:
            solid = poly[parent_key]
            if heirarchy[parent_key] == []:
                solid_region.append(solid)
            else:
                child_list = heirarchy[parent_key]
                child_union = unary_union([poly[child] for child in child_list])
                solid = solid.difference(child_union)
                dont_fill += child_list
                solid_region.append(solid)

    return solid_region


def find_nth_closest_point(end_point, start_points, n):
    tree = cKDTree(start_points)
    
    if len(start_points) < n:
        return None  # Not enough points available
    
    distance, indices = tree.query(end_point, k=n)  # Get n nearest points
    if n == 1:
        return start_points[indices], distance
    return start_points[indices[-1]], distance[-1]

    
def find_nearest_boundary_point(point, boundary_tree, all_boundary_vertices, vertex_to_index_map):
    distance, nearest_idx = boundary_tree.query(point)
    nearest_boundary_point = tuple(np.round((all_boundary_vertices[nearest_idx]), 2))

    nearest_poly_idx, _ = vertex_to_index_map.get(nearest_boundary_point)
    
    return nearest_boundary_point, nearest_poly_idx, distance

def calculate_path_distance(path):
    path_array = np.array(path)
    
    differences = np.diff(path_array, axis=0)
    
    distances = np.linalg.norm(differences, axis=1)
    
    return np.sum(distances)


def repoint_polygon(poly, n_points):
    perimeter = poly.exterior.length  
    spacing = perimeter / n_points    
    new_points = []
    for i in range(n_points):
        point = poly.exterior.interpolate(i * spacing)  # Get point at distance i * spacing
        new_points.append((point.x, point.y))
    new_points.append(new_points[0])  # Close the polygon
    return Polygon(new_points)

def is_line_in_non_solid_region(start, end, solid_region):
    line = LineString([start, end])
    
    # Handle single polygon or list of polygons
    if isinstance(solid_region, Polygon):
        solid_regions = [solid_region]
    else:
        solid_regions = solid_region
    
    # Check if line is fully contained within any solid region
    if any(solid.contains(line) for solid in solid_regions):
        return False
    
    # Check if line is in any hole
    for solid in solid_regions:
        for interior in solid.interiors:
            hole = Polygon(interior)
            if line.intersects(hole) or hole.contains(line):
                return True
    
    # Check boundary condition: return True only if line intersects boundary but does not lie entirely on it
    for solid in solid_regions:
        exterior = Polygon(solid.exterior)
        if not exterior.contains(line) and line.intersects(exterior.boundary) and not line.difference(exterior.boundary).is_empty:
            return True
    
    return False
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------



def is_valid_connection(end_point, next_start_point, normal, line_spacing, boundary_tree, all_boundary_vertices, vertex_to_index_map, max_normal_dist=1.0, dot_tolerance=0.5):
    """Check if two points can be connected without a z-lift."""
    connection_vec = np.array(next_start_point) - np.array(end_point)
    normal_dist = abs(np.dot(connection_vec, normal))
    
    if normal_dist > line_spacing + max_normal_dist:
        return False
    
    _, current_polygon_index, _ = find_nearest_boundary_point(end_point, boundary_tree, all_boundary_vertices, vertex_to_index_map)
    _, next_polygon_index, _ = find_nearest_boundary_point(next_start_point, boundary_tree, all_boundary_vertices, vertex_to_index_map)
    if current_polygon_index != next_polygon_index:
        return False
    # Optional: Dot product comparison along normal direction
    end_proj = np.dot(end_point, normal)
    next_proj = np.dot(next_start_point, normal)
    
    if np.isclose(end_proj, next_proj, atol=dot_tolerance):
        return False
    
    return True

def group_lines_continuously(lines, visited_lines, line_spacing,boundary_tree,all_boundary_vertices,vertex_to_index_map):
    """
    Group unvisited lines into continuous paths such that no z-lift is required within any group.
    
    Returns:
    - groups: list of lists of indices, each forming a continuous path
    """
    if len(visited_lines) == len(lines):
        return [], {}
    unvisited = [i for i in range(len(lines)) if i not in visited_lines]
    
    # Use first valid line to determine normal direction
    for i in unvisited:
        direction = np.array(lines[i][1]) - np.array(lines[i][0])
        if np.linalg.norm(direction) > 1e-10:
            break
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)

    groups = []
    used = set()
    
    while len(used) < len(unvisited):
        # Start a new group with the first unused line
        for i in unvisited:
            if i not in used:
                current_idx = i
                break
        
        group = [current_idx]
        used.add(current_idx)
        current_end = lines[current_idx][1]
        
        while True:
            found_next = False
            candidates = [j for j in unvisited if j not in used]
            min_dist = float('inf')
            best_idx = None
            
            for j in candidates:
                next_start = lines[j][0]
                if is_valid_connection(current_end, next_start, normal, line_spacing,boundary_tree,all_boundary_vertices,vertex_to_index_map):
                    dist = np.linalg.norm(np.array(current_end) - np.array(next_start))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = j
            
            if best_idx is not None:
                group.append(best_idx)
                used.add(best_idx)
                current_end = lines[best_idx][1]
                found_next = True
            else:
                break
        
        groups.append(group)
    
    representatives = {}
    for group_id, group in enumerate(groups):
        # dot_products = {i: np.dot(lines[i][0], normal) for i in group}
        # sorted_group = sorted(group, key=lambda x: dot_products[x])
        if len(group) == 1:
            representatives[group_id] = lines[group[0]][0]
        else:
            minn,maxx = group.index(min(group)), group.index(max(group))
            representatives[group_id] = [lines[group[minn]][0], lines[group[maxx]][0]]

    return groups, representatives

def are_connected(line1, line2, tolerance=3e-1):
    coords1 = np.array(line1.xy).T
    coords2 = np.array(line2.xy).T
    return (
        np.allclose(coords1[0], coords2[0], atol=tolerance) or
        np.allclose(coords1[0], coords2[-1], atol=tolerance) or
        np.allclose(coords1[-1], coords2[0], atol=tolerance) or
        np.allclose(coords1[-1], coords2[-1], atol=tolerance)
    )

def merge_linestrings(geoms, tolerance=3e-1, global_merge=False):
    final_set = []
    prev_coords = None
    prev_geom = None
    merged_exists = False
    merged_line = None
    if global_merge:

        unvisited = set(range(len(geoms)))
        final_set = []

        while unvisited:
            idx = unvisited.pop()
            current_group = [geoms[idx]]
            changes = True

            while changes:
                changes = False
                to_check = list(unvisited)
                for j in to_check:
                    for line_in_group in current_group:
                        if are_connected(line_in_group, geoms[j], tolerance):
                            current_group.append(geoms[j])
                            unvisited.remove(j)
                            changes = True
                            break

            merged = linemerge(MultiLineString(current_group))
            final_set.append(merged)

        return final_set
    
    else:
        for i, curr_geom in enumerate(geoms):
            if isinstance(curr_geom, LineString):
                curr_coords = np.array(curr_geom.xy).T
                if i == 0:
                    prev_coords = curr_coords
                    prev_geom = curr_geom
                else:
                    if np.allclose(prev_coords[0], curr_coords[0], atol=tolerance) or np.allclose(prev_coords[-1], curr_coords[0], atol=tolerance):
                        if merged_exists:
                            temp_line = MultiLineString([merged_line, curr_geom])
                            merged_line = line_merge(temp_line)
                        else:
                            temp_line = MultiLineString([prev_geom, curr_geom])
                            merged_line = line_merge(temp_line)
                            merged_exists = True
                    else:
                        if merged_exists:
                            final_set.append(merged_line)
                            merged_line = None
                            merged_exists = False
                        else:
                            final_set.append(prev_geom)
                    prev_coords = curr_coords
                    prev_geom = curr_geom
        
        if merged_exists: 
            final_set.append(merged_line)
        else:
            if prev_geom is not None:
                final_set.append(prev_geom)
        
        return final_set

#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------


def connect_points(zigzag_lines, poly_list, line_spacing, theta, solid_region):
    tool_path = []
    sub_path=[]
    polygon_map = {} 
    all_boundary_vertices = [] 
    vertex_to_index_map = {}
    visited_lines = set()
    connections=[]
    z_lift = False
    z_lift_travel = []
    groups = [] 

    direction_of_zigzag_lines = np.array(zigzag_lines[0][1]) - np.array(zigzag_lines[0][0])
    # Normal vector (perpendicular to direction)
    normal_to_zigzag_lines = np.array([-direction_of_zigzag_lines[1], direction_of_zigzag_lines[0]])
    # Normalize the normal vector
    normal_to_zigzag_lines = normal_to_zigzag_lines / np.linalg.norm(normal_to_zigzag_lines)

    start_point_to_index_map = {tuple(line[0]): idx for idx, line in enumerate(zigzag_lines)} #to get line index from start point
    end_point_to_index_map = {tuple(line[-1]): idx for idx, line in enumerate(zigzag_lines)} #to get line index from end point
    for i, poly in enumerate(poly_list): 
        boundary_vertices = np.array(np.round((poly.exterior.coords) , 2))
        all_boundary_vertices.append(boundary_vertices) 
        polygon_map[i] = boundary_vertices
        for idx, vtx in enumerate(boundary_vertices):
            v_tuple= tuple(np.round(vtx, 2))
            vertex_to_index_map[v_tuple] = (i, idx)

    all_boundary_vertices = np.vstack(all_boundary_vertices)  
    boundary_tree = cKDTree(all_boundary_vertices) #to find nearest boundary point

    end_point = zigzag_lines[0][-1] #starting with line 0
    visited_lines.add(0) #adding initial line to visited_lines set
    next_line_index = 1
    sub_path.extend(zigzag_lines[0]) #adding initial line to tool path
    while len(visited_lines) < len(zigzag_lines):
        unvisited_indices = [i for i in range(len(zigzag_lines)) if i not in visited_lines] 
        unvisited_start_points = zigzag_lines[unvisited_indices, 0] #list of all unvisited start points
        search_attempts=1
        valid_closest_start_point = True
        z_lift = False
        while True:
            end_line_index = end_point_to_index_map[tuple(end_point)]
            if(search_attempts > len(unvisited_start_points)): #no valid start point available
                z_lift = True
                valid_closest_start_point = False
                break
            closest_start_point, distance_between_joining_points = find_nth_closest_point(end_point, unvisited_start_points, search_attempts) 
            start_point_dot_product = np.dot(closest_start_point, normal_to_zigzag_lines)
            end_point_dot_product = np.dot(end_point, normal_to_zigzag_lines)
            start_line_index = start_point_to_index_map[tuple(closest_start_point)]
            end_line_index = end_point_to_index_map[tuple(end_point)]
            start_line_midpoint = (zigzag_lines[start_line_index][0] + zigzag_lines[start_line_index][-1]) / 2
            end_line_midpoint = (zigzag_lines[end_line_index][0] + zigzag_lines[end_line_index][-1]) / 2
            end_point_boundary, current_polygon_index, _ = find_nearest_boundary_point(end_point, boundary_tree, all_boundary_vertices, vertex_to_index_map)
            closest_start_point_boundary, next_polygon_index, _ = find_nearest_boundary_point(closest_start_point, boundary_tree, all_boundary_vertices, vertex_to_index_map)
            y2_y1 = closest_start_point[-1] - end_point[-1]
            x2_x1 = closest_start_point[0] - end_point[0]
            if x2_x1 == 0:
                x2_x1 = 1e-10
            slope = y2_y1/x2_x1
            m_deg = np.rad2deg(np.arctan(slope))
            relative_m = m_deg - theta 
            rel_rad = np.deg2rad(relative_m)
            normal_dist = distance_between_joining_points * np.sin(rel_rad)
            normal_dist = np.abs(normal_dist)
            if (normal_dist > line_spacing+ 5e-1): # no valid ADJACENT LINE start point available 
                z_lift = True
                valid_closest_start_point = False
                break
            elif(current_polygon_index != next_polygon_index):
                search_attempts+=1
                continue
            elif np.isclose(start_point_dot_product,end_point_dot_product, atol=1e-2) or is_line_in_non_solid_region(start_line_midpoint,end_line_midpoint,solid_region): #closest start point is valid
                if is_line_in_non_solid_region(closest_start_point,end_point,solid_region):
                    search_attempts+=1
                else:
                    break
            else: # valid closest start point found
                break

        if valid_closest_start_point:

            end_point_boundary, current_polygon_index, _ = find_nearest_boundary_point(end_point, boundary_tree, all_boundary_vertices, vertex_to_index_map)
            closest_start_point_boundary, next_polygon_index, _ = find_nearest_boundary_point(closest_start_point, boundary_tree, all_boundary_vertices, vertex_to_index_map)
            if(current_polygon_index == next_polygon_index):
                polygon_boundary = polygon_map[current_polygon_index]
                end_point_tuple = end_point_boundary  # Convert point to tuple for dictionary lookup
                closest_start_point_tuple = closest_start_point_boundary  # Convert point to tuple for dictionary lookup
                # Retrieve indices using precomputed map
                _, end_point_vertex_idx = vertex_to_index_map[end_point_tuple]
                _, closest_start_point_vertex_idx = vertex_to_index_map[closest_start_point_tuple]

                if end_point_vertex_idx < closest_start_point_vertex_idx:
                    pathA = polygon_boundary[end_point_vertex_idx:closest_start_point_vertex_idx + 1]
                    pathB = np.vstack((polygon_boundary[closest_start_point_vertex_idx:], polygon_boundary[:end_point_vertex_idx + 1]))
                else:
                    pathA = polygon_boundary[closest_start_point_vertex_idx:end_point_vertex_idx + 1]
                    pathB = np.vstack((polygon_boundary[end_point_vertex_idx:], polygon_boundary[:closest_start_point_vertex_idx + 1]))

                path = pathA if calculate_path_distance(pathA) < calculate_path_distance(pathB) else pathB

                # find distance between path[0] and closest_start_point
                start_to_path_distance = np.linalg.norm(np.array(path[0]) - np.array(closest_start_point))
                end_to_path_distance = np.linalg.norm(np.array(path[0]) - np.array(end_point))
                if start_to_path_distance < end_to_path_distance:
                    path = path[::-1]

                sub_path.extend(path)
                closest_start_point_tuple = tuple(closest_start_point)
                next_line_index = start_point_to_index_map[closest_start_point_tuple]
                sub_path.extend(zigzag_lines[next_line_index])
                connections.append(path)

            closest_start_point_tuple = tuple(closest_start_point)
            next_line_index = start_point_to_index_map[closest_start_point_tuple]
            visited_lines.add(next_line_index)
            visited_lines.add(end_line_index)
            end_point = zigzag_lines[next_line_index][-1]
        elif z_lift:
            tool_path.append(sub_path)
            sub_path = []
            groups,representatives = group_lines_continuously(zigzag_lines, visited_lines, line_spacing,boundary_tree,all_boundary_vertices,vertex_to_index_map)
            if not groups:
                return tool_path
            shortest_dist = float('inf')
            l=0
            for key,value in representatives.items():

                if isinstance(value, np.ndarray):
                    dist = math.dist(end_point, value)
                    l = 1
                else:
                    dist1 = math.dist(end_point, value[0])
                    dist2 = math.dist(end_point, value[1])
                    dist = min(dist1,dist2)
                    l=-1
                if dist < shortest_dist:
                    shortest_dist = dist
                    if l > 0:
                        new_start = value
                    else:
                        new_start = value[0] if dist1 < dist2 else value[1]
            z_lift_travel.append([end_point, list(new_start)])
            next_line_index = start_point_to_index_map[tuple(new_start)]
            end_point = zigzag_lines[next_line_index][-1]
            visited_lines.add(next_line_index)
            sub_path.extend(zigzag_lines[next_line_index])
            
    if sub_path:
        tool_path.append(sub_path)
    return tool_path

#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------


def zigzag():
    filename = "tst01.STL"
    stl_mesh = trimesh.load(filename, process=False)
    angle_radians = np.deg2rad(90)
    # rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [0, 0, 1])
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [1, 0, 0])
    stl_mesh.apply_transform(rotation_matrix)
    # slice_result = stl_mesh.section(plane_origin=[0, 0, 20], plane_normal=[0, 0, 1])

    ti=1
    slice_thickness = 2
    z_min, z_max = stl_mesh.bounds[:,2] # Get the min and max z values of the mesh
    slice_every = np.arange(z_min,z_max,step=slice_thickness) # Create an array of z values to slice at
    slices = stl_mesh.section_multiplane(plane_origin=[0,0,z_min] , plane_normal=[0,0,1], heights=slice_every)
    print(f"number of slices: {len(slices)}") # Slice the mesh
    for slice_result in slices:
        if slice_result is not None:
            s=time.time()
            poly_list = []
            if len(slice_result.entities) > 1:
                poly_area = []
                for i, contour in enumerate(slice_result.entities):
                    poly = Polygon(slice_result.vertices[contour.points][:, :2])
                    poly_list.append(poly)
                    poly_area.append(poly.area)
                sorted_indices = np.argsort(poly_area)[::-1]
                heirarchy_graph = {}

                for i in range(len(sorted_indices)):
                    heirarchy_graph[sorted_indices[i]] = []
                for i in range(1, len(sorted_indices)):
                    curr_idx = sorted_indices[i]

                    for j in range(i - 1, -1, -1):
                        larger = sorted_indices[j]
                        if poly_list[larger].contains(poly_list[curr_idx]):
                            heirarchy_graph[larger].append(curr_idx)
                            break
                solid = find_solid(heirarchy_graph, poly_list)
            elif len(slice_result.entities) == 1:
                for contour in (slice_result.entities):
                    poly = Polygon(slice_result.vertices[contour.points][:, :2])
                    poly_list.append(poly)
                solid = [Polygon(slice_result.vertices[slice_result.entities[0].points][:, :2])]

            zigzag_lines = []
            line_spacing = 4
            theta = 30

            alt_line = 0 
            
            for solid_region in solid:
                min_x, min_y,max_x,max_y = solid_region.bounds 
                minn = np.array([min_x,min_y])
                maxx = np.array([max_x,max_y])
                line_length = np.linalg.norm(minn-maxx) *2
                theta_rad = np.deg2rad(theta)
                direction = np.array([np.cos(theta_rad), np.sin(theta_rad)]) 
                normal = np.array([-direction[1], direction[0]])
                center = np.array([max_x,min_y])
                if theta > 90:
                    normal = np.array([direction[1], -direction[0]])
                    center = np.array([min_x,min_y])
                direction = direction / np.linalg.norm(direction)
                normal = normal / np.linalg.norm(normal) 
                while True:
                    distance = np.dot(center - np.array([min_x, min_y]), normal)
                    if distance > line_length:
                        break
                    start = center - ((line_length/2)* direction)
                    end = center + ((line_length/2)*direction)
                    center = center + (line_spacing * normal)
                    if alt_line%2==0:
                        line = LineString([list(start),list(end)])
                    else: 
                        line = LineString([list(end),list(start)])
                    intersection = solid_region.intersection(line)

                    # Check if the intersection is a valid line (not empty)
                    if not intersection.is_empty:
                        if isinstance(intersection, LineString):
                            coords = np.array(intersection.xy).T
                            if len(coords) >= 2:  # Ensure it has at least start and end points
                                zigzag_lines.append(np.array([coords[0], coords[-1]]))
                        elif isinstance(intersection, MultiLineString):
                            final_set = merge_linestrings(intersection.geoms)
                            finale = merge_linestrings(final_set, global_merge=True)
                            for lines in finale: 
                                start = lines.coords[0]
                                end = lines.coords[-1]
                                if alt_line==0:
                                    zigzag_lines.append(np.array([end, start]))
                                else:
                                    zigzag_lines.append(np.array([start,end]))
                           
                    alt_line += 1

            zigzag_lines = np.array(zigzag_lines)
            poly_list = [repoint_polygon(poly, n_points=200) for poly in poly_list]
            tool_path = connect_points(zigzag_lines, poly_list, line_spacing, theta,solid)
            e=time.time()
            print(f"slice number {ti} took {e-s} seconds")
            ti+=1
        
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------


def anime(tool_path):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Animated Tool Path')
    ax.grid(True)
    ax.set_aspect('equal')  # Equal scaling for x and y axes

    # Set axis limits based on tool_path data
    all_points = np.vstack([np.array(segment) for segment in tool_path])
    x_min, x_max = all_points[:, 0].min() - 5, all_points[:, 0].max() + 5
    y_min, y_max = all_points[:, 1].min() - 5, all_points[:, 1].max() + 5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Initialize line objects for each sub-path with unique colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(tool_path)))  # Distinct colors
    sub_path_lines = [ax.plot([], [], '-', color=colors[i], linewidth=1.5,
                              label=f'Sub-path {i+1}' if i < 10 else '')[0]
                      for i in range(len(tool_path))]
    
    # Initialize line object for z-lift transitions (red)
    z_lift_line = ax.plot([], [], '-', color='red', linewidth=1.5, label='Z-lift')[0]

    # Store plotted segments for each sub-path and z-lift transitions
    sub_path_segments = [[] for _ in tool_path]  # List of [x, y] pairs for each sub-path
    z_lift_segments = []  # List of [x, y] pairs for z-lift transitions

    def init():
        # Initialize all line objects with empty data
        for line in sub_path_lines:
            line.set_data([], [])
        z_lift_line.set_data([], [])
        return sub_path_lines + [z_lift_line]

    def update(frame):
        # Extract frame information
        segment_idx, point_idx, frame_type = frame

        if frame_type == 'segment':
            # Plot a segment within the current sub-path
            segment = tool_path[segment_idx]
            x = [segment[point_idx][0], segment[point_idx + 1][0]]
            y = [segment[point_idx][1], segment[point_idx + 1][1]]
            sub_path_segments[segment_idx].append(([x[0], x[1]], [y[0], y[1]]))

        elif frame_type == 'z_lift':
            # Plot a z-lift transition from the last point of segment_idx to the first point of segment_idx+1
            if segment_idx + 1 < len(tool_path):
                current_last = tool_path[segment_idx][-1]
                next_first = tool_path[segment_idx + 1][0]
                x = [current_last[0], next_first[0]]
                y = [current_last[1], next_first[1]]
                z_lift_segments.append(([x[0], x[1]], [y[0], y[1]]))

        # Update all line objects to maintain visibility
        for i, line in enumerate(sub_path_lines):
            if sub_path_segments[i]:
                x_data = []
                y_data = []
                # Plot each segment individually to avoid last-to-first connection
                for x_seg, y_seg in sub_path_segments[i]:
                    x_data.extend([x_seg[0], x_seg[1], np.nan])  # Add nan to break lines
                    y_data.extend([y_seg[0], y_seg[1], np.nan])
                line.set_data(x_data, y_data)
            else:
                line.set_data([], [])

        # Update z-lift line
        if z_lift_segments:
            x_data = []
            y_data = []
            for x_seg, y_seg in z_lift_segments:
                x_data.extend([x_seg[0], x_seg[1], np.nan])
                y_data.extend([y_seg[0], y_seg[1], np.nan])
            z_lift_line.set_data(x_data, y_data)
        else:
            z_lift_line.set_data([], [])

        return sub_path_lines + [z_lift_line]

    # Generate frames: (segment_idx, point_idx, frame_type)
    frames = []
    for i, segment in enumerate(tool_path):
        # Add frames for sub-path segments
        for j in range(len(segment) - 1):
            frames.append((i, j, 'segment'))
        # Add frame for z-lift transition (except after the last sub-path)
        if i < len(tool_path) - 1:
            frames.append((i, None, 'z_lift'))

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                 blit=True, interval=50)  # 50ms per frame

    # Ensure final plot is correct
    def finalize_plot(event):
        # Update all sub-path lines
        for i, line in enumerate(sub_path_lines):
            if sub_path_segments[i]:
                x_data = []
                y_data = []
                for x_seg, y_seg in sub_path_segments[i]:
                    x_data.extend([x_seg[0], x_seg[1], np.nan])
                    y_data.extend([y_seg[0], y_seg[1], np.nan])
                line.set_data(x_data, y_data)
            else:
                line.set_data([], [])

        # Update z-lift line
        if z_lift_segments:
            x_data = []
            y_data = []
            for x_seg, y_seg in z_lift_segments:
                x_data.extend([x_seg[0], x_seg[1], np.nan])
                y_data.extend([y_seg[0], y_seg[1], np.nan])
            z_lift_line.set_data(x_data, y_data)
        else:
            z_lift_line.set_data([], [])

        # Redraw the canvas to ensure final state
        fig.canvas.draw()

    # Connect finalize_plot to the figure's close event
    fig.canvas.mpl_connect('close_event', finalize_plot)

    # Add legend
    ax.legend()

    # Show the animation
    plt.show()




def get_system_info():
    info = {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }
    return info


def log_results(filepath, vol, num_triangles, slice_thickness, height,slicing_time):
    info = get_system_info()
    with open('slice_time_logs.txt', 'a') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 40 + "\n")
        f.write(f"File: {filepath}\n")
        f.write(f"Volume: {vol:.4f} cubic mm \n")
        f.write(f"Number of Facets: {num_triangles}\n")
        f.write(f"Slice thickness: {slice_thickness} mm\n")
        f.write(f"Model height: {height} mm\n")
        f.write(f"Number of slices: {int(height/slice_thickness)}\n")
        f.write(f"Slicing time: {slicing_time:.4f} seconds\n")
        


if __name__ == "__main__":

    filepath = 'tst01.STL'  # Path to your STL file
    slice_thickness = 2  # Z-layer for slicing
    angle_radians = np.deg2rad(90)
    # rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [0, 0, 1])
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [1, 0, 0])
    stl_mesh = trimesh.load(filepath,process=False)
    vol = stl_mesh.volume
    num_triangles = len(stl_mesh.faces)
    _, z_max = stl_mesh.bounds[:,2]

    start_slicing = time.time()
    zigzag()
    end_slicing = time.time()
    slicing_time = end_slicing - start_slicing
    print(f"Total time taken: {end_slicing - start_slicing:.4f} seconds")
    log_results(filepath, vol, num_triangles, slice_thickness, z_max,slicing_time)