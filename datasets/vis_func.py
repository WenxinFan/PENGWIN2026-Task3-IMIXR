import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
import trimesh

def _get_id_colors(ids):
    """
    (内部函数) 为一组 ID 生成对应的高对比度颜色矩阵。
    使用 matplotlib 的 tab20 色卡保证相邻 ID 颜色区分明显。
    """
    unique_ids = np.unique(ids)
    cmap = plt.get_cmap("tab20")

    # 建立 ID 到颜色的映射字典
    id_to_color = {uid: cmap(i % 20)[:3] for i, uid in enumerate(unique_ids)}

    # 映射到所有点
    colors = np.zeros((len(ids), 3))
    for uid in unique_ids:
        colors[ids == uid] = id_to_color[uid]

    return colors

# Dark 10-color palette for better visualization
_DARK10_COLORS = [
    [0.1216, 0.4667, 0.7059],  # Dark blue
    [1.0000, 0.4980, 0.0549],  # Orange
    [0.1725, 0.6275, 0.1725],  # Green
    [0.8392, 0.1529, 0.1569],  # Red
    [0.5804, 0.4039, 0.7412],  # Purple
    [0.4980, 0.4980, 0.4980],  # Gray
    [0.0902, 0.7451, 0.8118],  # Cyan
    [0.6902, 0.3490, 0.1922],  # Brown
    [0.6510, 0.6510, 0.6510],  # Light gray
    [0.9686, 0.7137, 0.8235],  # Pink
]


def visualize_fragments(meshdict, save_path=None):
    """
    Visualize mesh fragments using trimesh with different colors for each fragment.

    Args:
        meshdict: Dictionary with bone_type -> {frag_name: {'mesh': trimesh, 'volume': float}}
        save_path: Optional path to save the screenshot
    """
    scene = trimesh.Scene()

    color_idx = 0

    for bone_type in ["SA", "LI", "RI"]:
        fragments = meshdict.get(bone_type, {})
        for frag_name, frag_data in fragments.items():
            mesh = frag_data['mesh']

            # Use dark10 palette cyclically
            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]

            # Apply color to the mesh
            mesh.visual.vertex_colors[:, :3] = (np.array(fragment_color) * 255).astype(np.uint8)
            scene.add_geometry(mesh, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    if save_path:
        # Save screenshot
        scene.save_image(save_path)
        print(f"Saved visualization to {save_path}")

    scene.show()
    return scene


def visualize_transformed_fragments(meshdict, predictions, save_path=None):
    """
    Visualize mesh fragments after applying predicted 4x4 transformations.

    Each fragment is colored with the same Dark10 palette as visualize_fragments(),
    so before/after comparison is consistent.

    Args:
        meshdict: Dictionary with bone_type -> {frag_name: {'mesh': trimesh, 'volume': float}}
        predictions: List of dicts with 'fragment_id' and 'transformation' (4x4 matrix).
        save_path: Optional path to save the screenshot.
    """
    # Build lookup: fragment_id -> 4x4 matrix
    transform_lookup = {}
    for pred in predictions:
        transform_lookup[str(pred["fragment_id"])] = np.array(pred["transformation"])

    scene = trimesh.Scene()
    color_idx = 0

    for bone_type in ["SA", "LI", "RI"]:
        fragments = meshdict.get(bone_type, {})
        for frag_name, frag_data in fragments.items():
            mesh = frag_data['mesh'].copy()

            # Apply predicted transformation if available
            if frag_name in transform_lookup:
                mesh.apply_transform(transform_lookup[frag_name])

            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]
            mesh.visual.vertex_colors[:, :3] = (np.array(fragment_color) * 255).astype(np.uint8)
            scene.add_geometry(mesh, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    if save_path:
        scene.save_image(save_path)
        print(f"Saved visualization to {save_path}")

    scene.show()
    return scene


def visualize_pointcloud_by_part(pointclouds, points_per_part, title="Point Cloud"):
    """
    Visualize a flat point cloud colored by fragment using Dark10 palette.

    Args:
        pointclouds: (N, 3) numpy array of point coordinates.
        points_per_part: list/array of point counts per fragment.
        title: window title string.
    """
    scene = trimesh.Scene()
    offset = 0

    for i, count in enumerate(points_per_part):
        if count <= 0:
            continue
        pts = pointclouds[offset:offset + count]
        color = _DARK10_COLORS[i % len(_DARK10_COLORS)]
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        pc = trimesh.PointCloud(pts, colors=np.tile(color_uint8, (len(pts), 1)))
        scene.add_geometry(pc, geom_name=f"part_{i}")
        offset += count

    scene.show()
    return scene


def visualize_pointcloud_fragments(sampled_data, bone_type_to_id=None):
    """
    Visualize sampled point cloud fragments with different colors using trimesh.

    Args:
        sampled_data: Dictionary from sub_sample with 'coords' and 'normals' per fragment
        bone_type_to_id: Optional dict mapping bone type to ID, defaults to {"SA": 0, "LI": 1, "RI": 2}
    """
    if bone_type_to_id is None:
        bone_type_to_id = {"SA": 0, "LI": 1, "RI": 2}
    id_to_bone = {v: k for k, v in bone_type_to_id.items()}

    scene = trimesh.Scene()

    color_idx = 0

    for bone_type, fragments in sampled_data.items():
        for frag_name, frag_data in fragments.items():
            coords = frag_data['coords']

            if len(coords) == 0:
                continue

            # Use dark10 palette cyclically
            fragment_color = _DARK10_COLORS[color_idx % len(_DARK10_COLORS)]
            point_cloud = trimesh.PointCloud(coords, colors=(np.array(fragment_color) * 255).astype(np.uint8))
            scene.add_geometry(point_cloud, geom_name=f"{bone_type}_{frag_name}")
            color_idx += 1

    scene.show()
    return scene


def visualize_pointclouds_by_bone(pointclouds_gt, point_bone_ids, recon_tgt=None):
    """
    版本 1：按【骨头 (Bone)】上色可视化。
    同一个 bone_id 的所有点（即使它断成了多个碎块）都会显示相同的颜色。

    参数:
        pointclouds_gt: 真实点云 [N, 3]
        point_bone_ids: 每个点对应的匿名骨头 ID [N,]
        recon_tgt: 重建点云 [N, 3] (可选)
    """
    # 强制转为 numpy
    if hasattr(pointclouds_gt, "cpu"): pointclouds_gt = pointclouds_gt.detach().cpu().numpy()
    if hasattr(point_bone_ids, "cpu"): point_bone_ids = point_bone_ids.detach().cpu().numpy()
    if recon_tgt is not None and hasattr(recon_tgt, "cpu"): recon_tgt = recon_tgt.detach().cpu().numpy()

    colors = _get_id_colors(point_bone_ids)
    all_vis_object = []

    # 1. 创建 GT 点云 (左侧)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointclouds_gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)
    all_vis_object.append(pcd_gt)

    # 2. 创建 Recon 点云 (右侧并排)
    if recon_tgt is not None:
        pcd_recon = o3d.geometry.PointCloud()

        # 自动计算包围盒宽度，将 Recon 沿 X 轴平移，避免颜色重叠导致 Z-fighting
        shift_x = np.max(pointclouds_gt[:, 0]) - np.min(pointclouds_gt[:, 0]) + 0.1
        shifted_recon = recon_tgt.copy()
        shifted_recon[:, 0] += shift_x * 1.2  # 1.2 倍宽度作为间隔

        pcd_recon.points = o3d.utility.Vector3dVector(shifted_recon)
        pcd_recon.colors = o3d.utility.Vector3dVector(colors)  # 使用完全相同的颜色映射
        all_vis_object.append(pcd_recon)

    o3d.visualization.draw_geometries(all_vis_object,
                                      window_name="点云比较 (按 Bone 上色) - 左: GT, 右: Recon",
                                      width=1600, height=800)


def visualize_pointclouds_by_part(pointclouds_gt, points_per_part, recon_tgt=None):
    """
    版本 2：按【碎块 (Part)】上色可视化。
    同一个碎块的点具有相同的颜色，不同碎块颜色截然不同。

    参数:
        pointclouds_gt: 真实点云 [N, 3]
        points_per_part: 每个碎块的点数列表/数组
        recon_tgt: 重建点云 [N, 3] (可选)
    """
    # 强制转为 numpy
    if hasattr(pointclouds_gt, "cpu"): pointclouds_gt = pointclouds_gt.detach().cpu().numpy()
    if hasattr(points_per_part, "cpu"): points_per_part = points_per_part.detach().cpu().numpy()
    if recon_tgt is not None and hasattr(recon_tgt, "cpu"): recon_tgt = recon_tgt.detach().cpu().numpy()

    # 根据 points_per_part 动态生成点级的 part_ids
    # 注意过滤掉 padding 出来的 0
    valid_counts = [int(c) for c in points_per_part if c > 0]
    part_ids = np.concatenate([np.full(c, i, dtype=np.int32) for i, c in enumerate(valid_counts)])

    colors = _get_id_colors(part_ids)
    all_vis_object = []

    # 1. 创建 GT 点云 (左侧)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointclouds_gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)
    all_vis_object.append(pcd_gt)

    # 2. 创建 Recon 点云 (右侧并排)
    if recon_tgt is not None:
        pcd_recon = o3d.geometry.PointCloud()

        shift_x = np.max(pointclouds_gt[:, 0]) - np.min(pointclouds_gt[:, 0]) + 0.1
        shifted_recon = recon_tgt.copy()
        shifted_recon[:, 0] += shift_x * 1.2

        pcd_recon.points = o3d.utility.Vector3dVector(shifted_recon)
        pcd_recon.colors = o3d.utility.Vector3dVector(colors)
        all_vis_object.append(pcd_recon)

    o3d.visualization.draw_geometries(all_vis_object,
                                      window_name="点云比较 (按 Part 上色) - 左: GT, 右: Recon",
                                      width=1600, height=800)



# 使用示例（不用改，直接替换原函数即可）
from matplotlib import cm
def visualize_fracture_graph(pointclouds_gt: np.ndarray, points_per_part: np.ndarray,
                             graph: np.ndarray, num_parts: int,
                             window_name: str = "Fracture Graph Visualization"):
    """
    可视化碎块及其邻接关系

    Args:
        pointclouds_gt: 所有碎块的点云拼接 [N, 3]
        points_per_part: 每个碎块的点数 [P]
        graph: 邻接矩阵 [max_parts, max_parts]
        num_parts: 实际碎块数量
    """
    if isinstance(num_parts, (np.ndarray, np.generic)):
        if num_parts.size == 1:
            num_parts = int(num_parts.item())
        else:
            # If it's an array (e.g. from a batch), assume we want the first element
            # or it's actually the length of something
            num_parts = int(num_parts[0])
    elif hasattr(num_parts, 'item'):  # Handle torch.Tensor
        num_parts = int(num_parts.item())
    else:
        num_parts = int(num_parts)

        # [Fix] Ensure points_per_part is a proper numpy array
    if not isinstance(points_per_part, np.ndarray):
        points_per_part = np.array(points_per_part)

        # Safety check: ensure num_parts isn't larger than the available data
    if num_parts > len(points_per_part):
        print(f"[Warn] num_parts ({num_parts}) > len(points_per_part) ({len(points_per_part)}). Clipping.")
        num_parts = len(points_per_part)

        # 提取每个碎块的点云
    offsets = np.concatenate([[0], np.cumsum(points_per_part[:num_parts])])
    part_pointclouds = []

    for i in range(num_parts):
        start, end = offsets[i], offsets[i + 1]
        part_pointclouds.append(pointclouds_gt[start:end])

    # 创建颜色映射
    colors = cm.get_cmap('tab10', num_parts)

    # 创建可视化几何体
    geometries = []

    # 1. 绘制每个碎块的点云
    for i in range(num_parts):
        if len(part_pointclouds[i]) == 0:
            continue

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_pointclouds[i])

        # 为每个碎块分配不同颜色
        color = colors(i)[:3]  # 取RGB，忽略alpha
        pcd.paint_uniform_color(color)

        geometries.append(pcd)

        # 添加碎块标签
        centroid = np.mean(part_pointclouds[i], axis=0)
        text_mesh = _create_text_mesh(f"Part {i}", centroid, color)
        if text_mesh:
            geometries.append(text_mesh)

    # 2. 绘制邻接关系连线
    line_points = []
    line_indices = []
    line_colors = []

    line_count = 0
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            if graph[i, j]:
                # 计算两个碎块的质心
                centroid_i = np.mean(part_pointclouds[i], axis=0)
                centroid_j = np.mean(part_pointclouds[j], axis=0)

                line_points.append(centroid_i)
                line_points.append(centroid_j)

                line_indices.append([line_count * 2, line_count * 2 + 1])
                line_colors.append([1.0, 0.0, 0.0])  # 红色连线

                line_count += 1

    if line_points:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

    # 3. 添加坐标轴
    #coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #geometries.append(coordinate_frame)

    # 设置可视化参数
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)

    for geometry in geometries:
        vis.add_geometry(geometry)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])
    view_control.set_up([0, 1, 0])
    view_control.set_lookat([0, 0, 0])

    vis.run()
    vis.destroy_window()

def _create_text_mesh(text: str, position: np.ndarray, color: np.ndarray):
    """
    创建文本网格（简化版本，Open3D不支持直接文本，用球体代替）
    """
    try:
        # 用一个小球体代表标签位置
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        return sphere
    except:
        return None


def visualize_pointclouds_comparison(pointclouds_gt, recon_tgt=None, colors=None):
    """
    使用Open3D可视化两个点云的比较

    参数:
        pointclouds_gt: 真实点云 [N, 3]
        recon_tgt: 重建点云 [N, 3]
        colors: 可选的颜色列表，如果不提供则使用默认颜色
    """

    # 默认颜色设置
    if colors is None:
        color_gt = [0, 1, 0]  # 绿色 - 真实点云
        color_recon = [1, 0, 0]  # 红色 - 重建点云
    else:
        color_gt, color_recon = colors

    all_vis_object = []
    # 创建Open3D点云对象
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pointclouds_gt)
    pcd_gt.paint_uniform_color(color_gt)
    all_vis_object.append(pcd_gt)

    if recon_tgt is not None:
        pcd_recon = o3d.geometry.PointCloud()
        pcd_recon.points = o3d.utility.Vector3dVector(recon_tgt)
        pcd_recon.paint_uniform_color(color_recon)
        all_vis_object.append(pcd_recon)

    # 可视化
    o3d.visualization.draw_geometries(all_vis_object,
                                      window_name="点云比较: 绿色=真实, 红色=重建",
                                      width=1200, height=800)

def visualize_pointclouds(pointclouds_gt, pointclouds_normals_gt, fracture_surface):
    """
    显示两个图：
    1️⃣ 点云 + 断裂面高亮（灰 + 红）
    2️⃣ 点云 + 法向量箭头显示（红色线段）

    参数：
    - pointclouds_gt: [N,3] numpy array
    - pointclouds_normals_gt: [N,3] numpy array
    - fracture_surface: [N,] bool 或 0/1 标记断裂面
    """

    # ---------- 图1: 灰 + 红 (断裂面高亮) ----------
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pointclouds_gt)

    colors = np.tile(np.array([0.7, 0.7, 0.7]), (len(pointclouds_gt), 1))
    colors[fracture_surface.astype(bool)] = np.array([1.0, 0.0, 0.0])
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd1], window_name="Fracture Surface (Gray + Red)")

    # ---------- 图2: 点云 + 法向量 ----------
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(pointclouds_gt)
    # pcd2.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色点云
    #
    # geometries2 = [pcd2]
    #
    # # 法向量箭头（用 LineSet）
    # step = max(1, len(pointclouds_gt)//2000)  # 控制抽样密度
    # arrow_points = []
    # arrow_lines = []
    # arrow_colors = []
    #
    # scale = 0.1  # 箭头长度比例
    # for i, (p, n) in enumerate(zip(pointclouds_gt[::step], pointclouds_normals_gt[::step])):
    #     start = p
    #     end = p + n * scale
    #     arrow_points.append(start)
    #     arrow_points.append(end)
    #     arrow_lines.append([2 * i, 2 * i + 1])
    #     arrow_colors.append([1.0, 0.0, 0.0])  # 红色箭头
    #
    # if arrow_points:
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(np.array(arrow_points))
    #     line_set.lines = o3d.utility.Vector2iVector(np.array(arrow_lines))
    #     line_set.colors = o3d.utility.Vector3dVector(np.array(arrow_colors))
    #     geometries2.append(line_set)
    #
    # o3d.visualization.draw_geometries(geometries2, window_name="Point Cloud + Normals (Arrows)")

def visualize_pointclouds_and_fractures(pointclouds_gt, pointclouds_normals_gt, fracture_surface):
    """
    鲁棒的可视化函数 V2：
    1. [修复] 自动计算箭头长度，适应不同单位 (mm/cm/normalized)。
    2. [修复] 强制法向量归一化，防止长短不一。
    3. [优化] 增加采样步长，防止变成"刺猬"。
    """

    # -------------------------------------------------------------
    # 1. 数据清洗 (保持不变)
    # -------------------------------------------------------------
    def sanitize(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.dtype != np.float64 and x.dtype != np.int32 and x.dtype != bool:
            x = x.astype(np.float64)
        return np.ascontiguousarray(x)

    pts = sanitize(pointclouds_gt)
    nrms = sanitize(pointclouds_normals_gt)
    labels = sanitize(fracture_surface).flatten()

    if pts.ndim == 3:
        pts = pts.reshape(-1, 3)
        nrms = nrms.reshape(-1, 3)
        labels = labels.reshape(-1)

    print(f"[Visualizer] Points: {pts.shape}")

    # -------------------------------------------------------------
    # 2. [关键] 自适应计算参数
    # -------------------------------------------------------------
    # 计算包围盒对角线长度，作为物体尺度的参考
    bbox_min = pts.min(0)
    bbox_max = pts.max(0)
    object_scale = np.linalg.norm(bbox_max - bbox_min)

    # 设定箭头长度为物体尺寸的 2% ~ 5%
    arrow_len = object_scale * 0.03
    print(f"[Auto-Scale] Object Size: {object_scale:.2f}, Arrow Length: {arrow_len:.4f}")

    # -------------------------------------------------------------
    # 3. 窗口 1: 断裂面 (Fracture Surface)
    # -------------------------------------------------------------
    pcd_surf = o3d.geometry.PointCloud()
    pcd_surf.points = o3d.utility.Vector3dVector(pts)

    colors = np.zeros_like(pts)
    colors[:] = [0.7, 0.7, 0.7]  # 默认灰
    mask = (labels > 0.5)
    colors[mask] = [1.0, 0.0, 0.0]  # 断裂面红
    pcd_surf.colors = o3d.utility.Vector3dVector(colors)

    # -------------------------------------------------------------
    # 4. 窗口 2: 法向量 (Normals)
    # -------------------------------------------------------------
    vis_objs = []

    # 底图点云
    pcd_norm = o3d.geometry.PointCloud()
    pcd_norm.points = o3d.utility.Vector3dVector(pts)
    pcd_norm.paint_uniform_color([0.6, 0.6, 0.6])
    pcd_norm.normals = o3d.utility.Vector3dVector(nrms)
    vis_objs.append(pcd_norm)

    # --- 生成箭头 (LineSet) ---

    # [优化] 动态步长：保证屏幕上大概只有 1000-2000 个箭头，不至于密集恐惧症
    target_arrow_count = 10000
    step = max(1, len(pts) // target_arrow_count)

    sample_pts = pts[::step]
    sample_nrms = nrms[::step]

    # [修复] 强制归一化法向量 (防止输入数据本身长度不对)
    # np.linalg.norm 可能遇到 0，加个 epsilon 防止除零
    #norms = np.linalg.norm(sample_nrms, axis=1, keepdims=True)
    #sample_nrms = sample_nrms / (norms + 1e-8)
    #sample_nrms = norms
    # 构建线段
    starts = sample_pts
    ends = sample_pts + sample_nrms * arrow_len

    line_points = np.concatenate([starts, ends], axis=0)
    N = len(starts)
    lines = np.stack([np.arange(N), np.arange(N) + N], axis=1)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    line_set.paint_uniform_color([1.0, 0.2, 0.2])  # 亮红色箭头

    vis_objs.append(line_set)

    print(f">> Displaying Window. Press 'q' to close.")
    # 先显示断裂面
    o3d.visualization.draw_geometries([pcd_surf], window_name="1. Fracture Surface (Red)", width=800, height=600,
                                      left=50, top=50)
    # 再显示法向
    o3d.visualization.draw_geometries(vis_objs, window_name="2. Normals (Arrows)", width=800, height=600, left=900,
                                      top=50)

