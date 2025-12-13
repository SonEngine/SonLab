import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from pytorch3d.io import load_ply
from pathlib import Path


class MeshWithVertexPanel:
    def __init__(self, mesh_path: Path):
        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Mesh + Vertices", 1280, 720)

        # 3D View
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        # Right panel (scrollable)
        self.panel = gui.ScrollableVert(0, gui.Margins(8, 8, 8, 8))
        self.panel.preferred_width = 420

        self.panel.add_child(gui.Label("Vertices (PyTorch3D load_ply)"))
        self.text = gui.TextEdit()
        self.text.enabled = False  # read-only처럼 동작
        self.panel.add_child(self.text)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        # Load mesh + show in 3D
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("mesh", mesh, mat)

        # bounds = mesh.get_axis_aligned_bounding_box()
        # self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

        # Load vertices/faces via PyTorch3D and show in GUI
        # vertices, faces = load_ply(str(mesh_path))
        self.text.text_value = "VERTICES"

        app.run()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_w = int(self.panel.preferred_width)

        # 오른쪽 패널 / 왼쪽 3D 뷰 영역 분할
        self.panel.frame = gui.Rect(r.x + r.width - panel_w, r.y, panel_w, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    mesh_file = (root_dir / "../../models/cube.ply").resolve()

    MeshWithVertexPanel(mesh_file)
