import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from pathlib import Path


class Panel:
    def __init__(self, mesh_file):

        print("panel init")
        print("mesh file path : ", mesh_file)

        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Mesh + Vertices", 1280, 720)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        self.panel = gui.ScrollableVert(0, gui.Margins(8, 8, 8, 8))
        self.panel.add_child(gui.Label("Vertices (PyTorch3D load_ply)"))
        self.panel.preferred_width = 420
        self.text = gui.TextEdit()
        self.panel.add_child(self.text)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.on_layout)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))

        self.scene_widget.scene.add_geometry("mesh", mesh, mat)
        app.run()

    def on_layout(self, layout_context):
        r = self.window.content_rect
        panel_w = int(self.panel.preferred_width)

        self.panel.frame = gui.Rect(r.x + r.width - panel_w, r.y, panel_w, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)


if __name__ == "__main__":

    work_dir = Path(__file__).parent
    mesh_file = (work_dir / "../../models/cube.ply").resolve()

    panel = Panel(mesh_file)
