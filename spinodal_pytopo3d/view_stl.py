"""Open an STL in an interactive PyVista window (drag=rotate, scroll=zoom).

Usage: python -m spinodal_pytopo3d.view_stl <path.stl>
"""
import sys

import pyvista as pv


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "spinodal_pytopo3d/results/truss_microstructure.stl")
    mesh = pv.read(path)
    print(f"{path}: {mesh.n_cells:,} triangles")
    pl = pv.Plotter(window_size=[1500, 950])
    pl.set_background("white")
    pl.add_mesh(mesh, color="#4DB6AC", smooth_shading=True, specular=0.4,
                specular_power=25, ambient=0.3)
    pl.add_mesh(pv.Box(bounds=mesh.bounds).extract_all_edges(),
                color="black", line_width=1)
    pl.add_axes(xlabel="x1", ylabel="x2", zlabel="x3")
    pl.add_text("truss spinodal microstructure  |  drag = rotate, scroll = zoom",
                font_size=10, color="black")
    try:
        pl.enable_eye_dome_lighting()
    except Exception:
        pass
    pl.show(title="STL viewer")


if __name__ == "__main__":
    main()
