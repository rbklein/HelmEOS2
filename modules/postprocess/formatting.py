from prep_jax import *
from config.conf_geometry import *

from modules.geometry.grid import GRID_SPACING

import numpy as np
import vtk

from vtkmodules.util import numpy_support

def velocity_vti(u, name):
    v = u[1:(N_DIMENSIONS+1)] / u[0]
    
    # ---- user inputs ----
    vti_path = "output/velocity_cell_" + name + ".vti"

    origin  = (0.0, 0.0, 0.0)
    spacing = GRID_SPACING
    array_name = "velocity_" + name


    # Use float32 to cut file/memory footprint (recommended)
    v = v.astype(np.float32, copy=False)

    _, nx, ny, nz = v.shape  # cells

    img = vtk.vtkImageData()
    img.SetOrigin(*origin)
    img.SetSpacing(*spacing)
    img.SetDimensions(nx + 1, ny + 1, nz + 1)  # point dims

    # Flatten each component in VTK-friendly ordering and stack into N×3 tuples
    ux = np.ravel(v[0], order="F")
    uy = np.ravel(v[1], order="F")
    uz = np.ravel(v[2], order="F")

    vec = np.column_stack((ux, uy, uz))  # shape (nx*ny*nz, 3)

    vtk_vec = numpy_support.numpy_to_vtk(vec, deep=True)
    vtk_vec.SetName(array_name)

    # Attach as CELL vector data
    cell_data = img.GetCellData()
    cell_data.AddArray(vtk_vec)
    cell_data.SetActiveVectors(array_name)   # mark it as the active vector field

    # (Optional) also store magnitude as a scalar for coloring
    mag = np.sqrt(ux*ux + uy*uy + uz*uz).astype(np.float32, copy=False)
    vtk_mag = numpy_support.numpy_to_vtk(mag, deep=True)
    vtk_mag.SetName(f"{array_name}_mag")
    cell_data.AddArray(vtk_mag)
    cell_data.SetActiveScalars(f"{array_name}_mag")

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vti_path)
    writer.SetInputData(img)
    writer.SetCompressorTypeToZLib()
    ok = writer.Write()
    if not ok:
        raise RuntimeError("Failed to write VTI")

    print("Wrote:", vti_path)