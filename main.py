from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom, label
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import json

def main():

    #Location containing our downloaded DICOM files
    data_folder = Path("data/raw")

    #Search through all folders inside the data/raw 
    #and find every file ending in .dcm . rglob is a method that allows you to search for files recursively in a directory and its subdirectories. The "*.dcm" pattern specifies that we are looking for files with the .dcm extension, which is the standard file format for DICOM files.
    diccom_files = list(data_folder.rglob("*.dcm"))

    print("Number of DICOM files found:", len(diccom_files))

    #Look at the first file python found
    first_file = diccom_files[0]

    print("First file: ")
    print(first_file)

    #Read the DICOM file 
    ds = pydicom.dcmread(first_file)

    print("\n--- DICOM INFORMATION ---")

    print("Modality:", ds.Modality)
    print("Rows:", ds.Rows)
    print("Columns:", ds.Columns)
    print("Instance Number:", ds.InstanceNumber)
    print("Pixel Spacing:", ds.PixelSpacing)
    print("Slice Thickness:", ds.SliceThickness)


    image = ds.pixel_array

    print("\n--- IMAGE INFORMATION ---")

    print("Image shape:", image.shape)
    print("Image data type:", image.dtype)
    print("Minimum pixel storage value:", image.min())
    print("Maximum pixel storage value:", image.max())

    #Look at one specific pixel
    print("Pixel value at (100, 100):", image[100, 100])


    #CT rescale information

    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)

    print("\n--- CT RESCALE INFORMATION ---")
    print("Rescale Slope:", slope)
    print("Rescale Intercept:", intercept)

    #Convert pixel values to Hounsfield Units (HU)
    hu_image = image.astype("float32") * slope + intercept

    print("\n--- HU IMAGE INFORMATION ---")
    print("Minimum HU value:", hu_image.min())
    print("Maximum HU value:", hu_image.max())
    print("HU value at (100, 100):", hu_image[100, 100])


    #Display the CT image using Hounsfield Units (HU)

    plt.imshow(
        hu_image,
        cmap="gray",
        # vmin=-160,
        # vmax=240
    )

    plt.title("CT Image in Hounsfield Units (HU)")
    plt.axis("off")
    # plt.show()

    #Soft tissue windowing is a technique used in medical imaging, particularly in computed tomography (CT) scans, to enhance the visibility of soft tissues in the images. It involves adjusting the window level and window width settings of the image display to optimize the contrast and brightness for specific tissue types.
    window_level = 40
    window_width = 400

    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)

    print("n\--- SOFT TISSUE WINDOWING ---")
    print("Window Level:", window_level)
    print("Window Width:", window_width)
    print("Lower HU:", lower)
    print("Upper HU:", upper)

    plt.imshow(
        hu_image,
        cmap="gray",
        vmin=lower,
        vmax=upper
    )

    plt.title("CT - Soft Tissue Window")
    plt.axis("off")
    plt.show()

#-------------------------------------------
#LOAD ALL DICOMM SLICES
#-------------------------------------------

    slices = []
    for file in diccom_files:

        #read each DICOM file
        ds_slice = pydicom.dcmread(file)

        #store the DICOM dataset
        slices.append(ds_slice)

    print("\---3D SERIES INFORMATION ---")
    print("Slices loaded :", len(slices))

    #Slices is a list containing 120 dicom datasets. Each dataset represents a single slice of the CT scan. The datasets contain metadata and pixel data for each slice, allowing us to reconstruct the 3D volume of the scanned area.

    #The order returned by the filesystem may not be the same as the order of the slices in the CT scan. To ensure that we have the correct order, we can sort the slices based on their InstanceNumber attribute, which indicates the position of each slice in the scan.

    for ds_slice in slices[:5]:
        print(
            "Instance:",
            ds_slice.InstanceNumber,
            "| Position:",
            ds_slice.ImagePositionPatient
        )


    #Sort slices based according their physical Z position in the patient's body. The ImagePositionPatient attribute contains the (x, y, z) coordinates of the top-left corner of the image slice in the patient's coordinate system. By sorting the slices based on their z-coordinate, we can ensure that they are ordered correctly from top to bottom (or vice versa) in the 3D volume.
    slices.sort(
        key=lambda ds_slice: ds_slice.ImagePositionPatient[2],
        reverse=True
    )

    print("\n--- SORTED SLICES POSITIONS ---")

    for ds_slice in slices[:5]:
        print(
            "Instance:",
            ds_slice.InstanceNumber,
            "| Z Position:",
            ds_slice.ImagePositionPatient[2]

        )

    hu_slices = []
    for ds_slices in slices:

        #Get the stored pixel values from the DICOM dataset
        pixel_array = ds_slices.pixel_array.astype("float32")

        #Get this slice's rescale information
        slope = float(ds_slices.RescaleSlope)
        intercept = float(ds_slices.RescaleIntercept)

        #Convert pixel values to Hounsfield Units (HU)
        hu_slice = pixel_array * slope + intercept

        #Add the converted slice to our list
        hu_slices.append(hu_slice)

        #Important Line: Stack all the slices together to create a 3D volume. The np.stack function takes a sequence of arrays 
        # (in this case, the list of HU slices) and stacks them along a new axis (axis=0), resulting in a 3D array where each slice is along the first dimension.  
    volume = np.stack(hu_slices, axis=0)

    print("\n---3D CT VOLUME INFORMATION ---")
    print("Volume shape:", volume.shape)
    print("Data type:", volume.dtype)
    print("Minimum HU", volume.min())
    print("Maximum HU", volume.max())

        #You started with 120 independent 2D slices, each with a shape of (512, 512). 
        # After stacking them together, you now have a 3D volume with a shape of (120, 512, 512),
        #  where the first dimension represents the number of slices, and the other two dimensions represent the height and width of each slice.


    #Next, we'll acess the volume with things like #axial, coronal, and sagittal views. These are standard anatomical planes used in medical imaging to visualize the human body from different perspectives.

    #--------------------------------
    # MULTI-PLANAR RECONSTRUCTION (MPR)
    #--------------------------------  

    #Find the center index of each dimension
    z_center = volume.shape[0] // 2  # Axial view (top-down)
    y_center = volume.shape[1] // 2  # Coronal view (front-back)
    x_center = volume.shape[2] // 2  # Sagittal view (side view)

    print("\n --- VOLUME CENTER ---")
    print("Z center:", z_center)
    print("Y center:", y_center)
    print("X center:", x_center)


    axial = volume[z_center, :, :]
    coronal = volume[:, y_center, :]
    sagittal = volume[:, :, x_center]


    plt.imshow(
        axial,
        cmap="gray",
        vmin=-160,
        vmax=240
    )

    plt.title("Axial CT - Middle Slice")
    plt.axis("off")
    plt.show()

    #Next we'll dispaly the cor recons, you'll notice that the coronal and sagittal views are not as clear as the axial view. 
    # This is because the original CT scan was acquired in the axial plane, and the coronal and sagittal views are reconstructed from the axial slices. 
    # The resolution in these planes may be lower, leading to less sharp images.
    #That problem will intriduce us to voxel spacing and resampling, which is a technique used to adjust the resolution of medical images to achieve isotropic voxel spacing. 
    # Isotropic voxels have equal dimensions in all three directions (x, y, z), which is important for accurate 3D visualization and analysis.

    

    plt.imshow(
        coronal,
        cmap="gray",
        vmin=-160,
        vmax=240
    )

    plt.title("Coronal CT Reconstruction")
    plt.axis("off")
    plt.show()

    #The coronal image will look squished because the original CT scan was acquired with anisotropic voxel spacing,
    #  meaning that the distance between slices (z-axis) is different from the pixel spacing within each slice (x and y axes).
    #To fix this, we can resample the volume to achieve isotropic voxel spacing,
    
    #Cornal Image

    z1 = float(slices[0].ImagePositionPatient[2])
    z2 = float(slices[1].ImagePositionPatient[2])

    z_spacing = abs(z2 - z1)
    y_spacing = float(slices[0].PixelSpacing[0])
    x_spacing = float(slices[0].PixelSpacing[1])

    print("\n--- VOXEL SPACING ---")
    print("Z spacing:", z_spacing, "mm")
    print("Y spacing:", y_spacing, "mm")
    print("X spacing:", x_spacing, "mm")


    #---------------------------------
    #Correct the aspect ratio of the coronal view
    #---------------------------------

    #The z pixels represent 2.5mm,
    #while the X pixels represent 0.66mm.
    #We calculate their phjyscail size ratio to correct the aspect ratio of the coronal view.

    aspect_ratio = z_spacing / x_spacing

    print("\n--- Cornal View Aspect Ratio ---")
    print("Coronal view aspect ratio:", aspect_ratio)

    #Display  the cornal image using the 
    #correct physical protportions.

    plt.imshow(
        coronal,
        cmap="gray",
        vmin=-160,
        vmax=240,
        aspect=aspect_ratio
    )

    plt.title("Coronal CT - Middle Slice (Corrected Aspect Ratio)")
    plt.axis("off")
    plt.show()

    sagittal = volume[:, :, x_center]

    #--------------------------------
    #Sagittal View
    #--------------------------------

    #Correct the aspect ratio of the sagittal view
    sagittal_aspect_ratio = z_spacing / y_spacing

    print("\n--- Sagittal View Aspect Ratio ---")
    print("Sagittal view aspect ratio:", sagittal_aspect_ratio)

    plt.imshow(
        sagittal,
        cmap="gray",
        vmin=-160,
        vmax=240,
        aspect=sagittal_aspect_ratio
    )

    plt.title("Sagittal CT - Middle Slice (Corrected Aspect Ratio)")
    plt.axis("off")
    plt.show()


    #--------------------------------
    #3D  Volume RESAMPLING
    #--------------------------------

    #Desired isotopic voxel spacing in millimeters (mm)
    new_spacing = 1.0

    #Calculate how much each dimension needs to chnage
    z_zoom = z_spacing / new_spacing
    y_zoom = y_spacing / new_spacing
    x_zoom = x_spacing / new_spacing

    print("\n--- RESAMPLING FACTORS ---")
    print("Z zoom factor:", z_zoom)
    print("Y zoom factor:", y_zoom)
    print("X zoom factor:", x_zoom)


    #Resample the 3D CT Volume
    #zoom() is doing two important things: it resizes the volume based on the 
    # specified zoom factors for each dimension (z, y, x),
    #  and it applies linear interpolation to estimate the values of the
    #  new voxels in the resampled volume. The result is a new 3D volume with isotropic voxel spacing, 
    # which can be used for more accurate 3D visualization and analysis.
    resampled_volume =zoom(
        volume,
        (z_zoom, y_zoom, x_zoom),
        #order=1 means linear interpolation, which is a method of estimating new data points within the range of a discrete set of known data points. In the context of image resampling, linear interpolation calculates the value of a new pixel based on the values of its neighboring pixels. It is a common choice for resampling because it provides a good balance
        #  between image quality and computational efficiency.
        order=1  # Linear interpolation
    )

    print("\n--- RESAMPLED VOLUME INFORMATION ---")
    print("Original volume shape:", volume.shape)
    print("Resampled volume shape:", resampled_volume.shape)


    #--------------------------------
    #View Sampled Volume
    #-------------------------------

    #Find the center of the resampled volume
    new_z_center = resampled_volume.shape[0] // 2
    new_y_center = resampled_volume.shape[1] // 2
    new_x_center = resampled_volume.shape[2] // 2

    print("\n--- RESAMPLED VOLUME CENTER ---")
    print("Z center:", new_z_center)
    print("Y center:", new_y_center)
    print("X center", new_x_center)

    resampled_coronal = resampled_volume[:,new_y_center,:]

    plt.imshow(
        resampled_coronal,
        cmap="gray",
        vmin=-160,
        vmax=240
    )

    plt.title("Coronal CT - Resampled to 1mm")
    plt.axis("off")
    plt.show()

    #Notice we are not writing aspect=... anymore. Thats because the new volume is approx 1mm,1mm,1mm
    #The voxel are isotropic now

    #-----------------------------
    #MPR FROM RESAMPLED VOLUME
    #--------------------------------------

    resampled_axial = resampled_volume[new_z_center,:,:]
    resampled_coronal = resampled_volume[:, new_y_center, :]
    resampled_sagittal = resampled_volume[:,:,new_x_center]

    plt.imshow(
        resampled_sagittal,
        cmap="gray",
        vmin=-160,
        vmax=240

    )

    plt.title("Sagittal CT - Resampled to 1mm")
    plt.axis("off")
    plt.show()

    #DICOM spacing -> Physical geometry -> resampling _> stardardized voxel grid

    #Step 16- HU Based on Tissue Segmentation

    #-------------------------
    # BONE SEGMENTATION
    #----------------------------   

    #Define a smiple HU threshold for bone

    bone_threshold = 300

    #Every voxel >= 300 HU becomes True
    #Every voxel < 300 HU becomes False

    bone_mask = resampled_volume >= bone_threshold

    print("\n--- BONE SEGMENTATION ---")
    print("Bone threshold:", bone_threshold, "HU")
    print("Mask shape:", bone_mask.shape)
    print("Mask data type", bone_mask.dtype)

    #---------------------------
    #DISPLAY BONE MASK
    #---------------------------

    #Extract the middle slice from the 3D bone mask
    bone_axial = bone_mask[new_z_center, :, :]

    plt.imshow(
        bone_axial,
        cmap="gray"
    )

    plt.title("Bone Segmentation Mask - Axial")
    plt.axis("off")
    plt.show()

    # --------------------------------
    # BONE SEGMENTATION OVERLAY
    # --------------------------------

    resampled_axial = resampled_volume[new_z_center, :, :]

    # Display the CT first
    plt.imshow(
        resampled_axial,
        cmap="gray",
        vmin=-160,
        vmax=240
    )
    # Create a red overlay with transparency
    red_overlay = np.zeros(
        (*bone_axial.shape, 4),
        dtype=np.float32
    )

    # Set red channel to maximum
    red_overlay[bone_axial, 0] = 1.0

    # Set transparency only for bone pixels
    red_overlay[bone_axial, 3] = 0.5

    plt.imshow(red_overlay)

    # # Hide pixels that are not classified as bone
    # bone_overlay = np.ma.masked_where(~bone_axial, bone_axial)

    # plt.imshow(
    #     bone_overlay,
    #     cmap="Reds",
    #     alpha=0.5
# )

    plt.title("CT with Bone Segmentation Overlay")
    plt.axis("off")
    plt.show()

    #--------------------------------
    # Step 17 - CONNECTED COMPONENTS
    #--------------------------------

    labeled_bone, number_of_components = label(bone_axial)

    print("\n--- CONNECTED COMPONENTS ---")
    print("Number of connected components:", number_of_components)
    print("Labeled image shape:", labeled_bone.shape)

    #--------------------------------------------
    # STEP 17B - COMPONENT SIZES    
    #--------------------------------------------

    #Count pixels belonging to each label
    component_sizes = np.bincount(labeled_bone.ravel())

    #Exclude background (label 0)
    bone_component_sizes = component_sizes[1:]

    print("n\--- COMPONENT SIZES ---")

    for component_id, size in enumerate(bone_component_sizes, start = 1):
        print(f"Component {component_id}:{size} pixels")

    #Understanding the code
        # ravel() converts your 390 × 390 labeled image into a one-dimensional array.
        # np.bincount() counts how many times each component label appears.
        # component_sizes[1:] excludes label 0, which represents the background.
        # enumerate(..., start=1) prints each component's number alongside its pixel count.


    #-----------------------------------
    #STEP 17C - REMOVE SAMLL COMPONENTS
    #-----------------------------------

    #Minimum number of pixels required
    min_component_size = 10

    #Identity componenets that are large enough
    valid_components = np.where(
        component_sizes >= min_component_size
    )[0]

    # Exclude background label 0
    valid_components = valid_components[valid_components != 0]

    #Create cleaned binary mask
    cleaned_bone_mask = np.isin(
        labeled_bone,
        valid_components
    )

    print("\n--- CLEANED BONE MASK ---")
    print("Original components:", number_of_components)
    print("Remaining components:", len(valid_components))
    print("Cleaned mask shape:", cleaned_bone_mask.shape)

#Understanding the new functions
#np.where() identifies labels whose pixel counts meet our minimum threshold.
#valid_components stores the labels we want to retain.
#np.isin() checks whether each pixel belongs to one of those retained labels.
#cleaned_bone_mask becomes a new Boolean array with the small components removed.


#----------------------------------------
#STEP 17D - COMPARE SEGMENTATION
#-----------------------------------------

    plt.figure(figsize=(12,5))

    #Original bone segmentation
    plt.subplot(1,2,1)
    plt.imshow(bone_axial, cmap="gray")
    plt.title("Original Bone Mask")
    plt.axis("off")

    #Cleaned bone segmentation
    plt.subplot(1,2,2)
    plt.imshow(cleaned_bone_mask,cmap="gray")
    plt.title("Cleaned Bone Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Understanding the code
# - plt.figure(figsize=(12, 5)) creates a figure large enough to display two images.
# - plt.subplot(1, 2, 1) selects the first panel in a layout with one row and two columns.
# - plt.subplot(1, 2, 2) selects the second panel.
# - plt.tight_layout() adjusts spacing so the images and titles fit properly.

#------------------------------------------
#STEP 18 - BONE AREA MEASUREMENT
#-----------------------------------------

#Count pixels classified as bone
    bone_pixel_count = np.count_nonzero(cleaned_bone_mask)

    #Calculate the area of one resampled pixel
    pixel_area_mm2 = new_spacing * new_spacing

    #Calculate total segmented area
    bone_area_mm2 = bone_pixel_count * pixel_area_mm2

    #convert mm2 to cm2
    bone_area_cm2 = bone_area_mm2 / 100

    print("\n--- BONE AREA MEASUREMENT ---")
    print("Bone pixels:", bone_pixel_count)
    print("Bone area (mm²):", bone_area_mm2)
    print("Bone area (cm²):", bone_area_cm2)

    # --------------------------------
    # STEP 18B - 3D BONE VOLUME
    # --------------------------------\

    bone_voxel_count = np.count_nonzero(bone_mask)

    #Calculate the physcial volume of one voxel
    voxel_volume_mm3 = new_spacing ** 3

    #Calculate total segmeneted volume
    bone_volume_mm3 = bone_voxel_count * voxel_volume_mm3

    #Convert cubic millimeters to cubic centimeters
    bone_volume_cm3 = bone_volume_mm3 / 1000

    print("\n--- 3D BONE VOLUME ---")
    print("Segmented voxels:", bone_voxel_count)
    print("Voxel volume (mm³):", voxel_volume_mm3)
    print("Segmented volume (mm³):", bone_volume_mm3)
    print("Segmented volume (cm³):", bone_volume_cm3)

    #Understanding the calculation
# In 2D, we multiply by pixel area:
    # A = N{pixels} * S{x}  * S_{y}

# In 3D, we multiply by voxel volume:
    # V = N{voxels} * S{x}  * S{y} * S{z}

#Because we resampled to approximately 1 mm isotropic spacing, 
# each voxel represents approximately 1 mm³.

#-------------------------------------------
#Step 19B - MARCHING CUBES
#------------------------------------------

#Convery Boolean mask to numeric values
    bone_volume = bone_mask.astype(np.uint8)

    #Generate a 3D surface mesh
    vertices, faces, normals, values = marching_cubes(
        bone_volume,
        level=0.5,
        spacing=(new_spacing, new_spacing,new_spacing)
    )

    print("\n--- 3D MESH INFORMATION ---")
    print("Vertices shape:", vertices.shape)
    print("Faces shape:", faces.shape)
    print("Number of vertices:", len(vertices))
    print("Number of triangles:", len(faces))

#Important: We're generating a mesh from the full threshold-based mask, 
# not the cleaned 2D slice. The resulting surface may 
# contain disconnected structures or artifacts. That's fine for this learning stage


#Select every 20th triangle for faster visualization
    sampled_faces = faces[::20]

# Use the complete mesh
    # triangles = vertices[faces]

    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection="3d")

    # mesh = Poly3DCollection(
    #     triangles[:, :, [2, 1, 0]],
    #     alpha=1.0
    # )

    # mesh.set_facecolor("lightgray")
    # mesh.set_edgecolor("none")

    # ax.add_collection3d(mesh)

    # ax.set_xlim(vertices[:, 2].min(), vertices[:, 2].max())
    # ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    # ax.set_zlim(vertices[:, 0].min(), vertices[:, 0].max())

    # ax.set_xlabel("X (mm)")
    # ax.set_ylabel("Y (mm)")
    # ax.set_zlabel("Z (mm)")

    # ax.set_box_aspect((
    #     np.ptp(vertices[:, 2]),
    #     np.ptp(vertices[:, 1]),
    #     np.ptp(vertices[:, 0])
    # ))

    # ax.view_init(elev=20, azim=60)

    # ax.set_title("3D CT Bone Surface Reconstruction")

    # plt.show()

    # --------------------------------
# STEP 20D - PYVISTA 3D VISUALIZATION
# --------------------------------

# Convert vertices from (Z, Y, X) to (X, Y, Z)
    vertices_xyz = vertices[:, [2, 1, 0]]

    # PyVista requires each triangle to begin with 3
    face_data = np.hstack(
        (
            np.full((len(faces), 1), 3),
            faces
        )
    ).astype(np.int64)

    # Create a 3D surface mesh
    bone_mesh = pv.PolyData(
        vertices_xyz,
        face_data
    )

    print("\n--- PYVISTA MESH ---")
    print("Number of points:", bone_mesh.n_points)
    print("Number of faces:", bone_mesh.n_cells)

    # Display the mesh
    plotter = pv.Plotter()

    plotter.add_mesh(
        bone_mesh,
        color="ivory",
        smooth_shading=False
    )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

    # --------------------------------
    # STEP 21 - EXPORT 3D MODEL
    # --------------------------------

    #Create output directory
    output_folder = Path("results")
    output_folder.mkdir(parents=True, exist_ok=True)

    #Define STL file path
    stl_path = output_folder  / "ct_bone_reconstruction.stl"

    #save the complete 3D mesh
    bone_mesh.save(stl_path)

    print("\n--- STL EXPORT ---")
    print("STL file saved to:", stl_path.resolve())
    print("Number of triangles exported:", bone_mesh.n_cells)

    #Understanding the code
    # - Path("results") selects your project's results folder.
    # - mkdir(parents=True, exist_ok=True) ensures the folder exists.
    # - bone_mesh.save(stl_path) exports the complete triangular mesh in STL format.
    # - stl_path.resolve() displays the absolute file location.

    #Important: STL stores surface geometry but does not preserve the original
    #  CT intensities, DICOM metadata, or patient-space origin. 
    # This is a visualization/research output, not a clinically validated model.

#What the code does

#faces[::20] selects every 20th triangle, reducing the plotting workload. \
# This is only for visualization; your original mesh remains unchanged.
#vertices[sampled_faces] retrieves the coordinates of each triangle's three corners.
#Poly3DCollection constructs a collection of 3D polygons.
#triangles[:, :, [2, 1, 0]] changes the coordinate order from (Z, Y, X) to (X, Y, Z), matching our plot axes.
#set_box_aspect() keeps the displayed dimensions proportional.


# --------------------------------
# STEP 22A - SAVE PROCESSED DATA
# --------------------------------

#Create processed data folder
    processed_folder = Path("data/processed")
    processed_folder.mkdir(parents=True, exist_ok=True)

    #Save the resampled CT volume
    np.save(
        processed_folder / "ct_volume_1mm.npy",
        resampled_volume
    )
    np.save(
        processed_folder / "bone_mask_3d.npy",
        bone_mask
    )
    print("\n--- PROCESSED DATA SAVED ---")
    print("CT volume shape:", resampled_volume.shape)
    print("Bone mask shape:", bone_mask.shape)
    print("Files saved to:", processed_folder.resolve())

#What are we saving?
    # File	Contents
    # ct_volume_1mm.npy	Your 3D CT volume containing HU values
#   bone_mask_3d.npy	Your 3D Boolean segmentation mask

#Save Your Analysis Results
# Next, we'll save your measurements in a JSON file. JSON is commonly used in software 
# engineering and medical imaging pipelines to store structured information.

# --------------------------------
# STEP 22B - SAVE ANALYSIS RESULTS
# --------------------------------

    analysis_results = {
        "modality": "CT",
        "original_volume_shape": list(volume.shape),
        "resampled_volume_shape": list(resampled_volume.shape),
        "target_spacing_mm": new_spacing,
        "bone_threshold_hu": bone_threshold,
        "segmented_voxels": int(bone_voxel_count),
        "segmented_volume_cm3": float(bone_volume_cm3),
        "mesh_vertices": int(len(vertices)),
        "mesh_triangles": int(len(faces))
    }

    results_path = output_folder / "analysis_results.json"

    with open(results_path, "w") as file:
        json.dump(analysis_results, file, indent=4)

    print("\n--- ANALYSIS RESULTS SAVED ---")
    print("JSON file:", results_path.resolve())

#Understanding the new code
    # - analysis_results is a Python dictionary containing your measurements.
    # - list() converts NumPy shape tuples into JSON-compatible lists.
    # - int() and float() ensure the measurements are stored as standard Python numbers.
    # - json.dump() writes the dictionary to a file.
    # - indent=4 makes the JSON easy to read.

    #One important detail: this JSON records the target spacing, 
    # but not the full DICOM spatial geometry. That's sufficient for our current 
    # educational summary, not for clinical image registration.

#RUN: python -m pip freeze > requirements.txt. records the packages installed in your virtual environment, including their versions.



    # print("Medical Imaging AI Project")
    # print("Environment is working!")

if __name__ == "__main__":
    main()

