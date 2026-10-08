Medical Imaging CT Processing and 3D Bone Reconstruction

Project Overview

This project demonstrates an end-to-end medical image processing pipeline using Python. It transforms a series of CT DICOM images into a three-dimensional volume, performs basic bone segmentation using Hounsfield Units (HU), calculates quantitative measurements, and generates a 3D surface reconstruction.

The project combines medical imaging principles, digital image processing, and scientific computing.

Objectives

Load and interpret CT DICOM images and metadata.

Convert stored pixel values into Hounsfield Units.

Construct a 3D CT volume from axial slices.

Generate axial, coronal, and sagittal reconstructions.

Correct for anisotropic voxel spacing.

Resample CT data to approximately 1 mm isotropic resolution.

Perform threshold-based bone segmentation.

Apply connected-component labeling and cleanup to a 2D slice.

Calculate segmented area and volume.

Generate a 3D surface mesh using Marching Cubes.

Visualize the reconstruction interactively using PyVista.

Export the mesh to STL format.

Dataset

Source: The Cancer Imaging Archive (TCIA)
Collection: CPTAC-CCRCC
Imaging modality: CT
Series: KIDNEYS WO
Number of DICOM images: 120

The project uses publicly available, de-identified imaging data. The original DICOM files are not included in this repository.

Technologies

Python

NumPy

pydicom

SciPy

Matplotlib

scikit-image

PyVista / VTK

Processing Pipeline

DICOM ingestion: Read DICOM images and metadata.

HU conversion: Apply rescale slope and intercept.

Volume reconstruction: Sort slices by position and stack them into a 3D NumPy array.

Multiplanar reconstruction: Generate axial, coronal, and sagittal views.

Voxel resampling: Resample the CT volume to approximately 1 mm isotropic spacing.

Bone segmentation: Apply a threshold of 300 HU.

Connected components: Label regions and remove small components from a representative 2D slice.

Quantitative analysis: Calculate segmented area and 3D volume.

Surface reconstruction: Generate a triangular mesh using Marching Cubes.

3D visualization and export: Display the model with PyVista and save an STL file.

Results

Measurement

Result

Original CT volume

120 × 512 × 512

Resampled CT volume

300 × 390 × 390

Segmentation threshold

300 HU

Cleaned 2D segmented area

10.34 cm²

3D threshold-segmented volume

470.998 cm³

Mesh vertices

503,403

Mesh triangles

1,010,900

The 2D cleanup was performed on a single axial slice. The 3D volume and mesh were generated from the full threshold-based mask without equivalent 3D cleanup.

These measurements describe threshold-selected regions, not clinically validated anatomical bone volumes.

How to Run

Clone the repository.

Create and activate a Python virtual environment.

Install dependencies using pip install -r requirements.txt.

Obtain the appropriate DICOM dataset from TCIA and place the files inside data/raw/.

Run python main.py.

The program generates visualization windows and saves processed NumPy arrays, an STL model, and quantitative results.

Limitations and Future Improvements

Threshold-based segmentation can include non-bone high-density structures.

The 2D connected-component cleanup does not automatically apply to the full 3D mask.

Resampling introduces interpolated voxel values.

Full DICOM patient-space geometry is not preserved in the exported STL or NumPy arrays.

Future improvements could include 3D connected-component filtering, mesh smoothing, anatomical segmentation, and deep-learning approaches such as U-Net.

Educational Purpose

This project is intended for medical imaging software development and educational research. It is not validated for clinical diagnosis, treatment planning, or patient care.