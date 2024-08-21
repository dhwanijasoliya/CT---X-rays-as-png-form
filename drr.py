# ------------------------------------------------------------------------------
# Ref: https://github.com/kylekma/X2CT/blob/master/CT2XRAY/xraypro.py
# modified to create multiple xray views
# plastimatch used v 1.9.3 - download: http://plastimatch.org/
#
# Generate xrays (Digitally Reconstructed Radiographs - DRR)
# from CT scan (e.g. dicom files .dcm / or raw .mha raw)
# 
# ------------------------------------------------------------------------------
import subprocess
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import tempfile
import logging as logging_module
from multiprocess_tqdm import MPtqdm
import multiprocessing
import SimpleITK as sitk

logging = logging_module.getLogger(__name__)

def load_scan_mhda(path: Path) -> tuple[sitk.Image, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Load a CT scan from a .mha file.

    Args:
        path (Path): The path to the .mha file.

    Returns:
        tuple[sitk.Image, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]: The CT scan, origin, size, and spacing.
    """
    img_itk = sitk.ReadImage(str(path))
    return img_itk, img_itk.GetOrigin(), img_itk.GetSize(), img_itk.GetSpacing()

def get_center(
        origin: tuple[float, float, float],
        size: tuple[float, float, float],
        spacing: tuple[float, float, float]
    ) -> torch.Tensor:
    """Compute the center of the CT scan in world coordinates.

    Args:
        origin (tuple[float, float, float]): The origin of the CT scan in world coordinates.
        size (tuple[float, float, float]): The size of the CT scan in world coordinates.
        spacing (tuple[float, float, float]): The spacing of the CT scan in world coordinates.

    Returns:
        torch.Tensor: The center of the CT scan in world coordinates.
    """
    origin = torch.tensor(origin)
    size = torch.tensor(size)
    spacing = torch.tensor(spacing)
    center = origin + (size - 1) / 2 * spacing
    return center

def savepng(input_file: Path, output_file: Path, direction: int):
    """Save a pfm file as a png file.

    Args:
        input_file (Path): The path to the pfm file.
        output_file (Path): The path to save the png file to.
        direction (int): The direction to save the png file in.

    Raises:
        Exception: If the pfm file is malformed.
    """
    with input_file.open('rb') as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode('ascii') == 'PF':
            color = True    
        elif header.decode('ascii') == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
        
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        raw_data = np.reshape(data, shape)
        max_value = raw_data.max()
        im = (raw_data / max_value * 255).astype(np.uint8)
        # PA view should do additional left-right flip
        if direction == 1:
            im = np.fliplr(im)
        
        # plt.imshow(im, cmap=plt.cm.gray)
        plt.imsave(output_file, im, cmap=plt.cm.gray)
        # plt.imsave saves an image with 32bit per pixel, but we only need one channel
        image = cv2.imread(str(output_file.absolute()))
        gray = cv2.split(image)[0]
        cv2.imwrite(str(output_file.absolute()), gray)

class Plastimatch:
    def __init__(self, base_command: str):
        self.base_command = base_command

    def torch2string(self, array: torch.Tensor) -> str:
        """Convert a torch.Tensor to a string that is readable by plastimatch.

        Args:
            array (torch.Tensor): The tensor to convert.

        Returns:
            str: The string representation of the tensor. Readible by plastimatch.
        """
        return "\"" + " ".join((str(element.item()) for element in array)) + "\""
    
    def adjust_hu(self, input_file: Path, output_file: Path):
        logging.debug('Running plastimatch adjust for %s.', input_file)
        # Use "500 500" for chest
        # truncates the inputs to the range of [-1000,+1000]
        adjust_lst = [
            self.base_command,
            "adjust",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--pw-linear", "-inf,0,-1000,-1000,+1000,+1000,inf,0"
        ]
        #"-inf,0,-1000,-1000,+1000,+1000,inf,0"
        output = subprocess.check_output(adjust_lst)
        # output.check_returncode()
    
    def drr(
            self,
            input_file: Path,
            output_dir: Path,
            num_xrays: str,
            angle: str,
            sad: str,
            sid: str,
            resolution: str,
            detector_size: str,
            bg_color: str,
            center: str,
            multiple_view_mode: bool = True,
            frontal_dir: bool = True
        ):
        logging.debug('Running plastimatch drr for %s.', input_file)
        logging.debug('Center is %s.', center)
        # Use "500 500" for chest
        # use "350 350" for knee
        # Black bg: "0 255", white bg: "255 0"
        # If single view, choose frontal or lateral view
        if multiple_view_mode:
            drr_lst = [
                self.base_command,
                "drr",
                "-t", "pfm",
                "--algorithm", "uniform",
                "--gantry-angle", "0", # The gantry angle of the machine. Defines the initial rotation.
                "-N", angle, # The angle between subsequent views
                "-a", num_xrays, # the  number of x-rays to be generated
                "--sad", sad, # The source to axis distance (source to patient)
                "--sid", sid, # The source to detector distance (source to detector)
                "--autoscale", # Automatically rescale intensity
                "--autoscale-range", bg_color, # The range of the intensity to autoscale in
                "-r", resolution, # The detector resolution in mm (row col)
                "-o", center, # Isocenter position "x y z" in DICOM coordinates (mm)
                "-z", detector_size, # The physical size of the detector in format (in mm)
                "-P", "preprocess", # Choose HU conversion type {preprocess,inline,none}
                "-I", input_file, # Input file
                "-O", str(output_dir) + "/" # Output directory
            ]
            output = subprocess.check_output(drr_lst)
            # output.check_returncode()
        else:
            if frontal_dir:
                dir = "\"0 1 0\""
            else:
                dir = "\"1 0 0\""
            drr_lst = [
                self.base_command,
                "drr",
                "-t", "pfm",
                "--algorithm", "uniform",
                "--gantry-angle", "0",
                "-n", dir, # Direction
                "--sad", sad,
                "--sid", sid,
                "--autoscale",
                "--autoscale-range", bg_color,
                "-r", resolution,
                "-o", center,
                "-z", detector_size,
                "-P", "preprocess",
                "-I", input_file,
                "-O", str(output_dir) + "/"
            ]
            output = subprocess.run(drr_lst)
            output.check_returncode()


def generate_drr_from_ct(
        input_path: Path,
        output_path: Path,
        num_xrays: int,
        size: int,
        plastimatch: Plastimatch,
        preprocessing: bool = True
):
    """Generate DRR images from a CT scan.

    Args:
        input_path (Path): The path to the CT scan.
        output_path (Path): The directory path to save the DRR images to.
        num_xrays (int): The number of images to generate.
        size (int): The size of the DRR images. Generates (size x size x size).
        plastimatch (Plastimatch): The path of the plastimatch executable.
        preprocessing (bool, optional): Whether to preprocess the CT. Defaults to True.
    """
    # Set some options
    multiple_view_mode = True
    # Use "500 500" for chest
    # use "350 350" for knee
    # detector size for 256x256x256 ct scan is 500
    # detector size for 128x128x128 ct scan is 250, as we CROP into the ct to generate it
    scaled_detector_size = 500 // (256 / size)
    detector_size = f"\"{scaled_detector_size} {scaled_detector_size}\"" # The physical size of the detector in format (in mm)
    # Black bg: "0 255", white bg: "255 0"
    bg_color = "\"0 255\""
    # If single view, choose frontal or lateral view
    frontal_dir = True
    resolution = f"\"{size} {size}\""
    # If multiple view:
    angle = str(round(180 / num_xrays) % 180)

    ct, ct_origin, ct_size, ct_spacing = load_scan_mhda(input_path)
    center = get_center(ct_origin, ct_size, ct_spacing)

    # Note that the following values are sometimes missing from the CT
    # if missing, use sad=541, sid=949 for chest
    # DistanceSourceToPatient in mm
    source_patient_distance = round(float(ct.GetMetaData('DistanceSourceToPatient')))
    sad = str(source_patient_distance if source_patient_distance != 0 else 541)
    # DistanceSourceToDetector in mm
    source_detector_distance = round(float(ct.GetMetaData('DistanceSourceToDetector')))
    sid = str(source_detector_distance if source_detector_distance != 0 else 949)
    with tempfile.TemporaryDirectory(dir=output_path) as tmpdir:
        tmpfile = Path(tmpdir) / 'preprocess.mha'
        if preprocessing:
            plastimatch.adjust_hu(input_path, tmpfile)

        plastimatch.drr(
            input_file=tmpfile if preprocessing else input_path,
            output_dir=output_path,
            num_xrays=str(num_xrays),
            angle=angle,
            sad=sad,
            sid=sid,
            resolution=resolution,
            detector_size=detector_size,
            bg_color=bg_color,
            center=plastimatch.torch2string(center),
            multiple_view_mode=multiple_view_mode,
            frontal_dir=frontal_dir
        )

def drr_image_from_ct(ct_file: Path, output_path: Path, num_xrays: int, size: str, no_preprocess: bool, plastimatch: Plastimatch, tmpdir: str) -> None:
    """Generate DRR images from a CT scan.

    Args:
        ct_file (Path): The path to the CT scan.
        output_path (Path): The path to save the DRR images to.
        num_xrays (int): The number of xrays to generate.
        size (str): The size of the DRR images.
        no_preprocess (bool): If True, do not preprocess the CT scan.
        plastimatch (Plastimatch): The plastimatch object.
        tmpdir (str): The temporary directory to store intermediate files.
    """
    file_name = ct_file.stem
    drr_output = output_path / file_name
    drr_output.mkdir(exist_ok=True, parents=True)
    if no_preprocess:
        logging.debug('No preprocessing.')
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir_output:
        generate_drr_from_ct(
            input_path=ct_file,
            output_path=tmpdir_output,
            num_xrays=num_xrays,
            size=size,
            plastimatch=plastimatch,
            preprocessing=not no_preprocess
        )
        tmpdir_output_path = Path(tmpdir_output)
        logging.debug('Saving pngs from %s to %s.', tmpdir_output_path, drr_output)
        for index, file in enumerate(sorted(tmpdir_output_path.glob('*.pfm'))):
            savepng(input_file=file, output_file=drr_output / (str(index) + '.png'), direction=1)

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input directory containing .mha files.')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save DRRs to.')
    parser.add_argument('--plastimatch', type=str, required=True, help='Path to plastimatch executable.')
    parser.add_argument('--num-xrays', type=int, default=2, help='Number of xrays to generate.')
    parser.add_argument('--size', type=int, default=320, help='Size of the DRRs.')
    parser.add_argument('--no-preprocess', action='store_true', help='Do not preprocess the CT scans.')
    parser.add_argument('--log', type=str, default='INFO', help='Logging level.')
    parser.add_argument('--logfile', type=str, default=None, help='Log file path.')
    parser.add_argument('--tmpdir', type=str, default=None, help='Temporary directory to store intermediate files. Defaults to system default.')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers to use.')
    args = parser.parse_args()
    handlers = [logging_module.StreamHandler()]
    if args.logfile:
        handlers.append(logging_module.FileHandler(args.logfile, mode='w'))
    logging_module.basicConfig(
        level=args.log.upper(),
        handlers=handlers,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )
    input_path = Path(args.input)
    output_path = Path(args.output)
    plastimatch_command = args.plastimatch
    num_xrays = args.num_xrays
    plastimatch = Plastimatch(plastimatch_command)
    with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmpdir:
        ct_files = list(input_path.glob('*.mha'))
        with multiprocessing.Pool(processes=args.workers) as pool:
            MPtqdm.starmap(
                pool,
                drr_image_from_ct,
                ((ct_file, output_path, num_xrays, args.size, args.no_preprocess, plastimatch, tmpdir) for ct_file in ct_files),
                leave=True,
                total=len(ct_files),
                description='Generating DRRs',
            )

if __name__ == '__main__':
    main()
