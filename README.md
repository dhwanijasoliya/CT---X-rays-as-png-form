
## Overview

This project involves the use of Plastimatch to generate Digitally Reconstructed Radiographs (DRRs) from CT images. DRRs are synthetic radiographs used as reference images for verifying the correct setup position of a patient before radiation treatment. 

## DRR File Description

A DRR is a synthetic radiograph created from a CT scan, used in radiation treatment planning to ensure accurate patient positioning. The DRR is generated using the command-line tool provided by Plastimatch, which takes an input CT image in MHA format and produces output images in formats like PGM, PFM, or RAW.

### Basic DRR Command Usage

```bash
drr [options] [infile]
```

### Options

- **-A hardware**: Choose threading mode, "cpu" or "cuda". Default is "cpu".
- **-a num**: Generate a specific number of equally spaced angles.
- **-N angle**: Set the angle difference between neighboring images.
- **-nrm "x y z"**: Set the normal vector for the imaging panel.
- **-vup "x y z"**: Set the upward vector for the imaging panel.
- **-g "sad sid"**: Set the Source-Axis Distance (SAD) and Source-Image Distance (SID) in millimeters.
- **-r "r c"**: Set the output resolution in pixels.
- **-s scale**: Scale the intensity of the output.
- **-e**: Apply exponential mapping to output values.
- **-c "r c"**: Set the image center in pixels.
- **-z "s1 s2"**: Define the physical size of the imager in millimeters.
- **-w "r1 r2 c1 c2"**: Generate an image only for specified pixels in the window.
- **-t outformat**: Select output format: PGM, PFM, or RAW.
- **-i algorithm**: Choose the algorithm {exact, uniform}.
- **-o "o1 o2 o3"**: Set the isocenter position.
- **-I infile**: Specify the input file in MHA format.
- **-O outprefix**: Define the prefix for generated output files.

### Modes

- **Single Image Mode**: In this mode, you specify the geometry for a single image.
- **Rotational Mode**: Generates multiple images by rotating the source and imaging panel around the isocenter.

### Example Commands

- **Single Image Mode Example**:

```bash
drr -nrm "1 0 0" \
    -vup "0 0 1" \
    -g "1000 1500" \
    -r "1024 768" \
    -z "400 300" \
    -c "383.5 511.5" \
    -o "0 -20 -50" \
    input_file.mha
```

- **Rotational Mode Example**:

```bash
drr -N 20 \
    -a 18 \
    -g "1000 1500" \
    -r "1024 768" \
    -z "400 300" \
    -o "0 -20 -50" \
    input_file.mha
```

## Additional Guidelines for DRR Conversion

### Using the Plastimatch Command with Python

A Python-based approach can be used to automate and customize the DRR generation process. Below is an example command extracted from the Python file:

```bash
plastimatch drr -t pfm --algorithm uniform --gantry-angle 0 -N 90 -a 2 \
--sad 541 --sid 949 --autoscale --autoscale-range "0 255" -r "320 320" \
-o "-2.499603271484375 10.781600952148438 1122.805908203125" -z "624.0 624.0" \
-P preprocess -I <mha file> -O <out dir>
```

### Converting PFM to PNG Using OpenCV

If your output format is PFM, you can convert it to PNG using OpenCV. Here's how to do it:

```bash
python3 -c "import cv2, sys; read = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE); \
read = (read / read.max() * 255).astype('uint8'); out=sys.argv[1].rsplit('.')[0] + '.png'; \
cv2.imwrite(out, read); print('Saved', sys.argv[1], 'to', out)" <pfm file>
```

The provided Python program can also automatically convert PFM files to PNG using a similar method.
