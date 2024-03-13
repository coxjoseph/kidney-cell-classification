# Kidney Cell Classification
EEL 4938/5840 Class Project

## Project Overview
This project aims to develop an automated system for segmenting and classifying different cells within a histology 
kidney slice. 

## Group Members

- [Joseph Cox](#https://github.com/coxjoseph)
- [Andrea McPherson](#https://github.com/andreamm3)
- [Dylan Ogrodowsky](#https://github.com/)
- [Veronica Ramos](#https://github.com/VeronicaR-UF)

For any inquiries or further information, please contact [cox.j@ufl.edu](mailto:cox.j@ufl.edu).

## Installation

### Prerequisites
Before installation, ensure you have the following:
- Python 3.12 or later
- pip package manager

In addition, we recommend having enough RAM to open both your CODEX file and your Stained Image file
at the same time. While the classifier will still function without this, performance will be significantly
impacted, and output results will be worse. 

### Dependencies
#### Setting Up a Virtual Environment

We recommend using a virtual environment to manage the dependencies for your project. This helps to avoid potential 
conflicts with other Python packages you may have installed. To set up a virtual environment, run the following 
commands in your terminal to create an environment called `venv`:

```bash
python -m venv venv
```

and then activating the virtual environment using 
```ps1
venv\Scripts\activate
```
on Windows or 

```bash
source venv/bin/activate
```

on MacOS or Unix-like systems. 

#### Installing Dependencies

Once your virtual environment is activated, install all necessary dependencies by running:
```bash
pip install -r requirements.text
```
which will install all packages listed in `requirements.txt`.

### Preparing the Dataset
Ensure your dataset contains both a codex file and a Hematoxylin and Eosin (H&E) stained image, both in the `tiff` file 
format. 

### Running the Classifier
To start the classification process, run:

```bash
python classify.py --codex /path/to/codex.tiff --he /path/to/he_stained_image.tiff
```

Replace `/path/to/codex.tiff` and `/path/to/he_stained_image.tiff` with the actual path to your files. This script will 
segment, classify, and save the classification labels to a tiff file `classified_stain.tif`. Optionally, this output
file can be set using the `--output` or `-o` flag in the classification script.

## Contributing
We welcome contributions! Please read `CONTRIBUTING.md` for guidelines on how to submit contributions to this project.

## License
This project is licensed under the GPLv3 License - see the `LICENSE` file for details.

