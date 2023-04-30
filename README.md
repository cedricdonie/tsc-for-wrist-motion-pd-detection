# Time Series Classification for Detecting Parkinson's Disease from Wrist Motions
This is the source code for the paper titled _Time Series Classification for Detecting Parkinson's Disease from Wrist Motions_<sup>[[arXiv]](https://doi.org/10.48550/arXiv.2304.11265)</sup>.

> **Abstract**
>
> Parkinson’s disease (PD) is a neurodegenerative disease with frequently changing motor symptoms
> where continuous symptom monitoring enables more targeted treatment. Classical time series classification (TSC)
> and deep learning techniques have limited performance
> for PD symptom monitoring using wearable accelerometer data because PD movement patterns are complex, but
> datasets are small. We investigate InceptionTime and RandOm Convolutional KErnel Transform (ROCKET) because
> they are state-of-the-art for TSC and promising for PD
> symptom monitoring: InceptionTime’s high learning capacity is suited to modeling complex movement patterns while
> ROCKET is suited to small datasets. We used a random
> search to find the highest-scoring InceptionTime architecture and compared it to ROCKET with a ridge classifier
> and a multi-layer perceptron (MLP) on wrist motions of
> PD patients. We find that all approaches are suitable for
> estimating tremor severity and bradykinesia presence but
> struggle with detecting dyskinesia. ROCKET performs better for dyskinesia, whereas InceptionTime is slightly better
> for tremor and bradykinesia but has much higher variability
> in performance. Both outperform the MLP. In conclusion,
> both InceptionTime and ROCKET are suitable for continuous symptom monitoring, with the choice depending on the
> symptom of interest and desired robustness.

## Usage

Use `make help` to get an overview of options available.

### Getting Started
To get started, create a virtual environment (venv) with python 3.7.
You can use `make create_enviroment` for this.
To install the required packages, activate the environment and run `make requirements`.
Black is used for autoformatting, but is not project dependency; hence it is installed via the system package manager.
```
sudo apt-get install -y black
```

To build a single file without recreating any dependencies, you can sometimes run
```
source <(make --dry-run <target> | tail -n1)
```

### Data Download
Due to the usage restrictions, every user must download the files from Synapse, but the Makefile takes care of this provided you have the credentials configured.
You must set the enviroment variables `SYNAPSE_USERNAME` and `SYNAPSE_PASSWORD` to download the raw data.
The easiest method to set these environment variables is to create a `.env` file in the project root with the following contents.

```
SYNAPSE_USERNAME=<username>
SYNAPSE_PASSWORD=<password>
```

## Project Structure
This project follows a standard structure.
If you want to see how a file is created, just run `make -Bnd <filename>`.

------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    |                         Under <model>.pred/<dataset>.npz, the predictions of <model> on
    |                         <dataset> can be found.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project. See the Makefile for structure.
    │   ├── __init__.py    <- Makes src a Python module
    └── test               <- Scripts for testing the code (in addition to doctest)


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
