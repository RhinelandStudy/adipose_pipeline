# adipose_pipeline
This repository contains a Nipype wrapper for the FatSegNet tool available at [/Deep-MI/FatSegNet](https://github.com/Deep-MI/FatSegNet). FatSegNet is a automated tool for segmenting visceral and subcuteneous adipose tissue on fat images from a two-point Dixon sequence. 

If you use this wrapper please cite:

Estrada, Santiago, et al. "FatSegNet: A fully automated deep learning pipeline for adipose tissue segmentation on abdominal dixon MRI." Magnetic resonance in medicine 83.4 (2020): 1471-1483. [https:// doi.org/10.1002/mrm.28022](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28022)
```
@article{estrada2020fatsegnet,
  title={FatSegNet: A fully automated deep learning pipeline for adipose tissue segmentation on abdominal dixon MRI},
  author={Estrada, Santiago and Lu, Ran and Conjeti, Sailesh and Orozco-Ruiz, Ximena and Panos-Willuhn, Joana and Breteler, Monique MB and Reuter, Martin},
  journal={Magnetic resonance in medicine},
  volume={83},
  number={4},
  pages={1471--1483},
  year={2020},
  publisher={Wiley Online Library}
}
```

## Build docker image

```bash

docker build -t adipose_pipeline -f docker/Dockerfile .

```

## Or pull from docker hub

```bash

docker pull dznerheinlandstudie/rheinlandstudie:adipose_pipeline

```

## Run pipeline:

### Using docker

```bash

docker run --rm -v /path/to/inputdata:/input \
                -v /path/to/work:/work \
                -v /path/to//output:/output \
             dznerheinlandstudie/rheinlandstudie:adipose_pipeline \
             run_adipose_pipeline \
                -s /input \
                -w /work \
                -o /output -p 4 -t 2

```

The command line options are described briefly if the pipeline is started with only -h option.

### Using Singularity

The pipeline can be run with Singularity by running the singularity image as follows:

```bash


singularity build adipose_pipeline.sif docker://dznerheinlandstudie/rheinlandstudie:adipose_pipeline

```

When the singularit image is created, then it can be run as follows:


```bash

singularity run  -B /path/to/inputdata:/input \
                 -B /path/to/work:/work \
                 -B /path/to/output:/output \
            adipose_pipeline.sif \
            run_adipose_pipeline \ 
                      -s /input \
                      -w /work \
                      -o /output \ 
                      -p 4 -t 2
```


