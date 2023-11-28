# adipose_pipeline
Pipeline for adipose tissues segmentation from dixon body scans

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

export SINGULARITY_DOCKER_USERNAME=username
export SINGULARITY_DOCKER_PASSWORD=password

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


