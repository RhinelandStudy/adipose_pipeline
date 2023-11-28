FROM centos:centos7.9.2009

#----------------------------
# Install common dependencies
#----------------------------
RUN yum install -y -q bzip2 ca-certificates curl unzip bc dc gcc mono-mcs libX11\
    && yum clean packages \
    && rm -rf /var/cache/yum/* /tmp/* /var/tmp/*


#-------------------------------------------------
# Set paths and evn vars 
#-------------------------------------------------
ENV PATH=/opt/miniconda/envs/adipy3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia:/usr/local/cuda:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/lib:/lib64:$LD_LIBRARY_PATH \
    STARTUP="/opt/startup.sh"

COPY ./src /opt/
 
#-------------------------------------------------
# Install Miniconda, and set up Python environment
#-------------------------------------------------
RUN echo "Installing Miniconda and pipeline ..." \
    && curl -sSL -o miniconda.sh http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/miniconda \
    && rm -f miniconda.sh \
    && /opt/miniconda/bin/conda config --add channels conda-forge \
    && /opt/miniconda/bin/conda update -n base -c defaults conda \
    && /opt/miniconda/bin/conda create -y -n default python=3.5.5 \
    && source /opt/miniconda/bin/activate default \
    && conda clean -y --all \
    && pip install -U --no-cache-dir pip \
    && cd /opt \
    && pip install -r requirements.txt \
    && pip install -e .


RUN echo "Setting up entry point script..." \
    && if [ ! -f "$STARTUP" ]; then \
       echo '#!/usr/bin/env bash' >> $STARTUP; \
       echo 'echo "startup: setting environment"...' >> $STARTUP; \
       echo 'source /docker_environment' >> $STARTUP; \
       echo 'export LD_LIBRARY_PATH=/.singularity.d/libs:/lib64:$LD_LIBRARY_PATH' >> $STARTUP; \
       echo 'source /opt/miniconda/bin/activate default' >> $STARTUP; \
       echo 'IFS=";" read -ra COMMANDS <<< "$@"' >> $STARTUP; \
       echo 'for command in "${COMMANDS[@]}";do' >> $STARTUP; \
       echo ' echo "`date` : executing ${command}";' >> $STARTUP; \
       echo '   ${command}' >> $STARTUP; \
       echo 'done' >> $STARTUP; \
       fi \
    && chmod 755 $STARTUP 

ENTRYPOINT ["/opt/startup.sh"]
