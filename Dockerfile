FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda update --quiet --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN pip install altair_ally==0.1.1 deepchecks==0.19.1 pandera==0.27.0 vegafusion==2.0.3 vegafusion-python-embed==1.6.9 vl-convert-python==1.8.0 matplotlib==3.10.7
