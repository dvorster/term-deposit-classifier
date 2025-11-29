FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda update --quiet --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN pip install deepchecks==0.19.1 altair-ally==0.1.1 vega-datasets==0.9.0 vegafusion==2.0.3 vl-convert-python==1.8.0
