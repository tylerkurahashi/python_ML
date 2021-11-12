FROM jupyter/datascience-notebook:python-3.9.7

WORKDIR /workspace

# RUN conda create -n torch python=3.9
RUN conda install -c conda-forge notebook \
  pytorch==1.10 \
  scikit-learn==1.0.1 \
  seaborn==0.11.2

COPY . /workspace

ENTRYPOINT ["jupyter", "notebook", "--allow-root"]