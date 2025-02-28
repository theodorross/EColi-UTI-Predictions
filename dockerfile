FROM mambaorg/micromamba
RUN micromamba install -c conda-forge altair=5.4.0 matplotlib=3.9.2 numpy=2.1.3 scipy=1.14.1 openpyxl=3.1.5 pandas=2.2.2 scikit-learn=1.5.1 vl-convert-python=1.7.0 xgboost=2.0.3 -y
ENTRYPOINT [ "/opt/conda/bin/python" ]
