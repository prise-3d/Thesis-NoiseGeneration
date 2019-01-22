FROM pytorch/pytorch

# Update and add usefull package
RUN apt-get update
RUN apt-get install -y libglib2.0-dev libsm-dev libxrender-dev libxext-dev 

# Install JupyterLab 
RUN pip install jupyterlab && jupyter serverextension enable --py jupyterlab

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV LANG=C.UTF-8

# Expose Jupyter port & cmd 
EXPOSE 8888 
EXPOSE 6006

CMD jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/data --allow-root
