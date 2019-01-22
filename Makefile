current_dir := $(shell pwd)

build: 
	sudo docker build . --tag=jupyterlab-gan

run:
	sudo docker run -it -p 8888:8888 -p 6006:6006 -v $(current_dir):/data --name=noise jupyterlab-gan

clean:
	sudo docker container rm noise
