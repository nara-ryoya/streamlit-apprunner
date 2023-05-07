.PHONY: setup
setup:
	mkdir .torch_wheels && \
	cd .torch_wheels && \
	wget https://files.pythonhosted.org/packages/b6/b1/f562cb533751c272d23f605858cd17d6a6c50fa8cd3c1f99539e2acd359f/torch-2.0.0-cp310-cp310-manylinux1_x86_64.whl && \
	wget https://files.pythonhosted.org/packages/34/8d/43f36b6ff585de3ebd5df10ef07c4377b2cce9155f9810c7382427c1430e/torchvision-0.15.1-cp310-cp310-manylinux1_x86_64.whl && \
	cd ..