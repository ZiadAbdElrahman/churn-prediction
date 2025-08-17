# Makefile — force-run on fixed port (kill-if-busy)

IMAGE ?= churn-service:latest
CONTAINER_NAME ?= churn-service
PORT ?= 8000

.PHONY: build run stop logs reup

build:
	docker build -t $(IMAGE) .

run:
	@echo ">> Removing existing container (if any): $(CONTAINER_NAME)"
	-@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@echo ">> Starting container $(CONTAINER_NAME) on http://localhost:$(PORT)"
	docker run --rm  --name $(CONTAINER_NAME) \
	  -p $(PORT):8000 \
	  -v $(PWD)/artifacts:/app/artifacts \
	  -v $(PWD)/data:/app/data \
	  -v $(PWD)/config:/app/config \
	  $(IMAGE)

# View live logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop the container
stop:
	-@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@echo ">> Stopped (if it was running)."

# Rebuild image and run fresh
reup: build run