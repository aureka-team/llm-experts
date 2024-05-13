.PHONY: devcontainer-build


devcontainer-build:
	docker compose -f .devcontainer/docker-compose.yml build llm-experts-devcontainer


redis-start:
	docker compose -f .devcontainer/docker-compose.yml up -d llm-experts-redis

redis-stop:
	docker compose -f .devcontainer/docker-compose.yml stop llm-experts-redis

redis-flush:
	docker compose -f .devcontainer/docker-compose.yml exec llm-experts-redis redis-cli FLUSHALL
