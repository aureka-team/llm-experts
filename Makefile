.PHONY: devcontainer-build


devcontainer-build:
	[ -e .secrets/.env ] || touch .secrets/.env
	docker compose -f .devcontainer/docker-compose.yml build llm-experts-devcontainer


redis-start:
	docker compose -f .devcontainer/docker-compose.yml up -d llm-experts-redis

redis-stop:
	docker compose -f .devcontainer/docker-compose.yml stop llm-experts-redis

redis-flush:
	docker compose -f .devcontainer/docker-compose.yml exec llm-experts-redis redis-cli FLUSHALL


mongo-start:
	docker compose -f .devcontainer/docker-compose.yml up -d llm-experts-mongo

mongo-stop:
	docker compose -f .devcontainer/docker-compose.yml stop llm-experts-mongo

mongo-flush: mongo-stop
	$(info *** WARNING you are deleting all data from mongodb ***)
	sudo rm -r resources/db/mongo
	docker compose -f .devcontainer/docker-compose.yml up -d llm-experts-mongo
