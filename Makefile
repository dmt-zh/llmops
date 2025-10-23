ifeq (,$(wildcard .env))
$(error .env file is missing. Please create one based on .env.example. Run: "cp .env.example .env" and fill in the missing values.)
endif

include .env

# --- Default Values ---
CHECK_DIRS := .


# --- Run ---
help: # Display help message with available commands
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | while read -r l; do printf "\033[1;33m› $$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

llama-server-up: # Start inference of a local LLM with llama.cpp server
	@nohup llama-server -m ~/.llama.cpp/models/"$(LLM_MODEL_NAME)" --jinja --no-webui --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift > llama-server.log 2>&1 &
	@echo "Started LLM server with \""$(LLM_MODEL_NAME)"\""

llama-server-down: # Terminate llama.cpp server
	@pkill -f llama-server

services-up: # Start local services using Docker Compose
	@docker compose --env-file ./.env -f services/docker-compose.yml up -d

services-down: # Stop and remove local services
	@docker compose --env-file ./.env -f services/docker-compose.yml down

lint-check: # Check code for linting issues without making changes
	@uv run ruff check $(CHECK_DIRS)

