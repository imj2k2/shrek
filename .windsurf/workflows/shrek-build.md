---
description: workflow to instruct windsurf on steps to follow when it introduces change in shrek repo
---

- rebuild the containers automatically post change - use ./run_docker --rebuild command
- run ./run_docker.sh --logs > debug.log to gather docker logs in background
- execute postman tests to validate functional tests are passing post rebuild, use "newman run ../shrek_api_postman_collection_fixed.json -r htmlextra" inside test folder
- review docker logs in debug.log for errors and fix any issues identified