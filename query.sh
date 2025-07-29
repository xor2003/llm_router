#!/usr/bin/bash
set -x
curl http://localhost:4000/v1/chat/completions   -X POST   -H "Content-Type: application/json"   -d '{ "model": "my-router", "messages": [  {"role": "user", "content": "What is the capital of France?" }  ]  }'
