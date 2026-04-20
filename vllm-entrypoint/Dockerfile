FROM vllm/vllm-openai:latest

RUN pip install -q huggingface_hub[cli] --break-system-packages

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
