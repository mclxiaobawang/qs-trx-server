FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY freqtrade ./freqtrade
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home trx
USER trx
EXPOSE 8088
ENTRYPOINT ["trx-server"]
CMD ["--config", "/app/config/server.json"]
