FROM node:18-alpine
WORKDIR /app
COPY server/package.json server/package-lock.json* ./
RUN npm install --production
COPY server/ ./
COPY shards/model_config.json ./shards/
COPY shards/tokenizer_config.json ./shards/
ENV CDN_URL=https://github.com/Newton-ait/stream-llm/releases/download/v1.0.0
ENV PORT=7860
EXPOSE 7860
CMD ["node", "index.js"]
