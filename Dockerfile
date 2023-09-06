# ==== FRONTEND ====
FROM node:19-alpine AS builder

WORKDIR /frontend

# Copy the package.json and install dependencies
COPY app/package*.json ./
RUN npm install

# Copy rest of the files
COPY app/src ./src/
COPY app/* ./

# Build the project
RUN npx parcel build src/index.html --no-cache --no-source-maps

# ==== BACKEND ====
FROM thexjx/openplayground

WORKDIR /web/

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV XDG_CONFIG_HOME=/web/config

RUN pip install --no-cache-dir --upgrade pip

COPY server/ ./server/
COPY --from=builder /frontend/dist ./server/static/

# install python dependencies
RUN pip install -r ./server/requirements.txt

ENTRYPOINT ["python3", "-m", "server.app", "--host", "0.0.0.0", "--env", "/web/config/.env"]