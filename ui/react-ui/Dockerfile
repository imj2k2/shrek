FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy app files
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create empty logo files to prevent 404 errors
RUN touch /usr/share/nginx/html/logo192.png \
    && touch /usr/share/nginx/html/logo512.png

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
