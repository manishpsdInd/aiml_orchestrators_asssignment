FROM nginx:latest

# Remove the default configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy the custom Nginx configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Copy web application files (index.html, CSS, JS)
COPY webapp/ /usr/share/nginx/html/

# Expose port 8080 for serving web UI
EXPOSE 8080
