# Use an official, pre-built image that contains BOTH Python and Node.js
FROM nikolaik/python-nodejs:python3.11-nodejs20-slim

# Set the working directory inside the cloud server
WORKDIR /app

# Copy your requirements files first
COPY requirements.txt package.json ./

# Force the server to install your Python machine learning libraries
RUN pip install --no-cache-dir -r requirements.txt

# Force the server to install your Node packages
RUN npm install
RUN npm install ws

# Copy the rest of your dashboard files into the server
COPY . .

# Start the dashboard
CMD ["node", "server.js"]