# Server socket
bind = "0.0.0.0:10000"

# Worker processes
workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'

# Timeouts
timeout = 120
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'sp-crime-predictor'