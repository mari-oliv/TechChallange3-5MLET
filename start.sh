#!/bin/bash
exec gunicorn main:app -c gunicorn.conf.py