#!/bin/bash

# Use cp to copy the file to /etc/systemd/system/
sudo cp wf-dtm.service /etc/systemd/system/

# Use systemctl to enable the service. It will start on boot.
sudo systemctl enable wf-dtm.service

# Use systemctl to start the service immediately
sudo systemctl start wf-dtm.service