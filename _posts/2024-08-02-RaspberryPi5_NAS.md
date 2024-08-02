---
title: "Raspberry Pi for home NAS"
classes: wide 
categories:
  - Hardware
---

## What is NAS and why this is needed?

**Network Attached Storage (NAS)** is a storage device connected to a network that allows storage and retrieval of data from a centralized location for authorized network users and heterogeneous clients.

There are many types of data for backup. Each type has its own characteristics and requirements. 
1. Programming code
  - Git is an excellent choice for version control and backup of code. Since its sizes are not big, github repositories can handle what I need. 
2. Experimental data
  - Data generated from experiments in the Lab are saved in the lab data server and external hard drives.
  - Data that passed pre-processing processes are reconstructed as NWB format, then are saved in the internal storage in my desktop for analysis. 
3. Personal data (e.g., photos and videos)
  - Google Photos used to offer unlimited free storage for "high-quality" compressed photos and videos, especially for Pixel users. However, since June 2021, it has capped free storage at 15GB across Google Photos, Google Drive and Gmail. 

To replace "Google Photos" service (and more), I considered to make an automatic backup system from smartphones to a home NAS. 


## Implementing a Home NAS for Automatic Backup

Popular NAS brands include Synology, QNAP, and Western Digital. 

However, I thought that Raspberry Pi NAS can be a cost-effective solution. 

**Raspberry Pi** is a small, affordable single-board computer developed by the Raspberry Pi Foundation, a UK charity aimed at promoting computer science education. 
Raspberry Pi 5 offers improved processing power (Broadcom BCM2712 quad-core ARM Cortex A76 processor @ 2.4GHz), memory (up to 8GB), and connectivity options, including USB 3.0, HDMI, Wi-Fi, bluetooth, and Gigabit Ethernet. Raspberry Pi can run various operating systems, including Raspberry Pi OS, Ubuntu, and others. 

My main idea was to mount an external hard drive on the Raspberry Pi running Ubuntu 24.04, and then synchronize a folder from my phone's camera with a folder on the external hard drive.


## Parts needed

- Raspberry Pi 5
- Raspberry Pi Power Supply
- micro HDMI to HDMI cable
- micro SD card
- Raspberry Pi Active Cooler
- Raspberry Pi case


## Step-by-Step procedures
1. Set up Raspberry Pi with Ubuntu 24.04
  - Raspberry Pi Imager is the quick and easy way to install Raspberry Pi OS and other operating systems to a microSD card
  - If you download Ubuntu OS from a different source, make sure that it is made for ARM cpu.

2. Connect and mount the external hard drive

3. Install and configure Syncthing on Raspberry Pi
  - https://syncthing.net/

4. Install and configure Syncthing on smartphone
  - On Aug 1, both Sycnthing and Syncthing-fork were not available at Google Play store.
  - Check F-droid

5. Configure sync settings
  - phone and Raspberry pi should be in the same network
  - Set up one-way sync from phone (send-only) to Raspberry Pi (receive only)
  - enable "ignore delete" from both folder options

