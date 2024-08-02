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
