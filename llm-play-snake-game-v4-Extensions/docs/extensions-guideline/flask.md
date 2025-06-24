# Flask Web Integration Guide

This document explains how Flask is integrated into the Snake Game AI project to provide web-based interfaces for game visualization, control, and interaction across all tasks and extensions.

## 🎯 **Flask Integration Overview**

Flask serves as the web framework for:
- **Real-time game visualization** through web browsers
- **RESTful API endpoints** for game control and state access
- **Multi-mode support** (live games, replay, human play)
- **Cross-platform compatibility** without native app requirements
- **Remote access** to running experiments and training sessions

### **Key Flask Components:**
- **Web Controllers**: Handle HTTP requests and route to appropriate handlers
- **Template Engine**: Render dynamic HTML with game state
- **API Endpoints**: Provide JSON responses for AJAX requests
- **WebSocket Support**: Real-time updates for live game streaming
- **Static Assets**: CSS, JavaScript, and image resources

## 🏗️ **Flask Architecture Pattern**

TODO: Check the ROOT/web folder. Very important. It should be the same as for Task0