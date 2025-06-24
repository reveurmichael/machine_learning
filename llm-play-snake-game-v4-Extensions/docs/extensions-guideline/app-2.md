# Advanced Streamlit App Development for Extensions

This document covers advanced Streamlit patterns, architectures, and best practices used in the Snake Game AI project's extension applications (v0.03+ versions).

## 🎯 **Advanced Streamlit Architecture**

### **Multi-Tab Application Pattern**
Extensions v0.03 use sophisticated tab-based interfaces to organize different functionalities:


### VITAL
TODO: MAKE SURE THIS IDEA IS SPREAD ACROSS OTHER DOC MD FILES AND DOCSTRINGS AND COMMENTS.

streamlit app.py of streamlit is only for launching scripts in the folder "scripts", with 
subprocess 

# IMPORTANT
streamlit app.py is not for visualization of game states, is not for real time showing progress, is 
not 
for showing snake moves.

It's main idea is to launch scripts in the folder "scripts" with adjustable params, with 
subprocess. That's why for extensions v0.03 we will have a folder "dashboard" in the first place.