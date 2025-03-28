# 🚀 Machine Learning Course: Practice Session 1

## Environment Setup Guide

### 🌐 Internet Access Setup

#### For Students in China
![](../img/astrill.png)

If using Astrill:
- ❌ Ensure "Tunnel browsers only" is **NOT** checked
- ✅ This allows Terminal access to essential sites like Google and GitHub

#### Verify Your Connection
Test your internet connectivity with these commands:
```bash
# Test access to international sites
curl google.com
curl github.com

# Test access to local sites
curl baidu.com
```

### 📝 Essential Accounts Setup

#### GitHub Account
1. Create a GitHub account if you don't have one
2. For 2FA authentication, consider using:
   - [Google Authenticator](https://chromewebstore.google.com/detail/authenticator/bhghoamapcdpbohphigoooaddinpkbai)

#### Google Colab
Set up [Google Colab](https://colab.research.google.com/) as a backup environment:
- Runs in browser - no local installation needed
- Provides free GPU access
- Perfect fallback if local setup has issues

### 💻 Development Environment

#### Code Editor
- **Recommended**: VS Code or Cursor AI IDE
- Rich extension ecosystem for ML development
- Integrated terminal and Git support

> 💡 While PyCharm is powerful, VS Code/Cursor better aligns with our course workflow and typically presents fewer configuration issues.

#### Python Environment
We'll be using Python 3.10 with pip for package management:

```bash
# Verify your Python installation
python --version  # Should show 3.10.x

# Verify pip is working
pip --version
```
